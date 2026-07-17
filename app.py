import os
import re
import json
import zipfile
import io
import uuid
import threading
import time

# 加载 .env 文件（本地开发用，生产环境直接设置系统环境变量）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import fitz  # PyMuPDF
from flask import Flask, request, jsonify, send_file, render_template
from werkzeug.utils import secure_filename
from PIL import Image as PILImage
import storage  # 统一文件存储抽象层（本地 / Cloudflare R2 双模式）

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB
# 临时文件目录（本地模式存 uploads/；云端模式存 /tmp/）
app.config['UPLOAD_FOLDER'] = storage.local_tmp_path('') if storage.is_r2_mode() else os.path.join(os.path.dirname(__file__), 'uploads')
app.config['OUTPUT_FOLDER'] = storage.local_tmp_path('') if storage.is_r2_mode() else os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# 异步任务存储: {task_id: {status, progress, total, out_path, filename, error, r2_key}}
# 用文件持久化，防止多进程/重启后丢失
_tasks = {}
_tasks_lock = threading.Lock()
_TASKS_DIR = os.path.join(os.path.dirname(__file__), 'uploads', 'tasks') if not storage.is_r2_mode() else storage.local_tmp_path('tasks')
os.makedirs(_TASKS_DIR, exist_ok=True)


def _task_path(task_id):
    return os.path.join(_TASKS_DIR, f'{task_id}.json')


def _save_task(task_id, data):
    """写入任务状态到文件（原子写：先写临时文件，再 rename，防止读到半写状态）"""
    p   = _task_path(task_id)
    tmp = p + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f)
    os.replace(tmp, p)  # os.replace 在同一文件系统上是原子操作


def _load_task(task_id):
    """从文件读取任务状态"""
    p = _task_path(task_id)
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None

ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ============================================================
# 选择题题号检测（Cambridge MCQ：Arial,Bold，左边距，粗体）
# ============================================================
def detect_mcq_questions(doc):
    """
    检测选择题 PDF 中的题号。
    特征：Arial,Bold + x≈49.6 + bold(flags&16) + size≈11
    """
    questions = []
    seen_nums = set()

    for pg_i in range(doc.page_count):
        page = doc[pg_i]
        d = page.get_text("dict")

        for block in d.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span["text"].strip()
                    if not re.match(r'^\d{1,2}$', text):
                        continue
                    q_num = int(text)
                    if not (1 <= q_num <= 99):
                        continue
                    x0 = span["bbox"][0]
                    y0 = span["bbox"][1]
                    size = span["size"]
                    flags = span["flags"]

                    is_bold = (flags & 16) != 0
                    is_left_margin = x0 < 60
                    is_right_size = 9 <= size <= 14

                    if is_bold and is_left_margin and is_right_size:
                        if q_num not in seen_nums:
                            seen_nums.add(q_num)
                            questions.append({
                                "q_num": q_num,
                                "page_idx": pg_i,
                                "y_start": y0,
                                "x_start": x0
                            })

    questions.sort(key=lambda x: x["q_num"])
    return questions


# ============================================================
# 大题题号检测（Cambridge Structured：题号在页顶，"N text..." 格式）
# ============================================================
def detect_structured_questions(doc):
    """
    检测大题（结构化问答题）PDF 中的题号。
    特征：
    - 每道大题起始行格式为 "N " 开头（N为题号），位于页面顶部
    - x≈49.6, y≈63.8（页面内容起始位置）
    - 也需处理两位数题号（如"10 (a)..."）
    - 跨多页的题目：后续页没有新题号
    """
    questions = []
    seen_nums = set()

    for pg_i in range(doc.page_count):
        page = doc[pg_i]
        d = page.get_text("dict")

        # 获取该页所有span
        page_spans = []
        for block in d.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    page_spans.append(span)

        # 方法1：检测页面顶部区域（y<100）是否有 "N (a)..." 格式的block
        blocks = page.get_text("blocks")
        for b in blocks:
            x0, y0, x1, y1, text, _, btype = b
            if btype != 0:
                continue
            text_stripped = text.strip()
            # 大题特征：在页面顶部，以"数字 "或"数字\n"开头
            # y0在63-70范围，x0≈49.6
            if abs(x0 - 49.6) < 8 and 55 <= y0 <= 80:
                # 匹配 "1 \n..." 或 "10 (a)..." 格式
                m = re.match(r'^(\d{1,2})\s*[\n\r(]', text_stripped)
                if m:
                    q_num = int(m.group(1))
                    if 1 <= q_num <= 99 and q_num not in seen_nums:
                        seen_nums.add(q_num)
                        questions.append({
                            "q_num": q_num,
                            "page_idx": pg_i,
                            "y_start": y0,
                            "x_start": x0
                        })
                        break  # 每页最多一个大题题号

        # 方法2：检测span级别（处理题号单独成span的情况）
        for span in page_spans:
            text = span["text"].strip()
            if not re.match(r'^\d{1,2}$', text):
                continue
            q_num = int(text)
            if not (1 <= q_num <= 99):
                continue
            x0 = span["bbox"][0]
            y0 = span["bbox"][1]
            size = span["size"]

            # 大题题号：非粗体，x≈49.6，y≈63.8，size≈11
            is_left = abs(x0 - 49.6) < 8
            is_top = abs(y0 - 63.8) < 8
            is_right_size = 9 <= size <= 14

            if is_left and is_top and is_right_size and q_num not in seen_nums:
                seen_nums.add(q_num)
                questions.append({
                    "q_num": q_num,
                    "page_idx": pg_i,
                    "y_start": y0,
                    "x_start": x0
                })

    questions.sort(key=lambda x: x["q_num"])
    return questions


# ============================================================
# Edexcel 题号检测（支持 A-Level / IGCSE）
# ============================================================
def detect_edexcel_questions(doc):
    """
    检测 Edexcel 格式试卷（A-Level / IGCSE / IAL）的题号。

    Edexcel 特征：
    - Section A（MCQ）：题号格式 "N\\t题目文字"，x≈43，整个题目在同一 block
    - Section B（大题）：题号格式 "N\\t题目文字"，x≈43，多行 block
    - 页眉/页脚：页码在底部 y>790，"DO NOT WRITE" 为背景竖排文字
    - 子题号 (a)(b)(c) 以 x≈43 开头，格式 "(a)\\t" 或 "(a) "
    - 选项：A/B/C/D 开头，x≈70-80

    与 Cambridge 区别：
    - Cambridge：题号是独立 bold span，格式 "N " 后跟内容
    - Edexcel：题号和内容在同一 block，"N\\t文字" tab分隔
    """
    questions = []
    seen_nums = set()

    for pg_i in range(doc.page_count):
        page = doc[pg_i]
        blocks = page.get_text('blocks')

        for b in blocks:
            x0, y0, x1, y1, txt, bno, btype = b
            if btype != 0:
                continue
            txt_stripped = txt.strip()

            # 跳过页眉/页脚/背景文字
            if y0 > 790:   # 页码区域
                continue
            if 'DO NOT WRITE' in txt_stripped:
                continue
            if txt_stripped.startswith('*P') or txt_stripped.startswith('Turn over'):
                continue
            # Edexcel 页码行（单独数字在右侧）
            if re.match(r'^\d{1,2}$', txt_stripped) and x0 > 500:
                continue

            # 核心：Edexcel 题号格式 "N\t文字" 或 "N\n文字"
            # x0 约在 40-55 范围内
            if x0 < 60:
                m = re.match(r'^(\d{1,2})[\t\n\r]\s*\S', txt_stripped)
                if m:
                    q_num = int(m.group(1))
                    if 1 <= q_num <= 99 and q_num not in seen_nums:
                        seen_nums.add(q_num)
                        questions.append({
                            'q_num':    q_num,
                            'page_idx': pg_i,
                            'y_start':  y0,
                            'x_start':  x0,
                        })

    questions.sort(key=lambda q: q['q_num'])
    return questions


# ============================================================
# 自动识别试卷来源（Cambridge / Edexcel / Edexcel Maths）
# ============================================================
# Edexcel Maths 试卷代码 → unit 映射
_EDEXCEL_MATHS_CODE_MAP = {
    'WMA11': 'P1', 'WMA12': 'P2', 'WMA13': 'P3', 'WMA14': 'P4',
    'WFM01': 'FP1', 'WFM02': 'FP2', 'WFM03': 'FP3',
    'WMS01': 'S1',  'WMS02': 'S2',
    'WME01': 'M1',  'WME02': 'M2',
    'WDM01': 'D1',
}


def detect_edexcel_maths_unit(doc) -> str:
    """
    从封面（前2页）识别 Edexcel IAL 数学试卷的具体单元。
    返回：'P1'|'P2'|'P3'|'P4'|'FP1'|'FP2'|'FP3'|'S1'|'S2'|'M1'|'M2'|'unknown'

    识别优先级：
      1. 试卷代码 WMA13/01 → P3（最精确）
      2. 明确文字 "Pure Mathematics P3" → P3
      3. 孤立文字 "P3" → P3（兜底）
    """
    for pg_i in range(min(2, doc.page_count)):
        text = doc[pg_i].get_text()

        # 优先：试卷代码（最准确）
        m = re.search(r'(WMA\d{2}|WFM\d{2}|WMS\d{2}|WME\d{2}|WDM\d{2})', text)
        if m:
            code = m.group(1)
            unit = _EDEXCEL_MATHS_CODE_MAP.get(code)
            if unit:
                return unit

        # 次优：明确文字描述
        m2 = re.search(r'Pure Mathematics\s+P([1-4])', text)
        if m2:
            return f'P{m2.group(1)}'

        m3 = re.search(r'Further Pure Mathematics\s+(\d)', text)
        if m3:
            return f'FP{m3.group(1)}'

        m4 = re.search(r'Statistics\s+S([12])', text)
        if m4:
            return f'S{m4.group(1)}'

        m5 = re.search(r'Mechanics\s+M([12])', text)
        if m5:
            return f'M{m5.group(1)}'

        # 兜底：孤立 P1/P2/P3/P4 标识
        m6 = re.search(r'\bPure Mathematics\b.*?\bP([1-4])\b', text, re.DOTALL)
        if m6:
            return f'P{m6.group(1)}'

    return 'unknown'


def detect_paper_source(doc) -> str:
    """
    返回 'cambridge'、'edexcel' 或 'edexcel_maths'。
    通过封面/前几页文字关键词判断。
    Edexcel Maths (IAL Pure/Further Math) 优先在 edexcel 之前检测。
    """
    for pg_i in range(min(3, doc.page_count)):
        text = doc[pg_i].get_text()
        # 优先检测 Edexcel Maths：WMA/WFM/WPM 系列纯数试卷
        if re.search(r'WMA\d{2}/\d{2}|WFM\d{2}/\d{2}|WPM\d{2}/\d{2}', text):
            return 'edexcel_maths'
        if ('Pure Mathematics' in text or 'Further Mathematics' in text) and \
           ('Pearson' in text or 'Edexcel' in text):
            if re.search(r'P[1-4]|Unit [1-4]|Pure Math', text):
                return 'edexcel_maths'

    keywords_edexcel = ['Pearson', 'Edexcel', 'GCSE', 'IAL', 'International Advanced',
                        'WPH', 'WBI', 'WCH', 'WMA', 'Total for Question']
    keywords_cambridge = ['UCLES', 'Cambridge', 'Cambridge Assessment',
                          'Cambridge International', 'CIE']

    for pg_i in range(min(3, doc.page_count)):
        text = doc[pg_i].get_text()
        for kw in keywords_edexcel:
            if kw in text:
                return 'edexcel'
        for kw in keywords_cambridge:
            if kw in text:
                return 'cambridge'
    return 'cambridge'  # 默认 Cambridge


# ============================================================
# Edexcel Maths (P3-style) 题目检测
# 每道题占1页；题目页含 (N) 分值标记；空白答题页跳过
# ============================================================
def detect_edexcel_maths_questions(doc):
    """
    检测 Edexcel IAL Pure Mathematics (WMA13/P3 类) 试卷的题号边界。

    题号块格式（三种）：
      1. 'N.\t题干文字'  — tab 分隔，如 '1.\tA curve has equation'
      2. 'N.  题干文字'  — 双空格，如 '7.  y'（含图的题）
      3. 'N.'           — 独立块（前后有公式图块）

    位置特征：
      - x0 ≈ 42.5，范围 35~65
      - y0 通常 55~160（顶部区域）
      - 续页：'Question N continued' 不作为题号

    每题只取题目页（含题干+marks），不包含答题续页。
    返回 questions 列表，每项含 q_num / page_idx / y_start / x_start。
    """
    # 匹配三种题号格式
    Q_PAT_ALONE  = re.compile(r'^(\d{1,2})\.\s*$')          # '2.'
    Q_PAT_TAB    = re.compile(r'^(\d{1,2})\.\t')             # '1.\tFind'
    Q_PAT_SPACE  = re.compile(r'^(\d{1,2})\.\s{2,}')        # '7.  y'（2个以上空格）
    CONTINUED_PAT = re.compile(r'^Question\s+\d+\s+continued', re.IGNORECASE)

    questions = []
    seen_nums = set()

    for pg_i in range(1, doc.page_count):   # 跳过封面（第0页）
        page = doc[pg_i]
        blocks = page.get_text('blocks')
        for b in blocks:
            x0, y0, x1, y1, txt, bno, btype = b
            if btype != 0:
                continue
            ts = txt.strip()
            if not ts:
                continue
            # 排除续页行
            if CONTINUED_PAT.match(ts):
                break   # 续页标记是第一个内容块，无需继续扫描本页
            # 匹配题号
            m = (Q_PAT_ALONE.match(ts) or
                 Q_PAT_TAB.match(ts) or
                 Q_PAT_SPACE.match(ts))
            if not m:
                continue
            # 位置过滤：x 在题号列，y 在页面顶部区域
            if not (35 <= x0 <= 65 and 45 <= y0 <= 180):
                continue
            q_num = int(m.group(1))
            if not (1 <= q_num <= 30):
                continue
            if q_num in seen_nums:
                continue
            seen_nums.add(q_num)
            questions.append({
                'q_num':    q_num,
                'page_idx': pg_i,
                'y_start':  y0,
                'x_start':  x0,
            })
            break  # 每页只取第一个题号块

    questions.sort(key=lambda x: x['q_num'])
    return questions


# ============================================================
# 自动检测题型（兼容 Cambridge + Edexcel + Edexcel Maths）
# ============================================================
def detect_paper_type(doc):
    """
    自动判断试卷类型，返回：
    - 'mcq'           : Cambridge 纯选择题
    - 'structured'    : Cambridge 大题
    - 'edexcel_mcq'   : Edexcel 纯选择题（仅含Section A选择题）
    - 'edexcel'       : Edexcel 大题 / 混合题型
    - 'edexcel_maths' : Edexcel IAL Pure/Further Math (P1–P4)

    Edexcel 判断逻辑（改进版）：
      1. 如果发现 SECTION B 或 Section B → edexcel（混合卷）
      2. 如果发现 SECTION A 或 Section A 且有MCQ选项(A/B/C/D) → edexcel_mcq
      3. 否则检查题目内容：若题目块中包含 (a)/(b)/(c) 子题 → edexcel（纯大题）
      4. 再检查是否有 MCQ 选项格式 → edexcel_mcq
      5. 默认 → edexcel（保守，避免漏识别大题）
    """
    source = detect_paper_source(doc)

    # Edexcel Maths (IAL Pure/Further) — 直接返回专用类型
    if source == 'edexcel_maths':
        return 'edexcel_maths'

    if source == 'edexcel':
        full_text = '\n'.join(doc[pg_i].get_text() for pg_i in range(doc.page_count))

        # 优先判断：明确有 Section B → 混合卷，含大题
        if 'SECTION B' in full_text or 'Section B' in full_text:
            return 'edexcel'

        # 检查是否有子题格式 (a) (b) (c)，这是大题的标志
        # Edexcel 大题：block 以 "(a)\t" 开头，x≈42
        sub_q_count = 0
        mcq_option_count = 0
        for pg_i in range(doc.page_count):
            page = doc[pg_i]
            blocks = page.get_text('blocks')
            for b in blocks:
                x0, y0, x1, y1, txt, bno, btype = b
                if btype != 0 or y0 > 790:
                    continue
                txt_s = txt.strip()
                # 子题标志：(a)/(b)/(c) 开头，x≈42
                if x0 < 65 and re.match(r'^\([a-e]\)[\t\s]', txt_s):
                    sub_q_count += 1
                # MCQ 选项标志：A/B/C/D 单独一行，x=60-90
                if 60 <= x0 <= 95 and re.match(r'^[ABCD]\s+\S', txt_s):
                    mcq_option_count += 1

        # 有子题格式 → 大题卷
        if sub_q_count >= 3:
            return 'edexcel'
        # 有大量MCQ选项 → 选择题卷
        if mcq_option_count >= 8:
            return 'edexcel_mcq'
        # 默认保守处理：Edexcel 未知格式归入大题
        return 'edexcel'

    # Cambridge：原有逻辑
    for pg_i in range(min(2, doc.page_count)):
        text = doc[pg_i].get_text()
        if 'Multiple Choice' in text:
            return 'mcq'
        if 'Structured Questions' in text or 'structured' in text.lower():
            return 'structured'

    mcq_count = 0
    for pg_i in range(min(5, doc.page_count)):
        page = doc[pg_i]
        d = page.get_text('dict')
        for block in d.get('blocks', []):
            if block.get('type') != 0:
                continue
            for line in block.get('lines', []):
                for span in line.get('spans', []):
                    text = span['text'].strip()
                    if re.match(r'^\d{1,2}$', text):
                        if (span['flags'] & 16) != 0 and span['bbox'][0] < 60:
                            mcq_count += 1
    return 'mcq' if mcq_count > 3 else 'structured'


# ============================================================
# ============================================================
# 辅助：找页面有效内容的底部 y 坐标
# ============================================================
def _find_content_bottom(page, ph, margin_bottom=30):
    """
    找到页面上最后一个有效内容块的 y 坐标（底部），
    排除页脚（页码、版权等）。
    若找不到内容，返回 ph - margin_bottom。
    """
    best_bottom = ph - margin_bottom
    try:
        blocks = page.get_text("dict").get("blocks", [])
        # 找最后一个文本/图片块的 y1（排除页面最底部的小块，即页脚区）
        footer_zone = ph - 50  # 最后50pt认为是页脚区
        for b in blocks:
            by1 = b.get("bbox", [0,0,0,0])[3]
            if by1 < footer_zone and by1 > best_bottom:
                best_bottom = by1
        # 也考虑图片块
        for img in page.get_images(full=False):
            rects = page.get_image_rects(img[0])
            for r in rects:
                if r.y1 < footer_zone and r.y1 > best_bottom:
                    best_bottom = r.y1
    except Exception:
        pass
    return min(best_bottom + 10, ph - 5)


# 图片切割核心（通用，支持选择题和大题）
# ============================================================
def crop_question_image(doc, questions, q_idx, dpi=150, paper_type='mcq'):
    """
    截取单道题目的图像，支持跨页合并。

    选择题：题目在页面中间，按y坐标裁剪
    大题：题目从当前页顶部到下一题所在页顶部（整页截取）
    Edexcel Maths：只取题目页，截到最后一个 marks 下方（不含答题横线）
    """
    from PIL import Image

    scale = dpi / 72.0
    mat = fitz.Matrix(scale, scale)

    q = questions[q_idx]
    pg_start = q["page_idx"]

    # ── Edexcel Maths 专用：单页截到 marks，不含答题横线 ──
    if paper_type == 'edexcel_maths':
        page = doc[pg_start]
        pw, ph = page.rect.width, page.rect.height
        left  = 42   # 跳过左边框线（x=35~36.5），内容起始 x≈42.5
        right = min(pw - 36, 560)
        _right_lim = _detect_right_content_limit(page, pw, ph,
                                                  sample_y0=44, sample_y1=ph - 40)
        if _right_lim < right:
            right = _right_lim
        y_start = q.get("y_start", 0)
        top    = max(0, y_start - 8) if y_start > 10 else 48
        bottom = _find_edexcel_maths_question_bottom(page, ph)
        if bottom <= top + 10:
            bottom = ph - 25
        clip = fitz.Rect(left, top, right, bottom)
        pix  = page.get_pixmap(matrix=mat, clip=clip, colorspace=fitz.csRGB)
        data = pix.tobytes("png")
        w, h = pix.width, pix.height
        del pix
        return data, w, h
    y_top = q["y_start"] - 8  # 稍微往上留一点空间

    # 确定结束位置
    if q_idx + 1 < len(questions):
        next_q = questions[q_idx + 1]
        pg_end = next_q["page_idx"]
        y_end = next_q["y_start"] - 8
    else:
        # 最后一题：找到最后一个有实际内容的页面
        pg_end = _find_last_content_page(doc, pg_start)
        y_end = None  # 截到页尾（会在循环里处理）

    page_imgs = []

    for pg_i in range(pg_start, pg_end + 1):
        page = doc[pg_i]
        pw = page.rect.width
        ph = page.rect.height

        # 计算裁剪区域
        left = 30
        right = pw - 15

        # Edexcel Maths：检测右侧空白答题列并裁剪
        if paper_type in ('edexcel_maths', 'maths'):
            _right_lim = _detect_right_content_limit(
                page, pw, ph,
                sample_y0=(y_top if pg_i == pg_start else 55),
                sample_y1=ph
            )
            if _right_lim < pw - 15:
                right = _right_lim

        if pg_i == pg_start and pg_i == pg_end:
            # 同页
            top = max(0, y_top)
            if y_end is not None:
                bottom = min(ph, y_end)
            else:
                bottom = _find_content_bottom(page, ph)
        elif pg_i == pg_start:
            # 首页：从题号到页尾内容
            top = max(0, y_top)
            bottom = _find_content_bottom(page, ph)
        elif pg_i == pg_end:
            # 末页：从页顶内容区到下一题位置
            top = 55  # 跳过页眉
            if y_end is not None:
                bottom = min(ph, y_end)
            else:
                bottom = _find_content_bottom(page, ph)
        else:
            # 中间整页
            top = 55
            bottom = _find_content_bottom(page, ph)

        if bottom <= top + 10:
            continue

        clip = fitz.Rect(left, top, right, bottom)
        pix = page.get_pixmap(matrix=mat, clip=clip, colorspace=fitz.csRGB)
        page_imgs.append(pix)

    if not page_imgs:
        # fallback：截整页
        page = doc[pg_start]
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        return pix.tobytes("png"), pix.width, pix.height

    if len(page_imgs) == 1:
        return page_imgs[0].tobytes("png"), page_imgs[0].width, page_imgs[0].height

    # 多页垂直拼接
    imgs = []
    for pix in page_imgs:
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        imgs.append(img)

    # 分页之间加一条细分隔线
    sep_h = 3
    total_h = sum(im.height for im in imgs) + sep_h * (len(imgs) - 1)
    max_w = max(im.width for im in imgs)
    merged = Image.new("RGB", (max_w, total_h), (255, 255, 255))

    y_offset = 0
    for i, im in enumerate(imgs):
        merged.paste(im, (0, y_offset))
        y_offset += im.height
        if i < len(imgs) - 1:
            # 画分页线
            for py in range(y_offset, y_offset + sep_h):
                for px in range(max_w):
                    merged.putpixel((px, py), (220, 220, 220))
            y_offset += sep_h

    buf = io.BytesIO()
    merged.save(buf, format="PNG")
    return buf.getvalue(), merged.width, merged.height


def _detect_left_content_start(page, pw, ph, scan_up_to=90):
    """
    动态检测页面左侧装饰竖线/色条的右边界，返回内容区左起点 x（pt）。

    Edexcel 试卷常见格式：
      - 灰色内容区边框线 (x≈35pt, w=2pt) —— drawing 路径
      - 蓝色/彩色左侧括号竖线（某些版本，x≈20-65pt）—— drawing 路径
      - Edexcel P3/Maths 竖线分隔符 "|" 字符 (x≈64pt) —— 文字字符
      - Cambridge 试卷：无左侧装饰条，left=30 即可

    策略（优先级从高到低）：
      1. 扫描 drawing 路径，找最右的非黑彩色 "高竖线"（高度>50pt，x1<scan_up_to）
      2. 扫描左侧文字单词，检测竖线字符（"|"、"│" 等）的位置
      3. 扫描文字块左边界：若题号文字 x0 > 50pt，说明有较宽装饰区
      4. 默认返回 36（保守值，仅跳过 Edexcel 标准灰框线）

    返回：建议的 left x 值（pt），不小于 30，不大于 scan_up_to
    """
    best_left = 36  # 默认值

    try:
        # ── 策略1：通过 drawing 路径检测左侧装饰线 ──
        paths = page.get_drawings()
        for p in paths:
            rect = p.get('rect')
            if rect is None:
                continue
            x0, y0, x1, y1 = float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3])
            h = abs(y1 - y0)
            w_rect = abs(x1 - x0)

            # 只关注左侧区域的细长竖线（或宽色条）
            if x1 > scan_up_to:
                continue

            color = p.get('color') or p.get('fill')
            if color is None:
                continue

            # 排除无效颜色：纯黑（文字墨水）、纯白（背景框）、纯灰（Edexcel 灰色边框）
            r, g, b = (color[0] if len(color) > 0 else 0,
                       color[1] if len(color) > 1 else 0,
                       color[2] if len(color) > 2 else 0)
            # 纯黑
            if r < 0.1 and g < 0.1 and b < 0.1:
                continue
            # 近白色（背景/高光区域）
            if r > 0.9 and g > 0.9 and b > 0.9:
                continue
            # 近灰色（Edexcel 灰色边框线，r≈g≈b 且无明显彩色偏）
            max_ch = max(r, g, b)
            min_ch = min(r, g, b)
            if max_ch > 0 and (max_ch - min_ch) / max_ch < 0.15:
                # 饱和度极低 → 灰色系，跳过
                continue

            # 高度足够的元素（竖线/色条 h>50，或细高矩形宽<30 高>20）
            if h > 50 or (w_rect < 30 and h > 20):
                # 取该元素右边界+安全边距
                candidate = x1 + 3
                if candidate > best_left:
                    best_left = candidate

        # ── 策略2：检测左侧文字竖线字符（Edexcel P3/Maths 格式）──
        # 这类 PDF 用"|"字符作为左侧分隔竖线（x≈64pt），drawing 检测不到
        _VLINE_CHARS = {'|', '│', '┃', '‖', '｜'}
        words = page.get_text("words")
        for w in words:
            x0, y0, x1, y1, text = w[0], w[1], w[2], w[3], w[4]
            if x1 > scan_up_to:
                continue
            text_stripped = text.strip()
            if text_stripped in _VLINE_CHARS:
                # 竖线字符右边界+安全边距
                candidate = x1 + 4
                if candidate > best_left:
                    best_left = candidate

        # ── 策略3：扫描文字块左边界（内容区文字一般从 x>55 开始）──
        # 只做辅助验证，不覆盖策略1/2的结果
        if best_left <= 36:
            for w in words:
                x0_w = w[0]
                if 40 < x0_w < 120:
                    # 文字最左边界通常就是题号的 x，装饰线在这之左
                    if x0_w > 55:
                        # 题号文字从 >55pt 开始，说明有较宽的左侧装饰
                        candidate = x0_w - 2
                        if candidate > best_left:
                            best_left = candidate
                    break  # 只取最左边的一个文字块

    except Exception:
        pass

    # 限制范围：不小于30，不大于 scan_up_to
    return max(30, min(int(best_left) + 1, scan_up_to))


def _detect_content_image_bounds(page, pw, ph, min_img_width=200):
    """
    检测页面中嵌入的大型内容图片的边界框（用于 edexcel_maths 格式）。

    这类 PDF 将题目内容作为整页图片嵌入，图片居中放置：
      - x0 ≈ 167pt（内容图片左边界）
      - x1 ≈ 428pt（内容图片右边界）
      - y0 ≈ 74pt（内容图片顶部，蓝色页眉条下方）

    返回：(left, top, right) 或 None（若未检测到大图）
    """
    best = None
    best_area = 0
    for img in page.get_images(full=True):
        xref = img[0]
        native_w, native_h = img[2], img[3]
        # 跳过小图（logo、图标等）
        if native_w < min_img_width or native_h < 200:
            continue
        rects = page.get_image_rects(xref)
        for r in rects:
            area = (r.x1 - r.x0) * (r.y1 - r.y0)
            if area > best_area:
                best_area = area
                best = r
    if best is None:
        return None
    # 返回图片边界（稍微留 2pt 内边距）
    return (max(0, best.x0 - 2), max(0, best.y0), min(pw, best.x1 + 2))


def _detect_right_content_limit(page, pw, ph, sample_y0=None, sample_y1=None):
    """
    检测页面右侧是否存在空白答题列（Edexcel 题目右侧 'DO NOT WRITE' 区域）。
    策略（优先级从高到低）：
      1. 通过 'DO NOT WRITE IN THIS AREA' 文字块的左边界精确定位
      2. 通过纵向线条（竖线）的位置定位
      3. 若右侧 35% 无有效内容，固定裁至 68% 宽度
    返回：有效内容右边界 x（pt），默认 pw-15（不裁剪）
    """
    if sample_y0 is None: sample_y0 = 0
    if sample_y1 is None: sample_y1 = ph

    blocks = page.get_text('blocks')

    # ── 策略 1：找 'DO NOT WRITE' 文字块左边界 ──
    dnw_xs = []
    for b in blocks:
        bx0, by0, bx1, by1, txt, _, btype = b
        if btype != 0:
            continue
        if by1 < sample_y0 or by0 > sample_y1:
            continue
        if 'DO NOT WRITE' in txt.upper():
            # 这类文字通常是竖排在右侧空白列内，取其左边界
            if bx0 > pw * 0.50:   # 只取右半边的
                dnw_xs.append(bx0)

    if dnw_xs:
        # 取最左的 DO NOT WRITE 文字左边界，再往左 4pt
        candidate = min(dnw_xs) - 4
        if pw * 0.45 < candidate < pw - 20:
            return candidate

    # ── 策略 2：通过纵向线条位置 ──
    drawings = page.get_drawings()
    vertical_lines = []
    for d in drawings:
        for item in d.get('items', []):
            if item[0] == 'l':   # line segment
                p1, p2 = item[1], item[2]
                dx = abs(p2.x - p1.x)
                dy = abs(p2.y - p1.y)
                # 真正的竖线：高度 > 80pt，水平偏移 < 3pt
                if dy > 80 and dx < 3:
                    mid_x = (p1.x + p2.x) / 2
                    if mid_x > pw * 0.50:
                        vertical_lines.append(mid_x)
            elif item[0] == 're':  # rect — 有些实现用细矩形代替线
                r = item[1]
                rw = abs(r.x1 - r.x0)
                rh = abs(r.y1 - r.y0)
                if rh > 80 and rw < 4:  # 细高矩形 → 竖线
                    mid_x = (r.x0 + r.x1) / 2
                    if mid_x > pw * 0.50:
                        vertical_lines.append(mid_x)

    if vertical_lines:
        line_x = min(vertical_lines) - 4
        if pw * 0.45 < line_x < pw - 20:
            return line_x

    # ── 策略 3：右侧 35% 无有效内容则固定裁剪 ──
    SKIP_PATS = re.compile(
        r'DO NOT WRITE|BLANK PAGE|Turn over|©|UCLES|^\d{1,4}$|9702/',
        re.IGNORECASE
    )
    right_zone_x = pw * 0.65
    has_real_content = False
    for b in blocks:
        bx0, by0, bx1, by1, txt, _, btype = b
        if btype != 0:
            continue
        if by1 < sample_y0 or by0 > sample_y1:
            continue
        txt_clean = txt.strip()
        if not txt_clean or SKIP_PATS.search(txt_clean):
            continue
        if bx0 > right_zone_x:
            has_real_content = True
            break

    if not has_real_content:
        # 右侧无有效内容 → 裁到 70% 宽度
        return pw * 0.70

    return pw - 15


def _find_edexcel_maths_question_bottom(page, page_height):
    """
    Edexcel Maths 题目页专用：只截取题干，不含答题横线。

    策略：
      1. 在右侧（x > 430）找所有 marks 标记 (N) 的位置
      2. 取最后一个 marks 的 y1 作为截剪下边界（marks 后立即是答题横线）
      3. 加 8pt padding，确保括号完整显示
      4. 如果找不到 marks（题目页无分值），退回到找最后一个非横线内容块底部

    注意：只查找 x>430 的 marks，避免误匹配题干中的数学表达式 (N)。
    """
    blocks = page.get_text('blocks')
    # 右侧 marks：格式为单独的 '(N)' 块，x > 430，页脚区(y > ph-60)排除
    MARKS_PAT = re.compile(r'^\(\d+\)$')     # 精确匹配 '(3)' '(10)' 等

    marks_y1 = None
    for b in blocks:
        x0, y0, x1, y1, txt, bno, btype = b
        if btype != 0:
            continue
        ts = txt.strip()
        # 右侧 marks：x>430（避免误匹配题干内的括号）且不在页脚
        if x0 > 430 and y0 < page_height - 60 and MARKS_PAT.match(ts):
            marks_y1 = y1   # 取最后一个（持续更新）

    if marks_y1 is not None:
        # marks y1 + 8pt padding（保留括号完整显示空间）
        return min(marks_y1 + 8, page_height - 25)

    # 退回方案：找最后一个非横线、非 DO NOT WRITE、非页码的内容块底部
    last_y = 0
    for b in blocks:
        x0, y0, x1, y1, txt, bno, btype = b
        if btype != 0:
            continue
        ts = txt.strip()
        if not ts:
            continue
        if 'DO NOT WRITE' in ts:
            continue
        if y0 > page_height - 60:   # 页脚区域
            continue
        if re.match(r'^\d{1,3}$', ts):
            continue
        # 跳过答题横线（全是下划线）
        clean = ts.replace(' ', '').replace('\t', '').replace('\n', '')
        if clean and all(c == '_' for c in clean):
            continue
        last_y = max(last_y, y1)

    return min(last_y + 8, page_height - 25) if last_y > 50 else page_height - 25


def _find_last_content_page(doc, start_page):
    """
    从start_page开始，找最后一个有题目内容的页面。
    跳过空白页、版权页等。
    """
    last_content_page = start_page

    for pg_i in range(start_page, doc.page_count):
        page = doc[pg_i]
        text = page.get_text().strip()

        # 空白页
        if not text or 'BLANK PAGE' in text:
            break

        # 版权/结尾页
        if 'Permission to reproduce' in text:
            break

        # 页面有实际内容
        blocks = page.get_text("blocks")
        has_content = any(
            len(b[4].strip()) > 20 and
            b[1] < page.rect.height - 50 and
            '© UCLES' not in b[4] and
            '9702/' not in b[4]
            for b in blocks
        )

        if has_content:
            last_content_page = pg_i
        else:
            break

    return last_content_page


# ===================== API 路由 =====================

# ── 知识库文件路径 ──
_SYLLABUS_PATH = os.path.join(os.path.dirname(__file__), 'static', 'syllabus_9702.json')

def _load_syllabus():
    if os.path.exists(_SYLLABUS_PATH):
        with open(_SYLLABUS_PATH, encoding='utf-8') as f:
            return json.load(f)
    return None


# ═══════════════════════════════════════════════════════════
#  知识点匹配引擎
#  策略：为每个 L2 subtopic 配置关键词组，
#        从题目文本中 TF 加权计分，返回所有得分 >0 的 subtopic
# ═══════════════════════════════════════════════════════════

# 每条规则: (subtopic_id, [必须命中的关键词组], [加分关键词])
# 格式: (id, required_any, bonus_any)
# required_any: 只要命中其中一个即得基础分
# bonus_any:    额外命中加分（用于区分相似 topic）
_TOPIC_RULES = [
    # 1.1 Physical quantities
    ('1.1', ['physical quantity','base quantity','derived quantity','homogeneous','homogeneity',
             'scalar','vector','unit of'], []),
    # 1.2 SI units
    ('1.2', ['SI unit','SI base unit','base unit','kilogram','mole','candela',
             'kg m','m s','kg m2','fundamental unit'], ['unit']),
    # 1.3 Errors and uncertainties
    ('1.3', ['uncertainty','percentage uncertainty','error','random error','systematic error',
             'precision','accuracy','significant figure'], []),
    # 1.4 Scalars and vectors
    ('1.4', ['scalar','vector','resultant','component','resolve','resolution of','displacement',
             'velocity','acceleration','force','addition of vector'], []),

    # 2.1 Equations of motion / kinematics
    ('2.1', ['equation of motion','suvat','uniform acceleration','constant acceleration',
             'initial velocity','final velocity','distance travelled','displacement–time',
             'velocity–time','speed–time','acceleration–time','free fall','terminal velocity',
             'projectile','horizontal projection','v = u','v² = u²','s = ut'], []),

    # 3.1 Momentum and Newton's laws
    ('3.1', ["newton's law","newton's first","newton's second","newton's third",
             'momentum','rate of change of momentum','force and acceleration','F = ma',
             'net force','resultant force'], []),
    # 3.2 Non-uniform motion
    ('3.2', ['drag','air resistance','terminal','non-uniform','viscous','stokes',
             'velocity increases','reaches terminal'], []),
    # 3.3 Linear momentum conservation
    ('3.3', ['conservation of momentum','elastic collision','inelastic collision',
             'momentum conserved','collide','collision','explosion'], []),

    # 4.1 Turning effects / moments
    ('4.1', ['moment','torque','turning effect','principle of moment','couple',
             'pivot','lever','clockwise','anticlockwise'], []),
    # 4.2 Equilibrium
    ('4.2', ['equilibrium','net torque','zero resultant','in equilibrium',
             'centre of gravity','centre of mass','balanced'], []),
    # 4.3 Density and pressure
    ('4.3', ['density','pressure','ρ','upthrust','buoyancy','archimedes',
             'pascal','fluid pressure','P = ρgh','barometer'], []),

    # 5.1 Energy conservation
    ('5.1', ['conservation of energy','energy transfer','work done','power',
             'efficiency','W = Fd','work–energy theorem'], []),
    # 5.2 GPE and KE
    ('5.2', ['gravitational potential energy','kinetic energy','GPE','KE',
             'mgh','½mv²','interchange of energy','energy stored'], []),

    # 6.1 Stress and strain
    ('6.1', ['stress','strain','Young modulus','Young\'s modulus','tensile',
             'extension','cross-section','load–extension','force–extension'], []),
    # 6.2 Elastic and plastic
    ('6.2', ['elastic','plastic','deformation','Hooke','limit of proportionality',
             'elastic limit','permanent deformation','spring constant','k ='], []),

    # 7.1 Progressive waves
    ('7.1', ['progressive wave','amplitude','frequency','wavelength','period',
             'phase','wave speed','v = fλ','transverse','longitudinal',
             'wave equation'], []),
    # 7.2 Transverse and longitudinal
    ('7.2', ['transverse wave','longitudinal wave','compression','rarefaction',
             'displacement of particle','direction of propagation'], []),
    # 7.3 Doppler effect
    ('7.3', ['Doppler','doppler','doppler effect','observed frequency','source moving',
             'apparent frequency','blue shift','red shift'], []),
    # 7.4 Electromagnetic spectrum
    ('7.4', ['electromagnetic spectrum','EM spectrum','electromagnetic wave',
             'gamma ray','X-ray','ultraviolet','infrared','microwave','radio wave',
             'visible light','speed of light'], []),
    # 7.5 Polarisation
    ('7.5', ['polarisation','polarization','polarised','polarizer','Malus',
             'plane of polarisation','partially polarised'], []),

    # 8.1 Stationary waves
    ('8.1', ['stationary wave','standing wave','node','antinode','fundamental',
             'harmonic','resonance','string vibration','pipe open','pipe closed'], []),
    # 8.2 Diffraction
    ('8.2', ['diffraction','diffract','single slit','spread of wave',
             'aperture','obstacle'], []),
    # 8.3 Interference
    ('8.3', ['interference','superposition','constructive','destructive',
             'path difference','coherent','fringe','Young\'s double slit',
             'double slit','two-source'], []),
    # 8.4 Diffraction grating
    ('8.4', ['diffraction grating','grating','order of diffraction','d sin θ',
             'first order','second order','grating spacing'], []),

    # 9.1 Electric current
    ('9.1', ['electric current','charge','coulomb','I = Q/t','current flow',
             'conventional current','electron flow','drift velocity','charge carrier'], []),
    # 9.2 Potential difference and power
    ('9.2', ['potential difference','e.m.f','electromotive force','voltage','voltmeter',
             'electrical power','P = IV','P = I²R','V = IR','P = V²/R'], []),
    # 9.3 Resistance and resistivity
    ('9.3', ['resistance','resistivity','ohm','ohmic','R = ρl/A','cross-sectional area',
             'length of wire','ρ =','temperature coefficient'], []),

    # 10.1 Practical circuits
    ('10.1', ['internal resistance','terminal p.d.','lost volt','battery','cell',
              'ammeter','voltmeter placement','circuit diagram','series circuit',
              'parallel circuit'], []),
    # 10.2 Kirchhoff's laws
    ('10.2', ["kirchhoff",'Kirchhoff','sum of current','sum of e.m.f',
              'loop rule','junction rule'], []),
    # 10.3 Potential dividers
    ('10.3', ['potential divider','potentiometer','voltage divider','LDR','thermistor',
              'output voltage','ratio of resistance'], []),

    # 11.1 Atoms, nuclei and radiation
    ('11.1', ['nucleus','nucleon','proton','neutron','electron','atomic number',
              'mass number','nuclide','radioactive','α particle','β particle',
              'γ radiation','alpha','beta','gamma','isotope','nuclear'], []),
    # 11.2 Fundamental particles
    ('11.2', ['quark','lepton','hadron','meson','baryon','antiparticle',
              'neutrino','positron','fundamental particle'], []),

    # 12.1 Kinematics of uniform circular motion
    ('12.1', ['circular motion','angular velocity','angular speed','ω','radian',
              'angular frequency','period of rotation','revolutions per'], []),
    # 12.2 Centripetal acceleration
    ('12.2', ['centripetal','centripetal acceleration','centripetal force',
              'v²/r','rω²','directed towards centre'], []),

    # 13.1 Gravitational field
    ('13.1', ['gravitational field','field line','field strength','g =','free fall',
              'gravitational acceleration','weight'], []),
    # 13.2 Gravitational force between point masses
    ('13.2', ["newton's law of gravitation","law of gravitation",'gravitational force',
              'F = Gm1m2','inverse square law for gravity'], []),
    # 13.3 Gravitational field of a point mass
    ('13.3', ['gravitational field strength','GM/r²','point mass','radial field'], []),
    # 13.4 Gravitational potential
    ('13.4', ['gravitational potential','–GM/r','escape velocity','potential energy',
              'potential well','equipotential'], []),

    # 14.1 Thermal equilibrium
    ('14.1', ['thermal equilibrium','zeroth law','thermal energy','thermodynamics'], []),
    # 14.2 Temperature scales
    ('14.2', ['temperature scale','Celsius','kelvin','absolute temperature',
              'thermometric property','thermometer'], []),
    # 14.3 Specific heat capacity and latent heat
    ('14.3', ['specific heat capacity','specific latent heat','latent heat',
              'Q = mcΔT','boiling point','melting point','vaporisation','fusion'], []),

    # 15.1 The mole
    ('15.1', ['mole','Avogadro','number of molecules','amount of substance',
              'molar mass','N = nNA'], []),
    # 15.2 Equation of state
    ('15.2', ['equation of state','ideal gas','pV = nRT','gas law',
              'Boyle','Charles','pV/T','pressure–volume'], []),
    # 15.3 Kinetic theory
    ('15.3', ['kinetic theory','mean square speed','root mean square','r.m.s',
              '<c²>','molecular speed','Maxwell'], []),

    # 16.1 Internal energy
    ('16.1', ['internal energy','random kinetic energy of molecules',
              'increase in internal energy','thermal store'], []),
    # 16.2 First law of thermodynamics
    ('16.2', ['first law of thermodynamics','ΔU = q + w','heat supplied',
              'work done on gas','isothermal','adiabatic'], []),

    # 17.1 Simple harmonic motion
    ('17.1', ['simple harmonic','SHM','a = –ω²x','acceleration proportional',
              'sinusoidal','oscillation','restoring force'], []),
    # 17.2 Energy in SHM
    ('17.2', ['energy in SHM','potential energy in SHM','kinetic energy in SHM',
              'total energy in oscillation','½mω²A²'], []),
    # 17.3 Damping and resonance
    ('17.3', ['damping','damped','forced oscillation','resonance','driving frequency',
              'natural frequency','amplitude at resonance'], []),

    # 18.1 Electric fields
    ('18.1', ['electric field','field line','electric force','field strength',
              'E =','charge in electric field'], []),
    # 18.2 Uniform electric fields
    ('18.2', ['uniform electric field','parallel plate','E = V/d',
              'capacitor plate','electric field between'], []),
    # 18.3 Electric force between charges
    ('18.3', ["coulomb's law",'Coulomb','F = kq1q2','electric force between charges',
              'inverse square law for charge'], []),
    # 18.4 Electric field of a point charge
    ('18.4', ['electric field of point charge','E = kQ/r²','radial electric field',
              'point charge field'], []),
    # 18.5 Electric potential
    ('18.5', ['electric potential','V = kQ/r','potential at a point',
              'work done moving charge'], []),

    # 19.1 Capacitors
    ('19.1', ['capacitor','capacitance','C = Q/V','farad','parallel plate capacitor',
              'dielectric','charge stored'], []),
    # 19.2 Energy in capacitor
    ('19.2', ['energy stored in capacitor','½CV²','½QV','energy on capacitor'], []),
    # 19.3 Discharging capacitor
    ('19.3', ['discharging capacitor','charging capacitor','time constant','RC',
              'exponential decay','Q = Q0e','V = V0e'], []),

    # 20.1 Magnetic field
    ('20.1', ['magnetic field','magnetic flux density','tesla','flux density',
              'field line','magnetic field line','B ='], []),
    # 20.2 Force on current-carrying conductor
    ('20.2', ['force on conductor','F = BIl','current-carrying conductor',
              'Fleming','left-hand rule'], []),
    # 20.3 Force on moving charge
    ('20.3', ['force on moving charge','F = Bqv','charged particle in magnetic',
              'hall effect','cyclotron','circular path in magnetic'], []),
    # 20.4 Magnetic fields due to currents
    ('20.4', ['magnetic field due to current','solenoid','toroid','coaxial',
              'field inside solenoid','μ0','permeability'], []),
    # 20.5 Electromagnetic induction
    ('20.5', ['electromagnetic induction','faraday','lenz','induced e.m.f',
              'induced emf','flux linkage','magnetic flux','Φ =','rate of change of flux',
              'ε = –dΦ/dt'], []),

    # 21.1 Alternating currents
    ('21.1', ['alternating current','AC','root mean square','r.m.s value',
              'peak value','Irms','Vrms','I0','V0'], []),
    # 21.2 Rectification and smoothing
    ('21.2', ['rectification','rectifier','smoothing','diode','half-wave',
              'full-wave','smoothing capacitor'], []),

    # 22.1 Photon energy and momentum
    ('22.1', ['photon','E = hf','h = ','planck','momentum of photon',
              'p = h/λ','energy of photon'], []),
    # 22.2 Photoelectric effect
    ('22.2', ['photoelectric','work function','threshold frequency',
              'stopping potential','photoelectron','Ek = hf – φ'], []),
    # 22.3 Wave-particle duality
    ('22.3', ['wave-particle duality','de Broglie','electron diffraction',
              'λ = h/p','matter wave','duality'], []),
    # 22.4 Energy levels and spectra
    ('22.4', ['energy level','line spectrum','emission spectrum','absorption spectrum',
              'atomic spectrum','photon emission','transition between levels',
              'ground state','excited state'], []),

    # 23.1 Nuclear binding energy
    ('23.1', ['binding energy','mass defect','nuclear binding','E = mc²',
              'mass–energy','fission','fusion','binding energy per nucleon'], []),
    # 23.2 Radioactive decay
    ('23.2', ['radioactive decay','half-life','decay constant','activity',
              'A = λN','N = N0e','exponential decay of activity',
              'radioactive','random decay'], []),

    # 24.1 Ultrasound
    ('24.1', ['ultrasound','piezoelectric','acoustic impedance','pulse-echo',
              'ultrasonic','A-scan','B-scan'], []),
    # 24.2 X-rays
    ('24.2', ['X-ray','CT scan','attenuation','exponential attenuation',
              'I = I0e–μx','radiograph','contrast'], []),
    # 24.3 PET scanning
    ('24.3', ['PET','positron emission tomography','annihilation','tracer',
              'gamma camera','positron-electron'], []),

    # 25.1 Standard candles
    ('25.1', ['standard candle','Cepheid','luminosity','apparent magnitude',
              'absolute magnitude','distance modulus'], []),
    # 25.2 Stellar radii
    ('25.2', ['Stefan–Boltzmann','Wien','stellar radius','black body',
              'luminosity = ','surface temperature of star'], []),
    # 25.3 Hubble and Big Bang
    ('25.3', ['Hubble','Big Bang','recession speed','redshift','cosmic microwave',
              'age of universe','H0','Hubble constant'], []),
]

# 预编译：小写关键词 → subtopic_id 映射
_TOPIC_KW_INDEX: dict[str, list[str]] = {}
for _sid, _req, _bon in _TOPIC_RULES:
    for _kw in _req + _bon:
        _TOPIC_KW_INDEX.setdefault(_kw.lower(), []).append(_sid)


def _extract_question_text(doc: fitz.Document, questions: list, q_idx: int,
                            paper_type: str) -> str:
    """提取一道题的所有文字（所有 slice 拼接）"""
    slices = _collect_question_slices(doc, questions, q_idx, paper_type)
    parts = []
    for src_page, clip in slices:
        parts.append(src_page.get_text('text', clip=clip))
    return ' '.join(parts)


def _extract_marks_hint(text: str, unit_filter: str = None) -> dict:
    """
    从题目文字中提取 (marks) 标注，如 "(3)" "(5 marks)" 等，
    统计各章节关键词在各段落的出现情况，结合该段落分值生成权重字典。
    返回: {chapter_id: weight_float}，weight 表示该章节对应分值占比。
    若无法解析分值，返回空字典（tag_question_topics 中 weight 默认为 1.0）。
    """
    # 匹配 (n) 或 (n marks) 形式的分值标注
    mark_pattern = re.compile(r'\((\d+)(?:\s*marks?)?\)', re.IGNORECASE)
    marks = [int(m) for m in mark_pattern.findall(text)]
    if not marks:
        return {}

    # 按括号分值把文本分段，统计每段的关键词命中
    # 简化策略：对整道题按关键词命中量，乘以各部分分值权重
    # 找出每个分值段的文字片段（按 (...) 分割）
    segments = mark_pattern.split(text)
    # segments 格式: [text_before_1st_mark, mark1_val, text_after_mark1, mark2_val, ...]

    prefix = (unit_filter + '-') if unit_filter else ''
    rules = [(sid, req, bon) for sid, req, bon in _MATHS_CHAPTER_RULES
             if not prefix or sid.startswith(prefix)]

    chapter_marks: dict[str, float] = {}
    total_marks_assigned = 0

    i = 0
    seg_texts = []
    seg_marks_list = []
    # segments 交替为文字和数字（来自 split）
    j = 0
    while j < len(segments):
        seg_text = segments[j]
        j += 1
        seg_mark = 0
        if j < len(segments):
            try:
                seg_mark = int(segments[j])
            except (ValueError, TypeError):
                pass
            j += 1
        if seg_mark > 0:
            seg_texts.append(seg_text)
            seg_marks_list.append(seg_mark)
            total_marks_assigned += seg_mark

    if not total_marks_assigned:
        return {}

    for seg_text, seg_mark in zip(seg_texts, seg_marks_list):
        seg_lower = seg_text.lower()
        for sid, req_kws, bon_kws in rules:
            hit = any(kw.lower() in seg_lower for kw in req_kws)
            hit = hit or any(kw.lower() in seg_lower for kw in bon_kws)
            if hit:
                chapter_marks[sid] = chapter_marks.get(sid, 0) + seg_mark

    if not chapter_marks:
        return {}

    # 归一化为权重（最高分分给最高 weight=2.0，其余按比例）
    max_m = max(chapter_marks.values())
    return {sid: 1.0 + (m / max_m) for sid, m in chapter_marks.items()}


def tag_question_topics(text: str, syllabus_type: str = 'cambridge',
                         unit_filter: str = None,
                         marks_hint: dict = None) -> list[dict]:
    """
    输入题目文本，返回匹配到的知识点列表。
    每项: {'id': 'P3-2', 'title': 'Trigonometry', 'score': 4}
    按 score 降序，只返回 score >= 1 的项（至少返回1项 fallback）。

    syllabus_type: 'cambridge' | 'edexcel_maths'
    unit_filter:   若指定（如 'P3'），只使用该 unit 的规则（仅对 edexcel_maths 有效）
    marks_hint:    {chapter_id: marks_weight} 分值权重提示，得分乘以权重

    Edexcel Maths 模式：使用章节级别规则（_MATHS_CHAPTER_RULES），
    返回一级标题 ID，如 'P3-2'（Trigonometry），不细化到子章节。
    保证：即使关键词未命中，也会返回分值最高章节作为 fallback。
    """
    text_lower = text.lower()
    scores: dict[str, int] = {}

    if syllabus_type == 'cambridge':
        rules = _TOPIC_RULES
    else:
        # Edexcel Maths：使用章节级别规则
        if unit_filter:
            prefix = unit_filter + '-'   # 'P3-'
            rules = [(sid, req, bon) for sid, req, bon in _MATHS_CHAPTER_RULES
                     if sid.startswith(prefix)]
        else:
            rules = _MATHS_CHAPTER_RULES

    def _kw_match(kw: str, text: str) -> bool:
        """关键词匹配：包含 .* 时用 regex，否则用字符串包含匹配。"""
        kw_l = kw.lower()
        if '.*' in kw_l or kw_l.startswith('^') or kw_l.endswith('$'):
            try:
                return bool(re.search(kw_l, text))
            except re.error:
                return kw_l in text
        return kw_l in text

    for sid, req_kws, bon_kws in rules:
        score = 0
        for kw in req_kws:
            if _kw_match(kw, text_lower):
                score += 2
        for kw in bon_kws:
            if _kw_match(kw, text_lower):
                score += 1
        if score > 0:
            # 按分值权重加成（marks_hint 里的权重越高，得分越高）
            weight = (marks_hint or {}).get(sid, 1.0)
            scores[sid] = score * weight

    # 加载考纲标题映射
    if syllabus_type == 'edexcel_maths':
        syllabus = _load_edexcel_maths_syllabus()
        # Edexcel Maths：从章节条目（topics 列表顶层）构建 title_map
        title_map: dict[str, str] = {}
        if syllabus:
            for t in syllabus['topics']:
                tid = str(t['id'])
                title_map[tid] = t.get('title', tid)

        if not scores:
            # ── fallback：无命中时，对所有规则做宽松单词匹配，取分最高 ──
            fallback_scores: dict[str, int] = {}
            words = set(re.findall(r'[a-z]{3,}', text_lower))
            for sid, req_kws, bon_kws in rules:
                sc = 0
                for kw in req_kws:
                    # 每个规则关键词分词后做单词级别匹配
                    for word in kw.lower().split():
                        if len(word) >= 4 and word in words:
                            sc += 1
                if sc > 0:
                    fallback_scores[sid] = sc
            if fallback_scores:
                # 取分值最高者
                best_sid = max(fallback_scores, key=lambda k: fallback_scores[k])
                scores = {best_sid: fallback_scores[best_sid]}
            else:
                # 完全兜底：取该 unit 第一章（微积分/代数等）
                if rules:
                    scores = {rules[0][0]: 1}

        result = []
        for sid, sc in sorted(scores.items(), key=lambda x: -x[1]):
            result.append({
                'id':    sid,
                'title': title_map.get(sid, sid),
                'score': sc,
            })
        # 最多返回 3 个（头栏空间有限；score 已降序）
        return result[:3]
    else:
        if not scores:
            return []
        syllabus = _load_syllabus()
        title_map: dict[str, str] = {}
        parent_map: dict[str, str] = {}
        if syllabus:
            for t in syllabus['topics']:
                tid = str(t['id'])
                for s in t.get('subtopics', []):
                    title_map[s['id']] = s['title']
                    parent_map[s['id']] = tid
        result = []
        for sid, sc in sorted(scores.items(), key=lambda x: -x[1]):
            result.append({
                'id':       sid,
                'title':    title_map.get(sid, sid),
                'parent_id': parent_map.get(sid, '0'),
                'score':    sc,
            })
        return result


# ──────────────────────────────────────────────────────────────
# Edexcel Maths (P1–P4) 章节级别匹配规则（一级标题）
# ──────────────────────────────────────────────────────────────
_MATHS_CHAPTER_RULES = [
    # ── P1 ──
    ('P1-1', ['algebra','indices','surds','quadratic','discriminant',
              'completing the square','inequality','polynomial',
              'binomial expansion','remainder theorem','factor theorem',
              'rationalise','irrational'], []),
    ('P1-2', ['coordinate geometry','straight line','gradient','midpoint',
              'circle','equation of circle','tangent to circle'], []),
    ('P1-3', ['arithmetic sequence','geometric sequence','series','sigma notation',
              'common difference','common ratio','sum to infinity'], []),
    ('P1-4', ['sine rule','cosine rule','radian','arc length','sector area',
              'trigonometric identity','trigonometric equation',
              'solve sin','solve cos','solve tan'], []),
    ('P1-5', ['exponential','logarithm','log','ln','natural log',
              'laws of logarithms'], []),
    ('P1-6', ['differentiation','derivative','dy/dx','stationary point',
              'tangent','normal','turning point','second derivative',
              'rate of change'], []),
    ('P1-7', ['integration','integral','area under','definite integral',
              'indefinite integral','antiderivative'], []),
    # ── P2 ──
    ('P2-1', ['proof','prove','disprove','counterexample','contradiction'], []),
    # ── P3 ──
    # P3-1  Algebraic Methods
    ('P3-1', ['partial fraction','algebraic fraction','long division','algebraic division',
              'improper fraction','remainder','quotient','denominator','numerator',
              'express as partial fractions','decompose'], []),
    # P3-2  Functions and Graphs
    ('P3-2', ['modulus','|f(x)|','f(|x|)','domain','range','inverse function',
              'f⁻¹','composite function','fog','gof','fg(x)','gf(x)',
              'absolute value','mapping','one-to-one','onto',
              'sketch the graph','transformation','stretch','translation',
              'solving modulus','modulus equation','modulus inequality'], []),
    # P3-3  Trigonometric Functions  (sec / cosec / cot / arcsin / arccos / arctan)
    ('P3-3', ['sec','cosec','cot','secant','cosecant','cotangent',
              'sec²','cot²','cosec²',
              '1+cot','1+tan','sec x','cosec x','cot x',
              'arcsin','arccos','arctan','arcsec','arccosec','arccot',
              'inverse trig','inverse sine','inverse cosine','inverse tangent',
              'trigonometric identity'], []),
    # P3-4  Trigonometric Addition Formulae  (compound / double angle / R-form)
    ('P3-4', ['double angle','compound angle','addition formula',
              'sin(A+B)','cos(A+B)','tan(A+B)','sin(A-B)','cos(A-B)',
              'sin2A','cos2A','tan2A','sin 2','cos 2',
              'R sin','R cos','harmonic form',
              'acos x + bsin x','a cos x + b sin x',
              'addition formulae','angle formulae','half angle',
              'sin A cos B','cos A sin B'], []),
    # P3-5  Exponentials and Logarithms
    ('P3-5', ['exponential','natural logarithm','ln','log','log₁₀','log10',
              'e^x','e^(ax','exponential model','exponential growth','exponential decay',
              'laws of logarithms','log equation','ln equation',
              'non-linear data','y = ab^x','y = ax^n',
              'equation of the model'], []),
    # P3-6  Differentiation
    ('P3-6', ['chain rule','product rule','quotient rule',
              'implicit differentiation','implicit','parametric differentiation',
              'parametric','differential equation','dy/dx','dy / dx',
              'rate of change','dx/dt','dy/dt','in terms of t',
              'differentiating','derivative','second derivative','d²y/dx²',
              'stationary point','turning point',
              'differentiate','d/dx','tangent','normal to the curve',
              'sin x differentiat','cos x differentiat',
              'ln x differentiat','e^x differentiat'], []),
    # P3-7  Integration
    ('P3-7', ['integrat','integral','area under','definite integral',
              'indefinite integral','antiderivative',
              'integration by substitution','integration by parts',
              'trapezium rule','volume of revolution',
              'standard integral','exact area','area of region',
              'let u =','u substitution','by parts',
              'using the identity','trig identity integrat',
              'reverse chain rule','area between',
              'using identit','cos2x','sin2x','sec2','cosec2'], []),
    # P3-8  Numerical Methods
    ('P3-8', ['iteration','iterative','change of sign','sign change',
              'decimal search','interval bisection','newton-raphson',
              'root','locating root','fixed point','convergence',
              'staircase diagram','cobweb diagram',
              'f(a)f(b) < 0','f(x) = 0'], []),
    # ── P4 ──
    ('P4-1', ['proof by contradiction'], []),
    ('P4-2', ['partial fraction','binomial series','(1+x)^n','general binomial',
              'ellipse','hyperbola','parabola','conic section'], []),
    ('P4-3', ['parametric equation','cartesian form'], []),
    ('P4-4', ['binomial expansion for any n','binomial series'], []),
    ('P4-5', ['implicit differentiation','parametric differentiation',
              'connected rates','higher derivative','d²y/dx²'], []),
    ('P4-6', ['volume of revolution','integration by parts',
              'integration by substitution','separable differential equation',
              'reduction formula','arc length'], []),
    ('P4-7', ['vector','scalar product','dot product','equation of plane',
              'direction vector','position vector','unit vector'], []),
]

# 向后兼容别名（旧代码中引用了 _MATHS_TOPIC_RULES 的地方不会报错）
_MATHS_TOPIC_RULES = _MATHS_CHAPTER_RULES

# ─────────────────────────────────────────
# Edexcel Maths 考纲加载（缓存）
# ─────────────────────────────────────────
_edexcel_maths_syllabus_cache = None

def _load_edexcel_maths_syllabus():
    global _edexcel_maths_syllabus_cache
    if _edexcel_maths_syllabus_cache is not None:
        return _edexcel_maths_syllabus_cache
    path = os.path.join(os.path.dirname(__file__), 'static', 'syllabus_edexcel_maths.json')
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            _edexcel_maths_syllabus_cache = json.load(f)
        return _edexcel_maths_syllabus_cache
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════
#  WMA13 (P3) 题目难度数据库
#  来源：2022 / 2023 / 2024 Examiner Reports
#
#  评分依据（关键词映射）：
#    ★     — "straightforward" / "very well done" / "nearly all"
#    ★★    — "familiar" / "accessible" / "well done" / "majority"
#    ★★★   — "mixed" / "some difficulty" / "common errors" / "challenging parts"
#    ★★★★  — "discriminating" / "quite challenging" / "less than quarter"
#    ★★★★★ — "most challenging" / "modal score 0" / "quarter scored 0" / "rarely completed"
#
#  key = (year: int, q_num: int)  —— 对应试卷年份和题号
# ═══════════════════════════════════════════════════════════════════
_P3_DIFFICULTY_DB = {
    # ── 2022 ──
    (2022, 1):  3,   # "considerable number left blank", Q1不会 — 偏难开局
    (2022, 2):  3,   # domain/range issues common, range misunderstood
    (2022, 3):  2,   # "accessible to most", product rule well known
    (2022, 4):  3,   # (a) accessible; (b) "discriminating", diff+logs combined
    (2022, 5):  3,   # method marks easy, accuracy marks more demanding
    (2022, 6):  4,   # "quite challenging", modal score = full marks (~25%)
    (2022, 7):  3,   # "more accessible", but (d) "most challenging part"
    (2022, 8):  4,   # "challenging", modal score 0 (16%), transformations
    (2022, 9):  5,   # "most challenging", modal score 0 (~25%), trig proof

    # ── 2023 ──
    (2023, 1):  2,   # "familiar start", well done, iterative procedure
    (2023, 2):  2,   # "widely accessible", 75%+ scored 7/8 marks
    (2023, 3):  4,   # "more challenging than expected", many couldn't connect parts
    (2023, 4):  3,   # context Q, part (c) most commonly dropped
    (2023, 5):  3,   # "good access if could get started", ~25% scored 0/1
    (2023, 6):  3,   # "good access" but q-value context very discriminating
    (2023, 7):  3,   # "accessible to vast majority" but wide spread, part (c) harder
    (2023, 8):  4,   # "widespread across all marks", <25% full marks, trig proof
    (2023, 9):  3,   # "generally answered very well", part (c) discriminating (~10% last mark)
    (2023, 10): 5,   # "rarely completed", ~20% scored 0, final 3 marks "very discriminating"

    # ── 2024 ──
    (2024, 1):  2,   # "appropriate start", "vast majority" correct
    (2024, 2):  3,   # "more demanding than expected", dx/dy vs dy/dx confusion
    (2024, 3):  4,   # "rather more challenging than expected", modulus; working steps penalised
    (2024, 4):  3,   # "generally well attempted", part (c) "far more discriminating"
    (2024, 5):  3,   # "answered well by majority" in (a); (b) "only small proportion" full marks
    (2024, 6):  2,   # "very well done", functions; domain mark often missed
    (2024, 7):  3,   # product rule well known; last two factorised marks "more discriminating"
    (2024, 8):  4,   # "proved challenging", exponential context; parts (c)(d) demanding
    (2024, 9):  4,   # quotient rule OK; part (c) "more of a challenge"; integration errors
}

# 各 year 内所有题目的综合平均 → 用于估算未在数据库中的题目
_P3_YEAR_AVG = {
    2022: sum(v for (y, q), v in _P3_DIFFICULTY_DB.items() if y == 2022) /
          sum(1 for (y, q) in _P3_DIFFICULTY_DB if y == 2022),
    2023: sum(v for (y, q), v in _P3_DIFFICULTY_DB.items() if y == 2023) /
          sum(1 for (y, q) in _P3_DIFFICULTY_DB if y == 2023),
    2024: sum(v for (y, q), v in _P3_DIFFICULTY_DB.items() if y == 2024) /
          sum(1 for (y, q) in _P3_DIFFICULTY_DB if y == 2024),
}

# Q-num → 跨年平均（position-based 估算）
_P3_QNUM_AVG = {}
for _qn in range(1, 11):
    _vals = [v for (y, q), v in _P3_DIFFICULTY_DB.items() if q == _qn]
    if _vals:
        _P3_QNUM_AVG[_qn] = sum(_vals) / len(_vals)


def rate_question_difficulty(q_num: int, year: int | None, source: str,
                              maths_unit: str | None) -> int | None:
    """
    返回题目难度星级 1–5，或 None（非 WMA13/P3 时）。

    匹配优先级：
      1. 精确匹配 (year, q_num)
      2. 同 q_num 跨年平均（四舍五入）
      3. 同年平均（兜底）
      4. 全局 P3 平均 ≈ 3
    """
    if source != 'edexcel_maths' or maths_unit != 'P3':
        return None

    # 1. 精确匹配
    if year and (year, q_num) in _P3_DIFFICULTY_DB:
        return _P3_DIFFICULTY_DB[(year, q_num)]

    # 2. 跨年 q_num 平均
    if q_num in _P3_QNUM_AVG:
        return max(1, min(5, round(_P3_QNUM_AVG[q_num])))

    # 3. 同年平均
    if year and year in _P3_YEAR_AVG:
        return max(1, min(5, round(_P3_YEAR_AVG[year])))

    # 4. 全局兜底
    return 3


def _extract_year_from_filename(filename: str) -> int | None:
    """从文件名中提取4位年份数字，如 'WMA13_October2023.pdf' → 2023"""
    if not filename:
        return None
    m = re.search(r'(20\d{2})', filename)
    return int(m.group(1)) if m else None


# 月份名称列表（全写和缩写）
_MONTH_NAMES = [
    'January','February','March','April','May','June',
    'July','August','September','October','November','December',
    'Jan','Feb','Mar','Apr','Jun','Jul','Aug','Sep','Oct','Nov','Dec',
]

def _extract_exam_date_label(doc, filename: str = '') -> str:
    """
    从PDF首页文字中提取考试日期标签，如 "October 2023"。
    策略：
    1. 在首页（前2页）文本中查找 "Monday/Tuesday/.../Sunday DD MonthName YYYY" 格式
    2. 也匹配 "MonthName YYYY" 简化格式
    3. 如从PDF提取失败，从文件名中提取年份+月份
    返回如 "October 2023"，失败返回 ''
    """
    months_pattern = '|'.join(_MONTH_NAMES)
    # 完整日期：Wednesday 18 October 2023
    full_date_re = re.compile(
        r'(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s+\d{1,2}\s+'
        r'(' + months_pattern + r')\s+(20\d{2})',
        re.IGNORECASE
    )
    # 简化日期：October 2023
    short_date_re = re.compile(
        r'\b(' + months_pattern + r')\s+(20\d{2})\b',
        re.IGNORECASE
    )

    for pg_i in range(min(2, doc.page_count)):
        text = doc[pg_i].get_text()
        m = full_date_re.search(text)
        if m:
            # 规范化月份为全写
            month_raw = m.group(1)
            year = m.group(2)
            return f'{month_raw.capitalize()} {year}'
        m2 = short_date_re.search(text)
        if m2:
            return f'{m2.group(1).capitalize()} {m2.group(2)}'

    # 从文件名中兜底提取
    if filename:
        # 文件名如 Questionpaper-Unit3WMA13-October2024.pdf
        fn_m = re.search(r'(' + months_pattern + r')[-_ ]?(20\d{2})', filename, re.IGNORECASE)
        if fn_m:
            return f'{fn_m.group(1).capitalize()} {fn_m.group(2)}'
        # 只有年份
        yr_m = re.search(r'(20\d{2})', filename)
        if yr_m:
            return yr_m.group(1)

    return ''

_multi_sessions = {}
_multi_sessions_lock = threading.Lock()
# R2 模式下 PDF 临时存在 /tmp/pdf_uploads/本地模式存在 uploads/multi/
_MULTI_DIR = storage.local_tmp_path('') if storage.is_r2_mode() else os.path.join(os.path.dirname(__file__), 'uploads', 'multi')
if not storage.is_r2_mode():
    os.makedirs(_MULTI_DIR, exist_ok=True)


def _session_key(session_id):
    """R2 Key 或本地 key，供 storage 模块使用"""
    return f'sessions/{session_id}.json'


def _save_session_to_disk(session_id, groups):
    """
    将 session 元数据持久化（R2 或本地）。
    不存储图片数据。
    """
    try:
        storage.store_json(_session_key(session_id), groups)
    except Exception:
        pass


def _load_session_from_disk(session_id):
    """
    从存储（R2 或本地）恢复 session 元数据。
    如果对应 PDF 文件不可访问，返回 None。
    """
    try:
        groups = storage.load_json(_session_key(session_id))
        if groups is None:
            return None
        # 检查 PDF 可用性：R2 模式检查 r2_key；本地模式检查文件存在
        for g in groups:
            if storage.is_r2_mode():
                r2_key = g.get('r2_key', '')
                if r2_key and not storage.exists(r2_key):
                    return None  # PDF 已从 R2 删除
            else:
                if not os.path.exists(g.get('path', '')):
                    return None  # PDF 本地文件不存在
        return groups
    except Exception:
        return None


def _get_session(session_id):
    """
    获取 session 数据：先查内存，内存没有则从存储恢复。
    R2 模式：恢复时若本地临时文件不存在，自动从 R2 下载 PDF。
    恢复成功后写回内存缓存。
    """
    if not session_id:
        return None
    with _multi_sessions_lock:
        sess = _multi_sessions.get(session_id)
        if sess is not None:
            return sess
    # 内存中没有，尝试从存储恢复
    sess = _load_session_from_disk(session_id)
    if sess is not None:
        # R2 模式：确保本地临时文件存在（供 fitz.open 使用）
        if storage.is_r2_mode():
            for g in sess:
                r2_key   = g.get('r2_key', '')
                tmp_path = g.get('path', '')
                if r2_key and tmp_path and not os.path.isfile(tmp_path):
                    storage.make_local_copy(r2_key, tmp_path)
        with _multi_sessions_lock:
            _multi_sessions[session_id] = sess
    return sess


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/syllabus', methods=['GET'])
def get_syllabus():
    """返回 9702 知识库目录"""
    data = _load_syllabus()
    if data is None:
        return jsonify({'error': '知识库未加载'}), 404
    return jsonify(data)


@app.route('/api/syllabus/edexcel_maths', methods=['GET'])
def get_edexcel_maths_syllabus():
    """
    返回 Edexcel IAL Pure Mathematics 知识库目录。
    可选 ?unit=P3 过滤，只返回该 unit 的 topics。
    """
    data = _load_edexcel_maths_syllabus()
    if data is None:
        return jsonify({'error': 'Edexcel Maths 知识库未加载'}), 404

    unit = request.args.get('unit')
    if unit:
        import copy
        filtered = copy.deepcopy(data)
        filtered['topics'] = [t for t in filtered['topics']
                               if t.get('unit') == unit]
        filtered['active_unit'] = unit
        return jsonify(filtered)

    return jsonify(data)


@app.route('/api/upload_multi', methods=['POST'])
def upload_multi():
    """
    多文件上传接口。
    支持同时上传多个PDF，自动逐一检测题型和考试局，按文件分组返回。
    前端传 files[] 数组（multipart），可附带 paper_type 覆盖。
    返回: {session_id, groups: [{source, paper_type, filename, total_questions, questions}]}
    """
    files = request.files.getlist('files[]')
    if not files:
        # 兼容单文件 'file' 字段
        f = request.files.get('file')
        if f:
            files = [f]
    if not files:
        return jsonify({'error': '没有文件'}), 400

    paper_type_hint = request.form.get('paper_type', 'auto')
    session_id = str(uuid.uuid4())
    groups = []

    for file in files:
        if not file.filename or not allowed_file(file.filename):
            continue
        safe = secure_filename(file.filename)
        # 先保存到本地临时目录（fitz 需要本地文件）
        tmp_path  = storage.local_tmp_path(f'{session_id}_{safe}')
        file.save(tmp_path)
        # 同时上传到 R2（R2 模式）或本地 uploads/multi/（本地模式）
        r2_key    = f'multi/{session_id}_{safe}'
        storage.upload_from_local(tmp_path, r2_key)
        save_path = tmp_path  # fitz 始终用本地临时文件

        try:
            doc = fitz.open(save_path)
            if paper_type_hint == 'auto':
                pt = detect_paper_type(doc)
            else:
                pt = paper_type_hint

            source = detect_paper_source(doc)
            questions = _detect_questions(doc, pt)

            # ── 检测具体 unit（Edexcel Maths 专用）──
            maths_unit = None
            if source == 'edexcel_maths':
                maths_unit = detect_edexcel_maths_unit(doc)

            # ── 从文件名提取年份（用于难度评级）──
            paper_year = _extract_year_from_filename(file.filename)

            # ── 从PDF内容或文件名提取考试日期标签 ──
            exam_date_label = _extract_exam_date_label(doc, file.filename)

            # ── 知识点标注 + 难度评级 ──
            if source == 'cambridge':
                for q_idx, q in enumerate(questions):
                    try:
                        txt = _extract_question_text(doc, questions, q_idx, pt)
                        q['topics'] = tag_question_topics(txt, 'cambridge')
                    except Exception:
                        q['topics'] = []
                    q['difficulty'] = None  # Cambridge 暂无难度数据
            elif source == 'edexcel_maths':
                # 只用该 unit 的规则（若 unit 已知）
                for q_idx, q in enumerate(questions):
                    try:
                        txt = _extract_question_text(doc, questions, q_idx, pt)
                        # 从题目文字中提取分值分布，用于权重加成
                        marks_hint = _extract_marks_hint(txt, unit_filter=maths_unit)
                        q['topics'] = tag_question_topics(txt, 'edexcel_maths',
                                                          unit_filter=maths_unit,
                                                          marks_hint=marks_hint)
                    except Exception:
                        q['topics'] = []
                    # 难度评级（1–5，仅 P3 有数据）
                    q_num = q.get('q_num') or (q_idx + 1)
                    q['difficulty'] = rate_question_difficulty(
                        q_num=q_num,
                        year=paper_year,
                        source=source,
                        maths_unit=maths_unit
                    )
            else:
                for q in questions:
                    q['topics'] = []
                    q['difficulty'] = None

            # ── 把考试日期写入每道题（供导出头栏使用）──
            for q in questions:
                q['exam_date'] = exam_date_label

            doc.close()

            groups.append({
                'filename':       file.filename,
                'path':           save_path,    # 本地临时路径（fitz 使用）
                'r2_key':         r2_key,        # 存储 key（R2 模式用于持久化）
                'source':         source,
                'paper_type':     pt,
                'maths_unit':     maths_unit,   # 'P1'|'P2'|'P3'|'P4'|None
                'exam_date':      exam_date_label,  # e.g. "October 2023" or ""
                'questions':      questions,
                'total_questions': len(questions),
                'total_pages':    fitz.open(save_path).page_count
            })
        except Exception as e:
            groups.append({
                'filename': file.filename,
                'path':     save_path,
                'r2_key':   r2_key,
                'source':   'unknown',
                'paper_type': 'unknown',
                'maths_unit': None,
                'questions': [],
                'total_questions': 0,
                'error': str(e)
            })

    if not groups:
        return jsonify({'error': '没有有效的PDF文件'}), 400

    # 持久化 session（内存 + 存储，服务重启后可恢复）
    with _multi_sessions_lock:
        _multi_sessions[session_id] = groups
    _save_session_to_disk(session_id, groups)

    # 兼容旧单文件接口（current.pdf / paper_type.txt），仅本地模式保留
    if groups and not storage.is_r2_mode():
        last = groups[-1]
        import shutil
        try:
            shutil.copy(last['path'], os.path.join(app.config['UPLOAD_FOLDER'], 'current.pdf'))
            with open(os.path.join(app.config['UPLOAD_FOLDER'], 'paper_type.txt'), 'w') as f:
                f.write(last['paper_type'])
        except Exception:
            pass

    return jsonify({
        'session_id': session_id,
        'groups': [{
            'filename':        g['filename'],
            'source':          g['source'],
            'paper_type':      g['paper_type'],
            'maths_unit':      g.get('maths_unit'),   # 'P1'|'P2'|'P3'|'P4'|None
            'exam_date':       g.get('exam_date', ''),  # e.g. "October 2023"
            'questions':       g['questions'],
            'total_questions': g['total_questions'],
            'total_pages':     g.get('total_pages', 0),
            'error':           g.get('error')
        } for g in groups]
    })


@app.route('/api/upload', methods=['POST'])
def upload_pdf():
    """单文件上传兼容接口（内部转发给 upload_multi）"""
    return upload_multi()


def _get_paper_type():
    """读取当前PDF的题型"""
    pt_file = os.path.join(app.config['UPLOAD_FOLDER'], 'paper_type.txt')
    if os.path.exists(pt_file):
        with open(pt_file) as f:
            return f.read().strip()
    return 'mcq'


def _detect_questions(doc, paper_type):
    """统一题号检测入口，兼容所有格式"""
    if paper_type == 'mcq':
        return detect_mcq_questions(doc)
    elif paper_type in ('edexcel', 'edexcel_mcq'):
        return detect_edexcel_questions(doc)
    elif paper_type == 'edexcel_maths':
        return detect_edexcel_maths_questions(doc)
    else:
        return detect_structured_questions(doc)


@app.route('/api/preview/<int:q_num>', methods=['GET'])
def preview_question(q_num):
    """预览单题图片，支持 ?session_id=&file_idx= 参数"""
    dpi      = int(request.args.get('dpi', 150))
    sess_id  = request.args.get('session_id')
    file_idx = int(request.args.get('file_idx', 0))

    if sess_id:
        sess = _get_session(sess_id)
        if not sess or file_idx >= len(sess):
            return jsonify({'error': 'session不存在'}), 404
        group      = sess[file_idx]
        save_path  = group['path']
        paper_type = group['paper_type']
        questions  = group['questions']
    else:
        save_path  = os.path.join(app.config['UPLOAD_FOLDER'], 'current.pdf')
        paper_type = _get_paper_type()
        questions  = None

    if not os.path.exists(save_path):
        return jsonify({'error': '请先上传PDF'}), 400

    try:
        doc = fitz.open(save_path)
        if questions is None:
            questions = _detect_questions(doc, paper_type)

        q_idx = next((i for i, q in enumerate(questions) if q['q_num'] == q_num), None)
        if q_idx is None:
            doc.close()
            return jsonify({'error': f'未找到第{q_num}题'}), 404

        img_bytes, w, h = crop_question_image(doc, questions, q_idx, dpi=dpi, paper_type=paper_type)
        doc.close()
        return send_file(io.BytesIO(img_bytes), mimetype='image/png', as_attachment=False)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/preview_b64/<int:q_num>', methods=['GET'])
def preview_question_b64(q_num):
    """
    预览单题图片，返回 JSON {img_b64, img_w, img_h}（base64 编码，用于题册保存）。
    支持 ?session_id=&file_idx= 参数。
    对于 imported/workbook 类型题目，直接从 session 返回已存储的 img_bytes_b64。
    对于原始 PDF 题目，渲染并返回。
    """
    import base64
    dpi      = int(request.args.get('dpi', 120))
    sess_id  = request.args.get('session_id')
    file_idx = int(request.args.get('file_idx', 0))

    if sess_id:
        sess = _get_session(sess_id)
        if not sess or file_idx >= len(sess):
            return jsonify({'error': 'session不存在'}), 404
        group      = sess[file_idx]
        save_path  = group['path']
        paper_type = group['paper_type']
        questions  = group['questions']

        # 对于 imported/workbook 类型，直接返回已存储的图片
        q_obj = next((q for q in questions if q['q_num'] == q_num), None)
        if q_obj and q_obj.get('img_bytes_b64'):
            return jsonify({
                'img_b64': q_obj['img_bytes_b64'],
                'img_w':   q_obj.get('img_w', 1000),
                'img_h':   q_obj.get('img_h', 500),
            })
    else:
        save_path  = os.path.join(app.config['UPLOAD_FOLDER'], 'current.pdf')
        paper_type = _get_paper_type()
        questions  = None

    if not os.path.exists(save_path):
        return jsonify({'error': '请先上传PDF'}), 400

    try:
        doc = fitz.open(save_path)
        if questions is None:
            questions = _detect_questions(doc, paper_type)

        q_idx = next((i for i, q in enumerate(questions) if q['q_num'] == q_num), None)
        if q_idx is None:
            doc.close()
            return jsonify({'error': f'未找到第{q_num}题'}), 404

        img_bytes, w, h = crop_question_image(doc, questions, q_idx, dpi=dpi, paper_type=paper_type)
        doc.close()
        return jsonify({
            'img_b64': base64.b64encode(img_bytes).decode(),
            'img_w':   w,
            'img_h':   h,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/preview_b64_batch', methods=['POST'])
def preview_question_b64_batch():
    """
    批量获取题目图片 base64，避免前端逐题串行请求导致超时。
    请求体 JSON: {
      "session_id": "...",
      "questions": [
        {"q_num": 1, "file_idx": 0},
        ...
      ],
      "dpi": 120
    }
    返回: {"results": [{"q_num":1,"file_idx":0,"img_b64":"...","img_w":...,"img_h":...}, ...]}
    """
    import base64
    data     = request.get_json(force=True) or {}
    sess_id  = data.get('session_id')
    dpi      = int(data.get('dpi', 120))
    items    = data.get('questions', [])   # list of {q_num, file_idx}

    results = []

    # 预先按 file_idx 分组，减少重复打开 PDF
    from collections import defaultdict
    groups_by_idx = defaultdict(list)
    for item in items:
        groups_by_idx[int(item.get('file_idx', 0))].append(int(item['q_num']))

    for file_idx, q_nums in groups_by_idx.items():
        # 确定本组文件路径和题目列表
        if sess_id:
            sess = _get_session(sess_id)
            if not sess or file_idx >= len(sess):
                for q_num in q_nums:
                    results.append({'q_num': q_num, 'file_idx': file_idx,
                                    'error': 'session不存在'})
                continue
            group      = sess[file_idx]
            save_path  = group['path']
            paper_type = group['paper_type']
            questions  = group['questions']
        else:
            save_path  = os.path.join(app.config['UPLOAD_FOLDER'], 'current.pdf')
            paper_type = _get_paper_type()
            questions  = None

        if not os.path.exists(save_path):
            for q_num in q_nums:
                results.append({'q_num': q_num, 'file_idx': file_idx,
                                'error': '文件不存在'})
            continue

        try:
            doc = fitz.open(save_path)
            if questions is None:
                questions = _detect_questions(doc, paper_type)

            for q_num in q_nums:
                # 若 workbook 类型已有缓存图片，直接返回
                q_obj = next((q for q in questions if q['q_num'] == q_num), None)
                if q_obj and q_obj.get('img_bytes_b64'):
                    results.append({
                        'q_num': q_num, 'file_idx': file_idx,
                        'img_b64': q_obj['img_bytes_b64'],
                        'img_w': q_obj.get('img_w', 1000),
                        'img_h': q_obj.get('img_h', 500),
                    })
                    continue

                q_idx = next((i for i, q in enumerate(questions) if q['q_num'] == q_num), None)
                if q_idx is None:
                    results.append({'q_num': q_num, 'file_idx': file_idx,
                                    'error': f'未找到第{q_num}题'})
                    continue

                img_bytes, w, h = crop_question_image(doc, questions, q_idx,
                                                      dpi=dpi, paper_type=paper_type)
                results.append({
                    'q_num': q_num, 'file_idx': file_idx,
                    'img_b64': base64.b64encode(img_bytes).decode(),
                    'img_w': w, 'img_h': h,
                })
            doc.close()
        except Exception as e:
            for q_num in q_nums:
                results.append({'q_num': q_num, 'file_idx': file_idx,
                                'error': str(e)})

    return jsonify({'results': results})


@app.route('/api/download/<int:q_num>', methods=['GET'])
def download_question(q_num):
    """下载单题图片，支持 ?session_id=&file_idx= 参数"""
    dpi      = int(request.args.get('dpi', 200))
    fmt      = request.args.get('format', 'png').lower()
    sess_id  = request.args.get('session_id')
    file_idx = int(request.args.get('file_idx', 0))

    # 解析来源：优先 session，否则 fallback 到 current.pdf
    if sess_id:
        sess = _get_session(sess_id)
        if not sess or file_idx >= len(sess):
            return jsonify({'error': 'session不存在或file_idx越界'}), 400
        group      = sess[file_idx]
        save_path  = group['path']
        paper_type = group['paper_type']
        questions  = group['questions']
    else:
        save_path  = os.path.join(app.config['UPLOAD_FOLDER'], 'current.pdf')
        if not os.path.exists(save_path):
            return jsonify({'error': '请先上传PDF'}), 400
        paper_type = _get_paper_type()
        questions  = None  # 延迟检测

    try:
        doc = fitz.open(save_path)
        if questions is None:
            questions = _detect_questions(doc, paper_type)

        q_idx = next((i for i, q in enumerate(questions) if q['q_num'] == q_num), None)
        if q_idx is None:
            doc.close()
            return jsonify({'error': f'未找到第{q_num}题'}), 404

        img_bytes, w, h = crop_question_image(doc, questions, q_idx, dpi=dpi, paper_type=paper_type)
        doc.close()

        if fmt in ('jpg', 'jpeg'):
            from PIL import Image
            img_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            buf = io.BytesIO()
            img_pil.save(buf, format='JPEG', quality=95)
            img_bytes = buf.getvalue()
            mimetype, ext = 'image/jpeg', 'jpg'
        else:
            mimetype, ext = 'image/png', 'png'

        return send_file(
            io.BytesIO(img_bytes), mimetype=mimetype,
            as_attachment=True, download_name=f'Q{q_num:02d}.{ext}'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/download_batch', methods=['POST'])
def download_batch():
    """批量下载选中题目为ZIP，支持 session_id + file_idx"""
    data     = request.json
    q_nums   = data.get('questions', [])
    dpi      = int(data.get('dpi', 200))
    fmt      = data.get('format', 'png').lower()
    sess_id  = data.get('session_id')
    file_idx = int(data.get('file_idx', 0))

    if not q_nums:
        return jsonify({'error': '未选择题目'}), 400

    # 解析来源
    if sess_id:
        sess = _get_session(sess_id)
        if not sess or file_idx >= len(sess):
            return jsonify({'error': 'session不存在或file_idx越界'}), 400
        group      = sess[file_idx]
        save_path  = group['path']
        paper_type = group['paper_type']
        questions  = group['questions']
    else:
        save_path  = os.path.join(app.config['UPLOAD_FOLDER'], 'current.pdf')
        if not os.path.exists(save_path):
            return jsonify({'error': '请先上传PDF'}), 400
        paper_type = _get_paper_type()
        questions  = None

    try:
        doc = fitz.open(save_path)
        if questions is None:
            questions = _detect_questions(doc, paper_type)

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
            for q_num in sorted(q_nums):
                q_idx = next((i for i, q in enumerate(questions) if q['q_num'] == q_num), None)
                if q_idx is None:
                    continue
                img_bytes, _, _ = crop_question_image(doc, questions, q_idx, dpi=dpi, paper_type=paper_type)
                if fmt in ('jpg', 'jpeg'):
                    from PIL import Image
                    img_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                    buf = io.BytesIO()
                    img_pil.save(buf, format='JPEG', quality=95)
                    img_bytes = buf.getvalue()
                    ext = 'jpg'
                else:
                    ext = 'png'
                zf.writestr(f'Q{q_num:02d}.{ext}', img_bytes)

        doc.close()
        zip_buf.seek(0)
        return send_file(zip_buf, mimetype='application/zip', as_attachment=True, download_name='questions.zip')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/download_all', methods=['GET'])
def download_all():
    """下载所有题目为ZIP，支持 ?session_id=&file_idx= 参数"""
    dpi      = int(request.args.get('dpi', 200))
    fmt      = request.args.get('format', 'png').lower()
    sess_id  = request.args.get('session_id')
    file_idx = int(request.args.get('file_idx', 0))

    # 解析来源
    if sess_id:
        sess = _get_session(sess_id)
        if not sess or file_idx >= len(sess):
            return jsonify({'error': 'session不存在或file_idx越界'}), 400
        group      = sess[file_idx]
        save_path  = group['path']
        paper_type = group['paper_type']
        questions  = group['questions']
    else:
        save_path  = os.path.join(app.config['UPLOAD_FOLDER'], 'current.pdf')
        if not os.path.exists(save_path):
            return jsonify({'error': '请先上传PDF'}), 400
        paper_type = _get_paper_type()
        questions  = None

    try:
        doc = fitz.open(save_path)
        if questions is None:
            questions = _detect_questions(doc, paper_type)

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
            for q_idx, q in enumerate(questions):
                img_bytes, _, _ = crop_question_image(doc, questions, q_idx, dpi=dpi, paper_type=paper_type)
                if fmt in ('jpg', 'jpeg'):
                    from PIL import Image
                    img_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                    buf = io.BytesIO()
                    img_pil.save(buf, format='JPEG', quality=95)
                    img_bytes = buf.getvalue()
                    ext = 'jpg'
                else:
                    ext = 'png'
                zf.writestr(f'Q{q["q_num"]:02d}.{ext}', img_bytes)

        doc.close()
        zip_buf.seek(0)
        return send_file(zip_buf, mimetype='application/zip', as_attachment=True, download_name='all_questions.zip')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _build_pdf_worker(task_id, save_path, paper_type_val, q_nums, dpi, layout, out_path,
                       preloaded_questions=None, cover_title=''):
    """
    后台线程：生成 PDF，进度写文件持久化（gunicorn 多线程安全）。
    preloaded_questions: 若已解析好，直接使用（多文件场景）。
    cover_title: 封面标题，非空时在 PDF 首页插入封面。
    """
    def upd(prog, status='running', error=None):
        data = {'status': status, 'progress': prog,
                'total': len(q_nums), 'out_path': out_path,
                'filename': (_load_task(task_id) or {}).get('filename', 'questions.pdf'),
                'error': error}
        _save_task(task_id, data)

    try:
        src_doc   = fitz.open(save_path)
        questions = preloaded_questions if preloaded_questions is not None \
                    else _detect_questions(src_doc, paper_type_val)

        PAGE_W, PAGE_H = 595, 842
        MARGIN, HEADER_H, GAP, FS = 36, 28, 12, 11
        out_doc = fitz.open()

        # ── 封面页（放在所有题目之前）──
        if cover_title:
            _generate_cover_page(out_doc, cover_title,
                                  page_w=PAGE_W, page_h=PAGE_H)

        if layout == 'two_per_page':
            _export_two_per_page(out_doc, src_doc, questions, q_nums, dpi,
                                 paper_type_val, PAGE_W, PAGE_H, MARGIN,
                                 HEADER_H, GAP, FS, progress_cb=lambda p: upd(p))
        else:
            # one_per_page：MCQ类型自动走多题共页打包逻辑（_export_mcq_packed）
            # 大题类型每题独页，逻辑由 _export_one_per_page 内部分派
            _export_one_per_page(out_doc, src_doc, questions, q_nums, dpi,
                                 paper_type_val, PAGE_W, PAGE_H, MARGIN,
                                 HEADER_H, GAP, FS, progress_cb=lambda p: upd(p))

        src_doc.close()
        # 直接保存到磁盘，避免 tobytes() 将整个 PDF 载入内存
        out_doc.save(out_path, garbage=4, deflate=True)
        out_doc.close()

        # R2 模式：上传到 R2，task 记录 r2_key
        r2_key = ''
        if storage.is_r2_mode():
            r2_key = f'output/{task_id}.pdf'
            storage.upload_from_local(out_path, r2_key)

        # 更新任务为 done，写入 r2_key
        t_data = _load_task(task_id) or {}
        t_data.update({'status': 'done', 'progress': len(q_nums),
                       'total': len(q_nums), 'out_path': out_path,
                       'r2_key': r2_key, 'error': None})
        _save_task(task_id, t_data)

    except Exception as e:
        import traceback; traceback.print_exc()
        upd(0, status='error', error=str(e))


def _export_b64_questions_to_pdf(out_doc, questions, q_nums,
                                  PAGE_W, PAGE_H, MARGIN, HEADER_H, GAP, FS,
                                  progress_cb=None, seq_start=0):
    """
    将题目 img_bytes_b64 直接渲染到 out_doc（用于题册类型，无原始 PDF 文件）。
    每道题一页（one_per_page 模式）。
    """
    import base64 as _b64
    q_map = {q['q_num']: q for q in questions}
    for i, q_num in enumerate(q_nums):
        q_obj = q_map.get(q_num)
        if not q_obj:
            continue
        b64 = q_obj.get('img_bytes_b64', '')
        if not b64:
            continue
        try:
            img_bytes = _b64.b64decode(b64)
            img_w = q_obj.get('img_w', 1) or 1
            img_h = q_obj.get('img_h', 1) or 1
            diff = q_obj.get('difficulty')
            topics = q_obj.get('topics', [])
            q_meta = {'difficulty': diff, 'topics': topics} if (diff is not None or topics) else None
            label = f'Q{q_num:02d}'

            page = out_doc.new_page(width=PAGE_W, height=PAGE_H)
            _place_jpeg_on_page(
                page, img_bytes, img_w, img_h,
                fitz.Rect(MARGIN, MARGIN, PAGE_W - MARGIN, PAGE_H - MARGIN),
                label, HEADER_H, GAP, FS, q_meta=q_meta
            )
        except Exception:
            pass
        if progress_cb:
            progress_cb(seq_start + i + 1)


def _build_pdf_merged_worker(task_id, groups_info, dpi, layout, out_path, total_q,
                              cover_title='', ordered_items=None):
    """
    后台线程：多文件合并导出 PDF。
    groups_info: [{path, paper_type, questions, q_nums, g_idx}]
    ordered_items: [{gIdx, q_num}] 全局有序列表（来自前端 exportItems，保留 sortOrder 排序）。
                   若提供则按此全局顺序逐题输出；否则按组顺序输出（降级模式）。
    cover_title: 封面标题，非空时在首页插入封面。
    """
    def upd(prog, status='running', error=None):
        data = {'status': status, 'progress': prog,
                'total': total_q, 'out_path': out_path,
                'filename': (_load_task(task_id) or {}).get('filename', 'questions.pdf'),
                'error': error}
        _save_task(task_id, data)

    PAGE_W, PAGE_H = 595, 842
    MARGIN, HEADER_H, GAP, FS = 36, 28, 12, 11

    try:
        out_doc = fitz.open()

        # ── 封面页 ──
        if cover_title:
            _generate_cover_page(out_doc, cover_title,
                                  page_w=PAGE_W, page_h=PAGE_H)

        # 构建 gIdx → {src_doc, questions, paper_type} 的映射（延迟打开）
        group_map = {}  # g_idx -> ginfo
        for ginfo in groups_info:
            group_map[ginfo['g_idx']] = ginfo

        if ordered_items:
            # ── 有序模式：按 ordered_items 全局顺序逐题输出 ──
            # 需要分组打开 src_doc（按 gIdx 缓存）
            src_docs = {}  # g_idx -> fitz.Document
            done_total = 0

            # 按 gIdx 预打开文件
            for ginfo in groups_info:
                gi = ginfo['g_idx']
                if os.path.exists(ginfo['path']):
                    src_docs[gi] = fitz.open(ginfo['path'])

            # 注意：MCQ 打包模式（多题共页）在全局顺序下需要特殊处理
            # 策略：先按 gIdx 分组收集有序 q_nums，逐组调用 _export_xxx
            # 但跨组题目顺序仍需全局维持 → 改为按 ordered_items 顺序，
            # 每遇到 gIdx 切换时关闭上一组段落，开新组段落
            # 简单可靠方案：全局按 ordered_items 顺序，以组为段依次处理
            # （同组相邻题目保持连续；跨组切换时自然分段）

            # 构建每个 gIdx 的有序 q_nums（保持 ordered_items 中的顺序）
            ordered_by_group = {}  # g_idx -> [q_num, ...]（按 ordered_items 顺序）
            for item in ordered_items:
                gi = item.get('gIdx', 0)
                qn = item.get('q_num')
                if qn is None:
                    continue
                if gi not in ordered_by_group:
                    ordered_by_group[gi] = []
                ordered_by_group[gi].append(qn)

            # 按 ordered_items 中 gIdx 的首次出现顺序处理各组
            seen_gi = []
            for item in ordered_items:
                gi = item.get('gIdx', 0)
                if gi not in seen_gi:
                    seen_gi.append(gi)

            for gi in seen_gi:
                if gi not in group_map or gi not in src_docs:
                    continue
                ginfo     = group_map[gi]
                questions = ginfo['questions']
                paper_type = ginfo['paper_type']
                q_nums_ordered = ordered_by_group.get(gi, [])
                if not q_nums_ordered:
                    continue

                done_before = done_total
                def cb(p, _done=done_before):
                    upd(_done + p)

                # 题册类型（无PDF文件，使用img_bytes_b64直接渲染）
                is_workbook = not ginfo['path'] or not os.path.exists(ginfo['path'])
                if is_workbook:
                    _export_b64_questions_to_pdf(
                        out_doc, questions, q_nums_ordered,
                        PAGE_W, PAGE_H, MARGIN, HEADER_H, GAP, FS,
                        progress_cb=cb, seq_start=done_total)
                elif gi in src_docs:
                    src_doc = src_docs[gi]
                    if layout == 'two_per_page':
                        _export_two_per_page(out_doc, src_doc, questions, q_nums_ordered, dpi,
                                             paper_type, PAGE_W, PAGE_H, MARGIN,
                                             HEADER_H, GAP, FS, progress_cb=cb,
                                             seq_start=done_total)
                    else:
                        _export_one_per_page(out_doc, src_doc, questions, q_nums_ordered, dpi,
                                             paper_type, PAGE_W, PAGE_H, MARGIN,
                                             HEADER_H, GAP, FS, progress_cb=cb,
                                             seq_start=done_total)
                done_total += len(q_nums_ordered)

            for sd in src_docs.values():
                sd.close()

        else:
            # ── 降级模式：按 groups_info 顺序，每组内按 q_nums 原顺序 ──
            done_total = 0
            for gi, ginfo in enumerate(groups_info):
                save_path  = ginfo['path']
                paper_type = ginfo['paper_type']
                questions  = ginfo['questions']
                q_nums     = ginfo['q_nums']

                if not q_nums:
                    continue

                # 题册类型（无PDF文件，使用img_bytes_b64直接渲染）
                is_workbook = not save_path or not os.path.exists(save_path)
                if is_workbook:
                    done_before = done_total
                    def cb_b64(p, _done=done_before):
                        upd(_done + p)
                    _export_b64_questions_to_pdf(
                        out_doc, questions, q_nums,
                        PAGE_W, PAGE_H, MARGIN, HEADER_H, GAP, FS,
                        progress_cb=cb_b64, seq_start=done_total)
                    done_total += len(q_nums)
                    continue

                src_doc = fitz.open(save_path)
                done_before = done_total
                def cb(p, _done=done_before):
                    upd(_done + p)

                if layout == 'two_per_page':
                    _export_two_per_page(out_doc, src_doc, questions, q_nums, dpi,
                                         paper_type, PAGE_W, PAGE_H, MARGIN,
                                         HEADER_H, GAP, FS, progress_cb=cb,
                                         seq_start=done_total)
                else:
                    _export_one_per_page(out_doc, src_doc, questions, q_nums, dpi,
                                         paper_type, PAGE_W, PAGE_H, MARGIN,
                                         HEADER_H, GAP, FS, progress_cb=cb,
                                         seq_start=done_total)

                src_doc.close()
                done_total += len(q_nums)

        out_doc.save(out_path, garbage=4, deflate=True)
        out_doc.close()

        # R2 模式：上传到 R2，task 记录 r2_key
        r2_key = ''
        if storage.is_r2_mode():
            r2_key = f'output/{task_id}.pdf'
            storage.upload_from_local(out_path, r2_key)

        # 更新任务为 done，写入 r2_key
        t_data = _load_task(task_id) or {}
        t_data.update({'status': 'done', 'progress': total_q,
                       'total': total_q, 'out_path': out_path,
                       'r2_key': r2_key, 'error': None})
        _save_task(task_id, t_data)

    except Exception as e:
        import traceback; traceback.print_exc()
        upd(0, status='error', error=str(e))


@app.route('/api/export_pdf', methods=['POST'])
def export_pdf():
    """
    启动后台 PDF 生成任务，立刻返回 task_id（不阻塞主线程）。
    支持三种模式：
      1. 单文件（旧接口）: {questions, dpi, filename, layout}
      2. 单文件 session: {session_id, file_idx, questions, dpi, filename, layout}
      3. 多文件合并: {session_id, merged:true, sel_by_group:[{gIdx, q_nums:[]}], dpi, filename, layout}
         sel_by_group 中每项包含 gIdx（组索引）和该组要导出的题号列表
    可选参数：
      cover_title: 封面页标题（如 "数学Edexcel P3题册"）；为空则不生成封面
    """
    data      = request.json
    dpi       = min(int(data.get('dpi', 150)), 150)
    filename  = (data.get('filename') or 'questions').strip()
    layout    = data.get('layout', 'one_per_page')
    sess_id   = data.get('session_id')
    merged    = data.get('merged', False)
    cover_title = data.get('cover_title', '').strip()   # 封面标题（可选）

    safe_name = re.sub(r'[\\/*?:"<>|]', '_', filename)
    if not safe_name.endswith('.pdf'):
        safe_name += '.pdf'

    task_id  = str(uuid.uuid4())
    # 导出 PDF 始终写到本地临时目录；R2 模式下后续再上传
    out_path = storage.local_tmp_path(f'exp_{task_id}.pdf')

    # ── 模式3：多文件合并 ──
    if merged and sess_id:
        sel_by_group  = data.get('sel_by_group', [])   # [{gIdx, q_nums}]
        ordered_items = data.get('ordered_items', [])  # [{gIdx, q_num}] 全局有序列表

        if not sel_by_group:
            return jsonify({'error': '未选择题目'}), 400

        sess = _get_session(sess_id)
        if not sess:
            return jsonify({'error': 'session不存在'}), 400

        groups_info = []
        total_q = 0
        for item in sel_by_group:
            g_idx  = item.get('gIdx', 0)
            q_nums = item.get('q_nums', [])
            if not q_nums or g_idx >= len(sess):
                continue
            g = sess[g_idx]
            groups_info.append({
                'path':       g['path'],
                'paper_type': g['paper_type'],
                'questions':  g['questions'],
                'q_nums':     q_nums,
                'g_idx':      g_idx,
            })
            total_q += len(q_nums)

        if total_q == 0:
            return jsonify({'error': '未选择题目'}), 400

        _save_task(task_id, {
            'status': 'running', 'progress': 0,
            'total': total_q, 'out_path': out_path,
            'filename': safe_name, 'error': None,
        })

        threading.Thread(
            target=_build_pdf_merged_worker,
            args=(task_id, groups_info, dpi, layout, out_path, total_q),
            kwargs={'cover_title': cover_title, 'ordered_items': ordered_items},
            daemon=True
        ).start()

        return jsonify({'task_id': task_id, 'total': total_q})

    # ── 模式1/2：单文件 ──
    q_nums    = data.get('questions', [])
    file_idx  = data.get('file_idx', 0)

    if not q_nums:
        return jsonify({'error': '未选择题目'}), 400

    # 确定使用哪个文件
    if sess_id:
        sess = _get_session(sess_id)
        if not sess or file_idx >= len(sess):
            return jsonify({'error': 'session不存在或file_idx越界'}), 400
        group = sess[file_idx]
        save_path  = group['path']
        pt_val     = group['paper_type']
        preloaded  = group['questions']
    else:
        save_path  = os.path.join(app.config['UPLOAD_FOLDER'], 'current.pdf')
        pt_val     = _get_paper_type()
        preloaded  = None

    if not os.path.exists(save_path):
        return jsonify({'error': '请先上传PDF'}), 400

    safe_name = re.sub(r'[\\/*?:"<>|]', '_', filename)
    if not safe_name.endswith('.pdf'):
        safe_name += '.pdf'

    task_id  = str(uuid.uuid4())
    out_path = os.path.join(app.config['UPLOAD_FOLDER'], f'exp_{task_id}.pdf')

    _save_task(task_id, {
        'status':   'running',
        'progress': 0,
        'total':    len(q_nums),
        'out_path': out_path,
        'filename': safe_name,
        'error':    None,
    })

    threading.Thread(
        target=_build_pdf_worker,
        args=(task_id, save_path, pt_val, q_nums, dpi, layout, out_path),
        kwargs={'preloaded_questions': preloaded, 'cover_title': cover_title},
        daemon=True
    ).start()

    return jsonify({'task_id': task_id, 'total': len(q_nums)})


@app.route('/api/export_pdf/progress/<task_id>', methods=['GET'])
def export_pdf_progress(task_id):
    t = _load_task(task_id)
    if not t:
        return jsonify({'error': '任务不存在'}), 404
    return jsonify({'status': t['status'], 'progress': t['progress'],
                    'total': t['total'], 'error': t['error']})


@app.route('/api/export_pdf/download/<task_id>', methods=['GET'])
def export_pdf_download(task_id):
    t = _load_task(task_id)
    if not t or t['status'] != 'done':
        return jsonify({'error': '文件未就绪'}), 400
    out_path = t['out_path']
    filename = t['filename']
    r2_key   = t.get('r2_key', '')

    # R2 模式：若本地临时文件不存在，先从 R2 下载
    if storage.is_r2_mode() and r2_key:
        if not os.path.isfile(out_path):
            ok = storage.make_local_copy(r2_key, out_path)
            if not ok:
                return jsonify({'error': '文件已过期，请重新生成'}), 404
    elif not os.path.exists(out_path):
        return jsonify({'error': '文件已过期，请重新生成'}), 404

    def _cleanup():
        time.sleep(120)
        try: os.remove(out_path)
        except Exception: pass
        # R2 模式：同时清理 R2 上的文件
        if storage.is_r2_mode() and r2_key:
            storage.delete_object(r2_key)
        try: os.remove(_task_path(task_id))
        except Exception: pass
    threading.Thread(target=_cleanup, daemon=True).start()

    return send_file(out_path, mimetype='application/pdf',
                     as_attachment=True, download_name=filename)


# ─────────────────────────────────────────────────────────────────────────────
# 功能4：手动更新题目知识点 / 难度（前端可调用）
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/api/update_question_meta', methods=['POST'])
def update_question_meta():
    """
    前端手动修改题目的知识点或难度时调用。
    body: {session_id, g_idx, q_num, field:'topics'|'difficulty', value: ...}
    topics value: [{id, title}] 或字符串 chapter_id
    difficulty value: 1-5 整数
    """
    data = request.json
    sess_id = data.get('session_id')
    g_idx   = data.get('g_idx', 0)
    q_num   = data.get('q_num')
    field   = data.get('field')
    value   = data.get('value')

    if not sess_id or q_num is None or field not in ('topics', 'difficulty'):
        return jsonify({'error': '参数不完整'}), 400

    sess = _get_session(sess_id)
    if not sess or g_idx >= len(sess):
        return jsonify({'error': 'session不存在'}), 400

    questions = sess[g_idx]['questions']
    q_obj = next((q for q in questions if q.get('q_num') == q_num), None)
    if q_obj is None:
        return jsonify({'error': f'题目 {q_num} 不存在'}), 404

    if field == 'difficulty':
        try:
            v = int(value)
            if not (1 <= v <= 5):
                raise ValueError
            q_obj['difficulty'] = v
        except (ValueError, TypeError):
            return jsonify({'error': '难度值必须为1-5整数'}), 400

    elif field == 'topics':
        # value 可以是 chapter_id 字符串、或完整 [{id,title,is_core,...}] 列表
        if isinstance(value, str):
            syllabus = _load_edexcel_maths_syllabus()
            title = value
            if syllabus:
                for t in syllabus['topics']:
                    if str(t['id']) == value:
                        title = t.get('title', value)
                        break
                    for s in t.get('subtopics', []):
                        if str(s['id']) == value:
                            title = s.get('title', value)
                            break
            q_obj['topics'] = [{'id': value, 'title': title, 'score': 99, 'is_core': True}]
        elif isinstance(value, list):
            # 直接接受前端传来的完整列表（含 is_core 字段）
            q_obj['topics'] = value
        else:
            return jsonify({'error': 'topics 格式错误'}), 400

    return jsonify({'ok': True, 'q_num': q_num, 'field': field,
                    'new_value': q_obj.get(field)})


# ─────────────────────────────────────────────────────────────────────────────
# 功能5：导入已导出的 PDF，重新切割题目图片
# ─────────────────────────────────────────────────────────────────────────────
_EXPORT_HEADER_HEIGHT_PT = 28   # 与 _place_jpeg_on_page 中的 INFO_H 保持一致

# 已导出PDF头栏的深蓝色（RGB浮点）—— 与 _place_jpeg_on_page 中保持一致
_HEADER_NAVY_COLOR = (0.09, 0.26, 0.55)  # (23, 66, 140) 约

# 难度名称映射
_DIFF_NAME_MAP = {
    'starter': 1, 'basic': 2, 'medium': 3, 'hard': 4, 'expert': 5
}


def _detect_exported_pdf_header(page):
    """
    检测页面是否有本工具导出的深蓝色头栏，并提取题号、难度、知识点ID。

    检测策略（全新版本）：
    1. 找页面上第一张图片的 y0 坐标
       - img.y0 > 60：说明图片上方有头栏 → 进行文字提取
       - img.y0 <= 60：无头栏（exported without q_meta），返回 None
    2. 用 get_text('text', clip=header_rect) 提取头栏区域的 ASCII 文本
       （新版导出用 insert_text 写入ASCII文字，可正常提取）
    3. 解析格式："Q05  |  Medium  |  *P3-4  Differentiation"
       - Q(\d+) → 题号
       - Starter/Basic/Medium/Hard/Expert → 难度
       - P\d+-\d+ → 知识点章节 ID

    返回 (q_num, difficulty, topic_id, topic_title_hint) 或 None。
    topic_title_hint 是从头栏文本中解析出的章节标题，作为辅助信息。
    """
    MARGIN = 36
    BAND_H = _EXPORT_HEADER_HEIGHT_PT

    # ── Step 1：找第一张图片的 y0 ──
    img_list = page.get_images(full=False)
    if not img_list:
        return None

    first_img_y0 = None
    for xref, *_ in img_list:
        rects = page.get_image_rects(xref)
        if rects:
            for r in rects:
                if first_img_y0 is None or r.y0 < first_img_y0:
                    first_img_y0 = r.y0
            break

    if first_img_y0 is None:
        return None

    # img.y0 ≤ 60 说明无头栏（图片紧贴页面顶部 margin）
    if first_img_y0 <= 60:
        return None

    # ── Step 2：提取头栏文字（头栏位于图片上方） ──
    # 头栏 y 范围：从 MARGIN 到 img.y0 稍上方
    header_top = MARGIN - 5
    header_bot = first_img_y0 + 2   # 留一点 padding
    header_rect = fitz.Rect(0, header_top, page.rect.width, header_bot)
    raw_text = page.get_text('text', clip=header_rect).strip()

    if not raw_text:
        # fallback：用 rawdict 颜色法（兼容旧格式）
        return _detect_header_legacy(page, header_rect)

    # ── Step 3：解析头栏文本 ──
    # 典型格式："Q05  |  Medium  |  *P3-4  Differentiation"
    # 也可能带星号（is_core）："Q05  |  ★ Medium  |  *P3-4  Differentiation"

    # 提取题号
    q_num_match = re.search(r'\bQ\s*(\d+)\b', raw_text, re.IGNORECASE)
    if not q_num_match:
        # 尝试更宽松的数字匹配（兼容旧格式）
        q_num_match = re.search(r'\b(\d+)\b', raw_text)
    if not q_num_match:
        return None
    q_num = int(q_num_match.group(1))

    # 提取难度
    difficulty = None
    text_lower = raw_text.lower()
    for dname, dval in _DIFF_NAME_MAP.items():
        if dname in text_lower:
            difficulty = dval
            break

    # 提取知识点 ID（格式：P3-4 或 *P3-4）
    topic_id_match = re.search(r'\*?([A-Z]\d+-\d+)', raw_text)
    topic_id = topic_id_match.group(1) if topic_id_match else None

    # 提取知识点标题（| 后面的文字，去掉ID前缀）
    topic_title_hint = None
    if topic_id:
        # 从 "P3-4  Differentiation" 或 "*P3-4  Differentiation" 中提取标题
        title_match = re.search(r'\*?' + re.escape(topic_id) + r'\s+(.+)', raw_text)
        if title_match:
            topic_title_hint = title_match.group(1).strip()

    return q_num, difficulty, topic_id, topic_title_hint


def _detect_header_legacy(page, header_rect):
    """
    旧版兼容：用颜色过滤法提取头栏文字（用于旧版导出PDF，header文字可能乱码）。
    返回 (q_num, difficulty, topic_id, None) 或 None。
    """
    NAVY_INT_LOW  = 0x0a1080
    NAVY_INT_HIGH = 0x2055ff

    full_text_spans = []
    try:
        td = page.get_text('rawdict', flags=fitz.TEXT_PRESERVE_WHITESPACE,
                           clip=header_rect)
        for block in td.get('blocks', []):
            if block.get('type') != 0:
                continue
            for line in block.get('lines', []):
                for span in line.get('spans', []):
                    col = span.get('color', 0)
                    if NAVY_INT_LOW <= col <= NAVY_INT_HIGH:
                        chars = span.get('chars', [])
                        char_str = ''.join(c.get('c', '') for c in chars)
                        full_text_spans.append(char_str)
    except Exception:
        pass

    if not full_text_spans:
        return None

    combined = ' '.join(full_text_spans)

    # 旧格式：数字在乱码字符之间 "? 1 ?"
    num_match = re.search(r'(?:^|\?|\s)(\d+)(?:\?|\s|$)', combined)
    if not num_match:
        num_match = re.search(r'\b(\d+)\b', combined)
    if not num_match:
        return None

    q_num = int(num_match.group(1))

    difficulty = None
    for dname, dval in _DIFF_NAME_MAP.items():
        if dname in combined.lower():
            difficulty = dval
            break

    topic_id_match = re.search(r'P(\d+)-(\d+)', combined)
    topic_id = f'P{topic_id_match.group(1)}-{topic_id_match.group(2)}' if topic_id_match else None

    return q_num, difficulty, topic_id, None


def _lookup_topic_in_syllabus(syllabus, topic_id, title_hint=None):
    """
    在 syllabus（edexcel_maths 格式）中查找 topic_id 对应的完整信息。
    先精确匹配 id，再用 title_hint 模糊匹配。
    返回 {'id', 'title', 'unit', 'is_core'} 或 None。
    """
    if not syllabus or not topic_id:
        return None
    for t in syllabus.get('topics', []):
        if str(t.get('id', '')) == topic_id:
            return {'id': topic_id, 'title': t.get('title', topic_id),
                    'unit': t.get('unit', ''), 'is_core': False}
        for sub in t.get('subtopics', []):
            if str(sub.get('id', '')) == topic_id:
                return {'id': topic_id, 'title': sub.get('title', topic_id),
                        'unit': t.get('unit', ''), 'is_core': False}
    # 未精确匹配，但有 title_hint → 用 hint 作为 title
    if title_hint:
        return {'id': topic_id, 'title': title_hint,
                'unit': '', 'is_core': False}
    return {'id': topic_id, 'title': topic_id, 'unit': '', 'is_core': False}


def _auto_detect_syllabus_type(topic_id):
    """根据 topic_id 格式自动推断大纲类型。"""
    if topic_id and re.match(r'^[A-Z]\d+-\d+', topic_id):
        return 'edexcel_maths'
    return 'unknown'


def _stitch_pixmaps_vertical(pixmaps):
    """
    将多个 fitz.Pixmap 垂直拼接成一张大图。
    要求所有 pixmap 宽度相同（不同则等比缩放到第一张宽度）。
    返回 PIL Image。
    """
    if not pixmaps:
        return None
    from PIL import Image as PILImage
    pil_imgs = []
    for pix in pixmaps:
        mode = 'RGB' if pix.n == 3 else 'RGBA'
        img = PILImage.frombytes(mode, (pix.width, pix.height), pix.samples)
        pil_imgs.append(img)

    target_w = pil_imgs[0].width
    resized = []
    for img in pil_imgs:
        if img.width != target_w:
            ratio = target_w / img.width
            new_h = int(img.height * ratio)
            img = img.resize((target_w, new_h), PILImage.LANCZOS)
        resized.append(img)

    total_h = sum(img.height for img in resized)
    merged = PILImage.new('RGB', (target_w, total_h), (255, 255, 255))
    y_off = 0
    for img in resized:
        merged.paste(img.convert('RGB'), (0, y_off))
        y_off += img.height
    return merged


@app.route('/api/import_exported_pdf', methods=['POST'])
def import_exported_pdf():
    """
    导入之前本工具导出的 PDF。

    识别规则（全新版本）：
    - 通过图片 y0 坐标判断是否有头栏：img.y0 > 60 → 有头栏
    - 用 get_text('text') 提取头栏 ASCII 文本（新版导出文字完全可提取）
    - 解析 "Q05  |  Medium  |  *P3-4  Differentiation" 格式
    - 用 P\d+-\d+ 格式自动识别为 edexcel_maths 大纲，在 syllabus 中精确定位
    - 同一题号的多页图片垂直拼接为完整题目图
    """
    if 'file' not in request.files:
        return jsonify({'error': '请上传 PDF 文件'}), 400
    f = request.files['file']
    if not f.filename or not f.filename.lower().endswith('.pdf'):
        return jsonify({'error': '仅支持 PDF 格式'}), 400

    safe = secure_filename(f.filename)
    session_id = str(uuid.uuid4())
    save_path = os.path.join(_MULTI_DIR, f'{session_id}_{safe}')
    f.save(save_path)

    try:
        import base64
        doc = fitz.open(save_path)
        MARGIN = 36
        RENDER_SCALE = 2.0
        mat = fitz.Matrix(RENDER_SCALE, RENDER_SCALE)

        # 加载 edexcel maths 大纲（用于知识点查找）
        syllabus = _load_edexcel_maths_syllabus()

        # ── 第一遍：按页识别题号，收集每题的页面列表 ──
        page_info = []
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            result = _detect_exported_pdf_header(page)
            if result is None:
                # 续页（无头栏）：属于上一题
                if page_info:
                    page_info.append({
                        'q_num':          page_info[-1]['q_num'],
                        'difficulty':     page_info[-1]['difficulty'],
                        'topic_id':       page_info[-1]['topic_id'],
                        'topic_title':    page_info[-1]['topic_title'],
                        'page_obj':       page,
                        'is_first':       False,
                        'img_y0':         None,
                    })
                # else: 跳过（可能是封面或无元数据页）
            else:
                q_num, difficulty, topic_id, topic_title_hint = result
                page_info.append({
                    'q_num':          q_num,
                    'difficulty':     difficulty,
                    'topic_id':       topic_id,
                    'topic_title':    topic_title_hint,
                    'page_obj':       page,
                    'is_first':       True,
                    'img_y0':         None,   # 稍后填充
                })

        if not page_info:
            doc.close()
            return jsonify({'error': '未能识别出题目，请确认这是本工具导出的PDF（需包含题目头栏）'}), 400

        # ── 第二遍：按题号分组并拼接图片 ──
        from collections import OrderedDict
        q_groups = OrderedDict()
        for pi in page_info:
            qn = pi['q_num']
            if qn not in q_groups:
                q_groups[qn] = {
                    'difficulty':  pi['difficulty'],
                    'topic_id':    pi['topic_id'],
                    'topic_title': pi['topic_title'],
                    'pages':       [],
                }
            q_groups[qn]['pages'].append(pi['page_obj'])

        questions = []
        for q_num, gdata in q_groups.items():
            pages      = gdata['pages']
            difficulty = gdata['difficulty']
            topic_id   = gdata['topic_id']
            topic_title_hint = gdata['topic_title']

            # 渲染每一页，裁掉头栏，只保留题目图片区域
            pixmaps = []
            for pg_obj in pages:
                PW = pg_obj.rect.width
                PH = pg_obj.rect.height

                # 找本页图片的真实起始 y（精确裁剪）
                img_list_pg = pg_obj.get_images(full=False)
                content_y0 = None
                if img_list_pg:
                    for xref, *_ in img_list_pg:
                        rects = pg_obj.get_image_rects(xref)
                        if rects:
                            content_y0 = rects[0].y0
                            break

                # fallback：如果找不到图片rect，根据 img_y0 > 60 推断
                if content_y0 is None:
                    content_y0 = MARGIN + _EXPORT_HEADER_HEIGHT_PT + 14  # ~76

                # 裁剪 rect：从图片顶端到页底 margin
                img_rect = fitz.Rect(MARGIN, content_y0, PW - MARGIN, PH - MARGIN)
                pix = pg_obj.get_pixmap(matrix=mat, clip=img_rect)
                pixmaps.append(pix)

            # 多页垂直拼接
            merged_pil = _stitch_pixmaps_vertical(pixmaps)
            for pix in pixmaps:
                del pix

            if merged_pil is None:
                continue

            buf = io.BytesIO()
            merged_pil.save(buf, format='JPEG', quality=92)
            img_bytes = buf.getvalue()
            img_w, img_h = merged_pil.size
            del merged_pil

            # ── 知识点解析：在 syllabus 中精确定位 ──
            topics = []
            if topic_id:
                # 自动确认大纲类型（P\d+-\d+ → edexcel_maths）
                topic_info = _lookup_topic_in_syllabus(syllabus, topic_id, topic_title_hint)
                if topic_info:
                    topics = [{
                        'id':      topic_info['id'],
                        'title':   topic_info['title'],
                        'score':   99,
                        'is_core': topic_info.get('is_core', False),
                    }]
                else:
                    # 未在 syllabus 找到，但仍保留 ID
                    topics = [{'id': topic_id,
                               'title': topic_title_hint or topic_id,
                               'score': 99, 'is_core': False}]

            questions.append({
                'q_num':         q_num,
                'page_idx':      0,
                'difficulty':    difficulty,
                'topics':        topics,
                'img_bytes_b64': base64.b64encode(img_bytes).decode(),
                'img_w':         img_w,
                'img_h':         img_h,
                'label':         f'Q{q_num:02d}',
            })

        doc.close()

        if not questions:
            return jsonify({'error': '未能识别出题目，请确认这是本工具导出的PDF'}), 400

        # 自动检测大纲类型（用第一题的 topic_id）
        detected_syllabus = 'edexcel_maths'
        if questions:
            first_topics = questions[0].get('topics', [])
            if first_topics:
                detected_syllabus = _auto_detect_syllabus_type(
                    first_topics[0].get('id', '')) or 'edexcel_maths'

        # 存入 multi_sessions（内存 + 磁盘持久化）
        _new_groups = [{
            'filename':        f.filename,
            'path':            save_path,
            'source':          'imported',
            'paper_type':      detected_syllabus,
            'maths_unit':      None,
            'questions':       questions,
            'total_questions': len(questions),
        }]
        with _multi_sessions_lock:
            _multi_sessions[session_id] = _new_groups
        _save_session_to_disk(session_id, _new_groups)

        return jsonify({
            'session_id':      session_id,
            'total_questions': len(questions),
            'questions':       questions,
            'source':          'imported',
            'filename':        f.filename,
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500



# 设计原则：
#   1. 不产生全题合并大图；按"源 PDF 页片段"逐个渲染，用完即 del
#   2. fitz.Pixmap → JPEG bytes → insert_image；不经过 PIL
#   3. 跨页题目 = 多个 PDF 输出页连续排布（而非把所有页先拼成一张大图）
# ─────────────────────────────────────────────────────────────────────────────

def _pixmap_to_jpeg_bytes(pix, quality=88):
    """fitz.Pixmap → JPEG bytes（直接用 fitz 压缩，不需要 PIL）"""
    try:
        return pix.tobytes("jpeg", jpg_quality=quality)
    except Exception:
        # 旧版 fallback
        return pix.tobytes("png")


def _draw_header(page, x0, y0, x1, label, header_h, font_sz):
    """已弃用：原蓝色标题栏，不再调用。保留仅供向后兼容。"""
    pass  # Request 1: 删除每题蓝色标题栏


def _collect_question_slices(src_doc, questions, q_idx, paper_type):
    """
    返回题目所跨的"源页片段"列表，每项：
      (page_obj, clip_rect)  —— clip_rect 单位：PDF pt
    不产生任何像素数据。

    支持格式：
    - Cambridge MCQ / Structured：内容区 x=30~pw-15
    - Edexcel（含 IAL/IGCSE）：内容区 x=38~548（避开两侧装饰条）
    - Edexcel Maths (P3-style)：整页模式，只取题目页（1页），跳过空白答题页
    """
    # ── Edexcel Maths：只取题目页，截到最后一个 marks 下方（不含答题横线） ──
    if paper_type == 'edexcel_maths':
        q = questions[q_idx]
        page = src_doc[q["page_idx"]]
        pw, ph = page.rect.width, page.rect.height

        # left: 跳过左边框线（x=35~36.5），内容起始 x≈42.5
        left  = 42
        right = min(pw - 36, 560)
        _right_lim = _detect_right_content_limit(page, pw, ph,
                                                  sample_y0=44, sample_y1=ph - 40)
        if _right_lim < right:
            right = _right_lim

        # top: 从题号 y_start 往上 8pt（y_start 由新检测函数提供，>0）
        # 如 y_start=0（旧兼容），则退回用 48
        y_start = q.get("y_start", 0)
        top = max(0, y_start - 8) if y_start > 10 else 48

        # bottom: 截到最后一个右侧 marks(N) 的 y1 + 8pt padding
        bottom = _find_edexcel_maths_question_bottom(page, ph)

        if bottom > top + 10:
            return [(page, fitz.Rect(left, top, right, bottom))]
        return []

    q = questions[q_idx]
    pg_start = q["page_idx"]
    y_top    = max(0, q["y_start"] - 8)

    if q_idx + 1 < len(questions):
        nq     = questions[q_idx + 1]
        pg_end = nq["page_idx"]
        y_end  = nq["y_start"] - 8
    else:
        pg_end = _find_last_content_page(src_doc, pg_start)
        y_end  = None

    # Edexcel 有左右两侧的 "DO NOT WRITE" 装饰条，裁掉边缘
    is_edexcel = paper_type in ('edexcel', 'edexcel_mcq')

    slices = []
    for pg_i in range(pg_start, pg_end + 1):
        page = src_doc[pg_i]
        pw, ph = page.rect.width, page.rect.height

        if is_edexcel:
            left, right = 36, min(pw - 36, 550)   # 避开 Edexcel 两侧装饰条
        else:
            left, right = 30, pw - 15

        if pg_i == pg_start and pg_i == pg_end:
            top    = y_top
            bottom = _find_content_bottom(page, ph) if y_end is None else min(ph, y_end)
        elif pg_i == pg_start:
            top    = y_top
            bottom = _find_content_bottom(page, ph)
        elif pg_i == pg_end:
            top    = 50 if is_edexcel else 55   # Edexcel 页眉较窄
            bottom = _find_content_bottom(page, ph) if y_end is None else min(ph, y_end)
        else:
            top    = 50 if is_edexcel else 55
            bottom = _find_content_bottom(page, ph)

        if bottom > top + 10:
            slices.append((page, fitz.Rect(left, top, right, bottom)))

    return slices


def _render_slice_to_jpeg(src_page, clip_rect, dpi):
    """
    将源 PDF 的一个矩形片段渲染为 JPEG bytes。
    渲染完立即 del pixmap，最小化内存驻留时间。
    返回 (jpeg_bytes, px_width, px_height)
    """
    scale = dpi / 72.0
    mat   = fitz.Matrix(scale, scale)
    pix   = src_page.get_pixmap(matrix=mat, clip=clip_rect, colorspace=fitz.csRGB)
    w, h  = pix.width, pix.height
    data  = _pixmap_to_jpeg_bytes(pix)
    del pix   # 立即释放
    return data, w, h


# ─────────────────────────────────────────────────────────────────────────────
# Logo & 封面页
# ─────────────────────────────────────────────────────────────────────────────
_LOGO_PATH = os.path.join(os.path.dirname(__file__), 'static', 'logo_yuanxuetong_new.png')
_LOGO_BYTES = None  # 延迟加载

def _get_logo_bytes():
    """加载企业 Logo（PNG）字节，懒加载并缓存。"""
    global _LOGO_BYTES
    if _LOGO_BYTES is None:
        if os.path.exists(_LOGO_PATH):
            with open(_LOGO_PATH, 'rb') as fp:
                _LOGO_BYTES = fp.read()
    return _LOGO_BYTES


def _draw_logo_on_page(page, page_w, margin=36, logo_max_w=80, logo_max_h=28):
    """
    在页面右上角绘制企业 Logo。
    logo_max_w, logo_max_h: Logo 显示最大尺寸(pt)，等比缩放。
    """
    logo_bytes = _get_logo_bytes()
    if not logo_bytes:
        return
    try:
        # 获取原始尺寸用于等比计算
        from PIL import Image as PILImage
        img = PILImage.open(io.BytesIO(logo_bytes))
        orig_w, orig_h = img.size
        scale = min(logo_max_w / orig_w, logo_max_h / orig_h)
        draw_w = orig_w * scale
        draw_h = orig_h * scale
        # 右上角位置
        x1 = page_w - margin / 2
        y0 = margin / 2 - 2
        rect = fitz.Rect(x1 - draw_w, y0, x1, y0 + draw_h)
        stream = io.BytesIO(logo_bytes)
        page.insert_image(rect, stream=stream)
    except Exception:
        pass  # Logo 加载失败不影响主流程


def _generate_cover_page(out_doc, title_text, subtitle_text=None,
                          page_w=595, page_h=842, margin=60):
    """
    生成一张封面页并追加到 out_doc。
    封面设计：
      - 全页深蓝色顶部装饰带（占 1/3 页高）
      - Logo 居中显示在装饰带内
      - 标题（白色，大字）
      - 副标题（浅灰，中字）
      - 底部署名
    """
    logo_bytes = _get_logo_bytes()
    page = out_doc.new_page(width=page_w, height=page_h)

    NAVY     = (0.09, 0.26, 0.55)   # 深蓝
    WHITE    = (1.0, 1.0, 1.0)
    LIGHT    = (0.85, 0.92, 1.0)    # 浅蓝
    GOLD     = (1.0, 0.82, 0.22)    # 金色
    DARK_TXT = (0.15, 0.20, 0.35)   # 深色文字

    # ── 顶部深蓝装饰带 ──
    band_h = page_h * 0.42
    page.draw_rect(fitz.Rect(0, 0, page_w, band_h),
                   color=NAVY, fill=NAVY)

    # 装饰线（金色细线）
    y_line = band_h + 6
    page.draw_line(fitz.Point(margin, y_line),
                   fitz.Point(page_w - margin, y_line),
                   color=GOLD, width=2.5)

    # ── Logo（居中于装饰带内）──
    if logo_bytes:
        try:
            from PIL import Image as PILImage
            img = PILImage.open(io.BytesIO(logo_bytes))
            orig_w, orig_h = img.size
            max_logo_w = page_w * 0.40
            max_logo_h = band_h * 0.30
            scale = min(max_logo_w / orig_w, max_logo_h / orig_h)
            lw = orig_w * scale
            lh = orig_h * scale
            cx = page_w / 2
            ly = band_h * 0.18
            logo_rect = fitz.Rect(cx - lw/2, ly, cx + lw/2, ly + lh)
            page.insert_image(logo_rect, stream=io.BytesIO(logo_bytes))
        except Exception:
            pass

    # ── 公司名（装饰带内，Logo 下方）──
    company_y = band_h * 0.52
    page.insert_text(
        (page_w / 2 - 60, company_y),
        '淵學通  YUANXUETONG',
        fontsize=16, color=LIGHT, fontname='helv'
    )

    # ── 主标题（装饰带下方，大字）──
    title_y = band_h + 68
    # 根据标题长度选择字号（英文字符约 6pt/char，中文约 12pt/char）
    title_len_est = sum(12 if '\u4e00' <= c <= '\u9fff' else 7 for c in title_text)
    avail_w = page_w - 2 * margin
    title_fs = min(36, max(20, int(avail_w / max(title_len_est / 36, 1))))
    # 水平居中估算
    title_x = margin
    page.insert_text(
        (title_x, title_y),
        title_text,
        fontsize=title_fs, color=DARK_TXT, fontname='helv'
    )

    # 标题下方金色横线
    page.draw_line(fitz.Point(margin, title_y + 8),
                   fitz.Point(page_w - margin, title_y + 8),
                   color=GOLD, width=1.0)

    # ── 副标题 ──
    if subtitle_text:
        page.insert_text(
            (margin, title_y + 36),
            subtitle_text,
            fontsize=13, color=(0.4, 0.45, 0.55), fontname='helv'
        )

    # ── 底部信息区 ──
    bottom_y = page_h - margin - 30
    # 底部细分隔线
    page.draw_line(fitz.Point(margin, bottom_y - 10),
                   fitz.Point(page_w - margin, bottom_y - 10),
                   color=(0.75, 0.80, 0.90), width=0.8)
    page.insert_text(
        (margin, bottom_y + 10),
        'Powered by 淵學通 · YUANXUETONG',
        fontsize=9, color=(0.55, 0.60, 0.70), fontname='helv'
    )

    # 右下角年份
    import datetime
    year_str = str(datetime.datetime.now().year)
    page.insert_text(
        (page_w - margin - 40, bottom_y + 10),
        year_str,
        fontsize=9, color=(0.55, 0.60, 0.70), fontname='helv'
    )

    # ── 底部装饰带（浅色）──
    page.draw_rect(fitz.Rect(0, page_h - 18, page_w, page_h),
                   color=NAVY, fill=NAVY)

    return page


def _draw_star_shape(page, cx, cy, r_outer, r_inner, filled=True,
                      fill_color=(0.96, 0.62, 0.04),
                      stroke_color=(0.96, 0.62, 0.04)):
    """
    在 page 上用矢量多边形绘制一颗5角星。
    cx, cy: 中心坐标（pt）
    r_outer: 外半径，r_inner: 内半径
    filled: True=实心（金黄），False=空心（描边）
    """
    import math
    pts = []
    for i in range(10):
        angle = math.radians(-90 + i * 36)   # 从12点方向开始
        r = r_outer if i % 2 == 0 else r_inner
        pts.append(fitz.Point(cx + r * math.cos(angle),
                               cy + r * math.sin(angle)))
    shape = page.new_shape()
    shape.draw_polyline(pts + [pts[0]])
    if filled:
        shape.finish(fill=fill_color, color=stroke_color, width=0.3, closePath=True)
    else:
        shape.finish(fill=(1, 1, 1), color=stroke_color, width=0.8, closePath=True)
    shape.commit()


def _draw_difficulty_stars(page, x_start, y_center, n_filled, n_total=5,
                             r_outer=5.5, r_inner=2.2, gap=1.5):
    """
    在 page 上绘制 n_total 颗星，前 n_filled 颗实心，其余空心。
    返回绘制完成后的 x 右边界。
    """
    x = x_start + r_outer
    for i in range(n_total):
        filled = (i < n_filled)
        _draw_star_shape(page, x, y_center, r_outer, r_inner, filled=filled)
        x += r_outer * 2 + gap
    return x - gap  # 最右边界


def _place_jpeg_on_page(out_page, jpeg_bytes, img_w, img_h,
                         area_rect, label, header_h, gap, font_sz,
                         show_header=False, q_meta=None, draw_logo=True):
    """
    把一段 JPEG 图像放入输出 PDF 页的 area_rect 区域。
    白色背景，干净排版：
      - 顶部信息行：题号 | ★★★ 难度 | 知识点... | 年份 | Logo（右对齐）
      - 无蓝色背景色块，仅用细线和文字颜色区分
      - 图片紧跟信息行下方，左对齐，宽度铺满
    q_meta: dict {difficulty, topics, exam_date} 或 None
    """
    x0, y0, x1, y1 = area_rect.x0, area_rect.y0, area_rect.x1, area_rect.y1

    INFO_H  = 28        # 信息行高度(pt)
    SEP_Y   = 1.5       # 信息行底部分隔线厚度
    IMG_GAP = 6         # 分隔线到图片之间的间距

    # 定义颜色
    C_LABEL    = (0.10, 0.25, 0.55)   # 题号：深蓝
    C_STAR_ON  = (0.95, 0.65, 0.10)   # 实心星：金黄
    C_STAR_OFF = (0.75, 0.80, 0.85)   # 空心星：浅灰
    C_DIFF     = (0.35, 0.50, 0.75)   # 难度文字：蓝灰
    C_TOPIC_A  = (0.10, 0.50, 0.60)   # 核心知识点：青蓝
    C_TOPIC_B  = (0.40, 0.55, 0.65)   # 普通知识点：灰蓝
    C_DATE     = (0.50, 0.50, 0.50)   # 年份：中灰
    C_SEP_LINE = (0.75, 0.82, 0.90)   # 分隔线：浅蓝灰
    C_DIVIDER  = (0.82, 0.87, 0.92)   # 竖线分隔符：更浅

    text_y  = y0 + INFO_H - 9    # 文字基线（距底9pt，垂直居中效果）
    star_cy = y0 + INFO_H / 2    # 星形垂直居中

    cur_x   = x0 + 8

    if q_meta is not None:
        diff      = q_meta.get('difficulty')
        topics    = q_meta.get('topics') or []
        exam_date = q_meta.get('exam_date', '')

        # ── Logo 预计算（右侧占位）──
        logo_bytes = _get_logo_bytes() if draw_logo else None
        logo_draw_w = 0
        logo_draw_h = 0
        LOGO_MAX_H = INFO_H - 6   # logo高度最大22pt
        LOGO_MAX_W = 90
        LOGO_PAD_R = 8            # 右边距
        if logo_bytes:
            try:
                from PIL import Image as _PILImg
                _li = _PILImg.open(io.BytesIO(logo_bytes))
                _lw, _lh = _li.size
                _ls = min(LOGO_MAX_W / _lw, LOGO_MAX_H / _lh)
                logo_draw_w = _lw * _ls
                logo_draw_h = _lh * _ls
            except Exception:
                logo_draw_w = 70
                logo_draw_h = LOGO_MAX_H

        # Logo 右对齐
        if logo_draw_w > 0:
            lx1 = x1 - LOGO_PAD_R
            lx0 = lx1 - logo_draw_w
            ly0 = y0 + (INFO_H - logo_draw_h) / 2
            try:
                out_page.insert_image(
                    fitz.Rect(lx0, ly0, lx1, ly0 + logo_draw_h),
                    stream=io.BytesIO(logo_bytes)
                )
            except Exception:
                logo_draw_w = 0

        # 右侧可用终点（logo左边留8pt间距）
        right_end = x1 - logo_draw_w - LOGO_PAD_R - 8

        # ── 题号 ──
        lbl_fs = 11
        out_page.insert_text((cur_x, text_y), label,
                             fontsize=lbl_fs, color=C_LABEL, fontname='helv')
        cur_x += len(label) * 6.8 + 6

        # 竖线
        out_page.draw_line(fitz.Point(cur_x, y0 + 6),
                           fitz.Point(cur_x, y0 + INFO_H - 6),
                           color=C_DIVIDER, width=0.8)
        cur_x += 8

        # ── 星级难度 ──
        if diff and 1 <= diff <= 5:
            diff_names = {1:'Starter', 2:'Basic', 3:'Medium', 4:'Hard', 5:'Expert'}
            # 绘制 5 颗星（自定义颜色，不调 _draw_difficulty_stars）
            sr = 4.5    # 外半径
            si = 1.8    # 内半径
            sg = 1.5    # 间距
            sx = cur_x + sr
            for i in range(5):
                filled = (i < diff)
                _draw_star_shape(out_page, sx, star_cy, sr, si, filled=filled,
                                 fill_color=C_STAR_ON if filled else (1,1,1),
                                 stroke_color=C_STAR_ON if filled else C_STAR_OFF)
                sx += sr * 2 + sg
            cur_x = sx - sg + 4
            # 难度名
            dname = diff_names.get(diff, '')
            out_page.insert_text((cur_x, text_y), dname,
                                 fontsize=8, color=C_DIFF, fontname='helv')
            cur_x += len(dname) * 5.0 + 6

            # 竖线
            out_page.draw_line(fitz.Point(cur_x, y0 + 6),
                               fitz.Point(cur_x, y0 + INFO_H - 6),
                               color=C_DIVIDER, width=0.8)
            cur_x += 8

        # ── 知识点 ──
        # 预留右侧空间：日期 + logo
        date_reserve = (len(exam_date) * 5.2 + 14) if exam_date else 0
        topic_right  = right_end - date_reserve

        if topics:
            s_topics = sorted(topics,
                key=lambda t: (0 if t.get('is_core') else 1, -t.get('score', 0)))
            for ti, tp in enumerate(s_topics):
                tp_id    = tp.get('id', '')
                tp_title = tp.get('title', '')
                is_core  = tp.get('is_core', False)
                if not tp_id:
                    continue
                t_str = f'★{tp_id} {tp_title}' if is_core else f'{tp_id} {tp_title}'
                t_w   = len(t_str) * 5.0 + 6
                if cur_x + t_w > topic_right:
                    if ti < len(s_topics) and cur_x + 18 < topic_right:
                        out_page.insert_text((cur_x, text_y),
                                             f'+{len(s_topics)-ti}',
                                             fontsize=7, color=C_TOPIC_B, fontname='helv')
                    break
                tc = C_TOPIC_A if is_core else C_TOPIC_B
                out_page.insert_text((cur_x, text_y), t_str,
                                     fontsize=8, color=tc, fontname='helv')
                cur_x += t_w
                if ti < len(s_topics) - 1 and cur_x + 10 < topic_right:
                    out_page.insert_text((cur_x, text_y), ' / ',
                                         fontsize=7, color=C_DIVIDER, fontname='helv')
                    cur_x += 12

        # ── 年份（Logo 左侧右对齐）──
        if exam_date:
            dw   = len(exam_date) * 5.2 + 4
            dx   = right_end - dw
            if dx > cur_x + 4:
                out_page.insert_text((dx, text_y), exam_date,
                                     fontsize=8, color=C_DATE, fontname='helv')

        # ── 底部细分隔线 ──
        out_page.draw_line(
            fitz.Point(x0, y0 + INFO_H),
            fitz.Point(x1, y0 + INFO_H),
            color=C_SEP_LINE, width=SEP_Y
        )

        # 图片从信息行下方开始
        img_y0 = y0 + INFO_H + SEP_Y + IMG_GAP
    else:
        img_y0 = y0 + gap

    # ── 放置题目图片 ──
    img_area_w = x1 - x0
    img_area_h = y1 - img_y0 - gap
    if img_area_h < 10 or img_area_w < 10:
        return

    scale  = min(img_area_w / img_w, img_area_h / img_h)
    draw_w = img_w * scale
    draw_h = img_h * scale
    # 左对齐（与原版相同）
    img_rect = fitz.Rect(x0, img_y0, x0 + draw_w, img_y0 + draw_h)

    stream = io.BytesIO(jpeg_bytes)
    out_page.insert_image(img_rect, stream=stream)




def _estimate_slice_height_on_page(src_page, clip_rect, dpi, avail_w):
    """
    快速估算一个源 PDF 片段渲染到输出页后的显示高度（pt），不产生像素数据。
    用于 MCQ 打包布局的高度预算。
    """
    scale     = dpi / 72.0
    clip_w_pt = clip_rect.x1 - clip_rect.x0
    clip_h_pt = clip_rect.y1 - clip_rect.y0
    if clip_w_pt <= 0:
        return 0
    # 渲染后像素尺寸
    px_w = clip_w_pt * scale
    px_h = clip_h_pt * scale
    # 按 avail_w 等比缩放
    display_scale = avail_w / px_w
    return px_h * display_scale


def _export_one_per_page(out_doc, src_doc, questions, q_nums, dpi,
                          paper_type, PW, PH, M, HH, GAP, FS, progress_cb=None,
                          seq_start=0):
    """
    大题模式：每题独占一页（或多页）。
    seq_start: 全局导出序号起始值（0-based），用于头栏题号显示。
    """
    is_mcq_type = paper_type in ('mcq', 'edexcel_mcq')

    if is_mcq_type:
        # ── MCQ 打包模式：多题共页，按高度累积，溢出换页 ──
        _export_mcq_packed(out_doc, src_doc, questions, q_nums, dpi,
                           paper_type, PW, PH, M, GAP, progress_cb,
                           seq_start=seq_start)
        return

    # ── 大题模式：每题独占一页（或多页切片）──
    AVAIL_W = PW - 2 * M
    AVAIL_H = PH - 2 * M - 2 * GAP   # 去掉标题栏高度，直接用全部可用高度

    for done, q_num in enumerate(q_nums):  # 保持传入顺序，不再 sorted()
        q_idx = next((i for i, q in enumerate(questions) if q['q_num'] == q_num), None)
        if q_idx is None:
            if progress_cb: progress_cb(done + 1)
            continue

        q_obj  = questions[q_idx]
        # 用全局导出序号作为头栏题号，而非原始q_num
        export_seq = seq_start + done + 1
        label  = f'第 {export_seq} 题' if paper_type == 'structured' else f'Q{export_seq:02d}'
        slices = _collect_question_slices(src_doc, questions, q_idx, paper_type)

        # 构建本题的 meta（难度+知识点+年份），供头栏绘制
        q_meta = None
        diff      = q_obj.get('difficulty')
        topics    = q_obj.get('topics') or []
        exam_date = q_obj.get('exam_date', '')
        if diff is not None or topics or exam_date:
            q_meta = {'difficulty': diff, 'topics': topics, 'exam_date': exam_date}

        if not slices:
            if progress_cb: progress_cb(done + 1)
            continue

        # 先渲染第一片，决定是否需要分页
        first_src_page, first_clip = slices[0]
        jpeg0, w0, h0 = _render_slice_to_jpeg(first_src_page, first_clip, dpi)

        # 计算第一片在页面上的显示高度
        scale0    = AVAIL_W / w0
        fitted_h0 = h0 * scale0

        if len(slices) == 1 and fitted_h0 <= AVAIL_H:
            # ── 最简单情况：单片且放得下 ──
            page = out_doc.new_page(width=PW, height=PH)
            _place_jpeg_on_page(page, jpeg0, w0, h0,
                                fitz.Rect(M, M, PW-M, PH-M),
                                label, HH, GAP, FS, q_meta=q_meta)
        else:
            # ── 多片 或 单片但太高：逐片放到新 PDF 页 ──
            all_slices = slices[:]
            for si, (src_page, clip) in enumerate(all_slices):
                if si == 0:
                    jpeg, w, h = jpeg0, w0, h0
                else:
                    jpeg, w, h = _render_slice_to_jpeg(src_page, clip, dpi)
                # 每片单独一页（简单可靠）
                # 只在第一片画头栏，后续片不重复
                out_page = out_doc.new_page(width=PW, height=PH)
                _place_jpeg_on_page(out_page, jpeg, w, h,
                                    fitz.Rect(M, M, PW-M, PH-M),
                                    label, HH, GAP, FS,
                                    q_meta=(q_meta if si == 0 else None))
                del jpeg  # 立即释放本片内存

        if progress_cb: progress_cb(done + 1)


def _export_mcq_packed(out_doc, src_doc, questions, q_nums, dpi,
                        paper_type, PW, PH, M, GAP, progress_cb=None,
                        seq_start=0):
    """
    MCQ 多题共页打包布局。
    策略：
      - 每题只取第一个片段（MCQ 每题通常在单页内）
      - 按顺序累积高度，当前页放不下时换新页
      - 题目之间加小间隔 ITEM_GAP
    """
    AVAIL_W  = PW - 2 * M
    AVAIL_H  = PH - 2 * M          # 每页可用总高度
    ITEM_GAP = 10                  # 题目间距（pt）

    sorted_nums = list(q_nums)   # 保持传入顺序，不再 sorted()
    cur_page    = None
    cur_y       = M                # 当前页已使用的 y 位置
    done        = 0

    for q_num in sorted_nums:
        q_idx = next((i for i, q in enumerate(questions) if q['q_num'] == q_num), None)
        if q_idx is None:
            done += 1
            if progress_cb: progress_cb(done)
            continue

        slices = _collect_question_slices(src_doc, questions, q_idx, paper_type)
        if not slices:
            done += 1
            if progress_cb: progress_cb(done)
            continue

        # MCQ 只取第一片
        src_page, clip = slices[0]

        # 先估算高度，决定是否换页
        est_h = _estimate_slice_height_on_page(src_page, clip, dpi, AVAIL_W)
        est_h = max(est_h, 40)   # 最小高度保障

        # 如果当前页放不下（且不是刚开始的页），换新页
        if cur_page is None or (cur_y + est_h + ITEM_GAP > M + AVAIL_H and cur_y > M + 20):
            cur_page = out_doc.new_page(width=PW, height=PH)
            cur_y    = M

        # 渲染并放置
        jpeg, w, h = _render_slice_to_jpeg(src_page, clip, dpi)

        # 实际缩放高度（以像素 → pt 反算）
        scale_factor  = AVAIL_W / w
        actual_draw_h = h * scale_factor

        # 限制单题最大高度（避免超大题目撑满整页）
        if actual_draw_h > AVAIL_H - 2 * GAP:
            actual_draw_h = AVAIL_H - 2 * GAP

        export_seq = seq_start + done + 1
        q_obj_mc = questions[q_idx]
        diff_mc      = q_obj_mc.get('difficulty')
        topics_mc    = q_obj_mc.get('topics') or []
        exam_date_mc = q_obj_mc.get('exam_date', '')
        q_meta_mc = {'difficulty': diff_mc, 'topics': topics_mc, 'exam_date': exam_date_mc} \
                    if (diff_mc is not None or topics_mc or exam_date_mc) else None
        area = fitz.Rect(M, cur_y, PW - M, cur_y + actual_draw_h + 2 * GAP)
        _place_jpeg_on_page(cur_page, jpeg, w, h, area, f'Q{export_seq:02d}', 0, GAP, 11,
                            q_meta=q_meta_mc)
        del jpeg

        cur_y += actual_draw_h + 2 * GAP + ITEM_GAP

        done += 1
        if progress_cb: progress_cb(done)


def _export_two_per_page(out_doc, src_doc, questions, q_nums, dpi,
                          paper_type, PW, PH, M, HH, GAP, FS, progress_cb=None,
                          seq_start=0):
    """每页左右两列各放一题（MCQ 适用，大题建议 one_per_page）"""
    COL_GAP = 12
    COL_W   = (PW - 2 * M - COL_GAP) / 2
    COL_H   = PH - 2 * M

    sorted_nums = list(q_nums)   # 保持传入顺序，不再 sorted()
    done = 0
    for i in range(0, len(sorted_nums), 2):
        page = out_doc.new_page(width=PW, height=PH)
        for col, q_num in enumerate(sorted_nums[i:i+2]):
            q_idx = next((j for j, q in enumerate(questions) if q['q_num'] == q_num), None)
            if q_idx is None:
                done += 1
                if progress_cb: progress_cb(done)
                continue
            q_obj  = questions[q_idx]
            export_seq = seq_start + done + 1
            label  = f'Q{export_seq:02d}' if paper_type == 'mcq' else f'第{export_seq}题'
            # 构建本题 meta
            diff      = q_obj.get('difficulty')
            topics    = q_obj.get('topics') or []
            exam_date = q_obj.get('exam_date', '')
            q_meta = {'difficulty': diff, 'topics': topics, 'exam_date': exam_date} if (diff is not None or topics or exam_date) else None
            # 取第一个片段即可（two_per_page 主要用于 MCQ，单页题）
            slices = _collect_question_slices(src_doc, questions, q_idx, paper_type)
            if slices:
                src_page, clip = slices[0]
                jpeg, w, h = _render_slice_to_jpeg(src_page, clip, dpi)
                x0   = M + col * (COL_W + COL_GAP)
                area = fitz.Rect(x0, M, x0 + COL_W, M + COL_H)
                _place_jpeg_on_page(page, jpeg, w, h, area,
                                    label, HH, GAP, FS, q_meta=q_meta)
                del jpeg  # 立即释放
            done += 1
            if progress_cb: progress_cb(done)




# ─────────────────────────────────────────────────────────────────────────────
# 功能6：保存题册 / 读取题册
# ─────────────────────────────────────────────────────────────────────────────
_WORKBOOK_MARKER = '##YUANXUETONG_WORKBOOK_V1##'  # 题册识别标记

# ─────────────────────────────────────────────────────────────────────────────
# 功能7：题册图书馆（服务器端持久化存储）
# 目录结构（本地模式）: uploads/library/{exam_board}/{subject}/{wb_id}/
#          （R2 模式）: library/{exam_board}/{subject}/{wb_id}/ (R2 Key 前缀)
#   manifest.json  — 题册元数据（标题、题目列表、难度、知识点等）
#   q_{n}.jpg      — 各题图片
# ─────────────────────────────────────────────────────────────────────────────
# 本地模式使用本地目录，R2 模式使用 'library/' 前缀
if not storage.is_r2_mode():
    _LIBRARY_DIR = os.path.join(os.path.dirname(__file__), 'uploads', 'library')
    os.makedirs(_LIBRARY_DIR, exist_ok=True)
else:
    _LIBRARY_DIR = None  # R2 模式不使用本地 library 目录

_EXAM_BOARDS  = ['Edexcel', 'CAIE', 'AQA']
_SUBJECTS_MAP = {
    'Edexcel': ['数学 Maths',    '高数 Further Maths', '物理 Physics',
                '化学 Chemistry','生物 Biology',        '经济 Economics',
                '商业 Business', '会计 Accounting'],
    'CAIE':    ['数学 Maths',    '高数 Further Maths', '物理 Physics',
                '化学 Chemistry','生物 Biology',        '经济 Economics',
                '商业 Business', '会计 Accounting'],
    'AQA':     ['数学 Maths',    '高数 Further Maths', '物理 Physics',
                '化学 Chemistry','生物 Biology',        '经济 Economics',
                '商业 Business', '会计 Accounting'],
}


def _lib_key_prefix(board: str, subject: str, wb_id: str = '') -> str:
    """返回图书馆存储 key 前缀（R2 key 或本地目录路径）。"""
    safe_board   = re.sub(r'[^A-Za-z0-9_\-\u4e00-\u9fff ]', '', board).strip()
    safe_subject = re.sub(r'[^A-Za-z0-9_\-\u4e00-\u9fff ]', '', subject).strip()
    if storage.is_r2_mode():
        if wb_id:
            return f'library/{safe_board}/{safe_subject}/{wb_id}'
        return f'library/{safe_board}/{safe_subject}'
    else:
        if wb_id:
            return os.path.join(_LIBRARY_DIR, safe_board, safe_subject, wb_id)
        return os.path.join(_LIBRARY_DIR, safe_board, safe_subject)

# 保持兼容名称
_lib_path = _lib_key_prefix


@app.route('/api/library/boards', methods=['GET'])
def library_boards():
    """返回考试局和学科列表（固定结构）。"""
    return jsonify({'boards': _EXAM_BOARDS, 'subjects': _SUBJECTS_MAP})


@app.route('/api/library/list', methods=['GET'])
def library_list():
    """
    返回所有已保存的题册树形列表。
    结构: {tree: [{board, subjects: [{subject, workbooks: [{id, title, count, created_at, sort_order, ...}]}]}]}
    支持本地模式和 R2 模式。
    """
    tree = []
    for board in _EXAM_BOARDS:
        subjects_list = []
        for subject in _SUBJECTS_MAP.get(board, []):
            workbooks = []
            if storage.is_r2_mode():
                # R2 模式：扫描 library/{board}/{subject}/ 下的 manifest.json
                safe_board   = re.sub(r'[^A-Za-z0-9_\-\u4e00-\u9fff ]', '', board).strip()
                safe_subject = re.sub(r'[^A-Za-z0-9_\-\u4e00-\u9fff ]', '', subject).strip()
                prefix = f'library/{safe_board}/{safe_subject}/'
                keys   = storage.list_prefix(prefix)
                # 提取所有 wb_id（manifest.json 的父目录）
                seen_wb = set()
                for k in keys:
                    parts = k.split('/')
                    if len(parts) >= 5 and parts[-1] == 'manifest.json':
                        wb_id = parts[3]  # library/board/subject/wb_id/manifest.json
                        if wb_id not in seen_wb:
                            seen_wb.add(wb_id)
                            mfest_key = f'{prefix}{wb_id}/manifest.json'
                            m = storage.load_json(mfest_key)
                            if m:
                                workbooks.append({
                                    'id':           wb_id,
                                    'title':        m.get('title', wb_id),
                                    'count':        m.get('count', 0),
                                    'created_at':   m.get('created_at', ''),
                                    'sort_order':   m.get('sort_order', 'default'),
                                    'maths_unit':   m.get('maths_unit', ''),
                                    'syllabus_type':m.get('syllabus_type', ''),
                                    'exam_date':    m.get('exam_date', ''),
                                })
            else:
                # 本地模式：遍历本地目录
                safe_board   = re.sub(r'[^A-Za-z0-9_\-\u4e00-\u9fff ]', '', board).strip()
                safe_subject = re.sub(r'[^A-Za-z0-9_\-\u4e00-\u9fff ]', '', subject).strip()
                subj_path = os.path.join(_LIBRARY_DIR, safe_board, safe_subject)
                if os.path.isdir(subj_path):
                    for wb_id in sorted(os.listdir(subj_path)):
                        mfest = os.path.join(subj_path, wb_id, 'manifest.json')
                        if not os.path.isfile(mfest):
                            continue
                        try:
                            with open(mfest, 'r', encoding='utf-8') as f:
                                m = json.load(f)
                            workbooks.append({
                                'id':           wb_id,
                                'title':        m.get('title', wb_id),
                                'count':        m.get('count', 0),
                                'created_at':   m.get('created_at', ''),
                                'sort_order':   m.get('sort_order', 'default'),
                                'maths_unit':   m.get('maths_unit', ''),
                                'syllabus_type':m.get('syllabus_type', ''),
                                'exam_date':    m.get('exam_date', ''),
                            })
                        except Exception:
                            pass
            # 按创建时间降序
            workbooks.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            subjects_list.append({'subject': subject, 'workbooks': workbooks})
        tree.append({'board': board, 'subjects': subjects_list})
    return jsonify({'tree': tree})


@app.route('/api/library/save', methods=['POST'])
def library_save():
    """
    保存题册到图书馆（服务器端持久化）。

    支持两种模式：
    模式A（推荐）— 服务端裁图：
      前端传: {
        session_id,          # 上传 session
        title, board, subject, sort_order, maths_unit, syllabus_type,
        questions: [{
          q_num, file_idx,    # 定位题目（服务端裁图用）
          difficulty, topics, exam_date, source,
          # 可选：img_bytes_b64（若已有缓存图片，优先使用，跳过裁图）
        }]
      }

    模式B（兼容）— 前端传图：
      前端传: {
        title, board, subject, ...,
        questions: [{q_num, img_bytes_b64, img_w, img_h, ...}]
      }
    """
    import base64 as _b64
    data = request.json or {}
    session_id    = data.get('session_id', '')
    title         = (data.get('title') or '未命名题册').strip()
    board         = data.get('board', 'Edexcel')
    subject       = data.get('subject', '数学 Maths')
    sort_order    = data.get('sort_order', 'default')
    maths_unit    = data.get('maths_unit', '')
    syllabus_type = data.get('syllabus_type', 'edexcel_maths')
    questions     = data.get('questions', [])

    if not questions:
        return jsonify({'error': '没有题目数据'}), 400
    if board not in _EXAM_BOARDS:
        return jsonify({'error': f'未知考试局: {board}'}), 400

    # 获取 session（模式A需要）
    sess = None
    if session_id:
        sess = _get_session(session_id)

    wb_id      = str(uuid.uuid4())[:8]
    wb_prefix  = _lib_key_prefix(board, subject, wb_id)   # R2 key 前缀 或 本地目录
    # 本地模式需要创建目录
    if not storage.is_r2_mode():
        os.makedirs(wb_prefix, exist_ok=True)

    saved_q = []
    # 按 file_idx 分组，批量打开 PDF（避免同一文件反复 open/close）
    from collections import defaultdict
    file_idx_map = defaultdict(list)   # file_idx -> [(list_pos, q_dict)]
    q_has_b64    = {}                  # list_pos -> True/False（是否已有图片）

    for i, q in enumerate(questions):
        b64 = q.get('img_bytes_b64', '')
        if b64:
            q_has_b64[i] = True
        else:
            file_idx = int(q.get('file_idx', q.get('gIdx', 0)))
            file_idx_map[file_idx].append((i, q))

    # ── 步骤1：处理已有 b64 的题目（直接写图片文件）──
    img_results = {}   # list_pos -> (img_file, img_w, img_h)
    for i, q in enumerate(questions):
        if not q_has_b64.get(i):
            continue
        b64 = q.get('img_bytes_b64', '')
        img_fname = f'q_{i+1:03d}.jpg'
        try:
            raw = _b64.b64decode(b64)
            # 转成 JPEG（PNG 也存成 jpg）
            from PIL import Image as _PIL
            _im = _PIL.open(io.BytesIO(raw))
            buf = io.BytesIO()
            _im.convert('RGB').save(buf, format='JPEG', quality=88)
            # 统一用 storage 写入
            if storage.is_r2_mode():
                storage.store_bytes(f'{wb_prefix}/{img_fname}', buf.getvalue())
            else:
                img_path = os.path.join(wb_prefix, img_fname)
                with open(img_path, 'wb') as f:
                    f.write(buf.getvalue())
            img_results[i] = (img_fname, _im.width, _im.height)
        except Exception:
            img_results[i] = ('', 0, 0)

    # ── 步骤2：从 PDF session 裁图（模式A）──
    for file_idx, items in file_idx_map.items():
        if not sess or file_idx >= len(sess):
            # session 不可用，这些题目无法获取图片，记录失败
            for (i, q) in items:
                img_results[i] = ('', 0, 0)
            continue

        group      = sess[file_idx]
        save_path  = group['path']
        paper_type = group['paper_type']
        questions_meta = group['questions']

        if not os.path.exists(save_path):
            for (i, q) in items:
                img_results[i] = ('', 0, 0)
            continue

        try:
            doc = fitz.open(save_path)
            for (i, q) in items:
                q_num = int(q.get('q_num', 0))
                q_idx = next((qi for qi, qo in enumerate(questions_meta)
                              if qo['q_num'] == q_num), None)
                if q_idx is None:
                    img_results[i] = ('', 0, 0)
                    continue
                try:
                    img_bytes, w, h = crop_question_image(
                        doc, questions_meta, q_idx,
                        dpi=150, paper_type=paper_type
                    )
                    img_file = f'q_{i+1:03d}.jpg'
                    # PNG -> JPEG
                    from PIL import Image as _PIL
                    _im = _PIL.open(io.BytesIO(img_bytes))
                    buf = io.BytesIO()
                    _im.convert('RGB').save(buf, format='JPEG', quality=88)
                    # 统一用 storage 写入
                    if storage.is_r2_mode():
                        storage.store_bytes(f'{wb_prefix}/{img_file}', buf.getvalue())
                    else:
                        img_path = os.path.join(wb_prefix, img_file)
                        with open(img_path, 'wb') as f:
                            f.write(buf.getvalue())
                    img_results[i] = (img_file, w, h)
                except Exception as ce:
                    img_results[i] = ('', 0, 0)
            doc.close()
        except Exception as e:
            for (i, q) in items:
                img_results[i] = ('', 0, 0)

    # ── 步骤3：构建 manifest ──
    for i, q in enumerate(questions):
        img_file, img_w, img_h = img_results.get(i, ('', 0, 0))
        saved_q.append({
            'seq':        i + 1,
            'q_num':      q.get('q_num', i + 1),
            'file_idx':   int(q.get('file_idx', q.get('gIdx', 0))),
            'difficulty': q.get('difficulty'),
            'topics':     q.get('topics', []),
            'exam_date':  q.get('exam_date', ''),
            'source':     q.get('source', ''),
            'img_file':   img_file,
            'img_w':      img_w,
            'img_h':      img_h,
        })

    success_count = sum(1 for q in saved_q if q.get('img_file'))
    if success_count == 0:
        # 清理已写入的数据
        if storage.is_r2_mode():
            storage.delete_prefix(wb_prefix + '/')
        else:
            import shutil
            shutil.rmtree(wb_prefix, ignore_errors=True)
        return jsonify({'error': '所有题目图片获取失败，题册未保存。请确保在当前会话中保存（不要刷新页面后再保存）'}), 400

    manifest = {
        'version':       2,
        'id':            wb_id,
        'title':         title,
        'board':         board,
        'subject':       subject,
        'sort_order':    sort_order,
        'maths_unit':    maths_unit,
        'syllabus_type': syllabus_type,
        'session_id':    session_id,
        'count':         len(saved_q),
        'success_count': success_count,
        'created_at':    time.strftime('%Y-%m-%d %H:%M', time.localtime()),
        'questions':     saved_q,
    }
    # 写 manifest：R2 模式用 storage.store_json，本地模式写文件
    if storage.is_r2_mode():
        storage.store_json(f'{wb_prefix}/manifest.json', manifest)
    else:
        with open(os.path.join(wb_prefix, 'manifest.json'), 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    msg = f'题册已保存，共 {success_count} 题'
    if success_count < len(saved_q):
        msg += f'（{len(saved_q)-success_count} 题图片获取失败）'
    return jsonify({'ok': True, 'id': wb_id, 'title': title,
                    'count': success_count, 'total': len(saved_q), 'msg': msg})





def _register_workbook_session(wb_id: str, manifest: dict, questions_out: list) -> str:
    """
    将已加载的题册注册为一个虚拟 session，使 export_pdf 等接口可以使用。
    每道题的图片以 img_bytes_b64 存入 questions，
    export_pdf / preview_b64 会优先读取这个字段而不尝试打开 PDF。
    返回新的 session_id（'wb_' + wb_id）。
    """
    import base64 as _b64
    sess_id = f'wb_{wb_id}'
    # 构建虚拟 group（没有真实 PDF 路径，只有图片 base64）
    virt_questions = []
    for i, q in enumerate(questions_out):
        virt_questions.append({
            'q_num':         q.get('q_num', i + 1),
            'page_idx':      0,
            'difficulty':    q.get('difficulty'),
            'topics':        q.get('topics', []),
            'exam_date':     q.get('exam_date', ''),
            'source':        q.get('source', 'workbook'),
            'img_bytes_b64': q.get('img_bytes_b64', ''),
            'img_w':         q.get('img_w', 0),
            'img_h':         q.get('img_h', 0),
        })
    virt_group = {
        'filename':        manifest.get('title', wb_id) + '.pdf',
        'path':            '',                        # 无真实 PDF
        'r2_key':          '',
        'source':          'workbook',
        'paper_type':      manifest.get('syllabus_type', 'edexcel_maths'),
        'maths_unit':      manifest.get('maths_unit', None),
        'exam_date':       '',
        'questions':       virt_questions,
        'total_questions': len(virt_questions),
        'total_pages':     0,
    }
    with _multi_sessions_lock:
        _multi_sessions[sess_id] = [virt_group]
    return sess_id


@app.route('/api/library/load/<wb_id>', methods=['GET'])
def library_load(wb_id):
    """
    读取图书馆中的一个题册，返回题目列表（含图片 base64）。
    前端通过 GET /api/library/load/<wb_id>?board=Edexcel&subject=数学Maths
    """
    import base64 as _b64
    board   = request.args.get('board', '')
    subject = request.args.get('subject', '')
    wb_prefix = _lib_key_prefix(board, subject, wb_id)

    if storage.is_r2_mode():
        manifest = storage.load_json(f'{wb_prefix}/manifest.json')
        if manifest is None:
            return jsonify({'error': '题册不存在'}), 404
        questions_out = []
        for q in manifest.get('questions', []):
            img_file = q.get('img_file', '')
            b64 = ''
            if img_file:
                raw = storage.load_bytes(f'{wb_prefix}/{img_file}')
                if raw:
                    b64 = _b64.b64encode(raw).decode('ascii')
            questions_out.append({
                'seq':           q.get('seq', 0),
                'q_num':         q.get('q_num', 0),
                'difficulty':    q.get('difficulty'),
                'topics':        q.get('topics', []),
                'exam_date':     q.get('exam_date', ''),
                'source':        q.get('source', ''),
                'img_bytes_b64': b64,
                'img_w':         q.get('img_w', 0),
                'img_h':         q.get('img_h', 0),
            })
    else:
        mfest = os.path.join(wb_prefix, 'manifest.json')
        if not os.path.isfile(mfest):
            return jsonify({'error': '题册不存在'}), 404
        with open(mfest, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        questions_out = []
        for q in manifest.get('questions', []):
            img_file = q.get('img_file', '')
            b64 = ''
            if img_file:
                img_path = os.path.join(wb_prefix, img_file)
                if os.path.isfile(img_path):
                    with open(img_path, 'rb') as f:
                        b64 = _b64.b64encode(f.read()).decode('ascii')
            questions_out.append({
                'seq':           q.get('seq', 0),
                'q_num':         q.get('q_num', 0),
                'difficulty':    q.get('difficulty'),
                'topics':        q.get('topics', []),
                'exam_date':     q.get('exam_date', ''),
                'source':        q.get('source', ''),
                'img_bytes_b64': b64,
                'img_w':         q.get('img_w', 0),
                'img_h':         q.get('img_h', 0),
            })

    return jsonify({
        'ok':           True,
        'id':           wb_id,
        'session_id':   _register_workbook_session(wb_id, manifest, questions_out),
        'title':        manifest.get('title', ''),
        'board':        manifest.get('board', board),
        'subject':      manifest.get('subject', subject),
        'sort_order':   manifest.get('sort_order', 'default'),
        'maths_unit':   manifest.get('maths_unit', ''),
        'syllabus_type':manifest.get('syllabus_type', 'edexcel_maths'),
        'count':        manifest.get('count', 0),
        'created_at':   manifest.get('created_at', ''),
        'questions':    questions_out,
    })


@app.route('/api/library/delete/<wb_id>', methods=['DELETE'])
def library_delete(wb_id):
    """删除图书馆中的一个题册。"""
    board   = request.args.get('board', '')
    subject = request.args.get('subject', '')
    wb_prefix = _lib_key_prefix(board, subject, wb_id)
    if storage.is_r2_mode():
        # 检查 manifest 是否存在
        if not storage.exists(f'{wb_prefix}/manifest.json'):
            return jsonify({'error': '题册不存在'}), 404
        storage.delete_prefix(wb_prefix + '/')
    else:
        if not os.path.isdir(wb_prefix):
            return jsonify({'error': '题册不存在'}), 404
        import shutil
        shutil.rmtree(wb_prefix, ignore_errors=True)
    return jsonify({'ok': True})



def save_workbook():
    """
    把当前 session 的题目元数据（难度、知识点）嵌入 PDF 并保存为题册文件。
    题册格式：
      - 第1页：封面（含标题 + 题册标记 + JSON 元数据，用隐藏文本方式嵌入）
      - 后续页：每题图片（同导出格式，带头栏）
    前端传入: {session_id, workbook_title, syllabus_type, maths_unit, questions_meta}
    questions_meta: [{q_num, difficulty, topics, img_bytes_b64, img_w, img_h}]
    """
    data = request.json or {}
    sess_id       = data.get('session_id', '')
    title         = data.get('workbook_title', '题册').strip() or '题册'
    syllabus_type = data.get('syllabus_type', 'edexcel_maths')
    maths_unit    = data.get('maths_unit', '')
    questions_meta = data.get('questions_meta', [])

    if not questions_meta:
        return jsonify({'error': '没有题目数据'}), 400

    try:
        import base64, json as _json
        PAGE_W, PAGE_H = 595, 842
        MARGIN = 36
        BAND_H = _EXPORT_HEADER_HEIGHT_PT

        out_doc = fitz.open()

        # ── 封面页（含嵌入元数据）──
        cover_page = _generate_cover_page(out_doc, title,
                                           subtitle_text=f'共 {len(questions_meta)} 题',
                                           page_w=PAGE_W, page_h=PAGE_H)

        # 把元数据序列化嵌入封面的隐藏文字（白色极小字体，用于读取时解析）
        meta_obj = {
            'marker':       _WORKBOOK_MARKER,
            'title':        title,
            'syllabus':     syllabus_type,
            'unit':         maths_unit,
            'total':        len(questions_meta),
            'questions':    [{
                'q_num':     q.get('q_num'),
                'difficulty':q.get('difficulty'),
                'topics':    q.get('topics', []),
            } for q in questions_meta],
        }
        meta_json = _json.dumps(meta_obj, ensure_ascii=False)
        # 白色极小文字写入封面左下角（人眼不可见，程序可读）
        cover_page.insert_text(
            (2, PAGE_H - 2), meta_json[:3000],  # PyMuPDF单次限制
            fontsize=1, color=(1,1,1), fontname='helv'
        )
        if len(meta_json) > 3000:
            cover_page.insert_text(
                (2, PAGE_H - 1), meta_json[3000:6000],
                fontsize=1, color=(1,1,1), fontname='helv'
            )

        # ── 题目页 ──
        for q in questions_meta:
            q_num  = q.get('q_num', 0)
            diff   = q.get('difficulty')
            topics = q.get('topics', [])
            b64    = q.get('img_bytes_b64', '')
            img_w  = q.get('img_w', 1)
            img_h  = q.get('img_h', 1)

            if not b64:
                continue

            img_bytes = base64.b64decode(b64)
            q_meta = {'difficulty': diff, 'topics': topics} if (diff is not None or topics) else None
            label  = f'Q{q_num:02d}'

            out_page = out_doc.new_page(width=PAGE_W, height=PAGE_H)
            _place_jpeg_on_page(
                out_page, img_bytes, img_w, img_h,
                fitz.Rect(MARGIN, MARGIN, PAGE_W - MARGIN, PAGE_H - MARGIN),
                label, 28, 12, 11, q_meta=q_meta
            )

        task_id  = str(uuid.uuid4())
        out_path = os.path.join(app.config['UPLOAD_FOLDER'], f'wb_{task_id}.pdf')
        out_doc.save(out_path, garbage=4, deflate=True)
        out_doc.close()

        safe_title = re.sub(r'[\\/*?:"<>|]', '_', title)
        return send_file(out_path, mimetype='application/pdf',
                         as_attachment=True, download_name=f'{safe_title}.pdf')

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/load_workbook', methods=['POST'])
def load_workbook():
    """
    读取保存的题册 PDF，还原题目列表（包含元数据）。
    识别方式：
      1. 读取第1页封面，提取隐藏白色文字中的 JSON 元数据
      2. 从后续页提取题目图片（同 import_exported_pdf 逻辑）
    返回：同 import_exported_pdf 格式
    """
    if 'file' not in request.files:
        return jsonify({'error': '请上传题册 PDF'}), 400
    f = request.files['file']
    if not f.filename or not f.filename.lower().endswith('.pdf'):
        return jsonify({'error': '仅支持 PDF 格式'}), 400

    safe = secure_filename(f.filename)
    session_id = str(uuid.uuid4())
    save_path  = os.path.join(_MULTI_DIR, f'{session_id}_{safe}')
    f.save(save_path)

    try:
        import base64, json as _json
        doc = fitz.open(save_path)

        # ── 尝试解析封面元数据 ──
        meta_obj   = None
        wb_title   = ''
        syllabus   = 'edexcel_maths'
        maths_unit = ''
        q_meta_map = {}   # q_num -> {difficulty, topics}

        if len(doc) > 0:
            cover = doc[0]
            # 提取所有文字（包括隐藏的白色文字）
            full_text = cover.get_text('text')
            marker_pos = full_text.find(_WORKBOOK_MARKER)
            if marker_pos != -1:
                # 找到了题册标记，尝试解析 JSON
                json_start = full_text.find('{', marker_pos)
                if json_start != -1:
                    try:
                        meta_obj = _json.loads(full_text[json_start:])
                    except Exception:
                        # 可能被截断，尝试从多个 span 拼接
                        raw_spans = []
                        td = cover.get_text('dict')
                        for block in td.get('blocks', []):
                            if block.get('type') == 0:
                                for line in block.get('lines', []):
                                    for span in line.get('spans', []):
                                        if span.get('size', 99) < 2:
                                            raw_spans.append(span.get('text', ''))
                        combined = ''.join(raw_spans)
                        js_start = combined.find('{')
                        if js_start != -1:
                            try:
                                meta_obj = _json.loads(combined[js_start:])
                            except Exception:
                                pass

        if meta_obj and meta_obj.get('marker') == _WORKBOOK_MARKER:
            wb_title   = meta_obj.get('title', '')
            syllabus   = meta_obj.get('syllabus', 'edexcel_maths')
            maths_unit = meta_obj.get('unit', '')
            for qm in meta_obj.get('questions', []):
                qn = qm.get('q_num')
                if qn:
                    q_meta_map[qn] = {
                        'difficulty': qm.get('difficulty'),
                        'topics':     qm.get('topics', []),
                    }

        # ── 提取题目图片（跳过封面页）──
        MARGIN  = 36
        BAND_H  = _EXPORT_HEADER_HEIGHT_PT
        mat     = fitz.Matrix(2.0, 2.0)

        page_info = []
        start_page = 1 if (meta_obj and meta_obj.get('marker') == _WORKBOOK_MARKER) else 0

        for page_idx in range(start_page, len(doc)):
            page   = doc[page_idx]
            result = _detect_exported_pdf_header(page)
            if result is None:
                if page_info:
                    page_info.append({
                        'q_num':    page_info[-1]['q_num'],
                        'page_obj': page,
                    })
            else:
                q_num, _, _ = result
                page_info.append({'q_num': q_num, 'page_obj': page})

        from collections import OrderedDict
        q_groups = OrderedDict()
        for pi in page_info:
            qn = pi['q_num']
            q_groups.setdefault(qn, []).append(pi['page_obj'])

        edx_syllabus = _load_edexcel_maths_syllabus()
        questions = []
        for q_num, pages in q_groups.items():
            pixmaps = []
            for pg in pages:
                PW = pg.rect.width
                PH = pg.rect.height
                img_rect = fitz.Rect(MARGIN, MARGIN + BAND_H + 12, PW - MARGIN, PH - MARGIN)
                pix = pg.get_pixmap(matrix=mat, clip=img_rect)
                pixmaps.append(pix)

            merged_pil = _stitch_pixmaps_vertical(pixmaps)
            for pix in pixmaps:
                del pix

            if merged_pil is None:
                continue

            buf = io.BytesIO()
            merged_pil.save(buf, format='JPEG', quality=92)
            img_bytes = buf.getvalue()
            img_w, img_h = merged_pil.size

            # 优先使用元数据中的 difficulty/topics
            saved_meta = q_meta_map.get(q_num, {})
            difficulty = saved_meta.get('difficulty')
            topics     = saved_meta.get('topics', [])

            questions.append({
                'q_num':         q_num,
                'page_idx':      0,
                'difficulty':    difficulty,
                'topics':        topics,
                'img_bytes_b64': base64.b64encode(img_bytes).decode(),
                'img_w':         img_w,
                'img_h':         img_h,
                'label':         f'Q{q_num:02d}',
            })

        doc.close()

        if not questions:
            return jsonify({'error': '未识别出题目，请确认是本工具生成的题册'}), 400

        _wb_groups = [{
            'filename':        f.filename,
            'path':            save_path,
            'source':          'workbook',
            'paper_type':      syllabus,
            'maths_unit':      maths_unit,
            'questions':       questions,
            'total_questions': len(questions),
        }]
        with _multi_sessions_lock:
            _multi_sessions[session_id] = _wb_groups
        _save_session_to_disk(session_id, _wb_groups)

        return jsonify({
            'session_id':      session_id,
            'total_questions': len(questions),
            'questions':       questions,
            'source':          'workbook',
            'filename':        f.filename,
            'workbook_title':  wb_title,
            'syllabus':        syllabus,
            'maths_unit':      maths_unit,
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════
# 云端题库系统 v2
# 数据结构（R2 Key / 本地路径）：
#   cloud_db/subjects.json              — 全局学科+考试局元数据
#   cloud_db/{subject}/{board}/topics.json  — 该学科考试局的知识点树（一级/二级）
#   cloud_db/{subject}/{board}/{topic1}/{topic2}/{qid}.json  — 单题元数据
#   cloud_db/images/{qid}.jpg           — 题目图片（所有题目图片集中存放）
#   cloud_db/stats.json                 — 全局统计缓存（定期更新）
# ═══════════════════════════════════════════════════════════════════════════

# ── 学科配置 ──
_CLOUD_SUBJECTS = ['数学 Maths', '物理 Physics', '化学 Chemistry',
                   '生物 Biology', '高数 Further Maths', '经济 Economics',
                   '商业 Business', '会计 Accounting']
_CLOUD_BOARDS   = ['CAIE', 'Edexcel', 'AQA', 'OCR', 'IB', 'AP']

# 知识点树：从现有 syllabus JSON 中加载（cambridge / edexcel_maths），
# 运行时根据 board+subject 动态确定使用哪套知识点

def _cloud_prefix(subject: str = '', board: str = '', topic1: str = '',
                   topic2: str = '', qid: str = '') -> str:
    """生成云端题库的存储 key（R2 key 或本地路径）。"""
    def _safe(s):
        return re.sub(r'[^A-Za-z0-9_\-\u4e00-\u9fff. ]', '_', s).strip()

    parts = ['cloud_db']
    if subject: parts.append(_safe(subject))
    if board:   parts.append(_safe(board))
    if topic1:  parts.append(_safe(topic1))
    if topic2:  parts.append(_safe(topic2))
    if qid:     parts.append(qid)

    if storage.is_r2_mode():
        return '/'.join(parts)
    else:
        base = os.path.join(os.path.dirname(__file__), 'uploads')
        return os.path.join(base, *parts)


def _cloud_img_key(qid: str) -> str:
    """图片存储 key"""
    if storage.is_r2_mode():
        return f'cloud_db/images/{qid}.jpg'
    else:
        base = os.path.join(os.path.dirname(__file__), 'uploads', 'cloud_db', 'images')
        os.makedirs(base, exist_ok=True)
        return os.path.join(base, f'{qid}.jpg')


def _cloud_stats_key() -> str:
    if storage.is_r2_mode():
        return 'cloud_db/_stats.json'
    else:
        base = os.path.join(os.path.dirname(__file__), 'uploads', 'cloud_db')
        os.makedirs(base, exist_ok=True)
        return os.path.join(base, '_stats.json')


def _list_cloud_questions(subject: str, board: str,
                          topic1: str = '', topic2: str = '') -> list:
    """列出指定路径下所有题目元数据（JSON文件）的 key 列表。"""
    prefix = _cloud_prefix(subject, board, topic1, topic2)
    if storage.is_r2_mode():
        prefix_key = prefix + '/'
        all_keys   = storage.list_prefix(prefix_key)
        return [k for k in all_keys if k.endswith('.json') and not k.endswith('_meta.json')]
    else:
        results = []
        if not os.path.isdir(prefix):
            return results
        import glob as _glob
        pattern = os.path.join(prefix, '**', '*.json')
        for fp in _glob.glob(pattern, recursive=True):
            if not os.path.basename(fp).startswith('_'):
                results.append(fp)
        return results


def _load_cloud_question(key: str) -> dict | None:
    """从 key 加载单题元数据。"""
    if storage.is_r2_mode():
        return storage.load_json(key)
    else:
        try:
            with open(key, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None


def _save_cloud_question(key: str, data: dict):
    """保存单题元数据到 key。"""
    if storage.is_r2_mode():
        storage.store_json(key, data)
    else:
        os.makedirs(os.path.dirname(key), exist_ok=True)
        with open(key, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def _build_stats_cache() -> dict:
    """
    重新计算全局统计缓存，返回统计对象：
    {
      total: int,
      by_subject: {subject: {total, by_board: {board: {total, by_topic1: {t1: {total, by_topic2: {t2: int}}}}}}}
    }
    """
    stats = {'total': 0, 'by_subject': {}}

    for subj in _CLOUD_SUBJECTS:
        for board in _CLOUD_BOARDS:
            keys = _list_cloud_questions(subj, board)
            if not keys:
                continue
            subj_stats = stats['by_subject'].setdefault(subj, {'total': 0, 'by_board': {}})
            board_stats = subj_stats['by_board'].setdefault(board, {'total': 0, 'by_topic1': {}})

            for key in keys:
                q = _load_cloud_question(key)
                if not q:
                    continue
                t1 = q.get('topic1', '未分类')
                t2 = q.get('topic2', '未分类')
                t1_stats = board_stats['by_topic1'].setdefault(t1, {'total': 0, 'by_topic2': {}})
                t1_stats['by_topic2'][t2] = t1_stats['by_topic2'].get(t2, 0) + 1
                t1_stats['total'] += 1
                board_stats['total'] += 1
                subj_stats['total'] += 1
                stats['total'] += 1

    # 写缓存
    key = _cloud_stats_key()
    if storage.is_r2_mode():
        storage.store_json(key, stats)
    else:
        os.makedirs(os.path.dirname(key), exist_ok=True)
        with open(key, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False)
    return stats


def _get_stats_cache() -> dict:
    """读取统计缓存，不存在则返回空。"""
    key = _cloud_stats_key()
    if storage.is_r2_mode():
        return storage.load_json(key) or {}
    else:
        try:
            with open(key, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}


def _invalidate_stats():
    """删除统计缓存（下次读取时重建）。"""
    key = _cloud_stats_key()
    try:
        if storage.is_r2_mode():
            storage.delete_object(key)
        else:
            os.remove(key)
    except Exception:
        pass


# ── API: 获取云端题库基本配置 ──
@app.route('/api/cloud_library/config', methods=['GET'])
def cloud_library_config():
    """返回学科列表、考试局列表。"""
    return jsonify({
        'subjects': _CLOUD_SUBJECTS,
        'boards':   _CLOUD_BOARDS,
    })


# ── API: 获取全局/分级统计 ──
@app.route('/api/cloud_library/stats', methods=['GET'])
def cloud_library_stats():
    """
    返回云端题库统计（先查缓存，缓存失效时重算）。
    ?rebuild=1  强制重算
    """
    rebuild = request.args.get('rebuild', '0') == '1'
    if rebuild:
        stats = _build_stats_cache()
    else:
        stats = _get_stats_cache()
        if not stats:
            stats = _build_stats_cache()
    return jsonify(stats)


# ── API: 保存题目到云端题库 ──
@app.route('/api/cloud_library/save_questions', methods=['POST'])
def cloud_library_save_questions():
    """
    将选中的题目批量保存到云端题库。
    请求体：{
      session_id: str,
      questions: [{
        q_num: int,
        file_idx: int,
        subject: str,           # 学科
        board: str,             # 考试局
        topic1: str,            # 知识点一级标题
        topic2: str,            # 知识点二级标题（可为空）
        difficulty: int|null,   # 1-5
        topics: [...],          # 原始 topics 数组
        exam_date: str,         # "October 2023"
        source: str,            # 'cambridge'|'edexcel'|...
        paper_type: str,
        maths_unit: str,
        img_bytes_b64: str|null  # 若前端已有图片 base64，直接传；否则服务端裁图
      }]
    }
    返回：{saved: int, failed: int, ids: [...]}
    """
    import base64 as _b64
    data       = request.json or {}
    session_id = data.get('session_id', '')
    questions  = data.get('questions', [])

    if not questions:
        return jsonify({'error': '没有题目数据'}), 400

    sess = _get_session(session_id) if session_id else None

    from collections import defaultdict
    file_groups = defaultdict(list)  # file_idx -> [(list_pos, q)]
    b64_cache   = {}                 # list_pos -> b64 string

    for i, q in enumerate(questions):
        if q.get('img_bytes_b64'):
            b64_cache[i] = q['img_bytes_b64']
        else:
            file_groups[int(q.get('file_idx', q.get('gIdx', 0)))].append((i, q))

    # 从 PDF session 裁图
    for file_idx, items in file_groups.items():
        if not sess or file_idx >= len(sess):
            continue
        group      = sess[file_idx]
        save_path  = group['path']
        paper_type = group['paper_type']
        q_meta     = group['questions']
        if not os.path.exists(save_path):
            continue
        try:
            doc = fitz.open(save_path)
            for (i, q) in items:
                q_num = int(q.get('q_num', 0))
                q_idx = next((qi for qi, qo in enumerate(q_meta) if qo['q_num'] == q_num), None)
                if q_idx is None:
                    continue
                img_bytes, _, _ = crop_question_image(doc, q_meta, q_idx, dpi=150, paper_type=paper_type)
                from PIL import Image as _PIL
                _im = _PIL.open(io.BytesIO(img_bytes))
                buf = io.BytesIO()
                _im.convert('RGB').save(buf, format='JPEG', quality=88)
                b64_cache[i] = _b64.b64encode(buf.getvalue()).decode()
            doc.close()
        except Exception as e:
            print(f'[cloud_save] crop error file_idx={file_idx}: {e}')

    saved = 0
    failed = 0
    saved_ids = []

    for i, q in enumerate(questions):
        b64 = b64_cache.get(i, '')
        subject  = q.get('subject', '数学 Maths')
        board    = q.get('board', 'CAIE')
        topic1   = (q.get('topic1') or '未分类').strip()
        topic2   = (q.get('topic2') or '通用').strip()

        # 生成唯一题目ID
        qid = str(uuid.uuid4())[:12]

        # 保存图片
        if b64:
            try:
                img_data = _b64.b64decode(b64)
                img_key  = _cloud_img_key(qid)
                if storage.is_r2_mode():
                    storage.store_bytes(img_key, img_data)
                else:
                    with open(img_key, 'wb') as f:
                        f.write(img_data)
            except Exception as e:
                print(f'[cloud_save] img save error qid={qid}: {e}')
                failed += 1
                continue

        # 构建元数据（不含图片 base64，图片单独存）
        q_meta_save = {
            'qid':        qid,
            'q_num':      q.get('q_num'),
            'subject':    subject,
            'board':      board,
            'topic1':     topic1,
            'topic2':     topic2,
            'difficulty': q.get('difficulty'),
            'topics':     q.get('topics', []),
            'exam_date':  q.get('exam_date', ''),
            'source':     q.get('source', ''),
            'paper_type': q.get('paper_type', ''),
            'maths_unit': q.get('maths_unit', ''),
            'has_image':  bool(b64),
            'saved_at':   __import__('datetime').datetime.utcnow().isoformat(),
        }

        # 写元数据 JSON
        meta_key = _cloud_prefix(subject, board, topic1, topic2) + \
                   (f'/{qid}.json' if storage.is_r2_mode() else f'{os.sep}{qid}.json')
        try:
            _save_cloud_question(meta_key, q_meta_save)
            saved += 1
            saved_ids.append(qid)
        except Exception as e:
            print(f'[cloud_save] meta save error qid={qid}: {e}')
            failed += 1

    # 失效统计缓存
    if saved > 0:
        _invalidate_stats()

    return jsonify({'saved': saved, 'failed': failed, 'ids': saved_ids})


# ── API: 查询云端题库题目列表 ──
@app.route('/api/cloud_library/questions', methods=['GET'])
def cloud_library_questions():
    """
    分页查询某节点下的题目列表。
    参数：subject, board, topic1(可选), topic2(可选), page(默认1), per_page(默认30)
    返回：{questions: [...], total: int, page: int, pages: int}
    """
    subject  = request.args.get('subject', '')
    board    = request.args.get('board', '')
    topic1   = request.args.get('topic1', '')
    topic2   = request.args.get('topic2', '')
    page     = max(1, int(request.args.get('page', 1)))
    per_page = min(100, int(request.args.get('per_page', 30)))

    if not subject or not board:
        return jsonify({'error': '必须提供 subject 和 board'}), 400

    keys = _list_cloud_questions(subject, board, topic1, topic2)
    total = len(keys)
    start = (page - 1) * per_page
    end   = start + per_page
    page_keys = keys[start:end]

    questions = []
    for key in page_keys:
        q = _load_cloud_question(key)
        if q:
            # 附上图片 URL（供前端展示）
            q['img_url'] = f'/api/cloud_library/image/{q.get("qid", "")}'
            questions.append(q)

    return jsonify({
        'questions': questions,
        'total':     total,
        'page':      page,
        'pages':     max(1, (total + per_page - 1) // per_page),
    })


# ── API: 获取题目图片 ──
@app.route('/api/cloud_library/image/<qid>', methods=['GET'])
def cloud_library_image(qid):
    """返回云端题库中指定题目的图片（JPEG）。"""
    # 防注入
    qid = re.sub(r'[^A-Za-z0-9\-_]', '', qid)
    img_key = _cloud_img_key(qid)
    if storage.is_r2_mode():
        data = storage.load_bytes(img_key)
        if not data:
            return jsonify({'error': '图片不存在'}), 404
        return send_file(io.BytesIO(data), mimetype='image/jpeg')
    else:
        if not os.path.isfile(img_key):
            return jsonify({'error': '图片不存在'}), 404
        return send_file(img_key, mimetype='image/jpeg')


# ── API: 删除云端题库题目 ──
@app.route('/api/cloud_library/delete_question/<qid>', methods=['DELETE'])
def cloud_library_delete_question(qid):
    """删除单题（元数据 + 图片）。"""
    subject = request.args.get('subject', '')
    board   = request.args.get('board', '')
    topic1  = request.args.get('topic1', '')
    topic2  = request.args.get('topic2', '')

    qid = re.sub(r'[^A-Za-z0-9\-_]', '', qid)
    meta_key = _cloud_prefix(subject, board, topic1, topic2) + \
               (f'/{qid}.json' if storage.is_r2_mode() else f'{os.sep}{qid}.json')
    img_key  = _cloud_img_key(qid)

    if storage.is_r2_mode():
        storage.delete_object(meta_key)
        storage.delete_object(img_key)
    else:
        for p in [meta_key, img_key]:
            try: os.remove(p)
            except Exception: pass

    _invalidate_stats()
    return jsonify({'ok': True})


# ── API: 清空整个云端题库（危险操作，需要确认参数）──
@app.route('/api/cloud_library/clear_all', methods=['POST'])
def cloud_library_clear_all():
    """
    清空所有云端题库数据。
    需要请求体携带 {"confirm": "CLEAR_ALL"}
    """
    data = request.json or {}
    if data.get('confirm') != 'CLEAR_ALL':
        return jsonify({'error': '需要确认参数 confirm=CLEAR_ALL'}), 400

    if storage.is_r2_mode():
        storage.delete_prefix('cloud_db/')
    else:
        import shutil
        base = os.path.join(os.path.dirname(__file__), 'uploads', 'cloud_db')
        shutil.rmtree(base, ignore_errors=True)
        os.makedirs(base, exist_ok=True)

    _invalidate_stats()
    return jsonify({'ok': True, 'message': '已清空云端题库'})


# ── API: 获取四级目录树（含各层级题目数量） ──
@app.route('/api/cloud_library/tree', methods=['GET'])
def cloud_library_tree():
    """
    返回完整的四级目录树：
    学科 → 考试局 → 知识点一级 → 知识点二级，每级带题目数量。
    优先使用统计缓存；若缓存不存在则实时扫描。
    ?subject=  可过滤只返回该学科
    ?board=    可过滤只返回该考试局
    """
    filter_subject = request.args.get('subject', '')
    filter_board   = request.args.get('board', '')

    stats = _get_stats_cache()
    if not stats:
        stats = _build_stats_cache()

    by_subj = stats.get('by_subject', {})

    subjects_out = []
    for subj in _CLOUD_SUBJECTS:
        if filter_subject and subj != filter_subject:
            continue
        subj_data  = by_subj.get(subj, {})
        subj_total = subj_data.get('total', 0)

        boards_out = []
        for board in _CLOUD_BOARDS:
            if filter_board and board != filter_board:
                continue
            board_data  = subj_data.get('by_board', {}).get(board, {})
            board_total = board_data.get('total', 0)

            topics1_out = []
            for t1, t1_data in sorted(board_data.get('by_topic1', {}).items()):
                t1_total = t1_data.get('total', 0)
                topics2_out = []
                for t2, t2_cnt in sorted(t1_data.get('by_topic2', {}).items()):
                    topics2_out.append({'name': t2, 'count': t2_cnt})
                topics1_out.append({'name': t1, 'count': t1_total, 'subtopics': topics2_out})

            boards_out.append({
                'name':    board,
                'count':   board_total,
                'topics':  topics1_out,
            })

        subjects_out.append({
            'name':   subj,
            'count':  subj_total,
            'boards': boards_out,
        })

    return jsonify({
        'total':    stats.get('total', 0),
        'subjects': subjects_out,
    })


# ── API: 批量导入题目图片后保存到云端题库（支持从 session 批量推送）──
@app.route('/api/cloud_library/push_from_session', methods=['POST'])
def cloud_library_push_from_session():
    """
    从当前 session 推送指定题目到云端题库（供"保存到云端题库"按钮调用）。
    与 save_questions 的区别：此接口支持更丰富的批量配置。
    """
    return cloud_library_save_questions()


# ── API: 获取云端题库可用年份列表 ──
@app.route('/api/cloud_library/years', methods=['GET'])
def cloud_library_years():
    """
    返回云端题库中所有题目涉及的年份列表（从 exam_date 中提取）。
    可选参数：subject, board
    """
    subject = request.args.get('subject', '')
    board   = request.args.get('board', '')

    all_keys = _list_cloud_questions(subject, board, '', '')
    years = set()
    for key in all_keys:
        q = _load_cloud_question(key)
        if q:
            exam_date = q.get('exam_date', '')
            if exam_date:
                # 从 "October 2023" / "2023" / "2023-10" 等格式提取年份
                import re as _re
                m = _re.search(r'(20\d{2}|19\d{2})', str(exam_date))
                if m:
                    years.add(m.group(1))
    return jsonify({'years': sorted(years, reverse=True)})


# ── API: 从云端图库导入题目到 session ──
@app.route('/api/cloud_library/import_to_session', methods=['POST'])
def cloud_library_import_to_session():
    """
    按学科、考试局、年份筛选云端题目，返回一个新 session_id 供后续操作。
    请求体：{subject, board, years: [str] (空=全部)}
    返回：{session_id, groups, total_questions}
    """
    import base64 as _b64
    data    = request.json or {}
    subject = data.get('subject', '')
    board   = data.get('board', '')
    years   = data.get('years', [])   # 空列表 = 全部年份

    if not subject or not board:
        return jsonify({'error': '必须提供 subject 和 board'}), 400

    all_keys = _list_cloud_questions(subject, board, '', '')
    if not all_keys:
        return jsonify({'error': f'云端题库中没有 {subject} / {board} 的题目'}), 404

    # 过滤年份
    selected_qs = []
    for key in all_keys:
        q = _load_cloud_question(key)
        if not q:
            continue
        if years:
            exam_date = str(q.get('exam_date', ''))
            import re as _re
            m = _re.search(r'(20\d{2}|19\d{2})', exam_date)
            year_found = m.group(1) if m else ''
            if year_found not in years:
                continue
        selected_qs.append(q)

    if not selected_qs:
        return jsonify({'error': '按所选年份过滤后没有题目'}), 404

    # 加载每题图片（base64），构建虚拟题目列表
    virt_questions = []
    for i, q in enumerate(selected_qs):
        qid = q.get('qid', '')
        b64 = ''
        if qid:
            img_key = _cloud_img_key(qid)
            try:
                if storage.is_r2_mode():
                    raw = storage.load_bytes(img_key)
                    if raw:
                        b64 = _b64.b64encode(raw).decode()
                else:
                    if os.path.isfile(img_key):
                        with open(img_key, 'rb') as f:
                            b64 = _b64.b64encode(f.read()).decode()
            except Exception:
                pass

        virt_questions.append({
            'q_num':         i + 1,
            'page_idx':      0,
            'difficulty':    q.get('difficulty'),
            'topics':        q.get('topics', []),
            'exam_date':     q.get('exam_date', ''),
            'source':        q.get('source', 'cloud'),
            'paper_type':    q.get('paper_type', 'structured'),
            'maths_unit':    q.get('maths_unit', ''),
            'subject':       q.get('subject', subject),
            'board':         q.get('board', board),
            'topic1':        q.get('topic1', ''),
            'topic2':        q.get('topic2', ''),
            'img_bytes_b64': b64,
            'img_w':         0,
            'img_h':         0,
            '_cloud_qid':    qid,   # 保留原始云端 ID
        })

    # 注册为虚拟 session
    session_id = str(uuid.uuid4())
    virt_group = {
        'filename':        f'{subject}_{board}_云端导入.pdf',
        'path':            '',
        'r2_key':          '',
        'source':          'cloud',
        'paper_type':      selected_qs[0].get('paper_type', 'structured') if selected_qs else 'structured',
        'maths_unit':      None,
        'exam_date':       '',
        'questions':       virt_questions,
        'total_questions': len(virt_questions),
        'total_pages':     0,
    }
    with _multi_sessions_lock:
        _multi_sessions[session_id] = [virt_group]

    return jsonify({
        'session_id':      session_id,
        'total_questions': len(virt_questions),
        'groups': [{
            'filename':        virt_group['filename'],
            'source':          virt_group['source'],
            'paper_type':      virt_group['paper_type'],
            'maths_unit':      None,
            'exam_date':       '',
            'questions':       virt_questions,
            'total_questions': len(virt_questions),
            'total_pages':     0,
            'error':           None,
        }]
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
