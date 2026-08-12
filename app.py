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
app.config['UPLOAD_FOLDER'] = storage.local_tmp_path('') if storage.is_r2_mode() else storage.local_root()
app.config['OUTPUT_FOLDER'] = storage.local_tmp_path('') if storage.is_r2_mode() else os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# 异步任务存储: {task_id: {status, progress, total, out_path, filename, error, r2_key}}
# 用文件持久化，防止多进程/重启后丢失
_tasks = {}
_tasks_lock = threading.Lock()
_TASKS_DIR = os.path.join(storage.local_root(), 'tasks') if not storage.is_r2_mode() else storage.local_tmp_path('tasks')
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
    适用于 Cambridge 9702 及类似 structured paper 格式。

    题号特征（按优先级）：
    - 页面内容区顶部（y ≈ 55-100pt）出现独立题号 span（1-2位数字）
    - x ≈ 42-65pt（左边距，适当放宽以兼容不同印刷版本）
    - font size 9-14pt
    - 非粗体（Cambridge structured 题号通常非粗体）
    - 或：block 以 "数字 " 或 "数字\n" 开头，x 在合理范围内

    修复点：
    - 放宽 y 范围至 45-110（第一题可能在页面较高位置）
    - 放宽 x 范围至 40-75（兼容不同扫描/排版偏差）
    - 增加从页面顶部块文字中提取题号的逻辑
    - 增加 fallback：扫描所有页面的大字号题号
    """
    questions = []
    seen_nums = set()

    # ── 前置：识别 Data Booklet / Formulae 页，这些页不含题目 ──
    DATA_PAGE_KEYWORDS = {
        'data booklet', 'physics data', 'formulae', 'mathematical formulae',
        'data sheet', 'list of data', 'useful formulae', 'physical constants',
        'values of constants', 'mathematical data',
    }

    def _is_data_page(pg):
        """检测页面是否为 Data Sheet / Formulae 页（不含题目）"""
        try:
            pg_text_raw = pg.get_text().upper()
            # 页面前 40% 文字（通常是页眉/标题区）
            lines = pg_text_raw.split('\n')
            head_text = ' '.join(lines[:max(8, len(lines)//3)]).lower()
            for kw in DATA_PAGE_KEYWORDS:
                if kw in head_text:
                    return True
        except Exception:
            pass
        return False

    for pg_i in range(doc.page_count):
        page = doc[pg_i]
        ph   = page.rect.height

        # ── 跳过 Data Sheet / Formulae 页（9702 试卷前几页的常量/公式表）──
        if _is_data_page(page):
            continue

        try:
            d = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        except Exception:
            continue

        # ── 方法1：span 级别检测（最精确）──
        # 题号独立成 span，仅含1-2位数字，在页面内容区左上
        for block in d.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    txt = span["text"].strip()
                    if not re.match(r'^\d{1,2}$', txt):
                        continue
                    q_num = int(txt)
                    if not (1 <= q_num <= 30):
                        continue
                    x0   = span["bbox"][0]
                    y0   = span["bbox"][1]
                    size = span.get("size", 0)

                    # Cambridge structured paper 题号特征：
                    # - 左边距较小（40-75pt），不同印刷版本略有差异
                    # - y 在页面内容区上方（不低于页面中部）
                    # - font size 8.5-16pt（区分正文中偶现的小数字）
                    is_left         = 38 <= x0 <= 78
                    is_content_zone = 40 <= y0 <= ph * 0.55   # 页面上半区
                    is_q_size       = 8.5 <= size <= 16

                    if is_left and is_content_zone and is_q_size and q_num not in seen_nums:
                        seen_nums.add(q_num)
                        questions.append({
                            "q_num":    q_num,
                            "page_idx": pg_i,
                            "y_start":  y0,
                            "x_start":  x0
                        })

        # ── 方法2：block 文字开头检测（题号和内容在同一 block）──
        # 格式："1 \n题目文字..." 或 "10 (a)..."
        try:
            blocks_plain = page.get_text("blocks")
        except Exception:
            blocks_plain = []

        for b in blocks_plain:
            x0, y0, x1, y1, text, _, btype = b
            if btype != 0:
                continue
            text_s = text.strip()
            # 条件：左边距，页面内容区上部，以数字开头
            if not (38 <= x0 <= 78 and 40 <= y0 <= ph * 0.55):
                continue
            # 匹配 "1\n" / "1 " / "10 (a)" / "10\t" 等格式
            m = re.match(r'^(\d{1,2})\s*[\n\r\t (]', text_s)
            if not m:
                # 也尝试纯数字行（某些版本题号单独成 block）
                m = re.match(r'^(\d{1,2})\s*$', text_s)
            if m:
                q_num = int(m.group(1))
                if 1 <= q_num <= 30 and q_num not in seen_nums:
                    seen_nums.add(q_num)
                    questions.append({
                        "q_num":    q_num,
                        "page_idx": pg_i,
                        "y_start":  y0,
                        "x_start":  x0
                    })

    questions.sort(key=lambda x: x["q_num"])

    # ── 后处理：去除可能的误检（题号不连续且差距>2的后半段不可信）──
    if len(questions) >= 2:
        # 验证题号连续性（允许间隔1，剔除明显跳跃的孤立题号）
        valid = [questions[0]]
        for i in range(1, len(questions)):
            prev_num = valid[-1]["q_num"]
            curr_num = questions[i]["q_num"]
            if curr_num <= prev_num:
                continue  # 重复，跳过
            if curr_num - prev_num <= 3:  # 允许最多跳2题（有些题号不连续）
                valid.append(questions[i])
            elif curr_num == 1:
                # 可能是多份试卷，但此处处理单份，跳过
                pass
            else:
                valid.append(questions[i])  # 保守：不过滤，避免漏题
        questions = valid

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

    # ── 预检测 Source Booklet 开始页 ──
    # Economics U4 等 QP 内嵌 Source Booklet（页面含图表轴标签如 "32\n%"），
    # 会被正则误识别为 Q32 等题号。检测到 Source Booklet 开始页后停止扫描。
    # 检测策略：
    #   1. 'Sources for use with Section'：Source Booklet 内容页标题（最可靠）
    #   2. 'Source for use with Section'：变体拼写
    #   3. 'Source Booklet' + 'Do not return'：Source Booklet 封面（封面没有题目）
    # 注意：'Source Booklet' 单独出现可能在封面说明或题目中（"refer to Source Booklet"），
    #       需要结合其他特征判断，或只用更精确的 'Sources for use with Section'
    source_booklet_start_pg = None
    for _pg in range(doc.page_count):
        _pg_txt = doc[_pg].get_text()
        if ('Sources for use with Section' in _pg_txt or
                'Source for use with Section' in _pg_txt):
            source_booklet_start_pg = _pg
            break
        # Source Booklet 封面特征：同时含 'Source Booklet' 和 'Do not return'
        if 'Source Booklet' in _pg_txt and 'Do not return' in _pg_txt:
            source_booklet_start_pg = _pg
            break

    for pg_i in range(doc.page_count):
        # 到达 Source Booklet 开始页时停止扫描
        if source_booklet_start_pg is not None and pg_i >= source_booklet_start_pg:
            break
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
    # Pure Mathematics (IAL)
    'WMA11': 'P1', 'WMA12': 'P2', 'WMA13': 'P3', 'WMA14': 'P4',
    # Further Pure Mathematics (IAL)
    'WFM01': 'FP1', 'WFM02': 'FP2', 'WFM03': 'FP3',
    # Statistics (IAL) — WST prefix
    'WST01': 'S1',  'WST02': 'S2',
    # Legacy WMS prefix (some older papers)
    'WMS01': 'S1',  'WMS02': 'S2',
    # Mechanics (IAL)
    'WME01': 'M1',  'WME02': 'M2',
    # Decision Mathematics (IAL)
    'WDM11': 'D1',
    # Legacy WDM prefix
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
        m = re.search(r'(WMA\d{2}|WFM\d{2}|WMS\d{2}|WME\d{2}|WDM\d{2}|WST\d{2})', text)
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

        m6b = re.search(r'Decision Mathematics\s+D([1-9])', text)
        if m6b:
            return f'D{m6b.group(1)}'

        # 兜底：孤立 P1/P2/P3/P4 标识
        m6 = re.search(r'\bPure Mathematics\b.*?\bP([1-4])\b', text, re.DOTALL)
        if m6:
            return f'P{m6.group(1)}'

    return 'unknown'


def detect_paper_source(doc) -> str:
    """
    返回 'cambridge'、'edexcel'、'edexcel_maths'、'edexcel_economics' 或 'bpho'。
    通过封面/前几页文字关键词判断。
    优先级：bpho > edexcel_economics > edexcel_maths > edexcel > cambridge
    """
    for pg_i in range(min(3, doc.page_count)):
        text = doc[pg_i].get_text()
        # 最高优先：BPhO / British Physics Olympiad
        if ('British Physics Olympiad' in text or 'BRITISH PHYSICS OLYMPIAD' in text or
                'BPhO' in text or 'Physics Olympiad' in text):
            return 'bpho'
        # 最高优先：Edexcel Economics IAL 试卷代码 WEC11/WEC12/WEC13/WEC14
        if re.search(r'WEC1[1-4]', text):
            return 'edexcel_economics'
        # 次优：Edexcel Maths：WMA/WFM/WST/WME/WDM 系列试卷
        if re.search(r'WMA\d{2}/\d{2}|WFM\d{2}/\d{2}|WPM\d{2}/\d{2}|WST\d{2}/\d{2}|WME\d{2}/\d{2}|WDM\d{2}/\d{2}', text):
            return 'edexcel_maths'
        if ('Pure Mathematics' in text or 'Further Mathematics' in text or
                'Statistics' in text or 'Mechanics' in text or 'Decision Mathematics' in text) and \
           ('Pearson' in text or 'Edexcel' in text):
            if re.search(r'P[1-4]|FP[12]|S[12]|M[12]|D1|Unit [1-4]|Pure Math', text):
                return 'edexcel_maths'
        # Edexcel Economics 补充检测（文字描述）
        if ('Economics' in text) and ('Pearson' in text or 'Edexcel' in text):
            if re.search(r'Markets in action|Macroeconomic performance|Business behaviour|Developments in the global economy|UNIT 1|UNIT 2|UNIT 3|UNIT 4', text):
                return 'edexcel_economics'

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
# ============================================================
# BPhO (British Physics Olympiad) 专用题目检测
# BPhO Section 1 = 单个大题 Q1，以 (a)(b)(c)... 子题形式出现
# 每道子题 = 一个 q_num（a=1, b=2, ...）
# ============================================================
def detect_bpho_questions(doc):
    """
    检测 BPhO Round 1 Section 1 试卷的子题边界。

    支持两种格式：
    - 旧格式 (2010-11)：  a) Gas is contained...   [字母后直接跟右括号]
    - 新格式 (2012+)：    (a) The circuit...        [字母被括号包围]

    返回格式：
    [{q_num: 1, q_label: 'a', page_idx: N, y_start: Y, x_start: X, marks: M}, ...]
    q_num = ord(label) - ord('a') + 1  (a=1, b=2, ...)
    """
    MARKS_PAT = re.compile(r'\[(\d+)\]')

    # ── Step 1：判断格式（先扫描全文行，统计旧/新格式出现的字母集合）──────
    old_fmt_labels = set()   # 旧格式 "a) " 收集到的字母（排除罗马数字 i,v,x）
    new_fmt_labels = set()   # 新格式 "(a)" 收集到的字母
    for pg_i in range(doc.page_count):
        for ln in doc[pg_i].get_text().split('\n'):
            ln = ln.strip()
            # 旧格式：行首 单字母 + ')' + 空格 + 非空，排除易混罗马数字
            m_old = re.match(r'^([a-hj-np-z])\)\s+\S', ln)
            if m_old:
                old_fmt_labels.add(m_old.group(1).lower())
            # 新格式：行首 "(单字母)"
            m_new = re.match(r'^\(([a-z])\)', ln, re.IGNORECASE)
            if m_new:
                new_fmt_labels.add(m_new.group(1).lower())

    # 旧格式：若有 ≥3 个旧格式字母且 ≥ 新格式字母数，优先使用旧格式
    use_old_fmt = len(old_fmt_labels) >= 3 and len(old_fmt_labels) >= len(new_fmt_labels)

    if use_old_fmt:
        # 旧格式匹配：行首 单字母+右括号+空格
        LABEL_PAT = re.compile(r'^([a-hj-np-z])\)\s+\S', re.IGNORECASE)
    else:
        # 新格式匹配：行首 "(单字母)"
        LABEL_PAT = re.compile(r'^\(([a-z])\)', re.IGNORECASE)

    # 用于 has_sub_q 检测的多行版本（search 需要 MULTILINE 才能让 ^ 匹配每行）
    LABEL_PAT_ML = re.compile(LABEL_PAT.pattern, re.IGNORECASE | re.MULTILINE)

    # ── Step 2：收集所有命中条目（非 'i' 去重；'i' 收集全部候选后挑最佳）──
    all_hits = []
    seen_non_i = set()
    q1_found = False

    for pg_i in range(doc.page_count):
        page = doc[pg_i]
        ph = page.rect.height
        page_text = page.get_text()

        has_sub_q = bool(LABEL_PAT_ML.search(page_text))

        if not has_sub_q:
            if 'Important Constants' in page_text:
                continue
            if pg_i < 3 and 'Instructions' in page_text:
                continue

        if not q1_found:
            if (re.search(r'\bQ1\b', page_text) or
                    re.search(r'^Q\s*1', page_text, re.MULTILINE) or
                    has_sub_q):
                q1_found = True
            else:
                continue

        try:
            blocks = page.get_text("blocks")
        except Exception:
            continue

        for b in blocks:
            x0, y0, x1, y1, text, bno, btype = b
            if btype != 0 or y0 > ph - 30:
                continue
            text_s = text.strip()
            if not text_s:
                continue
            lines = [l.strip() for l in text_s.split('\n') if l.strip()]
            if not lines:
                continue

            m = LABEL_PAT.match(lines[0])
            if not m:
                continue

            label = m.group(1).lower()

            # 非 i 标签只取首次出现；i 标签收集全部（稍后挑最佳）
            if label != 'i':
                if label in seen_non_i:
                    continue
                seen_non_i.add(label)

            marks_in_block = MARKS_PAT.findall(text_s)
            marks = int(marks_in_block[-1]) if marks_in_block else None

            all_hits.append({
                'q_num':      ord(label) - ord('a') + 1,
                'q_label':    label,
                'page_idx':   pg_i,
                'y_start':    y0,
                'x_start':    x0,
                'marks':      marks,
                '_block_len': len(text_s),
            })

    # ── Step 3：处理 (i) 标签歧义 ────────────────────────────────────────────
    # (i) 既可能是主题目标签（字母序列 a-p 中的第9个），
    # 也可能是子问题的罗马数字编号 (i)(ii)(iii)。
    # 判断规则（三个条件均需满足）：
    #   A. 最长候选 block_len > 5（有实质内容）
    #   B. 'j' 在 non_i 标签集中（字母序列在 i 之后有 j，证明是字母序列成员）
    #   C. best_i 的文档位置在最后一个 'h' 块之后（若 h 存在）
    if not use_old_fmt:
        i_hits     = [h for h in all_hits if h['q_label'] == 'i']
        non_i_hits = [h for h in all_hits if h['q_label'] != 'i']
        non_i_labels = {h['q_label'] for h in non_i_hits}

        best_i = None
        if i_hits:
            best_i = max(i_hits, key=lambda h: h['_block_len'])
            # 条件 A：有实质内容
            if best_i['_block_len'] <= 5:
                best_i = None   # 最长也只是空标签，全丢
            # 条件 B：字母序列必须延伸到 j（没有 j 则 i 是罗马数字子问）
            elif 'j' not in non_i_labels:
                best_i = None   # 无 j，(i) 只是子问编号，丢弃
            else:
                # 条件 C：best_i 位置必须在最后一个 h 块之后（若 h 存在）
                h_hits = [h for h in non_i_hits if h['q_label'] == 'h']
                if h_hits:
                    last_h = max(h_hits, key=lambda h: (h['page_idx'], h['y_start']))
                    i_after_h = (
                        best_i['page_idx'] > last_h['page_idx'] or
                        (best_i['page_idx'] == last_h['page_idx'] and
                         best_i['y_start'] > last_h['y_start'])
                    )
                    if not i_after_h:
                        best_i = None  # (i) 在 h 之前，是罗马数字子问

        candidates = non_i_hits + ([best_i] if best_i else [])
    else:
        candidates = all_hits

    # ── Step 4：按页码+y坐标排序输出 ─────────────────────────────────────────
    questions = sorted(candidates, key=lambda q: (q['page_idx'], q['y_start']))
    for q in questions:
        q.pop('_block_len', None)

    # ── Step 5：补充分值（块内无 [N] 时从页面全文找）────────────────────────
    for q in questions:
        if q['marks'] is not None:
            continue
        pg_text = doc[q['page_idx']].get_text()
        pat = (re.escape(f"{q['q_label']})") if use_old_fmt
               else re.escape(f"({q['q_label']})"))
        m2 = re.search(pat + r'.*?\[(\d+)\]', pg_text, re.DOTALL)
        if m2:
            q['marks'] = int(m2.group(1))

    return questions


def _extract_bpho_year(doc, filename=''):
    """从BPhO试卷提取考试年份（如 2014-15 → '2014-15'，2011 → '2011'）"""
    # 先从文件名尝试
    m = re.search(r'(20\d{2}[-_]?\d{2,4})', filename)
    if m:
        yr = m.group(1).replace('_', '-')
        return yr

    # 从PDF文字提取
    for pg_i in range(min(3, doc.page_count)):
        text = doc[pg_i].get_text()
        # "British Physics Olympiad 2014-15" or "2014-2015"
        m = re.search(r'(20\d{2}[-–]\d{2,4})', text)
        if m:
            return m.group(1)
        m = re.search(r'(20\d{2})', text)
        if m:
            return m.group(1)
    return ''


def detect_edexcel_maths_questions(doc):
    """
    检测 Edexcel IAL Pure/Further/Statistics/Mechanics/Decision Mathematics 试卷的题号边界。
    适用全部单元：P1/P2/P3/P4/FP1/FP2/S1/S2/M1/M2/D1。

    识别策略（通用化，不依赖硬编码位置）：
      1. 真题号 span 特征：
           - 文字是 "N." 或 "N"（纯数字，1–15）
           - fontsize ≥ 10pt（区分题干内的上标/下标 ~7pt）
           - x0 ∈ [40, 85]（左边距，适当放宽兼容不同单元）
           - y0 ∈ [35, 750]（避开页脚页码，但不限制顶部，Q7等可能从较低位置开始）
           - 首次出现（seen_nums 去重）
      2. 续页标记 "Question N continued" → 标记跳过，不记录题号
      3. 去重+排序后返回

    返回 questions 列表，每项含 q_num / page_idx / y_start / x_start。
    """
    Q_SPAN_DOT   = re.compile(r'^(\d{1,2})\.$')    # '1.' '10.'
    Q_SPAN_PLAIN = re.compile(r'^(\d{1,2})$')       # '7'（无句点）
    CONTINUED_PAT = re.compile(r'^Question\s+\d+\s+continued', re.IGNORECASE)

    questions = []
    seen_nums = set()

    for pg_i in range(1, doc.page_count):   # 跳过封面（第0页）
        page = doc[pg_i]
        pw, ph = page.rect.width, page.rect.height

        try:
            blocks = page.get_text('dict', flags=fitz.TEXT_PRESERVE_WHITESPACE)['blocks']
        except Exception:
            blocks = []

        page_continued = False   # 本页是否是续页（有 "Question N continued"）

        for b in blocks:
            if b.get('type') != 0:
                continue
            for line in b.get('lines', []):
                line_txt = ''.join(s['text'] for s in line['spans']).strip()
                if CONTINUED_PAT.match(line_txt):
                    page_continued = True
                    break
            if page_continued:
                break

        if page_continued:
            continue   # 整页跳过，不在续页上重复记录题号

        # 遍历所有 span 找题号
        for b in blocks:
            if b.get('type') != 0:
                continue
            for line in b.get('lines', []):
                for span in line['spans']:
                    txt = span['text'].strip()
                    if not txt:
                        continue

                    m = Q_SPAN_DOT.match(txt) or Q_SPAN_PLAIN.match(txt)
                    if not m:
                        continue

                    q_num = int(m.group(1))
                    if not (1 <= q_num <= 15):
                        continue
                    if q_num in seen_nums:
                        continue

                    bbox    = span['bbox']   # (x0, y0, x1, y1)
                    x0, y0  = bbox[0], bbox[1]
                    fs      = span.get('size', 0)

                    # ── 位置过滤 ──
                    # x: 左边距题号列，[40,85] 兼容各单元
                    if not (40 <= x0 <= 85):
                        continue
                    # y: 不在页脚（页码区），页面顶部区不过度限制
                    if y0 > ph - 60:
                        continue
                    # fontsize: ≥ 10pt（真题号），排除上标/小字数字（~7pt）
                    if fs < 9.5:
                        continue

                    seen_nums.add(q_num)
                    questions.append({
                        'q_num':    q_num,
                        'page_idx': pg_i,
                        'y_start':  y0,
                        'x_start':  x0,
                    })

    questions.sort(key=lambda x: x['q_num'])
    return questions


# ============================================================
# 自动检测题型（兼容 Cambridge + Edexcel + Edexcel Maths）
# ============================================================
def detect_paper_type(doc):
    """
    自动判断试卷类型，返回：
    - 'mcq'                 : Cambridge 纯选择题
    - 'structured'          : Cambridge 大题
    - 'edexcel_mcq'         : Edexcel 纯选择题（仅含Section A选择题）
    - 'edexcel'             : Edexcel 大题 / 混合题型
    - 'edexcel_maths'       : Edexcel IAL Pure/Further Math (P1–P4)
    - 'edexcel_economics'   : Edexcel IAL Economics (WEC11/12/13/14)

    Edexcel 判断逻辑（改进版）：
      1. 如果发现 SECTION B 或 Section B → edexcel（混合卷）
      2. 如果发现 SECTION A 或 Section A 且有MCQ选项(A/B/C/D) → edexcel_mcq
      3. 否则检查题目内容：若题目块中包含 (a)/(b)/(c) 子题 → edexcel（纯大题）
      4. 再检查是否有 MCQ 选项格式 → edexcel_mcq
      5. 默认 → edexcel（保守，避免漏识别大题）
    """
    source = detect_paper_source(doc)

    # BPhO (British Physics Olympiad) — 直接返回专用类型
    if source == 'bpho':
        return 'bpho'

    # Edexcel Economics (IAL) — 直接返回专用类型
    if source == 'edexcel_economics':
        return 'edexcel_economics'

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


def _find_question_stem_bottom(page, ph, paper_type='structured', y_min=None, y_max=None):
    """
    通用题干结束位置检测：找到题目内容（题干 + 图表 + 子题）的真正底部，
    截止到答题区（密集横线/空白写答区）开始之前。

    适用于 Cambridge structured / Edexcel 大题（paper_type != 'edexcel_maths'）。
    Edexcel Maths 有专用函数 _find_edexcel_maths_question_bottom，不使用此函数。

    参数：
      y_min：若指定，只考虑 y >= y_min 的答题区文字标志（用于排除 Section header 中的全局指令）
      y_max：若指定，Phase 3 marks 扫描只考虑 y < y_max 的块（防止同页下一题的 Total 行覆盖）

    核心策略（按优先级）：
    1. 横线检测（最可靠，分两档）：
       - 单条超宽横线（>70% 页宽）：这本身就是答题区起始，立即截止
       - 连续2条普通宽横线（>30% 页宽，间距<32pt）：密集区起始，截止
    2. 答题区文字标志：Answer space / Write your answer / Do not write here 等
       （仅当 y >= y_min 时生效，避免 Section header 全局指令误触发）
    3. marks 标记辅助：若 marks 紧贴 cut_y 之前，以 marks y1+8 为下界（防截断）
       （同时应用 y_min/y_max 双向过滤，只考虑本题范围内的 marks 行）
    4. fallback：取所有非横线、非答题提示的最后一个内容块 y1
    """
    pw = page.rect.width

    # 答题区文字特征（扩展版，覆盖更多 Cambridge/Edexcel 格式）
    ANSWERZONE_RE = re.compile(
        r'^(answer\s+(space|here|in\s+the\s+space|on\s+the\s+grid|lines?|area)|'
        r'write\s+(your\s+)?(answer|working)|'
        r'leave\s+(this\s+)?(space\s+)?blank|'
        r'do\s+not\s+write\s+(in\s+this\s+space|here|outside)|'
        r'question\s+\d+\s+continued|'
        r'total\s+for\s+(question|this\s+question)\s*[\d\s]*|'
        r'\(total\s+for\s+(question|this\s+question)\s*[\d\s=a-z]*\)|'
        r'total\s+for\s+section\s+[a-z]\s*=|'
        r'additional\s+(answer\s+)?space|'
        r'use\s+(this\s+)?(page\s+)?space\s+(for\s+)?(your\s+)?(working|answer)|'
        r'space\s+for\s+(rough\s+)?working|'
        r'working\s+space|'
        r'show\s+your\s+working|'
        r'for\s+examiner[\'s]*\s+use|'
        r'examiner\s+only)',
        re.IGNORECASE
    )
    SKIP_RE = re.compile(
        r'^(DO NOT WRITE|Turn over|©|UCLES|Permission to reproduce|\d{1,4}$'
        r'|9702/|WMA\d{2}|WFM\d{2}|WST\d{2}|WME\d{2}|WDM\d{2}|WMS\d{2}|WPM\d{2}'
        r'|[A-Z]{2,4}\d{4,}|\s*$)',
        re.IGNORECASE
    )
    MARKS_PAT = re.compile(
        r'\(\d+\s*marks?\)$|\[\d+\]$|\(\d+\)$'
        r'|total\s+for\s+question\s+\d+\s*=\s*\d+\s*marks?',
        re.IGNORECASE
    )

    # ── 阶段0（最高优先级）：文字横线检测 ──
    # Edexcel Maths 等试卷的答题横线是文字字符 "____..."，不是 drawing 路径
    # 通用函数也加入此检测，作为辅助信号
    text_hline_y = None
    try:
        dict_blocks = page.get_text('dict')['blocks']
        text_hline_ys = []
        for b in dict_blocks:
            if b.get('type') != 0:
                continue
            for line in b.get('lines', []):
                for span in line.get('spans', []):
                    t = span['text']
                    # 判定为答题横线：下划线字符占比 > 80% 且总字符数 ≥ 10
                    if len(t) >= 10 and t.count('_') / len(t) > 0.80:
                        y0 = span['bbox'][1]
                        if 40 < y0 < ph - 40:
                            text_hline_ys.append(y0)
        if text_hline_ys:
            text_hline_ys.sort()
            text_hline_y = max(0, text_hline_ys[0] - 2)
    except Exception:
        pass

    # ── 阶段1：通过 drawing 找横线区域起始 y（最可靠信号）──
    # 分两档：超宽单条线（单独触发）+ 普通宽线密集区（2条以上）
    hline_start = None
    try:
        drawings = page.get_drawings()
        wide_hlines  = []   # 超宽线 dx > 70% 页宽，存 (mid_y, x_left)
        normal_hlines = []  # 普通宽线 dx > 30% 页宽
        for d in drawings:
            for item in d.get('items', []):
                if item[0] == 'l':
                    p1, p2 = item[1], item[2]
                    dy    = abs(p2.y - p1.y)
                    dx    = abs(p2.x - p1.x)
                    mid_y = (p1.y + p2.y) / 2
                    x_left = min(p1.x, p2.x)
                    if dy < 3 and 40 < mid_y < ph - 40:
                        if dx > pw * 0.70:
                            wide_hlines.append((mid_y, x_left))
                        elif dx > pw * 0.30:
                            normal_hlines.append(mid_y)
                elif item[0] == 're':
                    r     = item[1]
                    rh    = abs(r.y1 - r.y0)
                    rw    = abs(r.x1 - r.x0)
                    mid_y = (r.y0 + r.y1) / 2
                    x_left = min(r.x0, r.x1)
                    if rh < 4 and 40 < mid_y < ph - 40:
                        if rw > pw * 0.70:
                            wide_hlines.append((mid_y, x_left))
                        elif rw > pw * 0.30:
                            normal_hlines.append(mid_y)

        # 档位1：超宽单条横线 → 直接截止（答题区起始最可靠信号）
        # 过滤1：排除 y > ph - 60 的底部页脚装饰线（Edexcel 每页底部都有一条跨页分割线）
        # 过滤2：排除 y < y_min 的横线（属于上一道题的分隔线，不是本题的答题区边界）
        #        例：Q2 y_min=319, 页面上方 y=308.6 是 Q1/Q2 间分隔线，不应截断 Q2
        # 过滤3：排除 x_left > 65 的横线（图表内部网格线，如柱状图/折线图的坐标刻度）
        #        真正的题目分隔线从页面内容区左边距起始（x_left ≤ 65pt ≈ 43-45pt）
        #        图表内部线条有明显缩进（x_left ≈ 98pt 或更大），应排除
        wide_hlines_content = [y for y, x_left in wide_hlines
                               if y < ph - 60
                               and (y_min is None or y >= y_min)
                               and x_left <= 65]
        if wide_hlines_content:
            wide_hlines_content.sort()
            hline_start = max(0, wide_hlines_content[0] - 6)

        # 档位2：普通宽线，只要有连续2条（间距<32pt）即触发
        # ── 修复：排除数学/物理函数图网格线 ──
        # 图形网格特征：大量短间距线（间距 < 8pt），且分布在页面某个矩形区域内
        # 答题横线特征：少量长间距线（间距 10-30pt），分布较稀疏
        # 判断方法：若连续线条中最小间距 < 7pt，且线条数量 > 6 → 是网格图，不触发
        if hline_start is None and len(normal_hlines) >= 2:
            normal_hlines.sort()
            for idx in range(len(normal_hlines) - 1):
                y_a = normal_hlines[idx]
                y_b = normal_hlines[idx + 1]
                gap = y_b - y_a
                if gap < 32:
                    # 检查该组连续线是否是网格：统计相邻线的最小间距
                    # 若大量相邻线间距 < 7pt → 图形网格，跳过
                    group_lines = [y_a, y_b]
                    k = idx + 2
                    while k < len(normal_hlines) and normal_hlines[k] - group_lines[-1] < 32:
                        group_lines.append(normal_hlines[k])
                        k += 1
                    if len(group_lines) >= 2:
                        gaps = [group_lines[j+1] - group_lines[j] for j in range(len(group_lines)-1)]
                        min_gap = min(gaps)
                        # 网格判断：超过4条线 且 最小间距 < 6pt（密集网格）
                        is_graph_grid = (len(group_lines) > 4 and min_gap < 6)
                        if not is_graph_grid:
                            hline_start = max(0, y_a - 6)
                            break

        # 档位2b: 3条普通线即使间距稍大（<50pt）也触发（但排除网格）
        if hline_start is None and len(normal_hlines) >= 3:
            normal_hlines.sort()
            for idx in range(len(normal_hlines) - 2):
                y_a = normal_hlines[idx]
                y_b = normal_hlines[idx + 1]
                y_c = normal_hlines[idx + 2]
                if (y_b - y_a) < 50 and (y_c - y_b) < 50:
                    # 同样排除网格
                    min_gap = min(y_b - y_a, y_c - y_b)
                    if min_gap >= 6:  # 不是密集网格
                        hline_start = max(0, y_a - 6)
                        break
    except Exception:
        pass

    # 将文字横线信号并入 hline_start（取更早出现的）
    if text_hline_y is not None and text_hline_y > 40:
        if hline_start is None or text_hline_y < hline_start:
            hline_start = text_hline_y

    # ── 阶段2：答题区文字标志（扫描所有文字块）──
    answer_zone_y0 = ph
    try:
        blocks = page.get_text('blocks')
    except Exception:
        blocks = []

    for b in blocks:
        x0, y0, x1, y1, txt, bno, btype = b
        if btype != 0:
            continue
        if y0 < 40 or y0 > ph - 20:
            continue
        # y_min 过滤：忽略 y < y_min 的答题区标志
        # 作用：排除 Section header 中的全局性指令（如 "Write your answer in the space provided."）
        # 这类指令出现在页面顶部（y≈115），属于整页通用说明，不是具体题目的答题区起点
        if y_min is not None and y0 < y_min:
            continue
        ts = txt.strip()
        if ts and ANSWERZONE_RE.match(ts):
            answer_zone_y0 = min(answer_zone_y0, y0)

    # ── 综合：取横线信号 和 文字标志 中更早出现的 ──
    candidates = []
    if hline_start is not None:
        candidates.append(hline_start)
    if answer_zone_y0 < ph - 20:
        candidates.append(answer_zone_y0 - 4)

    if candidates:
        cut_y = min(candidates)   # 取最早出现的答题区边界

        # ── 阶段3（辅助）：marks 标记微调 ──
        # 若 marks 在 cut_y 之前（marks y1 < cut_y + 24pt 内），
        # 说明 marks 就在答题区入口处，需确保 marks 完整显示
        #
        # 修复：扫描时必须同时应用 y_min 过滤，只关注本题范围内的 marks 行。
        # 否则同页面的下一道题的 Total 行也会被扫到，把 marks_y1 推高，
        # 导致条件 marks_y1 < cut_y+24 变为 False，微调无法触发。
        # 例：Q1/Q2 同页，Q2的 "(Total for Q2)" y1=530 覆盖 Q1的 y1=308，
        #      530 > 302+24=326 → 不触发 → Q1的 Total 行被截断。
        marks_y1 = None
        for b in blocks:
            x0, y0, x1, y1, txt, bno, btype = b
            if btype != 0 or y0 > ph - 55:
                continue
            # y_min 过滤：只考虑本题起始以下的 marks 标记
            if y_min is not None and y0 < y_min:
                continue
            # y_max 过滤：只考虑本题范围以内的 marks 标记（防止下一题的 Total 行覆盖）
            if y_max is not None and y0 >= y_max:
                continue
            ts = txt.strip()
            if MARKS_PAT.search(ts) and x0 > pw * 0.35:
                marks_y1 = y1

        if marks_y1 is not None and marks_y1 < cut_y + 24:
            # marks 就在答题区前：用 marks 下边界（保证 marks 完整可见）
            cut_y = max(cut_y, marks_y1 + 8)

        if cut_y > 60:
            return min(cut_y, ph - 25)

    # ── 阶段4 fallback：扫描所有题目性内容块，取最后一个 y1 ──
    last_y = 0
    for b in blocks:
        x0, y0, x1, y1, txt, bno, btype = b
        if btype != 0:
            continue
        if y0 < 40 or y0 > ph - 40:
            continue
        ts = txt.strip()
        if not ts:
            continue
        if SKIP_RE.match(ts):
            continue
        if ANSWERZONE_RE.match(ts):
            continue
        clean = ts.replace(' ', '').replace('\n', '').replace('\t', '')
        if clean and all(c in '_-–—' for c in clean):
            continue
        if re.match(r'^\d{1,3}$', ts):
            continue
        last_y = max(last_y, y1)

    if last_y > 60:
        return min(last_y + 10, ph - 25)

    return _find_content_bottom(page, ph)


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

    # ── Edexcel Maths 专用：使用 _collect_question_slices() 支持跨页题目 ──
    # 与 PDF 导出逻辑完全一致（M1/M2/S1/S2/D1 等有大图跨页题同样生效）
    if paper_type == 'edexcel_maths':
        slices = _collect_question_slices(doc, questions, q_idx, 'edexcel_maths')
        if not slices:
            # fallback：截整页
            page = doc[pg_start]
            pix  = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            return pix.tobytes("png"), pix.width, pix.height

        if len(slices) == 1:
            src_page, clip = slices[0]
            pix  = src_page.get_pixmap(matrix=mat, clip=clip, colorspace=fitz.csRGB)
            data = pix.tobytes("png")
            w, h = pix.width, pix.height
            del pix
            return data, w, h

        # 多页拼接（跨页题目）
        imgs = []
        for src_page, clip in slices:
            pix = src_page.get_pixmap(matrix=mat, clip=clip, colorspace=fitz.csRGB)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            imgs.append(img)
            del pix

        sep_h = 3
        total_h = sum(im.height for im in imgs) + sep_h * (len(imgs) - 1)
        max_w   = max(im.width for im in imgs)
        merged  = Image.new("RGB", (max_w, total_h), (255, 255, 255))
        y_off = 0
        for i, im in enumerate(imgs):
            merged.paste(im, (0, y_off))
            y_off += im.height
            if i < len(imgs) - 1:
                y_off += sep_h
        buf = io.BytesIO()
        merged.save(buf, format='PNG')
        return buf.getvalue(), max_w, total_h
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

        # edexcel_economics 专项：非起始页如果是纯答题虚线页（dotted lines），跳过
        if pg_i != pg_start and paper_type == 'edexcel_economics':
            if _is_econ_dotted_answer_page(page, ph):
                continue

        # 计算裁剪区域
        left = 30
        right = pw - 15

        # Edexcel 大题：检测两侧装饰条
        if paper_type in ('edexcel', 'edexcel_mcq', 'edexcel_economics'):
            left, right = 36, min(pw - 36, 550)

        if pg_i == pg_start and pg_i == pg_end:
            # 同页：先用题干底部检测，再和 y_end 取 min（确保不含下一题）
            # 传入 y_min=y_top：让 _find_question_stem_bottom 忽略题目起始以上的
            # answer-zone 信号（如 Section header 全局指令、上一题的分隔横线）
            # 传入 y_max=y_end：Phase 3 marks 扫描只看本题范围，不扫下一题的 Total 行
            top = max(0, y_top)
            stem_bottom = _find_question_stem_bottom(page, ph, paper_type, y_min=y_top, y_max=y_end)
            if y_end is not None:
                if stem_bottom <= y_top:
                    bottom = min(ph, y_end)
                else:
                    bottom = min(stem_bottom, min(ph, y_end))
            else:
                bottom = stem_bottom if stem_bottom > y_top else ph - 25
        elif pg_i == pg_start:
            # 首页：从题号到题干底部（不含本页的答题区横线）
            # 传入 y_min=y_top：排除题目起始以上的 answer-zone 误判
            top = max(0, y_top)
            stem_bottom = _find_question_stem_bottom(page, ph, paper_type, y_min=y_top)
            bottom = stem_bottom if stem_bottom > y_top else ph - 25
        elif pg_i == pg_end:
            # 末页：从页顶到 min(题干底部, 下一题题号)
            top = 55  # 跳过页眉
            stem_bottom = _find_question_stem_bottom(page, ph, paper_type)
            if y_end is not None:
                bottom = min(stem_bottom, min(ph, y_end))
                # ★ 修复：对 edexcel_economics，若末页上无实质内容（只有 Section header），跳过
                # 例：Q6 pg_end=page7，top~y_end 区域只有 "SECTION B / Answer ALL questions..."
                if pg_i != pg_start and paper_type == 'edexcel_economics':
                    if not _has_question_content_in_range(page, top, y_end):
                        continue
            else:
                bottom = stem_bottom
        else:
            # 中间页（多页大题中间部分）：题干底部截止（不含答题区）
            top = 55
            bottom = _find_question_stem_bottom(page, ph, paper_type)

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
    适用所有单元：P1/P2/P3/P4/FP1/FP2/S1/S2/M1/M2/D1。

    策略（按优先级，三层信号）：
      1. 文字横线检测（最精准）：Edexcel Maths 答题横线是文字字符"____..."
         直接找第一条含≥10个下划线的 span，取其 y0 - 2 作为截止点
      2. 右侧 marks (N) 标记：取最后一个 marks y1 + 2pt
         （实测 marks_y1 到横线距离固定 ~3.5pt，+2 安全不越线）
      3. drawing 横线检测（兜底）：针对其他 Edexcel 系列可能有 drawing 线的情况
      综合：取三者中最小值（最保守截止点）
      4. fallback：找最后一个非横线内容块底部
    """
    pw = page.rect.width
    ph = page_height

    # ── 信号1（最高优先级）：文字横线检测 ──
    # Edexcel Maths 的答题横线是 "____..." 文字字符，不是 drawing 路径
    # 直接扫描所有 spans，找第一条含有大量下划线的行
    text_hline_y = None
    try:
        dict_blocks = page.get_text('dict')['blocks']
        spans_sorted = []
        for b in dict_blocks:
            if b.get('type') != 0:
                continue
            for line in b.get('lines', []):
                for span in line.get('spans', []):
                    t = span['text']
                    # 判定为答题横线：下划线字符占比 > 80% 且总字符数 ≥ 10
                    if len(t) >= 10 and t.count('_') / len(t) > 0.80:
                        spans_sorted.append(span['bbox'][1])  # y0
        if spans_sorted:
            spans_sorted.sort()
            # 取第一条答题横线的 y0，再退 2pt 作为截止点
            text_hline_y = max(0, spans_sorted[0] - 2)
    except Exception:
        pass

    # ── 信号2：右侧 marks (N) 标记 ──
    blocks = page.get_text('blocks')
    marks_x_min = max(350, pw * 0.60)
    MARKS_PAT = re.compile(r'^\(\d+\)$')

    marks_y1 = None
    for b in blocks:
        x0, y0, x1, y1, txt, bno, btype = b
        if btype != 0:
            continue
        ts = txt.strip()
        if x0 > marks_x_min and y0 < ph - 60 and MARKS_PAT.match(ts):
            marks_y1 = y1   # 取最后一个（持续更新）

    marks_bottom = None
    if marks_y1 is not None:
        # +2pt：实测 marks_y1 到横线固定 ~3.5pt，+2 不会越过横线
        marks_bottom = min(marks_y1 + 2, ph - 25)

    # ── 信号3（兜底）：drawing 横线检测（针对其他系列有 drawing 线的情况）──
    drawing_hline_y = None
    try:
        drawings = page.get_drawings()
        wide_hlines   = []
        normal_hlines = []
        for d in drawings:
            for item in d.get('items', []):
                if item[0] == 'l':
                    p1, p2 = item[1], item[2]
                    dy    = abs(p2.y - p1.y)
                    dx    = abs(p2.x - p1.x)
                    mid_y = (p1.y + p2.y) / 2
                    if dy < 3 and 40 < mid_y < ph - 40:
                        if dx > pw * 0.70:
                            wide_hlines.append(mid_y)
                        elif dx > pw * 0.30:
                            normal_hlines.append(mid_y)
                elif item[0] == 're':
                    r     = item[1]
                    rh    = abs(r.y1 - r.y0)
                    rw    = abs(r.x1 - r.x0)
                    mid_y = (r.y0 + r.y1) / 2
                    if rh < 4 and 40 < mid_y < ph - 40:
                        if rw > pw * 0.70:
                            wide_hlines.append(mid_y)
                        elif rw > pw * 0.30:
                            normal_hlines.append(mid_y)

        if wide_hlines:
            wide_hlines.sort()
            drawing_hline_y = max(0, wide_hlines[0] - 6)

        if drawing_hline_y is None and len(normal_hlines) >= 2:
            normal_hlines.sort()
            for idx in range(len(normal_hlines) - 1):
                y_a = normal_hlines[idx]
                y_b = normal_hlines[idx + 1]
                if (y_b - y_a) < 32:
                    drawing_hline_y = max(0, y_a - 6)
                    break

        if drawing_hline_y is None and len(normal_hlines) >= 3:
            normal_hlines.sort()
            for idx in range(len(normal_hlines) - 2):
                y_a = normal_hlines[idx]
                y_b = normal_hlines[idx + 1]
                y_c = normal_hlines[idx + 2]
                if (y_b - y_a) < 50 and (y_c - y_b) < 50:
                    drawing_hline_y = max(0, y_a - 6)
                    break
    except Exception:
        pass

    # ── 综合：收集所有有效信号，取最小值（最保守截止点）──
    candidates = []
    if text_hline_y is not None and text_hline_y > 60:
        candidates.append(text_hline_y)
    if drawing_hline_y is not None and drawing_hline_y > 60:
        candidates.append(drawing_hline_y)
    if marks_bottom is not None:
        candidates.append(marks_bottom)

    if candidates:
        result = min(candidates)
        # 安全检查：如果 marks 存在且 marks_bottom 比所有横线信号都大，
        # 说明题干（含 marks）完整可见，直接使用
        # 如果 marks_bottom 比横线信号更小，说明横线检测异常，
        # 以 marks_bottom 为准确保 marks 完整
        if marks_bottom is not None:
            # 确保至少包含最后一个 marks
            result = max(result, marks_bottom)
            # 但如果文字横线信号存在且在 marks_bottom 之后，以横线为准
            if text_hline_y is not None and text_hline_y < marks_bottom:
                result = text_hline_y
        return min(result, ph - 25)

    # ── fallback：找最后一个非横线、非 DO NOT WRITE、非页码的内容块底部 ──
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
        if y0 > ph - 60:   # 页脚区域
            continue
        if re.match(r'^\d{1,3}$', ts):
            continue
        # 跳过答题横线（全是下划线）
        clean = ts.replace(' ', '').replace('\t', '').replace('\n', '')
        if clean and all(c == '_' for c in clean):
            continue
        last_y = max(last_y, y1)

    return min(last_y + 8, ph - 25) if last_y > 50 else ph - 25


def _is_answer_writing_page(page):
    """
    判断一个 PDF 页面是否是"答题页"（供学生手写答案的空白/横线页）。
    这类页面在导出时应跳过，不应出现在题册 PDF 中。

    判断依据（满足任意一条即为答题页）：
    1. 页面文字包含 'BLANK PAGE' / 'THIS PAGE IS INTENTIONALLY LEFT BLANK'
    2. 去除页眉页脚/DO NOT WRITE/题目继续提示后，有效内容极少（≤15字符）
       且有3条以上横线绘制路径
    3. 有效内容极少（≤8字符）即视为空白页
    4. Edexcel Maths 答题续页特征：仅有 "Question N continued" + 大量横线
       （修复：逐行拆分混合块；SKIP_RE 含 Q\\d+；CONTINUE_RE 含 (Total N marks)/(Leave/blank)）

    注意：首页（包含题目编号）永远不应被本函数判为答题页（调用方保证）。
    """
    try:
        text_raw = page.get_text().strip()
    except Exception:
        return False

    # ── 规则 1: 明确标注的空白/答题页 ──
    upper = text_raw.upper()
    if 'BLANK PAGE' in upper:
        return True
    if 'THIS PAGE IS INTENTIONALLY LEFT BLANK' in upper:
        return True
    if 'INTENTIONALLY BLANK' in upper:
        return True

    ph = page.rect.height
    pw = page.rect.width

    # ── 规则 2: 分析文字块，剔除页眉/页脚/DO NOT WRITE 后的有效内容 ──
    try:
        blocks = page.get_text('blocks')
    except Exception:
        return False

    SKIP_RE = re.compile(
        r'^(DO NOT WRITE|Turn over|©|UCLES|\d{1,4}$|9702/|\*P|P\d{4,}[A-Z]'
        r'|WMA\d{2}|WFM\d{2}|WST\d{2}|WME\d{2}|WDM\d{2}|WMS\d{2}|WPM\d{2}'  # Edexcel Applied Maths 卷号
        r'|[A-Z]{2,4}\d{4,}'
        r'|Q\d{1,2}$'   # Edexcel Maths 续页底部题号标记（如 "Q1" "Q10"）
        r'|\s*$)',
        re.IGNORECASE
    )
    # 续页/答题区提示文字：不计为有效题目内容
    # 新增：(Total for Question N is X marks) 也是提示文字，不是题目
    CONTINUE_RE = re.compile(
        r'^(question\s+\d+\s+continued|'
        r'total\s+for\s+(question|this\s+question)[\s\d]*|'
        r'\(total\s+for\s+(question|this\s+question)[^)]*\)|'
        r'\(total\s+\d+\s+marks?\)|'   # (Total N marks) — Edexcel Maths 续页底部总分
        r'total\s+\d+\s+marks?|'         # Total N marks（无括号）
        r'answer\s+space|answer\s+in\s+the\s+space|'
        r'write\s+your\s+answer|'
        r'leave\s+blank|^leave$|^blank$|'  # "Leave blank" 也可能被拆成两个单独行
        r'additional\s+(answer\s+)?space|'
        r'do\s+not\s+write\s+(in\s+this\s+space|here|outside)|'
        r'for\s+examiner[\'s]*\s+use|examiner\s+only)',
        re.IGNORECASE
    )
    real_content_chars = 0
    underscore_chars   = 0

    for b in blocks:
        x0, y0, x1, y1, txt, bno, btype = b
        if btype != 0:
            continue
        # 跳过页眉（顶部 50pt）和页脚（底部 55pt，略微扩大以覆盖 Total for Question）
        if y1 < 50 or y0 > ph - 55:
            continue
        ts = txt.strip()
        if not ts:
            continue
        if SKIP_RE.match(ts):
            continue
        if CONTINUE_RE.match(ts):
            continue
        # 统计横线字符（下划线/破折线全组成的行）
        # 注意：Edexcel Maths 续页末尾可能有 "___...\nQ1" 的混合块
        # 逐行拆分处理，区分纯横线行和非横线行
        for line in ts.split('\n'):
            line_s = line.strip()
            if not line_s:
                continue
            line_clean = line_s.replace(' ', '').replace('\t', '')
            if line_clean and all(c in '_-–—' for c in line_clean):
                underscore_chars += len(line_clean)
            elif SKIP_RE.match(line_s) or CONTINUE_RE.match(line_s):
                pass  # 跳过（续页标记或页眉）
            else:
                real_content_chars += len(line_s)

    # 判断答题页：优先看下划线字符数量
    # 规则A：大量下划线（>200字符）+ 有效内容很少（≤ 50字符）→ 答题页
    # 这覆盖了 "Question N continued + 满页下划线 + Q1 + (Total N marks)" 的情况
    # 注意：(Total N marks) 已被 CONTINUE_RE 过滤，Q1 已被 SKIP_RE 过滤
    if underscore_chars > 200 and real_content_chars <= 50:
        return True

    # 规则B：几乎没有实质内容（≤ 15 字符）
    if real_content_chars <= 15:
        # 进一步检查：是否有大量横线（绘制路径）
        # 注意：物理/数学图形网格也有大量横线，需要区分
        try:
            drawings = page.get_drawings()
            hline_count  = 0
            grid_count   = 0   # 密集网格线计数（间距 < 6pt）
            all_hlines_y = []
            for d in drawings:
                for item in d.get('items', []):
                    if item[0] == 'l':
                        p1, p2 = item[1], item[2]
                        dy = abs(p2.y - p1.y)
                        dx = abs(p2.x - p1.x)
                        # 近似水平线，且跨度 > 页面宽度 30%
                        if dy < 3 and dx > pw * 0.30:
                            hline_count += 1
                            all_hlines_y.append((p1.y + p2.y) / 2)
                    elif item[0] == 're':
                        r = item[1]
                        rh = abs(r.y1 - r.y0)
                        rw = abs(r.x1 - r.x0)
                        # 横向细矩形（细线条）
                        if rh < 3 and rw > pw * 0.30:
                            hline_count += 1
                            all_hlines_y.append((r.y0 + r.y1) / 2)

            # 检测是否为图形网格：若有超过5条线且最小间距 < 6pt → 是网格图不是答题页
            if hline_count >= 5 and all_hlines_y:
                all_hlines_y.sort()
                gaps = [all_hlines_y[i+1] - all_hlines_y[i] for i in range(len(all_hlines_y)-1)]
                if gaps and min(gaps) < 6:
                    # 密集网格图 → 不是答题页
                    return False

            # 有 3 条以上横线且实质内容 ≤ 15 字符 → 答题页
            if hline_count >= 3:
                return True
        except Exception:
            pass
        # 即使没有横线，实质内容 ≤ 8 字符也认为是空白页
        if real_content_chars <= 8:
            return True

    return False


def _find_last_content_page(doc, start_page):
    """
    从start_page开始，找最后一个有题目内容的页面。
    跳过空白页、版权页、答题页等。
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

        # edexcel_economics 专项：材料页（Sources for use with Section C）
        # 材料页属于 Q12 的附属材料，不是 Q13/Q14 的题目内容，应停止扫描
        if ('Sources for use with Section' in text or
                'Source for use with Section' in text):
            break

        # Edexcel 封面/扉页："Pearson Edexcel International Advanced Level"
        # 这类页面是 Source Booklet 封面或考试开始页，不是题目内容
        if ('Pearson Edexcel International Advanced Level' in text or
                'Pearson Edexcel International Advanced Subsidiary' in text):
            break

        # Task4: 答题页（横线页/空白答题区）→ 跳过，不更新 last_content_page，但继续扫描
        if _is_answer_writing_page(page):
            continue

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

    syllabus_type: 'cambridge' | 'edexcel_maths' | 'edexcel_economics' | 'bpho'
    unit_filter:   若指定（如 'P3'），只使用该 unit 的规则（仅对 edexcel_maths 有效）
    marks_hint:    {chapter_id: marks_weight} 分值权重提示，得分乘以权重

    Edexcel Maths 模式：使用章节级别规则（_MATHS_CHAPTER_RULES），
    返回一级标题 ID，如 'P3-2'（Trigonometry），不细化到子章节。
    BPhO 模式：使用 _BPHO_TOPIC_RULES（物理12章），返回带 parent 的二级知识点。
    保证：即使关键词未命中，也会返回分值最高章节作为 fallback。
    """
    text_lower = text.lower()
    scores: dict[str, int] = {}

    if syllabus_type == 'cambridge':
        rules = _TOPIC_RULES
    elif syllabus_type == 'bpho':
        # BPhO：使用物理12章知识点规则
        rules = _BPHO_TOPIC_RULES
    elif syllabus_type == 'edexcel_economics':
        # Edexcel Economics：使用经济学章节规则
        if unit_filter:
            prefix = unit_filter + '-'   # 'U1-'
            rules = [(sid, req, bon) for sid, req, bon in _ECONOMICS_CHAPTER_RULES
                     if sid.startswith(prefix)]
        else:
            rules = _ECONOMICS_CHAPTER_RULES
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
    if syllabus_type == 'bpho':
        # ── BPhO 模式：使用内置 _BPHO_TOPIC_TITLES 映射，返回 L1/L2 两级结构 ──
        if not scores:
            # fallback：宽松单词匹配
            fallback_scores: dict[str, int] = {}
            words = set(re.findall(r'[a-z]{3,}', text_lower))
            for sid, req_kws, bon_kws in rules:
                sc = 0
                for kw in req_kws:
                    for word in kw.lower().split():
                        if len(word) >= 4 and word in words:
                            sc += 1
                if sc > 0:
                    fallback_scores[sid] = sc
            if fallback_scores:
                best_sid = max(fallback_scores, key=lambda k: fallback_scores[k])
                scores = {best_sid: fallback_scores[best_sid]}
            elif rules:
                # 完全兜底：用第一章 Kinematics
                scores = {rules[0][0]: 1}

        result = []
        for sid, sc in sorted(scores.items(), key=lambda x: -x[1]):
            # sid 格式：'BPhO-N-M'（二级），parent_id 为 'BPhO-N'（一级）
            parts = sid.split('-')
            if len(parts) == 3:
                parent_id = f'BPhO-{parts[1]}'
            else:
                parent_id = sid
            sub_title    = _BPHO_TOPIC_TITLES.get(sid, sid)
            parent_title = _BPHO_TOPIC_TITLES.get(parent_id, parent_id)
            result.append({
                'id':           sid,
                'title':        sub_title,
                'parent_id':    parent_id,
                'parent_title': parent_title,
                'score':        sc,
            })
        return result[:3]

    elif syllabus_type in ('edexcel_maths', 'edexcel_economics'):
        if syllabus_type == 'edexcel_maths':
            syllabus = _load_edexcel_maths_syllabus()
        else:
            syllabus = _load_edexcel_economics_syllabus()
        # 构建章节 title_map 和 subtopic 结构
        title_map: dict[str, str] = {}     # chapter_id → title
        chapter_subtopics: dict[str, list] = {}  # chapter_id → [{id, title}]
        if syllabus:
            for t in syllabus['topics']:
                tid = str(t['id'])
                title_map[tid] = t.get('title', tid)
                chapter_subtopics[tid] = t.get('subtopics', [])

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

        # ── 章节内 subtopic 精确匹配 ──
        # 对每个已命中的章节，用 subtopic 标题关键词在题目文本中进一步定位
        # 返回的 topic 条目带 parent_id（章节 id）和 parent_title（章节 title）
        # 这使得 q.topics[] 结构与 Cambridge 一致，cloud save 可做两级分类
        def _find_best_subtopic(chapter_id: str) -> dict | None:
            subs = chapter_subtopics.get(chapter_id, [])
            if not subs:
                return None
            best_sub = None
            best_sc  = 0
            for sub in subs:
                sub_title = sub.get('title', '').lower()
                # 用 subtopic title 中的单词（≥4字符）匹配题目文本
                sc = 0
                for word in re.findall(r'[a-z]{4,}', sub_title):
                    if word in text_lower:
                        sc += 1
                if sc > best_sc:
                    best_sc = sc
                    best_sub = sub
            return best_sub if best_sc > 0 else None

        result = []
        for sid, sc in sorted(scores.items(), key=lambda x: -x[1]):
            chapter_title = title_map.get(sid, sid)
            best_sub = _find_best_subtopic(sid)
            if best_sub:
                # 有 subtopic 匹配：返回 subtopic 级别的条目
                # id = subtopic id（P3-1.1），parent_id = chapter id（P3-1）
                result.append({
                    'id':           best_sub['id'],
                    'title':        best_sub.get('title', best_sub['id']),
                    'parent_id':    sid,
                    'parent_title': chapter_title,
                    'score':        sc,
                    '_has_sub':     True,   # 内部标记，用于排序
                })
            else:
                # 无 subtopic 匹配：返回章节级别（保持向后兼容）
                result.append({
                    'id':    sid,
                    'title': chapter_title,
                    'score': sc,
                    '_has_sub': False,
                })

        # ── 二级排序：有 subtopic 精确匹配的条目优先排在前面 ──
        # 这样 topics[0] 总是最具体的知识点，而不是通用章节（如 M1-1 数学建模）
        result.sort(key=lambda x: (0 if x.get('_has_sub') else 1, -x['score']))
        # 清除内部排序标记
        for r in result:
            r.pop('_has_sub', None)

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
    # ── FP1 ──
    ('FP1-1', ['complex number','imaginary','real part','imaginary part','argand',
               'modulus-argument','argument of z','conjugate','|z|',
               'locus','loci','complex roots'], []),
    ('FP1-2', ['roots of quadratic','sum of roots','product of roots',
               'alpha.*beta','beta.*alpha','symmetric function'], []),
    ('FP1-3', ['numerical method','interval bisection','linear interpolation',
               'newton-raphson','fixed point iteration'], []),
    ('FP1-4', ['parabola','rectangular hyperbola','focus','directrix',
               'conic','parametric','coordinate system'], []),
    ('FP1-5', ['matrix','determinant','inverse matrix','singular',
               'simultaneous equation.*matrix','matrix equation',
               'eigenvalue','eigenvector'], []),
    ('FP1-6', ['transformation','rotation','reflection','enlargement',
               'matrix.*transformation','invariant','stretch'], []),
    ('FP1-7', ['series','sum of squares','sum of cubes','standard series',
               r'r\^2', r'r\^3','summation','sigma'], []),
    ('FP1-8', ['proof by induction','induction','base case','inductive step',
               'assume.*true for n=k','true for n=k+1'], []),
    # ── FP2 ── (对齐 syllabus_edexcel_maths.json: FP2-1..FP2-8)
    ('FP2-1', ['inequalities','inequality.*algebraic','modulus inequality',
               'rational inequality'], []),
    ('FP2-2', ['series','method of differences','partial fractions.*series'], []),
    ('FP2-3', ['complex number.*further','de moivre','nth roots of unity',
               "exponential form","euler","e^{i","modulus-argument form","locus.*complex"], []),
    # FP2-4: Further Argand Diagrams (loci in complex plane, transformations)
    ('FP2-4', ['argand diagram','loci.*complex plane','complex.*locus',
               'half-line','circle.*complex','|z - a|','arg(z','z - z1',
               'perpendicular bisector.*complex','transformation.*complex'], []),
    # FP2-5: First-Order Differential Equations
    ('FP2-5', ['first order differential equation','first-order differential',
               'integrating factor','separable differential','exact equation',
               'dy/dx.*y','first order.*ode'], []),
    # FP2-6: Second-Order Differential Equations
    ('FP2-6', ['second order differential equation','second-order differential',
               'complementary function','particular integral','auxiliary equation',
               'd²y/dx²','second order.*ode'], []),
    # FP2-7: Maclaurin and Taylor Series
    ('FP2-7', ['maclaurin series','taylor series','power series expansion',
               'series expansion'], []),
    # FP2-8: Polar Coordinates
    ('FP2-8', ['polar coordinate','polar curve','area.*polar',
               'r = f(theta)','cardioid','rose curve',
               'convert.*polar','polar.*cartesian'], []),
    # ── M1 ──
    # M1-1: Mathematical Models in Mechanics
    # 只匹配真正讨论"建立模型/验证模型/建模假设"的题，不匹配泛用力学关键词
    ('M1-1', ['mathematical model','validate.*model','improve.*model',
              'state.*assumption','list.*assumption','modelling assumption',
              'limitations of the model','comment on.*model',
              'rigid body assumption','particle assumption'], []),
    ('M1-2', ['constant acceleration','suvat','v = u + at','s = ut',
              'v² = u²','velocity-time graph','displacement-time',
              'kinematics','free fall','acceleration due to gravity',
              'uniform acceleration'], []),
    ('M1-3', ['vector.*velocity','vector.*force','resultant vector',
              'column vector','i.*j component','bearing','component form',
              'unit vector','position vector','direction of motion',
              'express.*vector'], []),
    ('M1-4', ['newton','f = ma','equation of motion','dynamics','thrust',
              'tension','newton.s.*law','mass.*acceleration',
              'connected particles','pulley','resistance to motion',
              'net force','resultant force'], []),
    ('M1-5', ['friction','normal reaction','coefficient of friction',
              'limiting friction','rough surface','resolve.*forces',
              'inclined plane','frictional force','smooth surface.*friction'], []),
    ('M1-6', ['momentum','impulse','conservation of momentum','collision',
              'impact','explosion','i = mv - mu','change in momentum',
              'perfectly elastic','coefficient of restitution'], []),
    ('M1-7', ['equilibrium','statics','lami.*theorem','triangle of forces',
              'concurrent','resolve.*equilibrium','in equilibrium',
              'system is in equilibrium'], []),
    ('M1-8', ['moment','torque','couple','turning effect','clockwise',
              'anticlockwise','beam','uniform rod','sum of moments',
              'pivot','fulcrum'], []),
    # ── M2 ── (对齐 syllabus_edexcel_maths.json: M2-1..M2-6)
    ('M2-1', ['projectile','horizontal component','vertical component',
              'trajectory','range.*projectile','maximum height',
              'time of flight'], []),
    # M2-2: Variable Acceleration (using calculus/integration in kinematics)
    ('M2-2', ['variable acceleration','acceleration.*function of time',
              'v = ds/dt','a = dv/dt','integrate.*velocity','differentiate.*displacement',
              'x = \\int v','v = \\int a','acceleration varies','non-constant acceleration'], []),
    # M2-3: Centres of Mass
    ('M2-3', ['centre of mass','centroid','composite body',
              'lamina','uniform.*lamina','non-uniform','center of mass',
              'centre of gravity','suspended'], []),
    # M2-4: Work and Energy
    ('M2-4', ['work done','energy','kinetic energy','potential energy',
              'conservation of energy','power','work-energy theorem',
              'joule','watt','gravitational pe','elastic pe'], []),
    # M2-5: Impulses and Collisions (includes coefficient of restitution, elastic)
    ('M2-5', ['impulse','coefficient of restitution','elastic collision',
              'inelastic collision','hooke.*law','natural length',
              'elastic string','modulus of elasticity','extension',
              'i = mv - mu','conservation of momentum','collision'], []),
    # M2-6: Statics of Rigid Bodies
    ('M2-6', ['statics.*rigid body','toppling','sliding','tilting',
              'rigid body.*equilibrium','centre of mass.*equilibrium',
              'overturning','limiting equilibrium.*rod'], []),
    # ── S1 ──
    ('S1-1', ['mathematical model','statistical model','population','sample',
              'assumption.*model'], []),
    ('S1-2', ['mean','median','mode','standard deviation','variance','quartile',
              'interquartile range','skewness','range.*data','outlier'], []),
    ('S1-3', ['histogram','frequency density','stem.*leaf','box plot',
              'cumulative frequency','scatter diagram','representation'], []),
    ('S1-4', ['probability','venn diagram','tree diagram','conditional probability',
              'independent event','mutually exclusive','p(a|b)','p(a and b)',
              'p(a or b)','complement'], []),
    ('S1-5', ['correlation','regression','product moment','pmcc',
              'line of best fit','scatter','bivariate','y on x','x on y',
              'least squares'], []),
    ('S1-6', ['discrete random variable','probability distribution','expectation',
              'expected value','e(x)','var(x)','probability function',
              'discrete uniform'], []),
    ('S1-7', ['normal distribution','standard normal','z-score','phi','z table',
              'standardise','n(mu,sigma','symmetry.*normal'], []),
    # ── S2 ── (对齐 syllabus_edexcel_maths.json: S2-1..S2-7)
    ('S2-1', ['binomial distribution','b(n,p)','binomial probability',
              'number of successes','bernoulli'], []),
    ('S2-2', ['poisson distribution','po(lambda)','poisson probability',
              'mean = variance','rare event'], []),
    # S2-3: Approximations (normal approx to binomial/poisson)
    ('S2-3', ['normal approximation','continuity correction','approximate.*normal',
              'approximate.*binomial','approximate.*poisson',
              'np > 5','nq > 5','large n'], []),
    # S2-4: Continuous Random Variables
    ('S2-4', ['continuous random variable','probability density function','pdf',
              'f(x)','cumulative distribution function','cdf',
              'f(x) = 0 outside','p(x < ','p(x > '], []),
    # S2-5: Continuous Uniform Distribution
    ('S2-5', ['continuous uniform distribution','rectangular distribution',
              'uniform over','u(a,b)','uniform distribution'], []),
    # S2-6: Sampling and Sampling Distributions
    ('S2-6', ['sampling distribution','central limit theorem','distribution of sample mean',
              'unbiased estimator','sample variance','estimation',
              'confidence interval','point estimate'], []),
    # S2-7: Hypothesis Testing
    ('S2-7', ['hypothesis test','null hypothesis','alternative hypothesis',
              'h_0','h_1','significance level','critical region',
              'p-value','test statistic','one-tailed','two-tailed'], []),
    # ── D1 ──
    ('D1-1', ['algorithm','flow chart','bubble sort','quick sort','bin packing',
              'first fit','full bin','order of algorithm','complexity'], []),
    ('D1-2', ['graph','network','node','vertex','edge','arc','degree',
              'bipartite','matching','complete graph','cycle','path'], []),
    ('D1-3', ['kruskal','prim','dijkstra','minimum spanning tree','shortest path',
              'minimum connector','route inspection','chinese postman'], []),
    ('D1-4', ['route inspection','chinese postman','eulerian','semi-eulerian',
              'traversable','odd vertex'], []),
    ('D1-5', ['travelling salesman','upper bound','lower bound','nearest neighbour',
              'hamilton cycle'], []),
    ('D1-6', ['critical path','activity network','early time','late time',
              'float','critical activity','precedence','gantt chart',
              'resource histogram'], []),
    ('D1-7', ['linear programming','objective function','feasible region',
              'constraint','vertex.*optimal','simplex','inequalit.*region'], []),
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


# ─────────────────────────────────────────
# Edexcel Economics 考纲加载（缓存）
# ─────────────────────────────────────────
_edexcel_economics_syllabus_cache = None

def _load_edexcel_economics_syllabus():
    global _edexcel_economics_syllabus_cache
    if _edexcel_economics_syllabus_cache is not None:
        return _edexcel_economics_syllabus_cache
    path = os.path.join(os.path.dirname(__file__), 'static', 'syllabus_edexcel_economics.json')
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            _edexcel_economics_syllabus_cache = json.load(f)
        return _edexcel_economics_syllabus_cache
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────
# Edexcel Economics 知识点关键词匹配规则
# 格式：(topic_id, [必须关键词], [加分关键词])
# 每命中1个必须关键词+2分，加分关键词+1分
# ─────────────────────────────────────────────────────────────────────
_ECONOMICS_CHAPTER_RULES = [
    # ── Unit 1 ──────────────────────────────────────────────────────
    ('U1-1', ['ceteris paribus','economic model','positive statement','normative statement',
              'scarcity','opportunity cost','production possibility','PPF',
              'specialisation','division of labour','free market','command economy',
              'mixed economy','renewable resource','non-renewable'],
             ['economic problem','finite resources','unlimited wants','economic good','free good']),
    ('U1-2', ['demand','demand curve','shift in demand','consumer surplus',
              'price elasticity of demand','PED','income elasticity','YED',
              'cross elasticity','XED','marginal utility','diminishing marginal utility',
              'rational','utility maximisation','substitute','complement',
              'normal good','inferior good','income elastic','price inelastic','price elastic'],
             ['consumer behaviour','habitual','inertia','framing','herding','bias']),
    ('U1-3', ['supply','supply curve','shift in supply','price elasticity of supply','PES',
              'indirect tax','specific tax','ad valorem','subsidy','elastic supply',
              'inelastic supply','short run','long run'],
             ['technology','natural disaster','producer','cost of production']),
    ('U1-4', ['equilibrium','market equilibrium','excess demand','excess supply',
              'price mechanism','consumer surplus','producer surplus',
              'indirect tax','subsidy','incidence','rationing','signalling','incentive'],
             ['equilibrium price','equilibrium quantity','market forces','surplus','shortage']),
    ('U1-5', ['market failure','externality','external cost','external benefit',
              'social cost','social benefit','private cost','private benefit',
              'negative externality','positive externality','public good','free rider',
              'non-rival','non-excludable','asymmetric information','moral hazard',
              'market bubble','speculation','welfare loss','merit good','demerit good'],
             ['imperfect information','information gap','insurance','banking','housing']),
    ('U1-6', ['government intervention','government failure','tradeable pollution permit',
              'property rights','regulation','maximum price','minimum price',
              'state provision','regulatory capture','information gap','unintended consequence'],
             ['greenhouse gas','carbon tax','pollution','health','education','transport']),

    # ── Unit 2 ──────────────────────────────────────────────────────
    ('U2-1', ['GDP','GNI','gross domestic product','gross national income','economic growth',
              'inflation','deflation','disinflation','CPI','consumer price index',
              'unemployment','employment','balance of payments','current account',
              'real GDP','nominal GDP','per capita','purchasing power parity','PPP',
              'recession','output gap'],
             ['living standards','ILO','frictional unemployment','structural unemployment',
              'demand deficient','real wage','measuring inflation']),
    ('U2-2', ['aggregate demand','AD curve','consumption','investment','government expenditure',
              'net exports','X minus M','savings ratio','marginal propensity',
              'disposable income','interest rate','consumer confidence','wealth effect'],
             ['C+I+G','component of AD','shift in AD','movement along AD']),
    ('U2-3', ['aggregate supply','AS curve','SRAS','LRAS','short-run aggregate supply',
              'long-run aggregate supply','Keynesian','classical','potential output'],
             ['shift in AS','movement along AS','cost of production','productivity']),
    ('U2-4', ['circular flow','national income','injection','withdrawal','multiplier',
              'marginal propensity to consume','MPC','MPS','MPT','MPM',
              'multiplier formula','equilibrium national output'],
             ['savings','taxation','imports','exports','investment injection']),
    ('U2-5', ['actual growth','potential growth','economic growth','output gap',
              'positive output gap','negative output gap','export-led growth','FDI',
              'benefits of growth','costs of growth','living standards'],
             ['innovation','labour force','productivity','environment','inequality']),
    ('U2-6', ['fiscal policy','monetary policy','supply-side policy','macroeconomic objective',
              'inflation target','interest rate','quantitative easing','central bank',
              'government spending','taxation','reflationary','deflationary',
              'Phillips curve','unemployment inflation trade-off'],
             ['demand-side policy','lender of last resort','balanced budget',
              'income equality','balance of payments equilibrium']),

    # ── Unit 3 ──────────────────────────────────────────────────────
    ('U3-1', ['types of business','private sector','public sector','co-operative',
              'joint venture','SME','merger','takeover','organic growth',
              'vertical integration','horizontal integration','conglomerate',
              'business objective','profit maximisation','revenue maximisation',
              'sales maximisation','principal-agent','satisficing','divorce of ownership'],
             ['demerger','constraints on growth','economies of scale business']),
    ('U3-2', ['total revenue','average revenue','marginal revenue','total cost',
              'average cost','marginal cost','fixed cost','variable cost',
              'diminishing returns','law of diminishing returns','economies of scale',
              'diseconomies of scale','minimum efficient scale','internal economies',
              'external economies','normal profit','supernormal profit','loss',
              'short-run cost','long-run cost','shutdown point'],
             ['X-inefficiency','communication problem','coordination problem']),
    ('U3-3', ['perfect competition','monopolistic competition','oligopoly','monopoly',
              'monopsony','contestability','sunk cost','barrier to entry',
              'price discrimination','game theory','collusion','cartel',
              'price leadership','predatory pricing','limit pricing','concentration ratio',
              'allocative efficiency','productive efficiency','dynamic efficiency',
              'natural monopoly','price war','non-price competition'],
             ['market structure','profit maximising','MR=MC','interdependence',
              'Nash equilibrium','price maker','price taker']),
    ('U3-4', ['labour market','demand for labour','supply of labour','wage rate',
              'derived demand','elasticity of demand for labour','elasticity of supply of labour',
              'geographical immobility','occupational immobility','equilibrium wage',
              'public sector wage','trade union'],
             ['labour mobility','minimum wage','migration','age distribution']),
    ('U3-5', ['competition policy','merger control','price regulation','profit regulation',
              'privatisation','deregulation','regulatory authority','regulatory capture',
              'minimum wage','maximum wage','government intervention business',
              'discrimination','exploitation','monopsony power'],
             ['competition commission','anti-trust','quality standard','performance target']),

    # ── Unit 4 ──────────────────────────────────────────────────────
    ('U4-1', ['globalisation','transnational company','TNC','foreign direct investment','FDI',
              'trade liberalisation','migration','globalisation cost','globalisation benefit',
              'transfer pricing','income inequality globalisation'],
             ['trading bloc','Soviet','opening up China','transport cost','communication']),
    ('U4-2', ['comparative advantage','absolute advantage','specialisation trade',
              'terms of trade','trade pattern','free trade','trading bloc',
              'WTO','world trade organisation','tariff','quota','subsidy domestic',
              'non-tariff barrier','trade creation','trade diversion',
              'customs union','common market','free trade area','economic union',
              'protectionism','infant industry','dumping'],
             ['gains from trade','restrictions on trade','Prebisch-Singer']),
    ('U4-3', ['balance of payments','current account','capital account','financial account',
              'exchange rate','floating exchange rate','fixed exchange rate','managed float',
              'appreciation','depreciation','devaluation','revaluation',
              'Marshall-Lerner condition','J-curve','purchasing power parity',
              'international competitiveness','relative unit labour cost','current account deficit',
              'current account surplus','speculation currency','capital flight'],
             ['relative productivity','export price','non-price factor','FDI flow']),
    ('U4-4', ['absolute poverty','relative poverty','inequality','Lorenz curve',
              'Gini coefficient','income distribution','wealth inequality','income inequality',
              'poverty line','aid','debt relief','welfare benefit'],
             ['education training','structural change','civil war','life expectancy']),
    ('U4-5', ['public expenditure','government spending','transfer payment','national debt',
              'fiscal deficit','fiscal surplus','automatic stabiliser','discretionary fiscal',
              'structural deficit','cyclical deficit','taxation','direct tax','indirect tax',
              'progressive tax','regressive tax','proportional tax','Laffer curve',
              'corporation tax','crowding out','debt servicing'],
             ['capital expenditure','current expenditure','intergenerational equity']),
    ('U4-6', ['HDI','human development index','developing country','emerging economy',
              'Harrod-Domar','savings gap','foreign currency gap','primary product dependency',
              'microfinance','infrastructure development','market-orientated',
              'interventionist','World Bank','IMF','international monetary fund',
              'NGO','Lewis model','Lewis dual sector','debt relief','aid'],
             ['corruption','governance','commodity price','demographic','access to credit']),
]


# ══════════════════════════════════════════════════════════════════════════════
# BPhO 知识点分类规则表
# 格式：(topic_id, [required_kws], [bonus_kws])
# topic_id 格式：'BPhO-N-M'  N=章节号(1-12), M=子话题号(1-4)
# 对应12个物理章节，每章3-4个二级知识点
# ══════════════════════════════════════════════════════════════════════════════
_BPHO_TOPIC_RULES = [

    # ── Ch1: Kinematics (运动学) ────────────────────────────────────────────
    ('BPhO-1-1',
     ['velocity', 'acceleration', 'displacement', 'suvat', 'uniform acceleration',
      'speed', 'distance-time', 'velocity-time', 'v-t graph', 'gradient of', 'area under'],
     ['constant acceleration', 'deceleration', 'rest', 'initial velocity', 'final velocity',
      'kinematic equation', 'motion equation']),

    ('BPhO-1-2',
     ['relative velocity', 'relative motion', 'frame of reference', 'intercept', 'catch up',
      'overtake', 'reference frame', 'relative speed'],
     ['observer', 'moving frame', 'pursuit', 'interception']),

    ('BPhO-1-3',
     ['projectile', 'horizontal component', 'vertical component', 'trajectory',
      'parabola', 'range', 'angle of projection', 'launch angle',
      'vector decomposition', 'resolve'],
     ['maximum height', 'time of flight', 'horizontal distance', 'launched at angle']),

    ('BPhO-1-4',
     ['calculus', 'differentiate', 'integrate', 'rate of change', 'variable acceleration',
      'angular velocity', 'angular speed', 'omega', 'linear speed', 'tangential',
      'v = rω', 'a = rα'],
     ['non-uniform', 'varying acceleration', 'derivative', 'integral of velocity']),

    # ── Ch2: Dynamics / Newton's Laws (动力学) ─────────────────────────────
    ('BPhO-2-1',
     ["newton's first law", "newton's second law", "newton's third law",
      'net force', 'resultant force', 'F = ma', 'equation of motion',
      'inertia', 'mass', 'force diagram', 'free body'],
     ['law of motion', 'unbalanced force', 'balanced force', 'static equilibrium',
      'dynamic equilibrium']),

    ('BPhO-2-2',
     ['friction', 'coefficient of friction', 'normal reaction', 'rough surface',
      'sliding', 'static friction', 'kinetic friction', 'limiting friction',
      'μ', 'frictional force'],
     ['smooth surface', 'grip', 'traction', 'braking force', 'skid']),

    ('BPhO-2-3',
     ['inclined plane', 'slope', 'component of weight', 'angle of incline',
      'tension', 'connected particles', 'pulley', 'atwood', 'string', 'rope',
      'normal force on slope'],
     ['wedge', 'ramp', 'smooth incline', 'rough incline', 'weight component']),

    ('BPhO-2-4',
     ['drag', 'air resistance', 'terminal velocity', 'resistive force',
      'stokes', 'viscous', 'viscosity', 'buoyancy', 'upthrust', 'archimedes'],
     ['fluid resistance', 'streamline', 'laminar', 'turbulent', 'velocity-dependent']),

    # ── Ch3: Work, Energy & Power (功、能量与功率) ──────────────────────────
    ('BPhO-3-1',
     ['work done', 'work-energy theorem', 'kinetic energy', 'potential energy',
      'gravitational potential energy', 'elastic potential energy',
      'W = Fd', 'KE', 'GPE', 'EPE', '½mv²', 'mgh'],
     ['work against', 'energy transfer', 'joule', 'stored energy']),

    ('BPhO-3-2',
     ['conservation of energy', 'energy conservation', 'mechanical energy',
      'total energy', 'energy dissipated', 'efficiency', 'useful energy',
      'heat loss', 'wasted energy', 'energy transformation'],
     ['isolated system', 'no energy loss', 'energy input', 'energy output']),

    ('BPhO-3-3',
     ['power', 'rate of work', 'watt', 'P = W/t', 'P = Fv',
      'instantaneous power', 'average power', 'horsepower', 'engine power'],
     ['output power', 'input power', 'power rating', 'electrical power']),

    # ── Ch4: Momentum & Collisions (动量与碰撞) ─────────────────────────────
    ('BPhO-4-1',
     ['momentum', 'conservation of momentum', 'impulse', 'change in momentum',
      'linear momentum', 'p = mv', 'J = FΔt', 'impulse-momentum theorem',
      'collision', 'explosion'],
     ['total momentum', 'before and after', 'recoil', 'rifle and bullet']),

    ('BPhO-4-2',
     ['elastic collision', 'perfectly elastic', 'kinetic energy conserved',
      'coefficient of restitution', 'newton\'s law of restitution', 'e ='],
     ['relative speed', 'approach', 'separation', 'head on', 'glancing']),

    ('BPhO-4-3',
     ['inelastic collision', 'perfectly inelastic', 'coalesce', 'stick together',
      'kinetic energy lost', 'energy lost in collision', 'crumple'],
     ['merge', 'couple', 'common velocity', 'energy dissipated in collision']),

    # ── Ch5: Circular Motion & Gravitation (圆周运动与万有引力) ─────────────
    ('BPhO-5-1',
     ['circular motion', 'centripetal force', 'centripetal acceleration',
      'a = v²/r', 'F = mv²/r', 'angular velocity', 'period', 'frequency',
      'radian', 'arc length', 'tangential speed', 'uniform circular'],
     ['banked curve', 'conical pendulum', 'roundabout', 'car on bend']),

    ('BPhO-5-2',
     ["newton's law of gravitation", 'gravitational force', 'F = GMm/r²',
      'gravitational field', 'gravitational field strength', 'g = GM/r²',
      'gravitational potential', 'V = -GM/r', 'escape velocity', 'escape speed'],
     ['inverse square law', 'gravitational constant', 'G', 'attraction between masses']),

    ('BPhO-5-3',
     ['orbital', 'orbit', 'satellite', "kepler's third law", 'T² ∝ r³',
      'geostationary', 'geosynchronous', 'orbital period', 'orbital radius',
      'circular orbit', 'elliptical orbit'],
     ['ISS', 'moon orbit', 'planetary motion', 'centripetal = gravitational']),

    # ── Ch6: SHM & Oscillations (简谐运动与振动) ────────────────────────────
    ('BPhO-6-1',
     ['simple harmonic motion', 'SHM', 'a = -ω²x', 'restoring force',
      'amplitude', 'angular frequency', 'displacement', 'x = A cos',
      'x = A sin', 'period of oscillation', 'frequency of oscillation'],
     ['equilibrium position', 'oscillation', 'vibration', 'sinusoidal']),

    ('BPhO-6-2',
     ['simple pendulum', 'T = 2π√(l/g)', 'mass-spring system', 'T = 2π√(m/k)',
      'spring constant', 'spring stiffness', 'Hooke\'s law', 'k', 'elastic',
      'pendulum period'],
     ['bob', 'string length', 'small angle approximation', 'natural frequency']),

    ('BPhO-6-3',
     ['resonance', 'forced oscillation', 'driving frequency', 'natural frequency',
      'damping', 'damped oscillation', 'critical damping', 'overdamped',
      'underdamped', 'Q factor', 'amplitude at resonance'],
     ['energy loss', 'oscillation decays', 'resonant frequency', 'driver']),

    # ── Ch7: Waves & Optics (波动与光学) ────────────────────────────────────
    ('BPhO-7-1',
     ['wave', 'wavelength', 'frequency', 'wave speed', 'v = fλ',
      'transverse wave', 'longitudinal wave', 'amplitude', 'period',
      'phase', 'phase difference', 'wavefront', 'intensity'],
     ['crest', 'trough', 'compression', 'rarefaction', 'wave equation']),

    ('BPhO-7-2',
     ['reflection', 'refraction', 'Snell\'s law', 'n₁sinθ₁ = n₂sinθ₂',
      'refractive index', 'total internal reflection', 'critical angle',
      'lens', 'focal length', 'mirror', 'image', 'object distance',
      '1/v + 1/u', 'magnification', 'converging', 'diverging'],
     ['angle of incidence', 'angle of refraction', 'optical fibre', 'prism']),

    ('BPhO-7-3',
     ['interference', 'diffraction', 'superposition', 'path difference',
      'constructive interference', 'destructive interference',
      'double slit', 'diffraction grating', 'fringe', 'nλ = d sinθ',
      'coherent', 'monochromatic'],
     ['bright fringe', 'dark fringe', 'fringe spacing', 'grating equation',
      'standing wave', 'stationary wave', 'node', 'antinode']),

    # ── Ch8: Electricity & Circuits (电学与电路) ─────────────────────────────
    ('BPhO-8-1',
     ['current', 'voltage', 'resistance', "ohm's law", 'V = IR',
      'series circuit', 'parallel circuit', 'resistor', 'conductor',
      'kirchhoff', 'KVL', 'KCL', 'potential divider', 'voltmeter', 'ammeter',
      'power dissipated', 'P = I²R', 'P = V²/R'],
     ['loop equation', 'junction rule', 'battery', 'electric circuit',
      'total resistance', 'equivalent resistance']),

    ('BPhO-8-2',
     ['capacitor', 'capacitance', 'C = Q/V', 'charge stored', 'energy stored',
      'E = ½CV²', 'charging', 'discharging', 'time constant', 'τ = RC',
      'capacitor in series', 'capacitor in parallel'],
     ['dielectric', 'plate separation', 'plates', 'farad', 'exponential decay']),

    ('BPhO-8-3',
     ['EMF', 'electromotive force', 'internal resistance', 'terminal voltage',
      'ε = V + Ir', 'short circuit', 'variable resistance',
      'lost volts', 'efficiency of cell', 'battery circuit'],
     ['cell', 'source of EMF', 'external circuit', 'load resistance', 'ammeter reads']),

    # ── Ch9: Electromagnetism (电磁学) ──────────────────────────────────────
    ('BPhO-9-1',
     ['electric field', 'E = F/q', 'electric field strength',
      'magnetic field', 'magnetic flux density', 'B field', 'tesla',
      'F = qE', 'uniform field', 'field lines', 'potential difference'],
     ['coulomb', 'charge', 'electric force', 'field between plates']),

    ('BPhO-9-2',
     ['electromagnetic induction', 'Faraday\'s law', 'Lenz\'s law',
      'induced EMF', 'flux', 'magnetic flux', 'Φ = BA',
      'rate of change of flux', 'transformer', 'mutual inductance',
      'self inductance', 'solenoid'],
     ['coil', 'changing field', 'flux linkage', 'NΦ', 'cutting field lines']),

    ('BPhO-9-3',
     ['Lorentz force', 'F = BIL', 'F = BqV', 'force on conductor',
      'force on charge', 'motor effect', 'Hall effect',
      'cyclotron', 'charged particle in field',
      'radius of circular path', 'r = mv/Bq'],
     ['velocity selector', 'mass spectrometer', 'cathode ray', 'electron beam']),

    # ── Ch10: Thermal Physics (热学) ────────────────────────────────────────
    ('BPhO-10-1',
     ['ideal gas', 'gas law', 'pV = nRT', 'Boyle\'s law', 'Charles\'s law',
      'pressure law', 'Gay-Lussac', 'mole', 'Avogadro', 'molecular mass',
      'p₁V₁/T₁', 'p₂V₂/T₂', 'gas pressure', 'gas temperature', 'gas volume'],
     ['number of moles', 'kelvin', 'absolute temperature', 'gas constant R',
      'amount of substance', 'STP', 'withdrawn', 'remaining gas']),

    ('BPhO-10-2',
     ['internal energy', 'first law of thermodynamics', 'ΔU = Q - W',
      'isothermal', 'adiabatic', 'isobaric', 'isochoric',
      'heat capacity', 'specific heat capacity', 'Q = mcΔT',
      'latent heat', 'Q = mL', 'thermodynamics cycle', 'carnot'],
     ['work done by gas', 'heat absorbed', 'heat rejected', 'thermal efficiency']),

    ('BPhO-10-3',
     ['conduction', 'convection', 'radiation', 'thermal radiation',
      'Stefan-Boltzmann', 'P = σAT⁴', 'Newton\'s law of cooling',
      'thermal conductivity', 'heat flow rate', 'blackbody', 'emissivity',
      'Wien\'s law', 'peak wavelength'],
     ['insulation', 'U-value', 'heat loss', 'cooling rate', 'temperature gradient']),

    # ── Ch11: Nuclear & Radioactivity (核物理与放射性) ──────────────────────
    ('BPhO-11-1',
     ['radioactive decay', 'alpha', 'beta', 'gamma', 'half-life', 't½',
      'activity', 'decay constant', 'λ', 'A = λN', 'N = N₀e^(-λt)',
      'background radiation', 'ionising radiation', 'Geiger', 'count rate'],
     ['radioactive source', 'decay series', 'daughter nucleus', 'parent nucleus',
      'radiation dose', 'becquerel']),

    ('BPhO-11-2',
     ['nuclear reaction', 'fission', 'fusion', 'chain reaction',
      'nuclear equation', 'proton number', 'nucleon number',
      'mass number', 'atomic number', 'isotope', 'nuclide',
      'conservation of nucleon number', 'conservation of charge'],
     ['moderator', 'control rod', 'reactor', 'critical mass', 'neutron']),

    ('BPhO-11-3',
     ['binding energy', 'mass defect', 'nuclear mass', 'E = mc²',
      'mass-energy equivalence', 'unified mass unit', 'u',
      'binding energy per nucleon', 'nuclear stability curve',
      'energy released in nuclear', 'Q value'],
     ['MeV', 'einstein equation', 'atomic mass unit', 'iron peak',
      'energy from fission', 'energy from fusion']),

    # ── Ch12: Modern Physics (近代物理) ─────────────────────────────────────
    ('BPhO-12-1',
     ['photoelectric effect', 'photon', 'E = hf', 'work function', 'threshold frequency',
      'Planck constant', 'h', 'quantum', 'de Broglie', 'wave-particle duality',
      'λ = h/p', 'stopping potential', 'photoelectron',
      'kinetic energy of electron', 'hf = φ + KE_max'],
     ['photoemission', 'electron volt', 'eV', 'electromagnetic spectrum',
      'X-ray', 'UV', 'intensity', 'frequency threshold']),

    ('BPhO-12-2',
     ['atomic spectra', 'energy level', 'emission spectrum', 'absorption spectrum',
      'ground state', 'excited state', 'ionisation energy', 'photon emission',
      'Bohr model', 'hydrogen spectrum', 'line spectrum',
      'energy level diagram', 'transition'],
     ['Lyman', 'Balmer', 'Paschen', 'series', 'spectral line', 'quantum number']),

    ('BPhO-12-3',
     ['special relativity', 'time dilation', 'length contraction',
      'Lorentz factor', 'γ', 'relativistic mass', 'relativistic momentum',
      'rest mass energy', 'E₀ = m₀c²', 'relativistic kinetic energy',
      'speed of light', 'invariant'],
     ['reference frame', 'proper time', 'proper length', 'twin paradox',
      'muon experiment', 'simultaneity']),
]


# BPhO 知识点 ID → 一级/二级标题映射（用于前端展示）
_BPHO_TOPIC_TITLES = {
    # Ch1 Kinematics
    'BPhO-1':   '1. 运动学 (Kinematics)',
    'BPhO-1-1': '1.1 匀变速直线运动与图象 (suvat & v-t Graphs)',
    'BPhO-1-2': '1.2 相对运动与追及问题 (Relative Motion)',
    'BPhO-1-3': '1.3 抛体运动与矢量分解 (Projectile & Vectors)',
    'BPhO-1-4': '1.4 变加速与角/线速度 (Calculus & Angular)',
    # Ch2 Dynamics
    'BPhO-2':   '2. 动力学 (Dynamics)',
    'BPhO-2-1': '2.1 牛顿定律 (Newton\'s Laws)',
    'BPhO-2-2': '2.2 摩擦力 (Friction)',
    'BPhO-2-3': '2.3 斜面与绳拉问题 (Inclined Planes & Pulleys)',
    'BPhO-2-4': '2.4 阻力与浮力 (Drag & Buoyancy)',
    # Ch3 Work/Energy
    'BPhO-3':   '3. 功与能 (Work & Energy)',
    'BPhO-3-1': '3.1 功与动能定理 (Work-Energy Theorem)',
    'BPhO-3-2': '3.2 能量守恒 (Conservation of Energy)',
    'BPhO-3-3': '3.3 功率 (Power)',
    # Ch4 Momentum
    'BPhO-4':   '4. 动量与碰撞 (Momentum & Collisions)',
    'BPhO-4-1': '4.1 动量守恒与冲量 (Momentum & Impulse)',
    'BPhO-4-2': '4.2 弹性碰撞 (Elastic Collision)',
    'BPhO-4-3': '4.3 非弹性碰撞 (Inelastic Collision)',
    # Ch5 Circular/Gravity
    'BPhO-5':   '5. 圆周运动与引力 (Circular Motion & Gravity)',
    'BPhO-5-1': '5.1 匀速圆周运动 (Uniform Circular Motion)',
    'BPhO-5-2': '5.2 万有引力 (Gravitation)',
    'BPhO-5-3': '5.3 卫星运动与开普勒 (Orbital Mechanics)',
    # Ch6 SHM
    'BPhO-6':   '6. 简谐运动 (SHM & Oscillations)',
    'BPhO-6-1': '6.1 简谐运动方程 (SHM Equations)',
    'BPhO-6-2': '6.2 弹簧与摆 (Spring & Pendulum)',
    'BPhO-6-3': '6.3 共振与阻尼 (Resonance & Damping)',
    # Ch7 Waves
    'BPhO-7':   '7. 波动与光学 (Waves & Optics)',
    'BPhO-7-1': '7.1 波的基本性质 (Wave Properties)',
    'BPhO-7-2': '7.2 反射、折射与透镜 (Reflection & Refraction)',
    'BPhO-7-3': '7.3 干涉与衍射 (Interference & Diffraction)',
    # Ch8 Electricity
    'BPhO-8':   '8. 电路与电学 (Electricity & Circuits)',
    'BPhO-8-1': '8.1 电路与欧姆定律 (Circuits & Ohm\'s Law)',
    'BPhO-8-2': '8.2 电容器 (Capacitors)',
    'BPhO-8-3': '8.3 电动势与内阻 (EMF & Internal Resistance)',
    # Ch9 Electromagnetism
    'BPhO-9':   '9. 电磁学 (Electromagnetism)',
    'BPhO-9-1': '9.1 电场与磁场 (Electric & Magnetic Fields)',
    'BPhO-9-2': '9.2 电磁感应 (Electromagnetic Induction)',
    'BPhO-9-3': '9.3 洛伦兹力 (Lorentz Force)',
    # Ch10 Thermal
    'BPhO-10':   '10. 热学 (Thermal Physics)',
    'BPhO-10-1': '10.1 理想气体 (Ideal Gas Laws)',
    'BPhO-10-2': '10.2 热力学定律 (Laws of Thermodynamics)',
    'BPhO-10-3': '10.3 热传递 (Heat Transfer & Radiation)',
    # Ch11 Nuclear
    'BPhO-11':   '11. 核物理 (Nuclear Physics)',
    'BPhO-11-1': '11.1 放射性衰变 (Radioactive Decay)',
    'BPhO-11-2': '11.2 核反应 (Nuclear Reactions)',
    'BPhO-11-3': '11.3 质能方程与结合能 (Binding Energy & E=mc²)',
    # Ch12 Modern Physics
    'BPhO-12':   '12. 近代物理 (Modern Physics)',
    'BPhO-12-1': '12.1 光电效应与量子 (Photoelectric & Quantum)',
    'BPhO-12-2': '12.2 原子能级与光谱 (Atomic Spectra)',
    'BPhO-12-3': '12.3 相对论 (Special Relativity)',
}


# 经济学试卷代码 → 单元映射
_ECONOMICS_CODE_MAP = {
    'WEC11': 'U1',
    'WEC12': 'U2',
    'WEC13': 'U3',
    'WEC14': 'U4',
}


def detect_edexcel_economics_unit(doc) -> str:
    """
    从封面识别 Edexcel IAL 经济学试卷的具体单元。
    返回：'U1'|'U1A'|'U2'|'U2A'|'U3'|'U3A'|'U4'|'U4A'|'unknown'

    WEC 代码格式：
      WEC11/01  → U1     WEC11/01A → U1A
      WEC12/01  → U2     WEC12/01A → U2A
      WEC13/01  → U3     WEC13/01A → U3A
      WEC14/01  → U4     WEC14/01A → U4A
    同一年份可能同时存在 WEC11/01 和 WEC11/01A 两套卷子，必须区分。
    """
    for pg_i in range(min(2, doc.page_count)):
        text = doc[pg_i].get_text()
        # 精确匹配 WEC1x/数字[可选后缀字母]，如 WEC11/01A、WEC14/01
        m = re.search(r'WEC1([1-4])/\d+([A-Z])?', text)
        if m:
            unit_num = m.group(1)
            suffix   = m.group(2) or ''   # 'A' 或 ''
            return f'U{unit_num}{suffix}'
        # fallback：只含 WEC1x（无 /数字 部分）
        m2 = re.search(r'WEC1([1-4])', text)
        if m2:
            return f'U{m2.group(1)}'
        # 通过标题文字识别（无 WEC 代码时的兜底）
        if 'Markets in action' in text:
            return 'U1'
        if 'Macroeconomic performance' in text:
            return 'U2'
        if 'Business behaviour' in text:
            return 'U3'
        if 'Developments in the global economy' in text:
            return 'U4'
    return 'unknown'
#  Edexcel IAL Maths 题目难度数据库（来源：Examiner Report 关键词分析）
#  结构：_MATHS_DIFFICULTY_DB[unit_code][(year, q_num)] = 1–5
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

# ── WMA13 (P3) ──────────────────────────────────────────────────────
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

# ── 其他 unit 静态难度数据库（来自 zip 内 Report，2023年） ──────────────────

# WME01 (M1) – June2023
_WME01_DB = {
    (2023, 1): 2, (2023, 2): 2, (2023, 3): 2, (2023, 4): 1,
    (2023, 5): 1, (2023, 6): 2, (2023, 7): 2, (2023, 8): 2,
}
# WME02 (M2) – June2023
_WME02_DB = {
    (2023, 1): 2, (2023, 2): 3, (2023, 3): 2,
    (2023, 4): 2, (2023, 5): 2, (2023, 6): 3, (2023, 7): 3,
}
# WFM01 (FP1) – June2023
_WFM01_DB = {
    (2023, 1): 1, (2023, 2): 2, (2023, 3): 2, (2023, 4): 2,
    (2023, 5): 2, (2023, 6): 2, (2023, 7): 2, (2023, 8): 3, (2023, 9): 2,
}
# WFM02 (FP2) – June2023
_WFM02_DB = {
    (2023, 1): 1, (2023, 2): 3, (2023, 3): 2,
    (2023, 4): 1, (2023, 5): 2, (2023, 6): 2, (2023, 7): 3, (2023, 8): 2,
}
# WDM11 (D1) – June2023
_WDM11_DB = {
    (2023, 1): 3, (2023, 2): 2, (2023, 3): 2, (2023, 4): 4,
    (2023, 5): 2, (2023, 6): 3, (2023, 7): 2, (2023, 8): 5,
}
# WMA13 (P3) 2021 来自 Jan2021 Report
_P3_2021_DB = {
    (2021, 1): 3, (2021, 2): 2, (2021, 3): 1, (2021, 4): 2,
    (2021, 5): 2, (2021, 6): 1, (2021, 7): 2, (2021, 8): 2,
    (2021, 9): 2, (2021, 10): 3,
}

# 合并到统一DB，按 unit_code 索引
_MATHS_DIFFICULTY_DB: dict[str, dict] = {
    'WMA13': {**_P3_DIFFICULTY_DB, **_P3_2021_DB},
    'WME01': _WME01_DB,
    'WME02': _WME02_DB,
    'WFM01': _WFM01_DB,
    'WFM02': _WFM02_DB,
    'WDM11': _WDM11_DB,
    # P1/P2/P4/S1/S2等暂无数据，会用 _score_difficulty_report 动态提取
}

# Edexcel Maths unit -> unit_code 映射
_MATHS_UNIT_TO_CODE = {
    'P1': 'WMA11', 'P2': 'WMA12', 'P3': 'WMA13', 'P4': 'WMA14',
    'FP1': 'WFM01', 'FP2': 'WFM02', 'FP3': 'WFM03',
    'M1': 'WME01', 'M2': 'WME02',
    'S1': 'WST01', 'S2': 'WST02',
    'D1': 'WDM11',
}

# 运行时 Report 难度缓存（session 级别），key = (unit_code, year, q_num)
_report_difficulty_cache: dict[tuple, int] = {}
_report_cache_lock = __import__('threading').Lock()


def _score_difficulty_report(text: str) -> int:
    """
    基于 Examiner Report 单题段落文字，返回 1-5★ 难度。
    加权关键词评分后映射到 1-5。
    """
    t = text.lower()
    score = 0

    # ── 极难信号 ──
    if re.search(r'modal (score|mark) (was|of) 0\b', t):    score += 8
    if 'modal score of zero' in t:                           score += 8
    if 'attrition' in t:                                     score += 5
    if 'most challenging' in t:                              score += 4
    if 'challenging end to the paper' in t:                  score += 4
    if 'blank responses' in t:                               score += 2

    # ── 较难信号 ──
    if 'proved to be quite a challenging' in t:              score += 3
    if 'challenging for many' in t:                          score += 3
    if 'discriminated well' in t:                            score += 2
    if 'discriminating marks' in t:                          score += 2
    if 'more challenging' in t:                              score += 2
    if 'accuracy marks more demanding' in t:                 score += 3
    if 'accuracy marks' in t and 'demanding' in t:           score += 1
    if 'more mixed' in t or 'mixed success' in t:            score += 2
    if 'only a minority' in t:                               score += 2
    if 'very few were able to achieve full' in t:            score += 2
    if 'few students were able' in t:                        score += 2
    if 'considerable number' in t and 'left it blank' in t:  score += 3
    if 'good discriminator' in t or 'a good source of discriminat' in t:
                                                             score += 2

    # ── 简单信号 ──
    if 'largely accessible' in t:                            score -= 1
    if 'accessible to most students' in t:                   score -= 2
    if 'accessible to most' in t and 'accessible to most students' not in t:
                                                             score -= 1
    if 'access into all parts' in t:                         score -= 1
    if 'most accessible question' in t:                      score -= 3
    if 'well answered by most' in t:                         score -= 4
    if 'generally well answered' in t:                       score -= 3
    if 'well answered' in t and 'generally well answered' not in t:
                                                             score -= 2
    if 'most students scored full marks' in t:               score -= 3
    if 'most gained full marks' in t:                        score -= 3
    if 'most students gained full marks' in t:               score -= 3
    if 'straightforward' in t:                               score -= 3
    if 'familiar topic' in t:                                score -= 2
    if 'most students correctly' in t:                       score -= 2
    if 'most students were able' in t:                       score -= 1
    if 'majority were able' in t:                            score -= 1
    if 'majority were successful' in t:                      score -= 1
    if 'more accessible question' in t:                      score -= 2
    if 'most accessible' in t and 'most accessible question' not in t:
                                                             score -= 3
    if 'routine' in t:                                       score -= 1
    if 'good source of marks' in t:                          score -= 1
    if 'friendly starter' in t or 'good start' in t:        score -= 2
    if 'opening question' in t and 'success' in t:           score -= 1

    # 映射
    if   score >= 7:  return 5
    elif score >= 4:  return 4
    elif score >= 1:  return 3
    elif score >= -1: return 2
    else:             return 1


def extract_difficulty_from_report(doc) -> dict[int, int] | None:
    """
    解析 Examiner Report PDF，返回 {q_num: difficulty_1_to_5}。
    若文档不像 Examiner Report 则返回 None。
    """
    full_text = ''
    for i in range(min(12, doc.page_count)):
        full_text += doc[i].get_text() + '\n'

    # 必须含有 "Examiner" 或 "Report" 字样（报告特征）
    if 'examiner' not in full_text.lower() and 'report' not in full_text.lower():
        return None
    # 必须含有 "Question" 段落
    if not re.search(r'Question\s+\d', full_text, re.IGNORECASE):
        return None

    # 读取全文
    full_text = ''
    for i in range(doc.page_count):
        full_text += doc[i].get_text() + '\n'

    sections: dict[int, str] = {}
    pat = re.compile(r'Question\s+(\d+)\s*\n', re.IGNORECASE)
    matches = list(pat.finditer(full_text))
    for idx, m in enumerate(matches):
        q_num = int(m.group(1))
        start = m.end()
        end   = matches[idx + 1].start() if idx + 1 < len(matches) else len(full_text)
        txt   = full_text[start:end].strip()
        sections[q_num] = txt

    if not sections:
        return None

    result = {}
    for q_num, txt in sections.items():
        result[q_num] = _score_difficulty_report(txt)
    return result


def rate_question_difficulty(q_num: int, year: int | None, source: str,
                              maths_unit: str | None,
                              unit_code: str | None = None) -> int | None:
    """
    返回题目难度星级 1–5，或 None（非 edexcel_maths 时）。

    查找优先级：
      1. 运行时 Report 缓存（动态解析的）
      2. 静态 _MATHS_DIFFICULTY_DB 精确匹配 (year, q_num)
      3. 同 unit 同 q_num 跨年平均
      4. 同 unit 同年平均
      5. 全局 unit 平均 ≈ 3
    """
    if source != 'edexcel_maths':
        return None

    # 推断 unit_code
    if not unit_code and maths_unit:
        unit_code = _MATHS_UNIT_TO_CODE.get(maths_unit)

    # 1. 运行时 Report 缓存
    if unit_code and year:
        with _report_cache_lock:
            cached = _report_difficulty_cache.get((unit_code, year, q_num))
        if cached is not None:
            return cached

    # 2. 静态数据库精确匹配
    db = _MATHS_DIFFICULTY_DB.get(unit_code or '', {})
    if year and (year, q_num) in db:
        return db[(year, q_num)]

    # 3. 同 unit 同 q_num 跨年平均
    if unit_code and unit_code in _MATHS_DIFFICULTY_DB:
        vals = [v for (y, q), v in _MATHS_DIFFICULTY_DB[unit_code].items() if q == q_num]
        if vals:
            return max(1, min(5, round(sum(vals) / len(vals))))

    # 4. 同 unit 同年平均
    if unit_code and year and unit_code in _MATHS_DIFFICULTY_DB:
        vals = [v for (y, q), v in _MATHS_DIFFICULTY_DB[unit_code].items() if y == year]
        if vals:
            return max(1, min(5, round(sum(vals) / len(vals))))

    # 5. 全局兜底（edexcel maths 题目普遍中等）
    return 3




def _extract_year_from_filename(filename: str) -> int | None:
    """从文件名中提取4位年份数字，如 'WMA13_October2023.pdf' → 2023"""
    if not filename:
        return None
    m = re.search(r'(20\d{2})', filename)
    return int(m.group(1)) if m else None


def _extract_file_uuid(filename: str) -> str | None:
    """
    从文件名中提取 UUID 前缀（如果存在）。
    格式：{uuid}_{原始文件名}，例如：
      f767253f-6f06-4de5-a046-f62820eb7603_A-Level-EDEXCELQP-202505-Economics-U4.pdf
    返回 UUID 字符串，或 None（若文件名不含 UUID 前缀）。
    """
    if not filename:
        return None
    m = re.match(r'^([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})_', filename, re.IGNORECASE)
    return m.group(1).lower() if m else None


# 月份名称 → 标准月份编号（用于 session key 规范化）
_MONTH_TO_NUM = {
    'january':1,'february':2,'march':3,'april':4,'may':5,'june':6,
    'july':7,'august':8,'september':9,'october':10,'november':11,'december':12,
    'jan':1,'feb':2,'mar':3,'apr':4,'jun':6,'jul':7,
    'aug':8,'sep':9,'oct':10,'nov':11,'dec':12,
}

# Edexcel IAL 考试通常在 Jan / June / Oct 三个 session
# 规范化：把 October/Nov/Dec → 'oct'，May/Jun/Jul → 'jun'，Jan/Feb → 'jan'
_MONTH_NUM_TO_SESSION = {
    1:'jan', 2:'jan', 3:'jan',
    4:'jun', 5:'jun', 6:'jun', 7:'jun',
    8:'oct', 9:'oct', 10:'oct', 11:'oct', 12:'oct',
}

def _extract_exam_session(filename: str) -> str | None:
    """
    从文件名中提取考试 session 标识，格式为 '{year}_{session}'。
    例如：
      Questionpaper-Unit3(WMA13)-June2023.pdf  → '2023_jun'
      Markscheme-WMA11-October2022.pdf         → '2022_oct'
      WMA13_QP_Jan2021.pdf                     → '2021_jan'
      P3_June_2021_QP.pdf                      → '2021_jun'
      A-Level-EDEXCELQP-202505-Economics.pdf   → '2025_may'
      A-Level-EDEXCELQP-202510-Economics.pdf   → '2025_oct'
    返回 None 表示无法提取。
    """
    if not filename:
        return None
    fn = filename
    # 模式0：6位数字年月格式（如 202505 → 2025_may，202510 → 2025_oct）
    # 用于 A-Level-EDEXCELQP-202505-... 格式文件名
    m0 = re.search(r'(20\d{2})(0[1-9]|1[0-2])(?!\d)', fn)
    if m0:
        year_str = m0.group(1)
        month_num = int(m0.group(2))
        sess = _MONTH_NUM_TO_SESSION[month_num]
        return f'{year_str}_{sess}'
    # 模式1：MonthName + 4位年份（顺序，如 June2023 / June 2023 / June-2023）
    m = re.search(r'([A-Za-z]+)[-_ ]?(20\d{2})', fn, re.IGNORECASE)
    if m:
        month_str = m.group(1).lower()
        year_str  = m.group(2)
        if month_str in _MONTH_TO_NUM:
            mnum = _MONTH_TO_NUM[month_str]
            sess = _MONTH_NUM_TO_SESSION[mnum]
            return f'{year_str}_{sess}'
    # 模式2：4位年份 + MonthName（反序，如 2023_June / 2023June）
    m2 = re.search(r'(20\d{2})[-_ ]?([A-Za-z]+)', fn, re.IGNORECASE)
    if m2:
        year_str  = m2.group(1)
        month_str = m2.group(2).lower()
        if month_str in _MONTH_TO_NUM:
            mnum = _MONTH_TO_NUM[month_str]
            sess = _MONTH_NUM_TO_SESSION[mnum]
            return f'{year_str}_{sess}'
    # 只有年份：返回年份字符串（宽松匹配用）
    m3 = re.search(r'(20\d{2})', fn)
    if m3:
        return m3.group(1)  # 只有年份，无 session 区分
    return None


# 月份名称列表（全写和缩写），供 _extract_exam_date_label 使用
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
_MULTI_DIR = storage.local_tmp_path('') if storage.is_r2_mode() else os.path.join(storage.local_root(), 'multi')
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
                path = g.get('path', '')
                # workbook 虚拟组：path 为空或 source==workbook，跳过文件存在检查
                if g.get('source') == 'workbook':
                    continue
                if not path or not os.path.exists(path):
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


@app.route('/api/syllabus/bpho', methods=['GET'])
def get_bpho_syllabus():
    """
    返回 BPhO (British Physics Olympiad) 12章物理知识库目录。
    格式与 edexcel_maths 保持一致：{topics:[{id, title, subtopics:[{id, title}]}]}
    """
    topics = []
    # 按章节号排序：BPhO-1 … BPhO-12
    ch_ids = sorted(
        {k for k in _BPHO_TOPIC_TITLES if re.match(r'^BPhO-\d+$', k)},
        key=lambda x: int(x.split('-')[1])
    )
    for ch_id in ch_ids:
        ch_title = _BPHO_TOPIC_TITLES.get(ch_id, ch_id)
        ch_num   = ch_id.split('-')[1]
        # 子知识点：BPhO-N-1, BPhO-N-2, …
        sub_ids = sorted(
            {k for k in _BPHO_TOPIC_TITLES if re.match(rf'^BPhO-{ch_num}-\d+$', k)},
            key=lambda x: int(x.split('-')[2])
        )
        subtopics = [
            {'id': sid, 'title': _BPHO_TOPIC_TITLES[sid]}
            for sid in sub_ids
        ]
        topics.append({'id': ch_id, 'title': ch_title, 'subtopics': subtopics})
    return jsonify({'syllabus': 'bpho', 'topics': topics})


@app.route('/api/syllabus/edexcel_economics', methods=['GET'])
def get_edexcel_economics_syllabus():
    """
    返回 Edexcel IAL Economics 知识库目录。
    可选 ?unit=U1 过滤，只返回该 unit 的 topics。
    """
    data = _load_edexcel_economics_syllabus()
    if data is None:
        return jsonify({'error': 'Edexcel Economics 知识库未加载'}), 404

    unit = request.args.get('unit')
    if unit:
        import copy
        filtered = copy.deepcopy(data)
        filtered['topics'] = [t for t in filtered['topics']
                               if t.get('unit') == unit]
        filtered['active_unit'] = unit
        return jsonify(filtered)

    return jsonify(data)



def _is_markscheme_filename(filename: str) -> bool:
    """判断文件名是否为 Mark Scheme（支持多种命名格式）"""
    fn = filename.lower()
    return ('markscheme' in fn or 'mark_scheme' in fn or
            'mark scheme' in fn or '_ms_' in fn or
            fn.endswith('_ms.pdf') or '-ms-' in fn or '-ms.' in fn)


def _is_markscheme_by_content(doc) -> bool:
    """
    通过 PDF 首页内容判断是否为 Mark Scheme。
    Cambridge 9702 MS 首页固定包含 "MARK SCHEME" 字样。
    返回 True 表示该文件是 Mark Scheme。
    """
    try:
        # 只扫描前2页，避免遍历整个文档
        for pg_i in range(min(2, doc.page_count)):
            text = doc[pg_i].get_text()
            upper = text.upper()
            if 'MARK SCHEME' in upper:
                return True
    except Exception:
        pass
    return False


def _extract_9702_paper_info(filename: str) -> dict:
    """
    从 Cambridge 9702 文件名提取试卷信息。
    格式示例：9702_m20_qp_42.pdf / 9702_m20_ms_42.pdf
               9702_s21_qp_12.pdf / 9702_s21_ms_12.pdf
               9702_w19_qp_41.pdf / 9702_w19_ms_41.pdf
    返回: {
        'code': '9702',
        'session': 'm20',       # 考试季：m=Mar, s=Jun, w=Nov
        'type': 'qp' | 'ms',   # 试卷类型
        'variant': '42',         # 卷号
        'session_key': '9702_m20_42',  # 用于匹配 QP/MS 对
    }
    若无法解析，返回 {}
    """
    fn = filename.lower().rstrip('.pdf').replace('.pdf', '')
    # 匹配格式：9702_s21_qp_12 或 9702_m20_ms_42
    m = re.match(
        r'(9702)_([mswMSW]\d{2})_(qp|ms)_(\d{1,2}[a-z]?)',
        fn, re.IGNORECASE
    )
    if not m:
        return {}
    code    = m.group(1)
    session = m.group(2).lower()
    ptype   = m.group(3).lower()
    variant = m.group(4).lower()
    return {
        'code':        code,
        'session':     session,
        'type':        ptype,
        'variant':     variant,
        'session_key': f'{code}_{session}_{variant}',
    }


def _is_examiner_report_filename(filename: str) -> bool:
    """判断文件名是否为 Examiner Report（支持多种命名格式）"""
    fn = filename.lower()
    return ('examinerreport' in fn or 'examiner_report' in fn or
            'examiner report' in fn or '_er_' in fn or
            '-er-' in fn or 'examinersreport' in fn or
            'examiners_report' in fn)


def _extract_unit_from_filename(filename: str) -> str | None:
    """
    从文件名提取单元代码，例如：
      Markscheme-Unit1(WMA11)-June2023.pdf  → 'WMA11'
      Questionpaper-Unit3(WMA13)-Oct2022.pdf → 'WMA13'
      WST01_QP_June2023.pdf → 'WST01'
    返回映射后的 unit 字符串（P1/P2/S1 等）或 None。
    """
    m = re.search(r'\((W[A-Z]{2}\d{2})\)', filename, re.IGNORECASE)
    if m:
        code = m.group(1).upper()
        return _EDEXCEL_MATHS_CODE_MAP.get(code)
    m2 = re.search(r'\b(W[A-Z]{2}\d{2})\b', filename, re.IGNORECASE)
    if m2:
        code = m2.group(1).upper()
        return _EDEXCEL_MATHS_CODE_MAP.get(code)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Cambridge 9702 Physics Mark Scheme — 表格式答案提取
# ─────────────────────────────────────────────────────────────────────────────

def _detect_9702_ms_table(doc):
    """
    解析 Cambridge 9702 物理 Mark Scheme PDF 中的答案表格。

    实际格式（A4 纵向 PDF，rotation=90°横排显示）：
      - MediaBox = (0,0,595,842)：物理纸面是 A4 纵向
      - rotation=90 → 逻辑页面显示为横向（842×595）
      - get_text() 返回物理坐标（595×842 空间内）
      - get_pixmap(clip=...) 使用逻辑坐标（842×595 空间）
      - 坐标转换（rotation=90）：
          物理 (px, py) → 逻辑 (lx, ly) = (py, phys_w - px)
          其中 phys_w = mediabox.width = 595

      物理坐标布局（以 9702/42 March 2020 为例）：
        - 物理 y ≈ 55-72：Marks 行（B1/C1/A1/M1）—— 显示时在左侧
        - 物理 y ≈ 394-434：Answer 列头 —— 显示时在中部
        - 物理 y ≈ 475-720：实际答案文字内容
        - 物理 y ≈ 731-781：Question 列头 + 题号 —— 显示时在右侧
        - 物理 x ≈ 56-68：最左列（固定列头 Marks/Answer/Question）
        - 物理 x ≈ 80-370：各子题列（每列约 25pt 宽）
      每页只有一道大题（Q1-Q12 各占一页）

    返回：
      {q_num: [(page_idx, lx0, ly0, lx1, ly1), ...]}
      其中 lx0/ly0/lx1/ly1 是 get_pixmap(clip=...) 使用的逻辑坐标
    """
    Q_NUM_PAT = re.compile(r'^(\d{1,2})\b')    # 提取整数题号

    result = {}   # q_num(int) → [(pg_i, lx0, ly0, lx1, ly1)]

    for pg_i in range(doc.page_count):
        page = doc[pg_i]

        # 只处理 rotation=90 的页面（9702 MS 答案页）
        if page.rotation != 90:
            continue

        phys_w = page.mediabox.width    # 物理宽度 ≈ 595
        phys_h = page.mediabox.height   # 物理高度 ≈ 842

        try:
            blocks = page.get_text('dict', flags=fitz.TEXT_PRESERVE_WHITESPACE)['blocks']
        except Exception:
            continue

        # ── 收集该页所有文字行（物理坐标）──
        tlines = []
        for b in blocks:
            if b.get('type') != 0:
                continue
            for line in b.get('lines', []):
                ltxt = ''.join(s['text'] for s in line['spans']).strip()
                if ltxt:
                    bbox = line['bbox']
                    # 物理坐标：(px0, py0, px1, py1, txt)
                    tlines.append((bbox[0], bbox[1], bbox[2], bbox[3], ltxt))

        # ── 判断该页是否是答案页：需含 "Question" 标题（物理 x ≈ 56-70，物理 y > 700）──
        has_question_hdr = any(
            txt.strip() == 'Question' and px0 < 75 and py0 > 700
            for px0, py0, px1, py1, txt in tlines
        )
        if not has_question_hdr:
            continue

        # ── 找题号行（物理 y > 730，物理 x > 70）──
        q_col_pxs = {}   # q_num(int) → [px0, ...]
        for px0, py0, px1, py1, txt in tlines:
            if py0 < 730 or px0 < 70:
                continue
            # 跳过固定列头和版权行
            if txt.strip() in ('9702/42', '9702/41', '9702/43', '9702/44',
                                '© UCLES 2020', '© UCLES 2021', '© UCLES 2022',
                                '© UCLES 2023', '© UCLES 2024', '[Turn over',
                                'Question', 'Answer', 'Marks',
                                'PUBLISHED', 'SEEN', '^', '', ' '):
                continue
            m = Q_NUM_PAT.match(txt.strip())
            if m:
                qn = int(m.group(1))
                if 1 <= qn <= 30:
                    if qn not in q_col_pxs:
                        q_col_pxs[qn] = []
                    q_col_pxs[qn].append(px0)

        if not q_col_pxs:
            continue

        q_nums = sorted(q_col_pxs.keys())
        print(f'[9702_ms_table] Page {pg_i}: found q_nums={q_nums}')

        # ── 对每道大题，确定物理 x 范围，然后转换到逻辑坐标 ──
        # 物理 y 范围：从 Marks 顶部（约 54）到 Question 标题上方（约 730）
        # 物理 x 范围：从该题最小列 x - 5，到下一题最小列 x（或最大列 x + 30）
        phys_y_top = 54.0    # Marks 行顶部
        phys_y_bot = 726.0   # Question 列头上方

        for qi, qn in enumerate(q_nums):
            col_pxs = sorted(q_col_pxs[qn])
            phys_x_left  = max(68.0, min(col_pxs) - 5.0)

            if qi + 1 < len(q_nums):
                next_qn    = q_nums[qi + 1]
                next_pxs   = sorted(q_col_pxs[next_qn])
                phys_x_right = min(next_pxs) - 2.0
            else:
                phys_x_right = min(phys_w - 10.0, max(col_pxs) + 28.0)

            if phys_x_right - phys_x_left < 10:
                continue

            # 物理坐标 → 逻辑坐标（rotation=90）：
            #   lx = phys_y,  ly = phys_w - phys_x
            lx0 = phys_y_top
            lx1 = phys_y_bot
            ly0 = max(0.0, phys_w - phys_x_right)
            ly1 = min(page.rect.height, phys_w - phys_x_left)

            # 确保逻辑坐标合法
            if lx1 - lx0 < 10 or ly1 - ly0 < 5:
                print(f'[9702_ms] skip tiny logical clip q{qn}: lx={lx0:.1f}-{lx1:.1f}, ly={ly0:.1f}-{ly1:.1f}')
                continue

            if qn not in result:
                result[qn] = []
            result[qn].append((pg_i, lx0, ly0, lx1, ly1))

    print(f'[9702_ms_table] final result: {sorted(result.keys())} questions')
    return result


def _render_9702_ms_answers(ms_doc, dpi=150):
    """
    用 _detect_9702_ms_table() 定位 9702 MS 答案区，渲染为 JPEG base64。
    返回 {q_num: {'b64': str, 'w': int, 'h': int}}。
    结果坐标已转换为逻辑坐标，可直接传给 get_pixmap(clip=...)。
    """
    import base64 as _b64
    from PIL import Image as _PILImg

    table_slices = _detect_9702_ms_table(ms_doc)
    if not table_slices:
        return {}

    result = {}
    scale  = dpi / 72.0
    mat    = fitz.Matrix(scale, scale)

    for q_num, slices in table_slices.items():
        parts = []
        total_w, total_h = 0, 0
        try:
            for (pg_i, lx0, ly0, lx1, ly1) in slices:
                page    = ms_doc[pg_i]
                log_w   = page.rect.width
                log_h   = page.rect.height
                # 确保逻辑 clip 坐标合法，避免 "Invalid bandwriter header" 错误
                x0_c = max(0.0, min(float(lx0), float(lx1)))
                x1_c = min(log_w, max(float(lx0), float(lx1)))
                y0_c = max(0.0, min(float(ly0), float(ly1)))
                y1_c = min(log_h, max(float(ly0), float(ly1)))
                if x1_c - x0_c < 10 or y1_c - y0_c < 5:
                    print(f'[9702_ms] skip tiny clip q{q_num}: ({x0_c:.1f},{y0_c:.1f},{x1_c:.1f},{y1_c:.1f})')
                    continue
                clip = fitz.Rect(x0_c, y0_c, x1_c, y1_c)
                pix  = page.get_pixmap(matrix=mat, clip=clip, colorspace=fitz.csRGB)
                w, h = pix.width, pix.height
                data = _pixmap_to_jpeg_bytes(pix)
                del pix
                parts.append((data, w, h))
                total_w = max(total_w, w)
                total_h += h

            if not parts:
                continue

            if len(parts) == 1:
                jpeg_out, cw, ch = parts[0]
            else:
                canvas = _PILImg.new('RGB', (total_w, total_h), (255, 255, 255))
                y_off  = 0
                for (jpeg_part, pw_p, ph_p) in parts:
                    img_part = _PILImg.open(io.BytesIO(jpeg_part))
                    if pw_p != total_w:
                        img_part = img_part.resize(
                            (total_w, int(ph_p * total_w / pw_p)), _PILImg.LANCZOS)
                        ph_p = img_part.size[1]
                    canvas.paste(img_part, (0, y_off))
                    y_off += ph_p
                buf = io.BytesIO()
                canvas.save(buf, format='JPEG', quality=88)
                jpeg_out = buf.getvalue()
                cw, ch   = total_w, y_off

            result[q_num] = {
                'b64': _b64.b64encode(jpeg_out).decode('utf-8'),
                'w':   cw,
                'h':   ch,
            }
        except Exception as e:
            print(f'[9702_ms] render error q{q_num}: {e}')

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Edexcel Maths Mark Scheme — 表格式答案提取
# ─────────────────────────────────────────────────────────────────────────────

def _detect_edexcel_maths_ms_table(doc):
    """
    扫描 Edexcel Maths Mark Scheme PDF，按题切割答案区域。

    切割规则（严格按样板图三色标注）：
      - y_top = 表头行（蓝色：Question Number / Scheme / Marks）的 y0
      - q_num = 表头之后 Question Number 列（左列）第一个数字开头的文本，取首数字
                真实格式例：'1', '2(i)', '3(a)', '4(a)', '7(a)' → q_num = 1/2/3/4/7
      - y_bot = 该表格区间内 Marks 列最后一个 "Total N" 行的 y1
                （"Total N" 比 "(N marks)" 更可靠，两者都检测，优先 Total）
      - Notes 文字在表格外部（Total 行之后），不截取

    本文件结构（WMA13 Jan2021 Mark Scheme）：
      - 每道题单独一页（或连续两页），每页顶部有独立的表头
      - 总分标识为 "Total N"（如 "Total 3", "Total 6"），在 Marks 列右侧
      - 子题小计为 "(N)"，总分为 "Total N"

    返回：
      {q_num: [(page_idx, y_top, y_bottom, x_left, x_right), ...]}
    """
    # WMA/WFM 系列表头：Question / Scheme / Marks（三列完整）
    # WST/WMS 系列表头：Qu / Scheme / Marks（"Qu" 代替 "Question"）
    # WME/WDM 系列表头：Question + Number（分两行）/ Scheme / Marks
    HDR_WORDS  = {'question', 'scheme', 'marks'}
    # WST/WMS 系列备用：用 'qu' 替代 'question'
    HDR_WORDS_QU = {'qu', 'scheme', 'marks'}
    # 总分行：优先匹配 "Total N"（更精确），其次 "(N marks)" / "(N)"
    TOTAL_PAT  = re.compile(r'^Total\s+\d+$', re.IGNORECASE)
    PAREN_PAT  = re.compile(r'^\(\s*\d+\s*(?:marks?)?\s*\)$', re.IGNORECASE)
    # 题号：行首 1-2 位数字（后可接任意字符，包括 "(i)", "(a)" 等）
    Q_NUM_PAT  = re.compile(r'^(\d{1,2})\b')

    # ── Step 1：找每页的表头块 (page_idx, hdr_y0, hdr_y1) ──
    header_blocks = []
    for pg_i in range(doc.page_count):
        page = doc[pg_i]
        try:
            blocks = page.get_text('dict', flags=fitz.TEXT_PRESERVE_WHITESPACE)['blocks']
        except Exception:
            continue
        tlines = []
        for b in blocks:
            if b.get('type') != 0: continue
            for line in b.get('lines', []):
                ltxt = ''.join(s['text'] for s in line['spans']).strip().lower()
                if ltxt:
                    tlines.append((line['bbox'][1], line['bbox'][3],
                                   line['bbox'][0], line['bbox'][2], ltxt))
        tlines.sort(key=lambda x: x[0])
        n = len(tlines)
        i = 0
        while i < n:
            y0_i = tlines[i][0]
            ww = set(); wy0 = y0_i; wy1 = tlines[i][1]
            j = i
            while j < n:
                y0_j, y1_j, x0_j, x1_j, txt_j = tlines[j]
                if y0_j > y0_i + 40: break
                # 同时收集 HDR_WORDS 和 HDR_WORDS_QU 的词到 ww
                for hw in HDR_WORDS | HDR_WORDS_QU:
                    if hw in txt_j: ww.add(hw)
                wy0 = min(wy0, y0_j)
                wy1 = max(wy1, y1_j)
                j += 1
            # 满足完整表头（question/scheme/marks）或简化表头（qu/scheme/marks）
            if HDR_WORDS <= ww or HDR_WORDS_QU <= ww:
                header_blocks.append((pg_i, wy0, wy1))
                i = j
            else:
                i += 1

    if not header_blocks:
        return {}

    # ── Step 2：收集全部文本行（从第一个表头页开始）──
    first_pg = header_blocks[0][0]
    all_lines = []   # (pg_i, y0, y1, x0, x1, text)
    for pg_i in range(first_pg, doc.page_count):
        page = doc[pg_i]
        ph = page.rect.height
        try:
            blocks = page.get_text('dict', flags=fitz.TEXT_PRESERVE_WHITESPACE)['blocks']
        except Exception:
            continue
        for b in blocks:
            if b.get('type') != 0: continue
            for line in b.get('lines', []):
                bbox = line['bbox']
                y0, y1, x0, x1 = bbox[1], bbox[3], bbox[0], bbox[2]
                if y0 > ph - 28: continue
                ltxt = ''.join(s['text'] for s in line['spans']).strip()
                if ltxt:
                    all_lines.append((pg_i, y0, y1, x0, x1, ltxt))

    # ── Step 3：动态确定列宽 ──
    # Question Number 列右边界 = "Scheme" 文字 x0（在表头中）
    # Marks 列左边界 = "Marks" 文字 x0（在表头中）
    pw_def = doc[0].rect.width
    q_col_max_x = pw_def * 0.20   # 默认 20% 页宽
    marks_col_x = pw_def * 0.75   # 默认 75% 页宽

    for pg_i, hy0, hy1 in header_blocks[:3]:   # 用前3个表头取平均
        page = doc[pg_i]
        pw = page.rect.width
        try:
            blocks = page.get_text('dict', flags=fitz.TEXT_PRESERVE_WHITESPACE)['blocks']
        except Exception:
            continue
        for b in blocks:
            if b.get('type') != 0: continue
            for line in b.get('lines', []):
                ltxt = ''.join(s['text'] for s in line['spans']).strip().lower()
                bbox = line['bbox']
                y0_l = bbox[1]
                if not (hy0 - 2 <= y0_l <= hy1 + 4): continue
                # Scheme 列：含 "scheme"，不含 "question"
                if 'scheme' in ltxt and 'question' not in ltxt:
                    q_col_max_x = max(q_col_max_x, bbox[0] + 8)
                # Marks 列：仅含 "marks"，在右侧 55% 以后
                if ltxt.strip() == 'marks' and bbox[0] > pw * 0.55:
                    marks_col_x = min(marks_col_x, bbox[0] - 5)

    q_col_max_x = min(q_col_max_x, 220)
    marks_col_x = max(marks_col_x, pw_def * 0.60)

    # ── Step 4：为每个表头块确定 q_num + 总分行 y_bot ──
    # 搜索区间：
    #   q_num     搜索：从 hdr_y0 开始（题号在表头带内，与 Question/Scheme/Marks 同行区间）
    #   总分行    搜索：从 hdr_y1 开始（避免把表头 "Marks" 文字误认为总分）
    #   两者结束：下一表头 hdr_y0（或页末）
    search_regions = []
    for idx, (pg_i, hy0, hy1) in enumerate(header_blocks):
        if idx + 1 < len(header_blocks):
            n_pg, n_hy0, _ = header_blocks[idx + 1]
            search_regions.append((pg_i, hy0, hy1, n_pg, n_hy0))
        else:
            lp = doc.page_count - 1
            search_regions.append((pg_i, hy0, hy1, lp, doc[lp].rect.height - 28))

    answers = {}

    for (hdr_pg, hdr_y0, hdr_y1), (s_pg, q_start_y, total_start_y, e_pg, e_y) in \
            zip(header_blocks, search_regions):

        q_num          = None
        best_total_pg  = None
        best_total_y1  = None
        best_paren_pg  = None
        best_paren_y1  = None

        for pg_i, y0, y1, x0, x1, ltxt in all_lines:
            if pg_i < s_pg or pg_i > e_pg: continue
            if pg_i == e_pg and y0 >= e_y: continue

            # ── 找题号：Question Number 列最左侧 ──
            # WMA/WFM/WST 系列题号 x0 约在 46-55；WME/WDM 系列 x0 约在 90-97
            # x0 < 110 可覆盖所有系列，同时过滤正文中更靠右的数字
            # 搜索范围：表头页（hdr_pg），y ∈ [hdr_y0-2, hdr_y1+60]
            # +60 容差：WME/WDM 表格中题号有时位于表头带之后（有注释行间隔）
            if q_num is None and x0 < 110:
                if pg_i == hdr_pg and hdr_y0 - 2 <= y0 <= hdr_y1 + 60:
                    m = Q_NUM_PAT.match(ltxt.strip())
                    if m:
                        cand = int(m.group(1))
                        if 1 <= cand <= 30:
                            q_num = cand

            # ── 找总分行：Marks 列（x0 >= marks_col_x - 20）──
            # 从 hdr_y1 开始搜索（避免表头 "Marks" 文字干扰）
            if x0 >= marks_col_x - 20:
                if pg_i > s_pg or y0 >= total_start_y - 2:   # 从 hdr_y1 开始
                    s = ltxt.strip()
                    if TOTAL_PAT.match(s):    # "Total N" → 最优先
                        best_total_pg = pg_i
                        best_total_y1 = y1
                    if PAREN_PAT.match(s):    # "(N marks)" / "(N)" → 备用
                        best_paren_pg = pg_i
                        best_paren_y1 = y1

        if q_num is None:
            continue

        # 优先用 "Total N" 行；其次用最后一个 "(N)" 行
        if best_total_pg is not None:
            end_pg, end_y = best_total_pg, best_total_y1 + 4
        elif best_paren_pg is not None:
            end_pg, end_y = best_paren_pg, best_paren_y1 + 4
        else:
            end_pg, end_y = e_pg, e_y

        # ── Step 5：生成切片 ──
        slices = []
        for pg_i in range(hdr_pg, end_pg + 1):
            page = doc[pg_i]
            ph = page.rect.height
            pw = page.rect.width
            top    = hdr_y0 if pg_i == hdr_pg else 26
            bottom = end_y  if pg_i == end_pg  else (ph - 28)
            left   = 24
            right  = pw - 24
            if bottom > top + 6:
                slices.append((pg_i, top, bottom, left, right))

        if slices:
            answers[q_num] = slices

    return answers


def _render_ms_answers_from_table(ms_doc, dpi=150):
    """
    用 _detect_edexcel_maths_ms_table() 定位答案区，渲染为 JPEG base64。
    返回 {q_num: {'b64': str, 'w': int, 'h': int}}。
    若表格检测失败（返回空），调用方可退回旧逻辑。
    """
    import base64 as _b64
    from PIL import Image as _PILImg

    table_slices = _detect_edexcel_maths_ms_table(ms_doc)
    if not table_slices:
        return {}

    result = {}
    scale  = dpi / 72.0
    mat    = fitz.Matrix(scale, scale)

    for q_num, slices in table_slices.items():
        parts = []
        total_w, total_h = 0, 0
        try:
            for (pg_i, y_top, y_bot, x_left, x_right) in slices:
                page = ms_doc[pg_i]
                clip = fitz.Rect(x_left, y_top, x_right, y_bot)
                pix  = page.get_pixmap(matrix=mat, clip=clip, colorspace=fitz.csRGB)
                w, h = pix.width, pix.height
                data = _pixmap_to_jpeg_bytes(pix)
                del pix
                parts.append((data, w, h))
                total_w = max(total_w, w)
                total_h += h

            if not parts:
                continue

            if len(parts) == 1:
                jpeg_out, cw, ch = parts[0]
            else:
                canvas = _PILImg.new('RGB', (total_w, total_h), (255, 255, 255))
                y_off  = 0
                for (jpeg_part, pw, ph) in parts:
                    img_part = _PILImg.open(io.BytesIO(jpeg_part))
                    if pw != total_w:
                        img_part = img_part.resize(
                            (total_w, int(ph * total_w / pw)), _PILImg.LANCZOS)
                        ph = img_part.size[1]
                    canvas.paste(img_part, (0, y_off))
                    y_off += ph
                buf = io.BytesIO()
                canvas.save(buf, format='JPEG', quality=88)
                jpeg_out = buf.getvalue()
                cw, ch   = total_w, y_off

            result[q_num] = {
                'b64': _b64.b64encode(jpeg_out).decode('utf-8'),
                'w':   cw,
                'h':   ch,
            }
        except Exception:
            pass

    return result


def _render_ms_questions_b64(ms_doc, ms_questions, paper_type, dpi=150):
    """
    渲染 Mark Scheme 中所有题目为 JPEG base64。
    返回 {q_num: {'b64': str, 'w': int, 'h': int}} 字典。

    策略（优先级）：
      1. 对 edexcel_economics：使用专用函数 _render_edexcel_economics_ms_answers()
      2. 对 edexcel_maths：先尝试表格式检测（Question/Scheme/Marks 表头）
         → _render_ms_answers_from_table()
      3. 若表格检测失败，退回到旧逻辑：_collect_question_slices + 垂直拼接
    """
    import base64 as _b64

    # ── 优先：Edexcel Economics 专用渲染 ──
    if paper_type == 'edexcel_economics':
        return _render_edexcel_economics_ms_answers(ms_doc, dpi=dpi)

    # ── 优先：Edexcel Maths 表格式 MS ──
    if paper_type == 'edexcel_maths':
        table_result = _render_ms_answers_from_table(ms_doc, dpi=dpi)
        if table_result:
            return table_result
        # 表格检测失败时继续走下面的旧逻辑

    # ── 旧逻辑：逐题切片渲染 ──
    result = {}
    for q_idx, q in enumerate(ms_questions):
        q_num = q.get('q_num')
        if q_num is None:
            continue
        try:
            slices = _collect_question_slices(ms_doc, ms_questions, q_idx, paper_type)
            if not slices:
                continue
            # 将所有切片垂直拼接为一张图
            from PIL import Image as _PILImg
            parts = []
            total_w, total_h = 0, 0
            for src_page, clip in slices:
                jpeg, w, h = _render_slice_to_jpeg(src_page, clip, dpi)
                parts.append((jpeg, w, h))
                total_w = max(total_w, w)
                total_h += h
            if not parts:
                continue
            if len(parts) == 1:
                jpeg_combined, cw, ch = parts[0]
            else:
                # 垂直拼接多片
                canvas = _PILImg.new('RGB', (total_w, total_h), (255, 255, 255))
                y_off = 0
                for (jpeg_part, pw, ph) in parts:
                    img_part = _PILImg.open(io.BytesIO(jpeg_part))
                    # 若宽度不等则缩放对齐
                    if pw != total_w:
                        img_part = img_part.resize(
                            (total_w, int(ph * total_w / pw)), _PILImg.LANCZOS)
                        ph = img_part.size[1]
                    canvas.paste(img_part, (0, y_off))
                    y_off += ph
                buf = io.BytesIO()
                canvas.save(buf, format='JPEG', quality=85)
                jpeg_combined = buf.getvalue()
                cw, ch = total_w, y_off

            result[q_num] = {
                'b64': _b64.b64encode(jpeg_combined).decode('utf-8'),
                'w': cw,
                'h': ch
            }
        except Exception:
            pass
    return result



@app.route('/api/upload_multi', methods=['POST'])
def upload_multi():
    """
    多文件上传接口。
    支持同时上传多个PDF（QP + Mark Scheme混合），自动识别文件类型并关联。
    - 文件名含 'Markscheme'/'mark_scheme' 的识别为 MS，与同 unit QP 匹配
    - MS 匹配成功后，为每道 QP 题目添加 answer_b64/answer_w/answer_h
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

    # ── 第一遍：保存所有文件，分类为 QP、MS 和 Examiner Report ──
    qp_groups   = []   # 正常题目卷
    ms_registry = []   # Mark Scheme 信息列表
    er_registry = []   # Examiner Report 信息列表（用于难度填充）

    for file in files:
        if not file.filename or not allowed_file(file.filename):
            continue
        safe = secure_filename(file.filename)
        tmp_path = storage.local_tmp_path(f'{session_id}_{safe}')
        file.save(tmp_path)
        r2_key = f'multi/{session_id}_{safe}'
        storage.upload_from_local(tmp_path, r2_key)

        is_ms     = _is_markscheme_filename(file.filename)
        is_report = _is_examiner_report_filename(file.filename)

        try:
            doc = fitz.open(tmp_path)

            # ── 功能1: 若文件名未识别为MS，通过内容检测（MARK SCHEME字样）补充判断 ──
            if not is_ms and not is_report:
                if _is_markscheme_by_content(doc):
                    is_ms = True
                    print(f'[upload] Content-detected MS: {file.filename}')

            # ── 功能2: Cambridge 9702 文件名解析（9702_m20_qp_42 / 9702_m20_ms_42）──
            paper9702 = _extract_9702_paper_info(file.filename)
            if paper9702.get('type') == 'ms':
                is_ms = True

            if paper_type_hint == 'auto':
                pt = detect_paper_type(doc)
            else:
                pt = paper_type_hint

            source     = detect_paper_source(doc)
            maths_unit = None
            econ_unit  = None
            if source == 'edexcel_maths':
                # 先从文件名取 unit（更可靠），fallback 到内容检测
                maths_unit = (_extract_unit_from_filename(file.filename) or
                              detect_edexcel_maths_unit(doc))
            elif source == 'edexcel_economics':
                econ_unit = detect_edexcel_economics_unit(doc)

            # ── Task 2/1: Examiner Report 处理 ──
            if is_report:
                er_fn_unit = _extract_unit_from_filename(file.filename)
                er_unit    = er_fn_unit or maths_unit
                er_year    = _extract_year_from_filename(file.filename)
                # 尝试解析难度
                try:
                    diff_map = extract_difficulty_from_report(doc)
                    if diff_map:
                        er_code = _MATHS_UNIT_TO_CODE.get(er_unit or '', '')
                        if er_code and er_year:
                            with _report_cache_lock:
                                for q_num, diff in diff_map.items():
                                    _report_difficulty_cache[(er_code, er_year, q_num)] = diff
                            er_registry.append({
                                'filename': file.filename,
                                'unit':     er_unit,
                                'code':     er_code,
                                'year':     er_year,
                                'diff_map': diff_map,
                            })
                            print(f'[upload] ER parsed: {file.filename} unit={er_unit} year={er_year} '
                                  f'Q={sorted(diff_map.keys())} diffs={list(diff_map.values())}')
                except Exception as e:
                    print(f'[upload] ER parse error {file.filename}: {e}')
                doc.close()
                continue  # Report 不加入 qp_groups

            if is_ms:
                # Mark Scheme 类型判断：
                # - edexcel_economics MS：保持 edexcel_economics 类型，使用 econ_unit 匹配
                # - edexcel_maths MS：若文件名含 Edexcel Maths 单元代码，强制使用 edexcel_maths 类型
                ms_pt = pt
                ms_fn_unit = _extract_unit_from_filename(file.filename)

                if source == 'edexcel_economics':
                    # Economics MS：不覆盖为 edexcel_maths
                    ms_pt = 'edexcel_economics'
                    if not econ_unit:
                        econ_unit = detect_edexcel_economics_unit(doc)
                elif ms_fn_unit or source == 'edexcel_maths':
                    ms_pt = 'edexcel_maths'
                    if not maths_unit:
                        maths_unit = ms_fn_unit

                # ── 功能3: Cambridge 9702 物理 MS —— 解析 Question/Answer/Marks 表格 ──
                is_9702_ms = (paper9702.get('type') == 'ms' or
                              (source == 'cambridge' and _is_markscheme_by_content(doc)))
                if is_9702_ms and ms_pt not in ('edexcel_maths', 'edexcel_economics'):
                    ms_answers = _render_9702_ms_answers(doc, dpi=150)
                    if ms_answers:
                        print(f'[upload] 9702 MS parsed: {file.filename} → {sorted(ms_answers.keys())} questions')
                        ms_registry.append({
                            'filename':    file.filename,
                            'file_uuid':   _extract_file_uuid(file.filename),
                            'unit':        maths_unit,
                            'source':      source,
                            'answers':     ms_answers,
                            'session':     _extract_exam_session(file.filename),
                            'paper9702':   paper9702,
                        })
                        doc.close()
                        continue  # MS 不加入 qp_groups

                # 检测题目边界，渲染答案图片（Edexcel Maths / Economics / 通用格式）
                ms_questions = _detect_questions_ms(doc, ms_pt)
                ms_answers   = _render_ms_questions_b64(doc, ms_questions, ms_pt, dpi=150)
                print(f'[upload] MS parsed: {file.filename} type={ms_pt} '
                      f'answers={sorted(ms_answers.keys(), key=str) if ms_answers else []}')
                ms_registry.append({
                    'filename':   file.filename,
                    'file_uuid':  _extract_file_uuid(file.filename),
                    'unit':       maths_unit,
                    'econ_unit':  econ_unit,   # ← 新增：供 econ 匹配使用
                    'source':     source,
                    'answers':    ms_answers,  # {q_num or '12a': {b64, w, h}}
                    'session':    _extract_exam_session(file.filename),
                    'paper9702':  paper9702,
                })
                doc.close()
                continue  # MS 不加入 qp_groups

            # ── 正常题目卷处理 ──
            questions = _detect_questions(doc, pt)
            paper_year      = _extract_year_from_filename(file.filename)
            exam_date_label = _extract_exam_date_label(doc, file.filename)

            # BPhO：用专用年份提取覆盖通用提取
            if source == 'bpho':
                bpho_year = _extract_bpho_year(doc, file.filename)
                if bpho_year:
                    paper_year = bpho_year
                    exam_date_label = bpho_year

            # ── 知识点标注 + 难度评级 ──
            if source == 'cambridge':
                for q_idx, q in enumerate(questions):
                    try:
                        txt = _extract_question_text(doc, questions, q_idx, pt)
                        q['topics'] = tag_question_topics(txt, 'cambridge')
                    except Exception:
                        q['topics'] = []
                    q['difficulty'] = None
            elif source == 'edexcel_economics':
                for q_idx, q in enumerate(questions):
                    try:
                        txt = _extract_question_text(doc, questions, q_idx, pt)
                        q['topics'] = tag_question_topics(txt, 'edexcel_economics',
                                                          unit_filter=econ_unit)
                    except Exception:
                        q['topics'] = []
                    q['difficulty'] = None

                # ── 检测材料页（Sources for use with Section B/C）──
                # 通用策略：扫描文档中是否存在 "Sources for use with Section" 标志页
                # - U1/U2 通常有 "Sources for use with Section C" → 附加到 Q12
                # - U3 通常有 "Sources for use with Section B" → 附加到 Q7
                # - 通过检测标题页的 Section B/C 决定附加目标
                _has_sources_marker = any(
                    ('Sources for use with Section' in doc[_pi].get_text() or
                     'Source for use with Section' in doc[_pi].get_text())
                    for _pi in range(doc.page_count)
                )
                if _has_sources_marker:
                    try:
                        import base64 as _b64sc
                        from PIL import Image as _PILsc
                        _sources_pages = []
                        _in_sources = False
                        _sources_section = None   # 'B' 或 'C'，从标题页检测
                        _sc_mat = fitz.Matrix(150 / 72.0, 150 / 72.0)
                        for _pg_i in range(doc.page_count):
                            _pg_text = doc[_pg_i].get_text()
                            # 检测材料页开始标志，同时记录是 Section B 还是 C
                            if not _in_sources and (
                                    'Sources for use with Section' in _pg_text or
                                    'Source for use with Section' in _pg_text or
                                    'sources for use with section' in _pg_text.lower()):
                                _in_sources = True
                                # 判断 Section B / C
                                _tl = _pg_text.lower()
                                if 'section b' in _tl:
                                    _sources_section = 'B'
                                elif 'section c' in _tl:
                                    _sources_section = 'C'
                                else:
                                    # 默认根据 econ_unit 判断（U3/U3A 用 Section B，其余用 Section C）
                                    _sources_section = 'B' if econ_unit in ('U3', 'U3A') else 'C'
                            # 检测材料页结束（Acknowledgements 或页面数超限）
                            if _in_sources:
                                if 'Acknowledgements' in _pg_text or 'BLANK PAGE' in _pg_text:
                                    break
                                _pg = doc[_pg_i]
                                _pw, _ph = _pg.rect.width, _pg.rect.height
                                # 裁掉 Edexcel 两侧装饰条
                                _clip = fitz.Rect(36, 40, min(_pw - 36, 550), _ph - 25)
                                _pix = _pg.get_pixmap(matrix=_sc_mat, clip=_clip, colorspace=fitz.csRGB)
                                _sc_buf = io.BytesIO()
                                _PILsc.frombytes('RGB', [_pix.width, _pix.height], _pix.samples)\
                                      .save(_sc_buf, format='JPEG', quality=88)
                                _sources_pages.append({
                                    'b64': _b64sc.b64encode(_sc_buf.getvalue()).decode('utf-8'),
                                    'w': _pix.width,
                                    'h': _pix.height,
                                })
                                del _pix
                        # 根据 _sources_section 决定注入 Q7（Section B）还是 Q12（Section C）
                        if _sources_pages:
                            _target_qnum = 7 if _sources_section == 'B' else 12
                            _tq = next((q for q in questions if q.get('q_num') == _target_qnum), None)
                            if _tq is not None:
                                _tq['source_pages'] = _sources_pages
                                print(f'[econ_qp] 检测到 Sources for use with Section {_sources_section} '
                                      f'材料页 {len(_sources_pages)} 页 (unit={econ_unit})，'
                                      f'已附加到 Q{_target_qnum}')
                    except Exception as _sc_err:
                        print(f'[econ_qp] 材料页检测失败: {_sc_err}')
            elif source == 'edexcel_maths':
                unit_code = _MATHS_UNIT_TO_CODE.get(maths_unit or '', None)
                for q_idx, q in enumerate(questions):
                    try:
                        txt = _extract_question_text(doc, questions, q_idx, pt)
                        marks_hint = _extract_marks_hint(txt, unit_filter=maths_unit)
                        q['topics'] = tag_question_topics(txt, 'edexcel_maths',
                                                          unit_filter=maths_unit,
                                                          marks_hint=marks_hint)
                    except Exception:
                        q['topics'] = []
                    q_num = q.get('q_num') or (q_idx + 1)
                    q['difficulty'] = rate_question_difficulty(
                        q_num=q_num, year=paper_year,
                        source=source, maths_unit=maths_unit,
                        unit_code=unit_code
                    )
            elif source == 'bpho':
                # BPhO：按子题文本提取 + BPhO 物理知识点标注
                for q_idx, q in enumerate(questions):
                    try:
                        txt = _extract_question_text(doc, questions, q_idx, pt)
                        q['topics'] = tag_question_topics(txt, 'bpho')
                    except Exception:
                        q['topics'] = []
                    q['difficulty'] = None
                    # 将 q_label（字母）记录到 exam_date 作为副标题辅助信息
                    label = q.get('q_label', '')
                    if label:
                        q['q_label'] = label
            else:
                for q in questions:
                    q['topics'] = []
                    q['difficulty'] = None

            for q in questions:
                q['exam_date'] = exam_date_label

            doc.close()

            qp_groups.append({
                'filename':        file.filename,
                'file_uuid':       _extract_file_uuid(file.filename),
                'path':            tmp_path,
                'r2_key':          r2_key,
                'source':          source,
                'paper_type':      pt,
                'maths_unit':      maths_unit,
                'econ_unit':       econ_unit,
                'exam_date':       exam_date_label,
                'session':         _extract_exam_session(file.filename),
                'questions':       questions,
                'total_questions': len(questions),
                'total_pages':     fitz.open(tmp_path).page_count,
                'has_ms':          False,   # 更新后会设为 True
                'paper9702':       paper9702,  # 9702试卷信息，用于精确匹配MS
            })

        except Exception as e:
            qp_groups.append({
                'filename':        file.filename,
                'path':            tmp_path,
                'r2_key':          r2_key,
                'source':          'unknown',
                'paper_type':      'unknown',
                'maths_unit':      None,
                'questions':       [],
                'total_questions': 0,
                'has_ms':          False,
                'error':           str(e)
            })

    # ── 第二遍：将 MS 答案注入对应 QP 题目 ──
    # 匹配优先级（从高到低）：
    #   ⓪ 9702 文件名精确匹配（session_key = code_session_variant 完全一致）
    #   ① unit + session 精确匹配（如 P3 + 2023_jun）
    #   ② unit 匹配 + 只有一个同 unit QP 未匹配
    #   ③ unit 匹配 + 选得分最多（答案题号覆盖最多）的 QP
    #   ④ source 相同且只有一份 QP（兼容 Cambridge 等）
    #   ⑤ 同 source 中第一个尚未匹配的 QP（最终 fallback）
    unmatched_ms = []  # 记录未匹配的 MS 供前端提示

    def _inject_answers(grp, ms_answers, ms_filename):
        """
        将 ms_answers 注入到 grp 的题目中，返回注入数量。

        处理两种匹配：
        1. 直接 key 匹配：ms_answers 的 key 为整数 q_num，直接注入
        2. 子题聚合匹配：ms_answers 含 '12a'/'12b'... 字符串 key，
           合并后注入到 QP 中 q_num=12 的题目
        """
        import base64 as _b64
        from PIL import Image as _PILImg

        count = 0

        # 预处理：检测 ms_answers 中是否有子题 key（如 '12a', '12b'...）
        # 将子题分组：{q_num: [sub_key_sorted...]}
        sub_keys_by_q = {}  # {12: ['12a','12b','12c','12d','12e']}
        for k in ms_answers:
            if isinstance(k, str):
                m = re.match(r'^(\d+)([a-e])$', k)
                if m:
                    qn = int(m.group(1))
                    sub_keys_by_q.setdefault(qn, []).append(k)
        # 对子题按字母排序
        for qn in sub_keys_by_q:
            sub_keys_by_q[qn].sort()

        # 如果有子题分组，将子题图像转换为多页数组（每子题一页），不再垂直拼接
        # 这样 Q12 的答案在 UI 显示时每子题独立一页，PDF 导出时每子题一页纸
        # 与 Q13/Q14 多页逻辑一致（_insert_answer_pages 迭代 pages 数组）
        merged_answers = dict(ms_answers)
        for qn, sub_keys in sub_keys_by_q.items():
            pages_list = []
            first_b64, first_w, first_h = None, 0, 0
            for sk in sub_keys:
                ans = ms_answers.get(sk)
                if not ans:
                    continue
                # 检查该子题本身是否已有多页（如 12e 有3页）
                sub_pages = ans.get('pages')
                if sub_pages and isinstance(sub_pages, list) and len(sub_pages) > 1:
                    # 子题本身多页：逐页加入
                    for pg in sub_pages:
                        pages_list.append({'b64': pg['b64'], 'w': pg['w'], 'h': pg['h']})
                else:
                    # 子题单页：直接加入
                    pages_list.append({'b64': ans['b64'], 'w': ans['w'], 'h': ans['h']})
                if first_b64 is None:
                    first_b64, first_w, first_h = ans['b64'], ans['w'], ans['h']
            if not pages_list:
                continue
            if len(pages_list) == 1:
                # 只有一个子题且单页：保持单页格式（不用 pages 数组）
                merged_answers[qn] = {
                    'b64':   pages_list[0]['b64'],
                    'w':     pages_list[0]['w'],
                    'h':     pages_list[0]['h'],
                    'pages': None,
                }
            else:
                # 多个子题或多页：保存为 pages 数组，每子题一页
                merged_answers[qn] = {
                    'b64':   first_b64,    # 向后兼容：b64 取第一页
                    'w':     first_w,
                    'h':     first_h,
                    'pages': pages_list,   # 多页数组，前端/PDF 逐页显示
                }
            print(f'[inject_answers] Q{qn} sub_keys={sub_keys} → pages={len(pages_list)}')


        # 注入
        for q in grp['questions']:
            q_num = q.get('q_num')
            sub   = q.get('sub_label', '')
            # 优先：子题 key（如 '12a'）
            if sub:
                sub_key = f"{q_num}{sub[1]}"  # '(a)' → '12a'
                if sub_key in merged_answers:
                    ans = merged_answers[sub_key]
                    q['answer_b64']   = ans['b64']
                    q['answer_w']     = ans['w']
                    q['answer_h']     = ans['h']
                    q['answer_pages'] = ans.get('pages')  # 多页数组（可为 None）
                    count += 1
                    continue
            # 整数 key 直接匹配
            if q_num in merged_answers:
                ans = merged_answers[q_num]
                q['answer_b64']   = ans['b64']
                q['answer_w']     = ans['w']
                q['answer_h']     = ans['h']
                q['answer_pages'] = ans.get('pages')  # 多页数组（可为 None）
                count += 1

        grp['has_ms']  = True
        grp['ms_file'] = ms_filename
        return count

    def _score_match(grp, ms_answers):
        """计算 MS 与 QP 的题号覆盖得分（交集题数）"""
        qp_nums = {q.get('q_num') for q in grp['questions']}
        # ms_answers key 可能是整数或字符串（如 '12a'）
        ms_nums = set()
        for k in ms_answers:
            if isinstance(k, int):
                ms_nums.add(k)
            elif isinstance(k, str):
                m = re.match(r'^(\d+)', k)
                if m:
                    ms_nums.add(int(m.group(1)))
        return len(qp_nums & ms_nums)

    for ms_info in ms_registry:
        ms_unit      = ms_info['unit']
        ms_econ_unit = ms_info.get('econ_unit')    # ← 新增：经济学 unit
        ms_answers   = ms_info['answers']
        ms_source    = ms_info['source']
        ms_session   = ms_info.get('session')   # 如 '2023_jun' 或 '2023'
        ms_p9702     = ms_info.get('paper9702', {})
        matched      = False

        # ── 优先级⓪U：UUID 前缀精确匹配（最可靠，同 UUID = 同批次上传的配对文件）──
        # 文件名格式：{uuid}_{原始文件名}.pdf，同一对 QP+MS 共享相同 UUID 前缀
        ms_uuid = ms_info.get('file_uuid')
        if not matched and ms_uuid:
            for grp in qp_groups:
                if grp.get('file_uuid') == ms_uuid and not grp.get('has_ms'):
                    cnt = _inject_answers(grp, ms_answers, ms_info['filename'])
                    print(f'[MS match⓪U] UUID: {ms_info["filename"]} → {grp["filename"]} '
                          f'(uuid={ms_uuid}, answers={cnt})')
                    matched = True
                    break

        # ── 优先级⓪：9702 文件名精确匹配（session_key 完全一致）──
        # 例：9702_m20_ms_42 精确匹配 9702_m20_qp_42
        if ms_p9702.get('session_key'):
            ms_sk = ms_p9702['session_key']
            for grp in qp_groups:
                grp_p9702 = grp.get('paper9702', {})
                if (grp_p9702.get('session_key') == ms_sk and
                        grp_p9702.get('type') == 'qp'):
                    cnt = _inject_answers(grp, ms_answers, ms_info['filename'])
                    print(f'[MS match⓪] 9702 filename: {ms_info["filename"]} → {grp["filename"]} '
                          f'(session_key={ms_sk}, answers={cnt})')
                    matched = True
                    break

        # ── 优先级①E：Edexcel Economics econ_unit + session 完全匹配 ──
        if not matched and ms_econ_unit and ms_session:
            for grp in qp_groups:
                if (grp.get('econ_unit') == ms_econ_unit and
                        grp.get('session') == ms_session and
                        grp.get('source') == 'edexcel_economics'):
                    cnt = _inject_answers(grp, ms_answers, ms_info['filename'])
                    print(f'[MS match①E] {ms_info["filename"]} → {grp["filename"]} '
                          f'(econ_unit={ms_econ_unit}, session={ms_session}, answers={cnt})')
                    matched = True
                    break

        # ── 优先级②E：Edexcel Economics econ_unit 匹配，选得分最高 ──
        if not matched and ms_econ_unit:
            candidates = [g for g in qp_groups
                          if g.get('econ_unit') == ms_econ_unit
                          and g.get('source') == 'edexcel_economics'
                          and not g.get('has_ms')]
            if len(candidates) == 1:
                grp = candidates[0]
                cnt = _inject_answers(grp, ms_answers, ms_info['filename'])
                print(f'[MS match②E-single] {ms_info["filename"]} → {grp["filename"]} '
                      f'(econ_unit={ms_econ_unit}, answers={cnt})')
                matched = True
            elif len(candidates) > 1:
                best = max(candidates, key=lambda g: _score_match(g, ms_answers))
                best_score = _score_match(best, ms_answers)
                if best_score >= 0:   # econ MS 子题 key 可能是字符串，score 可能为 0
                    cnt = _inject_answers(best, ms_answers, ms_info['filename'])
                    print(f'[MS match②E-best] {ms_info["filename"]} → {best["filename"]} '
                          f'(econ_unit={ms_econ_unit}, score={best_score}, answers={cnt})')
                    matched = True

        # ── 优先级①：unit + session 完全匹配 ──
        if not matched and ms_unit and ms_session:
            for grp in qp_groups:
                if grp.get('maths_unit') == ms_unit and grp.get('session') == ms_session:
                    cnt = _inject_answers(grp, ms_answers, ms_info['filename'])
                    print(f'[MS match①] {ms_info["filename"]} → {grp["filename"]} '
                          f'(unit={ms_unit}, session={ms_session}, answers={cnt})')
                    matched = True
                    break

        # ── 优先级②：unit 匹配，选覆盖得分最高且尚未匹配的 QP ──
        if not matched and ms_unit:
            candidates = [g for g in qp_groups
                          if g.get('maths_unit') == ms_unit and not g.get('has_ms')]
            if len(candidates) == 1:
                grp = candidates[0]
                cnt = _inject_answers(grp, ms_answers, ms_info['filename'])
                print(f'[MS match②-single] {ms_info["filename"]} → {grp["filename"]} '
                      f'(unit={ms_unit}, answers={cnt})')
                matched = True
            elif len(candidates) > 1:
                # 选覆盖得分最高的
                best = max(candidates, key=lambda g: _score_match(g, ms_answers))
                best_score = _score_match(best, ms_answers)
                if best_score > 0:
                    cnt = _inject_answers(best, ms_answers, ms_info['filename'])
                    print(f'[MS match②-best] {ms_info["filename"]} → {best["filename"]} '
                          f'(unit={ms_unit}, score={best_score}, answers={cnt})')
                    matched = True

        # ── 优先级③：Cambridge 等非数学：source 相同且只有一份 QP ──
        if not matched and not ms_unit and not ms_econ_unit:
            same_src = [g for g in qp_groups if g['source'] == ms_source]
            if len(same_src) == 1:
                grp = same_src[0]
                cnt = _inject_answers(grp, ms_answers, ms_info['filename'])
                print(f'[MS match③] {ms_info["filename"]} → {grp["filename"]} '
                      f'(source={ms_source}, answers={cnt})')
                matched = True

        # ── 优先级④：覆盖得分最高的同 source 未匹配 QP（最终 fallback）──
        if not matched:
            fallback_cands = [g for g in qp_groups
                              if g['source'] in (ms_source, 'edexcel_maths', 'edexcel')
                              and not g.get('has_ms')]
            if not fallback_cands:
                # 若全部都已有 MS，也允许覆盖（例如重复上传 MS 文件时）
                fallback_cands = [g for g in qp_groups
                                  if g['source'] in (ms_source, 'edexcel_maths', 'edexcel')]
            if fallback_cands:
                best = max(fallback_cands, key=lambda g: _score_match(g, ms_answers))
                cnt = _inject_answers(best, ms_answers, ms_info['filename'])
                print(f'[MS match④-fallback] {ms_info["filename"]} → {best["filename"]} '
                      f'(answers={cnt})')
                matched = True

        if not matched:
            print(f'[MS unmatched] {ms_info["filename"]} (unit={ms_unit}, session={ms_session})')
            unmatched_ms.append(ms_info['filename'])

    groups = qp_groups

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
        'unmatched_ms': unmatched_ms,   # 未成功匹配任何 QP 的 MS 文件名列表
        'groups': [{
            'filename':        g['filename'],
            'source':          g['source'],
            'paper_type':      g['paper_type'],
            'maths_unit':      g.get('maths_unit'),
            'econ_unit':       g.get('econ_unit'),
            'exam_date':       g.get('exam_date', ''),
            'session':         g.get('session', ''),
            'has_ms':          g.get('has_ms', False),
            'ms_file':         g.get('ms_file', ''),
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


def detect_edexcel_economics_ms_questions(doc, econ_unit=None):
    """
    检测 Edexcel Economics Mark Scheme 中的题目边界。

    MS 文件结构：
      Section A (pg 3-4): 表格式，Q1-Q6，按 Answer 列块 y1 分隔
      U1/U2 Section B (pg 5-9): Q7-Q11，每页独立，顶部有 Question 行
      U3 Section B: 只有 Q7，但 Q7 包含子题 7(a)/7(b)/7(c)/7(d)/7(e)
      Section C (pg 10+): Q12(a)-Q12(e)，子题格式
      Section D: Q13-Q14，大作文格式

    参数：
      econ_unit: 'U1'|'U2'|'U3'|'U4'|None，若为 None 则从 doc 自动检测

    返回列表：[{q_num, sub_label, page_idx, y_start, y_end, pages, section}]
      - q_num: 整数题号（1-14）
      - sub_label: '' 或 '(a)'/'(b)'/'(c)'/'(d)'/'(e)'
      - page_idx: 题目起始页（0-based）
      - y_start: 起始 y 坐标
      - y_end: 结束 y 坐标（None 表示到页末）
      - pages: [(page_idx, y_start, y_end)] 多页列表（用于 Q12e/Q7e 等）
    """
    # 若未提供 econ_unit，自动从文档中检测
    if econ_unit is None:
        econ_unit = detect_edexcel_economics_unit(doc)
    questions = []

    # ── Section A: 表格式，Q1-Q6 ──
    # 识别策略（兼容 U1-U4，尤其解决 U2 左列合并单元格问题）：
    #
    # U2 PDF 特殊性：左列（x0≈62）的 PDF 块会把多行合并，导致 Q2-Q5 的题号
    # 不出现在任何块的第一行——只有 Q1("1 ") 和 Q6("6 ") 是纯数字块。
    #
    # 核心策略：优先使用右列 "The only correct answer is X" 块的 y0 作为行锚点。
    # 这些块在每一题对应一个，y0 精确对应该题行的视觉顶部。
    # 按文档顺序 (pg_i, y0) 排序后，第 i 个块就是第 Q(i+1) 题的锚点。
    #
    # 行范围：
    #   y_start = 该题锚点的 y0（用左列第一块的 y0 取 min 以覆盖更完整的行顶）
    #   y_end   = 下一题锚点的 y0 - 2（同页），或页底（末题）
    section_a_pages = []
    for pg_i in range(doc.page_count):
        text = doc[pg_i].get_text()
        if 'Section A' in text and pg_i >= 2:
            # 从 Section A 开始页往后扫，直到遇到 Section B 或超出
            # 重要：Q6 可能在「含 Section B」的同一页面（Section B 的起始同页包含Q6尾部）
            section_a_pages.append(pg_i)
            for ext_pg in range(pg_i + 1, min(pg_i + 5, doc.page_count)):
                ext_text = doc[ext_pg].get_text()
                # 如果这页包含 Section B，也把它加进来（Q6 可能在这里），然后停止
                if 'Section B' in ext_text:
                    section_a_pages.append(ext_pg)
                    break
                # 普通延续页（还在 Section A 范围内）
                section_a_pages.append(ext_pg)
            break

    _section_a_table = {}   # q_num -> (page_idx, y_row_start, y_row_end)

    for pg_i in section_a_pages:
        page = doc[pg_i]
        ph = page.rect.height
        pw = page.rect.width
        blocks = page.get_text('blocks')

        # 收集所有文本块，按 y0 排序
        text_blocks = []
        for b in blocks:
            x0, y0, x1, y1, txt, bno, btype = b
            if btype != 0:
                continue
            text_blocks.append((x0, y0, x1, y1, txt.strip()))
        text_blocks.sort(key=lambda b: b[1])

        # ── 右列 "correct answer" 锚点（核心策略）──
        # 收集右半页（x0 > page_width/3）含 "correct answer" / "only correct" 的块
        right_anchor_ys = []  # 每个元素：(y0, y1)
        for x0, y0, x1, y1, ts in text_blocks:
            if x0 > pw / 3.0:
                tsl = ts.lower()
                if 'only correct answer' in tsl or 'correct answer is' in tsl:
                    right_anchor_ys.append((y0, y1))

        # ── 左列辅助：找最左侧明确的纯数字题号块（用于调整行顶 y）──
        left_xs = sorted(set(round(b[0]) for b in text_blocks if b[0] < 150))
        q_col_thresh = (min(left_xs) + 50) if left_xs else 110

        # ── 若本页同时包含 Section B，找到 Section B 的起始 y（作为 Q6 行底上限）──
        # Section B 开头特征：独立的 'Section B' 标题块，或 'Question' 表头行（Section B question header）
        # ⚠️  注意：pg5 中 Q6 的 "C is not correct..." 块末尾会混入 "Section B\n" 文字
        #    不能用 'Section B' in ts 检测——要求是独立标题（ts 去除空白后就是 'Section B'）
        #    或者检测 "Question\n..." 这种 Section B 题目 header 块（x0<100, y0 在 Section A 表格区之后）
        page_text_full = page.get_text()
        section_b_y_limit = ph  # 默认：无 Section B → 用页底
        if 'Section B' in page_text_full:
            # 策略1：找文本去除空白后正好是 'Section B' 的独立标题块
            for x0, y0, x1, y1, ts in text_blocks:
                if re.match(r'^Section\s+B\s*$', ts):
                    section_b_y_limit = y0 - 2
                    break
            if section_b_y_limit == ph:
                # 策略2：找 "Question\n..." 块（Section B 每道题开头的题目 header）
                # 特征：块以 'Question' 开头，y0 > Section A 表格底部（一般 > 200），x0 < 100
                last_anchor_y1 = right_anchor_ys[-1][1] if right_anchor_ys else 200
                for x0, y0, x1, y1, ts in text_blocks:
                    if ts.startswith('Question') and y0 > last_anchor_y1 and x0 < 100:
                        section_b_y_limit = y0 - 2
                        break
            if section_b_y_limit == ph:
                # 策略3：找左列 '7 ' 开头的块（Q7 题号，Section B 第一题）
                last_anchor_y1 = right_anchor_ys[-1][1] if right_anchor_ys else 200
                for x0, y0, x1, y1, ts in text_blocks:
                    first = ts.split('\n')[0].strip()
                    if x0 < 100 and re.match(r'^7\s', first) and y0 > last_anchor_y1:
                        section_b_y_limit = y0 - 2
                        break
            if section_b_y_limit == ph:
                # 策略4（兜底）：找任何含 'Section B' 的块，取其 y0 作为上限
                # 适用于 U4 等格式：'Section B' 嵌在 Q6 的 (1) 标记块里
                # （如 '(1)\n\n\nSection B\n'），此时取该块的 y0 即可
                for x0, y0, x1, y1, ts in text_blocks:
                    if 'Section B' in ts:
                        section_b_y_limit = y0 - 2
                        break

        # 先用右列锚点策略构建本页 questions
        if right_anchor_ys:
            # 按 y0 排序（同页内应该已经有序）
            right_anchor_ys.sort(key=lambda a: a[0])
            n_anchors = len(right_anchor_ys)

            # 已找到的最大 q_num（跨页累计，用于判断本页从哪个题号继续）
            q_start_on_page = len(_section_a_table) + 1  # 本页第一题编号

            for ai, (ry0, ry1) in enumerate(right_anchor_ys):
                q_num = q_start_on_page + ai
                if q_num > 6:
                    break  # Section A 只有 Q1-Q6

                # 行顶：取右列锚点 y0，若左列有更早（y 更小）的块则用左列
                row_top = ry0
                for x0, y0, x1, y1, ts in text_blocks:
                    if x0 <= q_col_thresh and y0 < ry0 and ry0 - y0 < 30:
                        # 左列有稍早的块，说明行顶在此
                        row_top = min(row_top, y0)
                        break

                # ── 行底计算（修复：用行内最大y1+margin，而非下一题y0-2）──
                # 原逻辑 next_anchor_y0 - 2 会截断底部框线；
                # 新逻辑：取本行范围内（ry0 ~ next_row_y0）所有块的最大 y1，
                # 加上 6pt 边距（包含框线），但不超过 section_b_y_limit
                next_row_y0 = right_anchor_ys[ai + 1][0] if ai + 1 < n_anchors else section_b_y_limit
                # 收集本行内所有块的 y1（限制在 next_row_y0 以内，防止跨行大块污染）
                row_max_y1 = ry1  # 至少是右列锚点块自身的 y1
                for x0, y0, x1, y1, ts in text_blocks:
                    if ry0 - 5 <= y0 < next_row_y0:
                        # 用 min(y1, next_row_y0) 防止跨行块（如左列大块）溢出
                        row_max_y1 = max(row_max_y1, min(y1, next_row_y0))
                # 行底 = 行内最大y1 + 6pt，但不超过 section_b_y_limit
                row_bottom = min(row_max_y1 + 6, section_b_y_limit)

                if q_num not in _section_a_table:
                    _section_a_table[q_num] = (pg_i, row_top, row_bottom)

            print(f'[econ_ms] Section A pg{pg_i} (right-anchor strategy): '
                  f'anchors={n_anchors} q_range=Q{q_start_on_page}-Q{q_start_on_page+n_anchors-1} '
                  f'table_keys={list(_section_a_table.keys())}')

        else:
            # ── 兜底策略（U1 等左列纯数字清晰的情况）──
            q_blocks = []    # (q_num, y0)
            ans_blocks = []  # (y0, y1)

            for x0, y0, x1, y1, ts in text_blocks:
                first_line = ts.split('\n')[0].strip()
                if x0 <= q_col_thresh:
                    m = re.match(r'^(\d+)\s*$', first_line)
                    if m:
                        q_num = int(m.group(1))
                        if 1 <= q_num <= 6:
                            q_blocks.append((q_num, y0))
                    elif 'Question' in ts and re.search(r'(?:^|\n)\s*1\s*(?:\n|$)', ts):
                        if not any(q[0] == 1 for q in q_blocks):
                            q_blocks.insert(0, (1, y0))

                if ('correct answer' in ts.lower() or 'only correct' in ts.lower()):
                    ans_blocks.append((y0, y1))

            if not any(q[0] == 1 for q in q_blocks):
                for x0, y0, x1, y1, ts in text_blocks:
                    if x0 <= q_col_thresh and 40 < y0 < 250:
                        first = ts.split('\n')[0].strip()
                        if not re.match(r'(?i)^section|^question\s+number', first):
                            q_blocks.insert(0, (1, y0))
                            break

            q_blocks.sort(key=lambda x: x[1])

            for qi, (q_num, q_y0) in enumerate(q_blocks):
                row_bottom = ph
                for a_y0, a_y1 in sorted(ans_blocks):
                    if a_y0 >= q_y0 - 5:
                        row_bottom = a_y1
                        break
                if row_bottom == ph and qi + 1 < len(q_blocks):
                    row_bottom = q_blocks[qi + 1][1] - 2
                if q_num not in _section_a_table:
                    _section_a_table[q_num] = (pg_i, q_y0, row_bottom)

            print(f'[econ_ms] Section A pg{pg_i} (fallback strategy): '
                  f'q_blocks={[x[0] for x in q_blocks]} ans_blocks={len(ans_blocks)} '
                  f'table_keys={list(_section_a_table.keys())}')

    for q_num in sorted(_section_a_table.keys()):
        pg_i, y_start, y_end = _section_a_table[q_num]
        questions.append({
            'q_num':     q_num,
            'sub_label': '',
            'page_idx':  pg_i,
            'y_start':   y_start,
            'y_end':     y_end,
            'pages':     [(pg_i, y_start, y_end)],
            'section':   'A',
        })

    # ── Section B/C/D ──
    Q_NUM_PAT = re.compile(r'^(\d+)\s*$')
    Q12_SUB   = re.compile(r'^12\s*\(([a-e])\)', re.IGNORECASE)
    # U3/U4 专用：Q7 子题模式  "7(a)" / "7 (a)" / "7(a)" 等
    Q7_SUB    = re.compile(r'^7\s*\(([a-e])\)', re.IGNORECASE)

    # U3/U4 标志：Section B 只有 Q7 且含子题；Section C Q8-Q10 使用嵌入格式
    # U4A 与 U4 结构相同；U3A 与 U3 结构相同
    is_u3 = (econ_unit in ('U3', 'U3A'))
    is_u4 = (econ_unit in ('U4', 'U4A'))
    is_u3_or_u4 = is_u3 or is_u4
    # U4/U4A Q7 子题上限：只有 7(a)-7(d)（无 7(e)）；U3/U3A 有 7(a)-7(e)
    u4_q7_max_sub = 'd' if is_u4 else 'e'

    seen_q      = set(q['q_num'] for q in questions)
    seen_12subs = set()
    seen_7subs  = set()   # U3/U4 专用：已检测到的 Q7 子题
    q12e_pages  = []
    q12e_started = False
    q7e_pages   = []      # U3 Q7(e) 多页（U4 无 Q7(e)）
    q7e_started = False
    # U3/U4: 用于跨块检测 Q9 等（前一个块含 'Indicative content'）
    _prev_had_indicative = False

    for pg_i in range(doc.page_count):
        page = doc[pg_i]
        ph = page.rect.height
        pw = page.rect.width
        page_text = page.get_text()

        # 跳过 Section A 页
        if 'Section A' in page_text and pg_i < 6:
            continue

        # Q12(e) 多页追加（在 Section D 出现前）
        if q12e_started:
            if 'Section D' in page_text or pg_i >= 17:
                # Q12e 结束
                q12e_started = False
                questions.append({
                    'q_num':     12,
                    'sub_label': '(e)',
                    'page_idx':  q12e_pages[0][0],
                    'y_start':   q12e_pages[0][1],
                    'y_end':     None,
                    'pages':     list(q12e_pages),
                    'section':   'C',
                })
            else:
                if pg_i not in [p[0] for p in q12e_pages]:
                    q12e_pages.append((pg_i, 0, None))

        # U3: Q7(e) 多页追加（7(e) 有 14 分，通常跨多页，直到 Section C 或 Q12 出现）
        if q7e_started:
            pg_text_lower = page_text
            if ('Section C' in pg_text_lower or 'Section D' in pg_text_lower or
                    Q12_SUB.search(page_text)):
                # Q7e 结束
                q7e_started = False
                questions.append({
                    'q_num':     7,
                    'sub_label': '(e)',
                    'page_idx':  q7e_pages[0][0],
                    'y_start':   q7e_pages[0][1],
                    'y_end':     None,
                    'pages':     list(q7e_pages),
                    'section':   'B',
                })
            else:
                if pg_i not in [p[0] for p in q7e_pages]:
                    q7e_pages.append((pg_i, 0, None))

        blocks = page.get_text('blocks')
        _prev_had_indicative = False  # 重置跨块状态（每页重置）
        for b in blocks:
            x0, y0, x1, y1, txt, bno, btype = b
            if btype != 0:
                _prev_had_indicative = False
                continue
            ts = txt.strip()
            first_line = ts.split('\n')[0].strip()

            # ── U3/U4: Q7 子题检测（7(a)/7(b)/7(c)/7(d)/7(e)）──
            if is_u3_or_u4:
                m7 = Q7_SUB.match(first_line)
                if m7:
                    sub7 = m7.group(1).lower()
                    key7 = f'7{sub7}'
                    if key7 not in seen_7subs:
                        seen_7subs.add(key7)
                        if sub7 == 'e' and is_u3:
                            # Q7(e) 多页（U3 专有，14分），处理类似 Q12(e)
                            if not q7e_started:
                                q7e_started = True
                                q7e_pages = [(pg_i, y0, None)]
                        else:
                            questions.append({
                                'q_num':     7,
                                'sub_label': f'({sub7})',
                                'page_idx':  pg_i,
                                'y_start':   y0,
                                'y_end':     None,
                                'pages':     [(pg_i, y0, None)],
                                'section':   'B',
                            })
                    _prev_had_indicative = 'Indicative content' in ts
                    continue  # 匹配到 7(x) 子题块 → 跳过下面整题检测
                # m7 未匹配（即非 7(x) 格式）→ 继续走整题检测（Q8/Q9/Q10/Q11）

                # ── U3/U4 Section C: Q8/Q9/Q10 格式特殊 ──
                # 情况1：题号嵌在含 Indicative content 的块里
                #   1a（U3/U4 Q8,Q10）：同一块含 "Question" + "Indicative content\n{n}\n"
                #   1b（U4 Q8 分离）：块含 "Indicative content\n\n{n}\n"（无 Question）
                # 情况2（U4 Q9）：题号在独立小块 "\n{n}\nIndicative content guidance\n"
                #                 而前一个块含 'Indicative content'
                _emb_q_num = None
                if 'Indicative content' in ts:
                    # 情况1：块内嵌入数字，正则匹配 \nIndicative content\n(空行)\n{num}\n
                    _emb = re.search(r'\nIndicative content\s*\n\s*(\d+)\s*\n', ts)
                    if _emb:
                        _emb_q_num = int(_emb.group(1))
                if _emb_q_num is None and _prev_had_indicative and re.match(r'^\s*(\d+)\s*\n', ts):
                    # 情况2：当前块以独立数字行开头，且上一个块含 Indicative content
                    _m_num = re.match(r'^\s*(\d+)\s*\n', ts)
                    if _m_num:
                        _emb_q_num = int(_m_num.group(1))

                if _emb_q_num is not None and 8 <= _emb_q_num <= 11 and _emb_q_num not in seen_q:
                    questions.append({
                        'q_num':     _emb_q_num,
                        'sub_label': '',
                        'page_idx':  pg_i,
                        'y_start':   y0,
                        'y_end':     None,
                        'pages':     [(pg_i, y0, None)],
                        'section':   'C',   # U3/U4 Section C
                    })
                    seen_q.add(_emb_q_num)
                    _prev_had_indicative = 'Indicative content' in ts
                    continue

            # 更新跨块状态
            _prev_had_indicative = 'Indicative content' in ts

            # Q7-Q11: 纯数字块 x0<80（U1/U2）
            # U3/U4: Q7 整体块不检测（由 Q7_SUB 子题代替），Q8-Q11由上方嵌入块逻辑检测
            m = Q_NUM_PAT.match(first_line)
            if m and x0 < 80:
                q_num = int(m.group(1))
                # U3/U4 中跳过 q_num==7 整体（由 Q7 子题逻辑处理）
                skip_q7_whole = is_u3_or_u4 and q_num == 7
                if not skip_q7_whole and 7 <= q_num <= 11 and q_num not in seen_q:
                    questions.append({
                        'q_num':     q_num,
                        'sub_label': '',
                        'page_idx':  pg_i,
                        'y_start':   y0,
                        'y_end':     None,
                        'pages':     [(pg_i, y0, None)],
                        'section':   'B',
                    })
                    seen_q.add(q_num)
                elif q_num in (13, 14) and q_num not in seen_q:
                    questions.append({
                        'q_num':     q_num,
                        'sub_label': '',
                        'page_idx':  pg_i,
                        'y_start':   y0,
                        'y_end':     None,
                        'pages':     [(pg_i, y0, None)],
                        'section':   'D',
                    })
                    seen_q.add(q_num)

            # Q12 子题
            m12 = Q12_SUB.match(first_line)
            if m12:
                sub = m12.group(1).lower()
                key = f'12{sub}'
                if key not in seen_12subs:
                    seen_12subs.add(key)
                    if sub == 'e':
                        if not q12e_started:
                            q12e_started = True
                            q12e_pages = [(pg_i, y0, None)]
                    else:
                        questions.append({
                            'q_num':     12,
                            'sub_label': f'({sub})',
                            'page_idx':  pg_i,
                            'y_start':   y0,
                            'y_end':     None,
                            'pages':     [(pg_i, y0, None)],
                            'section':   'C',
                        })

    # Q12e 若未关闭
    if q12e_started and q12e_pages:
        questions.append({
            'q_num':     12,
            'sub_label': '(e)',
            'page_idx':  q12e_pages[0][0],
            'y_start':   q12e_pages[0][1],
            'y_end':     None,
            'pages':     list(q12e_pages),
            'section':   'C',
        })

    # U3: Q7e 若未关闭（扫描结束时未遇到 Section C/D）
    if q7e_started and q7e_pages:
        questions.append({
            'q_num':     7,
            'sub_label': '(e)',
            'page_idx':  q7e_pages[0][0],
            'y_start':   q7e_pages[0][1],
            'y_end':     None,
            'pages':     list(q7e_pages),
            'section':   'B',
        })

    # U3 兜底：若 Q7(d) 已找到但 Q7(e) 未找到，从 Q7(d) 所在页+1 向后扫到 Section C/D 前
    if is_u3 and '7e' not in seen_7subs:
        d_page_7 = None
        for q in questions:
            if q['q_num'] == 7 and q.get('sub_label') == '(d)':
                d_page_7 = q['page_idx']
                break
        if d_page_7 is not None:
            e7_start = d_page_7 + 1
            q7e_pages_fb = []
            for ext_pg in range(e7_start, doc.page_count):
                ext_t = doc[ext_pg].get_text()
                if ('Section C' in ext_t or 'Section D' in ext_t or
                        re.search(r'^12\s*\([a-e]\)', ext_t, re.MULTILINE)):
                    break
                y_s = 63 if ext_pg == e7_start else 0
                q7e_pages_fb.append((ext_pg, y_s, None))
            if q7e_pages_fb:
                questions.append({
                    'q_num':     7,
                    'sub_label': '(e)',
                    'page_idx':  q7e_pages_fb[0][0],
                    'y_start':   q7e_pages_fb[0][1],
                    'y_end':     None,
                    'pages':     q7e_pages_fb,
                    'section':   'B',
                })

    # Q12(e) 兜底：若 seen_12subs 仍无 '12e'，通过 Q12(d) 推断
    if '12e' not in seen_12subs:
        # Q12(d) 所在页 + 1 到 Section D 前
        d_page = None
        for q in questions:
            if q['q_num'] == 12 and q.get('sub_label') == '(d)':
                d_page = q['page_idx']
                break
        if d_page is not None:
            e_start = d_page + 1
            q12e_pages_fb = []
            for ext_pg in range(e_start, 17):
                if ext_pg >= doc.page_count:
                    break
                if 'Section D' not in doc[ext_pg].get_text():
                    y_s = 63 if ext_pg == e_start else 0
                    q12e_pages_fb.append((ext_pg, y_s, None))
            if q12e_pages_fb:
                questions.append({
                    'q_num':     12,
                    'sub_label': '(e)',
                    'page_idx':  q12e_pages_fb[0][0],
                    'y_start':   q12e_pages_fb[0][1],
                    'y_end':     None,
                    'pages':     q12e_pages_fb,
                    'section':   'C',
                })

    # Q13 兜底：若仍未找到，扫含 Section D 的页
    if 13 not in seen_q:
        for pg_i in range(doc.page_count):
            page = doc[pg_i]
            page_text = page.get_text()
            if 'Section D' in page_text:
                blocks = page.get_text('blocks')
                for b in blocks:
                    x0, y0, x1, y1, txt, bno, btype = b
                    if btype != 0:
                        continue
                    ts = txt.strip()
                    if re.search(r'(?:^|\n)13\s*\n', ts) and x0 < 80:
                        questions.append({
                            'q_num':     13,
                            'sub_label': '',
                            'page_idx':  pg_i,
                            'y_start':   y0,
                            'y_end':     None,
                            'pages':     [(pg_i, y0, None)],
                            'section':   'D',
                        })
                        seen_q.add(13)
                        break
                if 13 in seen_q:
                    break

    def _sort_key(q):
        sub_ord = {'': 0, '(a)': 1, '(b)': 2, '(c)': 3, '(d)': 4, '(e)': 5}
        return (q['q_num'], sub_ord.get(q.get('sub_label', ''), 99))

    questions.sort(key=_sort_key)
    return questions


def _render_edexcel_economics_ms_answers(ms_doc, dpi=150):
    """
    渲染 Edexcel Economics Mark Scheme 所有题目答案为 JPEG base64 字典。

    返回：{key: {'b64': str, 'w': int, 'h': int}}
      - Section A (Q1-Q6): key = 整数 1-6
      - Section B (Q7-Q11): key = 整数 7-11（U1/U2）
      - U3 Section B Q7子题: key = 字符串 '7a'/'7b'/'7c'/'7d'/'7e'
      - Section C Q12子题: key = 字符串 '12a'/'12b'/'12c'/'12d'/'12e'
      - Section D (Q13-Q14): key = 整数 13-14

    切割策略：
      Section A: 按表格横线行切，全宽渲染
      Section B/C/D 单页题: 从 y_start 到页末
      Q12(e)/Q7(e) 多页: 每页单独存储
      Q13/Q14 多页: 从起始页 y_start 到文档末，多页拼接
    """
    import base64 as _b64
    from PIL import Image as _PILImg

    # 从文档检测 econ_unit，传给题目检测函数
    ms_econ_unit = detect_edexcel_economics_unit(ms_doc)
    questions = detect_edexcel_economics_ms_questions(ms_doc, econ_unit=ms_econ_unit)
    if not questions:
        return {}

    # 收集 Q13/Q14 的多页范围
    # Q13: 从 page_idx 到 Q14 开始页前一页（或到文档最后正文页）
    # Q14: 从 page_idx 到文档最后正文页
    q13_info = next((q for q in questions if q['q_num'] == 13), None)
    q14_info = next((q for q in questions if q['q_num'] == 14), None)

    # 从文档末尾向前扫，跳过版权页（含 "Pearson Education"）和几乎空白页（< 80 字符）
    # 找到最后一页真正的学术内容页
    last_content_page = ms_doc.page_count - 1  # 先假设最后一页是正文
    for _pg_back in range(ms_doc.page_count - 1, -1, -1):
        _back_txt = ms_doc[_pg_back].get_text().strip()
        # 版权页：包含 "Pearson Education" 且正文内容很短
        # 空白页：几乎没有文字
        if len(_back_txt) < 80 or 'Pearson Education' in _back_txt:
            last_content_page = _pg_back - 1
        else:
            break  # 找到真正的内容页，停止向前扫
    last_content_page = max(last_content_page, 0)
    print(f'[econ_ms] last_content_page={last_content_page} (total={ms_doc.page_count})')

    if q13_info:
        q13_start_pg = q13_info['page_idx']
        q14_start_pg = q14_info['page_idx'] if q14_info else last_content_page + 1
        q13_pages = [(pg_i, (q13_info['y_start'] if pg_i == q13_start_pg else 0), None)
                     for pg_i in range(q13_start_pg, q14_start_pg)]
        q13_info = dict(q13_info)
        q13_info['pages'] = q13_pages

    if q14_info:
        q14_start_pg = q14_info['page_idx']
        q14_pages = [(pg_i, (q14_info['y_start'] if pg_i == q14_start_pg else 0), None)
                     for pg_i in range(q14_start_pg, last_content_page + 1)]
        q14_info = dict(q14_info)
        q14_info['pages'] = q14_pages

    # ── U3/U4 Section C 多页扩展：Q8/Q9/Q10 每题跨3页 ──
    # 策略：按题号排序，Q8的范围=Q8起始页到Q9起始页-1，以此类推
    # 最后一题(Q10)到 last_content_page
    if ms_econ_unit in ('U3', 'U3A', 'U4', 'U4A'):
        _sec_c_qs = sorted(
            [q for q in questions if q.get('section') == 'C' and isinstance(q.get('q_num'), int)],
            key=lambda x: x['q_num']
        )
        for _qi, _qc in enumerate(_sec_c_qs):
            _qn = _qc['q_num']
            _start_pg = _qc['page_idx']
            # 下一题起始页（若有）或 last_content_page+1
            if _qi + 1 < len(_sec_c_qs):
                _end_pg = _sec_c_qs[_qi + 1]['page_idx']
            else:
                _end_pg = last_content_page + 1
            _mp = [(pg_i, (_qc['y_start'] if pg_i == _start_pg else 0), None)
                   for pg_i in range(_start_pg, _end_pg)]
            if len(_mp) > 1:
                # 更新 questions 列表里对应题目的 pages
                for _q in questions:
                    if _q.get('q_num') == _qn and _q.get('section') == 'C':
                        _q['pages'] = _mp
                        print(f'[econ_ms] {ms_econ_unit} Q{_qn} Section C 多页扩展: {len(_mp)}页 (p{_start_pg+1}~p{_end_pg})')

    result = {}
    scale = dpi / 72.0
    mat = fitz.Matrix(scale, scale)

    for q in questions:
        q_num = q['q_num']
        sub   = q.get('sub_label', '')

        # 确定 result key
        if sub:
            key = f"{q_num}{sub[1]}"   # '(a)' → '12a'
        else:
            key = q_num

        # Q13/Q14 使用多页版本
        if q_num == 13 and q13_info:
            q = q13_info
        elif q_num == 14 and q14_info:
            q = q14_info

        section = q.get('section', '')
        pages_info = q.get('pages', [(q['page_idx'], q['y_start'], q.get('y_end'))])

        try:
            parts = []
            total_w, total_h = 0, 0

            for (pg_i, y_s, y_e) in pages_info:
                if pg_i >= ms_doc.page_count:
                    continue
                page = ms_doc[pg_i]
                ph = page.rect.height
                pw = page.rect.width

                if section == 'A':
                    x_left  = 50.0
                    x_right = pw - 20.0
                    y_top   = max(0.0, float(y_s) - 2)
                    y_bot   = min(ph, float(y_e) + 2) if y_e is not None else ph
                else:
                    x_left  = 40.0
                    x_right = pw - 20.0
                    y_top   = max(0.0, float(y_s) - 2) if y_s and float(y_s) > 0 else 0.0
                    y_bot   = min(ph, float(y_e)) if y_e is not None else ph - 20.0

                if y_bot <= y_top + 5:
                    continue

                clip = fitz.Rect(x_left, y_top, x_right, y_bot)
                pix  = page.get_pixmap(matrix=mat, clip=clip, colorspace=fitz.csRGB)
                w, h = pix.width, pix.height
                data = _pixmap_to_jpeg_bytes(pix)
                del pix
                if h > 5:
                    parts.append((data, w, h))
                    total_w = max(total_w, w)
                    total_h += h

            if not parts:
                continue

            if len(parts) == 1:
                # 单页：保持原有格式
                jpeg_out, cw, ch = parts[0]
                result[key] = {
                    'b64':   _b64.b64encode(jpeg_out).decode('utf-8'),
                    'w':     cw,
                    'h':     ch,
                    'pages': None,  # 单页不需要 pages 数组
                }
            else:
                # 多页：每页单独存储，不拼接
                pages_list = []
                for (jpeg_part, pw2, ph2) in parts:
                    pages_list.append({
                        'b64': _b64.b64encode(jpeg_part).decode('utf-8'),
                        'w':   pw2,
                        'h':   ph2,
                    })
                # b64 取第一页（向后兼容；前端优先用 pages 数组）
                first = parts[0]
                result[key] = {
                    'b64':   _b64.b64encode(first[0]).decode('utf-8'),
                    'w':     first[1],
                    'h':     first[2],
                    'pages': pages_list,  # 多页数组，前端逐页显示
                }

            print(f'[econ_ms] rendered Q{q_num}{sub} key={key} pages={len(pages_info)} h_total={sum(p[2] for p in parts)}px')
        except Exception as e:
            print(f'[econ_ms] render error Q{q_num}{sub}: {e}')

    return result


def _detect_questions_ms(doc, paper_type):
    """
    Mark Scheme 专用题号检测入口。
    MS 文件题号格式多为 'Question 1'、'Question 2' 或 '1.' / '1'。
    对 edexcel_maths 类型额外尝试 'Question N' 行级标题格式。
    edexcel_economics 使用专用函数 detect_edexcel_economics_ms_questions。
    """
    if paper_type == 'edexcel_maths':
        # 先用标准检测
        questions = detect_edexcel_maths_questions(doc)
        if questions:
            return questions
        # 退回：扫描 'Question N' 行（MS 常见格式）
        return _detect_ms_questions_by_header(doc)
    # edexcel_economics 专用检测（返回结果仅用于占位；实际渲染由专用函数完成）
    if paper_type == 'edexcel_economics':
        return detect_edexcel_economics_ms_questions(doc)
    return _detect_questions(doc, paper_type)


def _detect_ms_questions_by_header(doc):
    """
    扫描 MS 文件中 'Question N' 格式的题目标题行（Edexcel Maths MS 常见）。
    匹配格式：
      - 'Question 1'  / 'Question 12'
      - 'Question 1 (a)' 等变体（只取 N，忽略子题）
    返回 questions 列表（q_num / page_idx / y_start / x_start）。
    """
    Q_HDR = re.compile(r'^Question\s+(\d{1,2})\b', re.IGNORECASE)
    questions = []
    seen_nums = set()

    for pg_i in range(doc.page_count):
        page = doc[pg_i]
        ph   = page.rect.height
        try:
            blocks = page.get_text('dict', flags=fitz.TEXT_PRESERVE_WHITESPACE)['blocks']
        except Exception:
            blocks = []

        for b in blocks:
            if b.get('type') != 0:
                continue
            for line in b.get('lines', []):
                line_txt = ''.join(s['text'] for s in line['spans']).strip()
                m = Q_HDR.match(line_txt)
                if not m:
                    continue
                q_num = int(m.group(1))
                if not (1 <= q_num <= 20):
                    continue
                if q_num in seen_nums:
                    continue
                # 取第一个 span 的位置信息
                spans = [s for ln in b.get('lines', []) for s in ln.get('spans', [])]
                if not spans:
                    continue
                bbox = spans[0]['bbox']
                x0, y0 = bbox[0], bbox[1]
                if y0 > ph - 40:
                    continue
                seen_nums.add(q_num)
                questions.append({
                    'q_num':    q_num,
                    'page_idx': pg_i,
                    'y_start':  y0,
                    'x_start':  x0,
                })

    questions.sort(key=lambda x: x['q_num'])
    return questions


def _detect_questions(doc, paper_type):
    """统一题号检测入口，兼容所有格式"""
    if paper_type == 'mcq':
        return detect_mcq_questions(doc)
    elif paper_type in ('edexcel', 'edexcel_mcq', 'edexcel_economics'):
        return detect_edexcel_questions(doc)
    elif paper_type == 'edexcel_maths':
        return detect_edexcel_maths_questions(doc)
    elif paper_type == 'bpho':
        return detect_bpho_questions(doc)
    else:
        return detect_structured_questions(doc)



@app.route('/api/get_answer', methods=['GET'])
def get_answer():
    """
    按需返回单题的完整答案数据（含多页图片）以及 Section C 材料页。
    参数：session_id, file_idx, q_num
    返回：{b64, w, h, pages: [{b64,w,h},...] or null, source_pages: [...] or null}
    """
    sess_id  = request.args.get('session_id', '')
    file_idx = int(request.args.get('file_idx', 0))
    q_num    = int(request.args.get('q_num', 0))

    sess = _get_session(sess_id) if sess_id else None
    if not sess or file_idx >= len(sess):
        return jsonify({'error': 'session不存在'}), 404

    grp = sess[file_idx]
    q_obj = next((q for q in grp.get('questions', []) if q.get('q_num') == q_num), None)
    if not q_obj:
        return jsonify({'error': '题目不存在'}), 404

    ans_b64     = q_obj.get('answer_b64', '')
    ans_pages   = q_obj.get('answer_pages')   # None 或 [{b64,w,h},...]
    source_pages = q_obj.get('source_pages')  # Section C 材料页（仅 Q12 有）

    if not ans_b64:
        return jsonify({'error': '该题无答案'}), 404

    return jsonify({
        'b64':          ans_b64,
        'w':            q_obj.get('answer_w', 0),
        'h':            q_obj.get('answer_h', 0),
        'pages':        ans_pages,
        'source_pages': source_pages,  # Section C 材料页，前端展示在答案前
    })


# ─────────────────────────────── AI 解析接口 ───────────────────────────────
@app.route('/api/ai_solution', methods=['POST'])
def ai_solution():
    """
    调用 AI 视觉模型对题目图片进行解析，返回详细解题过程。
    OpenRouter(OR) + Gemini 全部并发竞速，取最快成功的那个。
    OR 遇到 429 rate-limit 时，Gemini 会自动顶上，无需等 OR 全部超时。
    """
    import os as _os
    import json as _json
    import urllib.request as _urllib_req
    import urllib.error  as _urllib_err
    from concurrent.futures import ThreadPoolExecutor as _TPE, as_completed as _as_completed

    OPENROUTER_API_KEY = _os.environ.get('OPENROUTER_API_KEY', '').strip()
    GEMINI_API_KEY     = _os.environ.get('GEMINI_API_KEY', '').strip()

    app.logger.info(f'[ai_solution] OR_KEY={OPENROUTER_API_KEY[:12]+"..." if OPENROUTER_API_KEY else "未设置"} '
                    f'GEMINI_KEY={"已设置" if GEMINI_API_KEY else "未设置"}')

    if not OPENROUTER_API_KEY and not GEMINI_API_KEY:
        return jsonify({'ok': False,
                        'error': '未配置 AI API Key。请在 Railway Variables 中添加 OPENROUTER_API_KEY 或 GEMINI_API_KEY'}), 400

    data    = request.get_json(force=True, silent=True) or {}
    img_b64 = data.get('img_b64', '')
    q_num   = data.get('q_num', '')
    context = data.get('context', 'BPhO 物理竞赛')
    ans_b64 = data.get('ans_b64', '')

    if not img_b64:
        return jsonify({'ok': False, 'error': '缺少题目图片'}), 400

    prompt_text = (
        "你是一位专业的英国物理竞赛（BPhO）解题专家。\n"
        f"题目背景：{context}\n\n"
        "【重要指令】请直接给出完整详细的解题过程，不要写任何介绍、问候语或自我介绍。直接从"## 解题过程"开始。\n\n"
        "请按以下格式输出：\n\n"
        "## 解题过程\n\n"
        "**【已知条件】**\n"
        "列出题目中的所有已知量（含数值和单位）\n\n"
        "**【求解目标】**\n"
        "明确需要求的量\n\n"
        "**【物理原理】**\n"
        "列出本题涉及的物理定律/公式（LaTeX格式，$$公式$$）\n\n"
        "**【逐步推导】**\n"
        "若有多问 (a)(b)(c) 或 (i)(ii)(iii)，每问单独一个小节，逐步推导，每步说明物理意义\n\n"
        "**【最终答案】**\n"
        "给出所有问的最终数值结果，注意单位\n\n"
        "**【解题要点】**\n"
        "1-3条解题关键点总结\n\n"
        "注意：所有公式使用LaTeX（行间公式 $$...$$ ，行内公式 $...$），中文解释，内容要完整详尽。"
    )

    errors_detail = []
    workers = []   # list of callables, each returns a result dict

    # ── OpenRouter workers ──────────────────────────────────────────
    if OPENROUTER_API_KEY:
        # 2026-08 确认支持图片的免费视觉模型（不含 openrouter/free 安全过滤路由）
        OR_MODELS = [
            "google/gemma-4-26b-a4b-it:free",    # Gemma4 26B ✅
            "google/gemma-4-31b-it:free",          # Gemma4 31B ✅
            "nvidia/nemotron-nano-12b-v2-vl:free", # Nemotron VL ✅
        ]

        def _make_or_worker(model_id):
            def _worker():
                try:
                    content = [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                    ]
                    if ans_b64:
                        content.append({"type": "text",
                                         "text": "\n以下是官方 Mark Scheme，请严格按照它核对答案："})
                        content.append({"type": "image_url",
                                         "image_url": {"url": f"data:image/jpeg;base64,{ans_b64}"}})
                    body = _json.dumps({
                        "model": model_id,
                        "messages": [{"role": "user", "content": content}],
                        "max_tokens": 4096,
                        "temperature": 0.2,
                    }).encode('utf-8')
                    req = _urllib_req.Request(
                        "https://openrouter.ai/api/v1/chat/completions",
                        data=body,
                        headers={
                            'Content-Type': 'application/json',
                            'Authorization': f'Bearer {OPENROUTER_API_KEY}',
                            'HTTP-Referer': 'https://yxt-question-bank.railway.app',
                            'X-Title': 'BPhO Question Bank',
                        },
                        method='POST'
                    )
                    with _urllib_req.urlopen(req, timeout=45) as resp:
                        result = _json.loads(resp.read().decode('utf-8'))

                    choices = result.get('choices', [])
                    if not choices:
                        err = result.get('error', {})
                        return {'ok': False, 'model': model_id, 'provider': 'openrouter',
                                'error': f'无choices: {err.get("message", str(result))[:200]}'}

                    solution      = (choices[0].get('message', {}).get('content', '') or '').strip()
                    finish_reason = choices[0].get('finish_reason', 'UNKNOWN')

                    if len(solution) < 80:
                        return {'ok': False, 'model': model_id, 'provider': 'openrouter',
                                'error': f'内容过短({len(solution)}字符): {solution[:120]}'}

                    return {'ok': True, 'model': model_id, 'provider': 'openrouter',
                            'solution': solution, 'finish_reason': finish_reason}

                except _urllib_err.HTTPError as e:
                    body_txt = e.read().decode('utf-8', errors='replace')
                    return {'ok': False, 'model': model_id, 'provider': 'openrouter',
                            'error': f'HTTP {e.code}: {body_txt[:200]}'}
                except Exception as e:
                    return {'ok': False, 'model': model_id, 'provider': 'openrouter',
                            'error': f'{type(e).__name__}: {str(e)}'}
            return _worker

        for _m in OR_MODELS:
            workers.append(_make_or_worker(_m))

    # ── Gemini workers ──────────────────────────────────────────────
    if GEMINI_API_KEY:
        GEMINI_MODELS = [
            "gemini-2.5-flash",
            "gemini-2.0-flash",
        ]

        gemini_parts = [
            {"text": prompt_text},
            {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}}
        ]
        if ans_b64:
            gemini_parts.append({"text": "\n以下是官方 Mark Scheme，请严格按照它核对答案："})
            gemini_parts.append({"inline_data": {"mime_type": "image/jpeg", "data": ans_b64}})
        gemini_payload = _json.dumps({
            "contents": [{"parts": gemini_parts}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 4096, "topP": 0.95}
        }).encode('utf-8')

        def _make_gemini_worker(model):
            def _worker():
                try:
                    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
                           f"{model}:generateContent?key={GEMINI_API_KEY}")
                    req = _urllib_req.Request(url, data=gemini_payload,
                                              headers={'Content-Type': 'application/json'},
                                              method='POST')
                    with _urllib_req.urlopen(req, timeout=60) as resp:
                        result = _json.loads(resp.read().decode('utf-8'))

                    candidates = result.get('candidates', [])
                    if not candidates:
                        return {'ok': False, 'model': model, 'provider': 'gemini',
                                'error': '无候选结果'}

                    finish_reason = candidates[0].get('finishReason', 'UNKNOWN')
                    parts_out     = candidates[0].get('content', {}).get('parts', [])
                    solution      = ''.join(p.get('text', '') for p in parts_out if 'text' in p).strip()

                    if not solution:
                        return {'ok': False, 'model': model, 'provider': 'gemini',
                                'error': '返回文本为空'}

                    if finish_reason == 'MAX_TOKENS':
                        solution += '\n\n> 警告：内容较长，已达到输出上限，解析可能不完整'

                    return {'ok': True, 'model': model, 'provider': 'gemini',
                            'solution': solution, 'finish_reason': finish_reason}

                except _urllib_err.HTTPError as e:
                    body_txt = e.read().decode('utf-8', errors='replace')
                    return {'ok': False, 'model': model, 'provider': 'gemini',
                            'error': f'HTTP {e.code}: {body_txt[:200]}'}
                except Exception as e:
                    return {'ok': False, 'model': model, 'provider': 'gemini',
                            'error': f'{type(e).__name__}: {str(e)}'}
            return _worker

        for _m in GEMINI_MODELS:
            workers.append(_make_gemini_worker(_m))

    # ── 全部 workers 并发竞速，取最快成功的那个 ──────────────────────
    # OR 429 时，Gemini 马上顶上；无需等 OR timeout
    first_success = None
    with _TPE(max_workers=max(len(workers), 1)) as executor:
        futures = {executor.submit(w): w for w in workers}
        for fut in _as_completed(futures):
            res = fut.result()
            if res.get('ok'):
                first_success = res
                for f in futures:
                    f.cancel()
                break
            else:
                tag = f"{res.get('provider','?')}/{res.get('model','?')}"
                errors_detail.append(f"{tag}: {res.get('error','未知错误')}")
                app.logger.warning(f'[ai_solution] x {errors_detail[-1]}')

    if first_success:
        sol = first_success['solution']
        fin = first_success['finish_reason']
        mid = first_success['model']
        if fin == 'length':
            sol += '\n\n> 警告：内容较长，已达到输出上限，解析可能不完整'
        app.logger.info(f'[ai_solution] OK {first_success["provider"]}/{mid} finish={fin} len={len(sol)}')
        return jsonify({'ok': True, 'solution': sol, 'model': mid,
                        'finish_reason': fin, 'provider': first_success['provider']})

    # 全部失败
    or_key_hint = f'OR前缀={OPENROUTER_API_KEY[:12]}...' if OPENROUTER_API_KEY else 'OR未设置'
    gem_hint    = 'Gemini已设置' if GEMINI_API_KEY else 'Gemini未设置'
    last = errors_detail[-1] if errors_detail else '无详情'
    app.logger.error(f'[ai_solution] 全部失败 [{or_key_hint} {gem_hint}]: {errors_detail}')
    return jsonify({
        'ok': False,
        'error': f'所有模型均失败，详情：{last}',
        'key_status': f'{or_key_hint} | {gem_hint}',
        'all_errors': errors_detail
    }), 500


@app.route('/api/env_check', methods=['GET'])
def env_check():
    """安全诊断：检查环境变量是否存在，不暴露完整key值。"""
    import os as _os
    keys_to_check = ['OPENROUTER_API_KEY', 'GEMINI_API_KEY', 'R2_BUCKET_NAME',
                     'R2_ACCOUNT_ID', 'R2_ACCESS_KEY_ID', 'R2_SECRET_ACCESS_KEY']
    result = {}
    for k in keys_to_check:
        v = _os.environ.get(k, '')
        v = v.strip()
        if v:
            result[k] = f'✅ 已设置 (长度={len(v)}, 前缀={v[:8]}...)'
        else:
            result[k] = '❌ 未设置或为空'
    return jsonify({'ok': True, 'env': result})


@app.route('/api/test_gemini', methods=['GET'])
def test_gemini():
    """诊断接口：测试 OpenRouter 和 Gemini 模型可用性。访问 /api/test_gemini 查看。"""
    import os as _os
    import json as _json
    import urllib.request as _urllib_req
    import urllib.error  as _urllib_err

    OPENROUTER_API_KEY = _os.environ.get('OPENROUTER_API_KEY', '').strip()
    GEMINI_API_KEY     = _os.environ.get('GEMINI_API_KEY', '').strip()
    results = []

    # 测试 OpenRouter
    if OPENROUTER_API_KEY:
        or_models = [
            "google/gemma-4-26b-a4b-it:free",
            "nvidia/nemotron-nano-12b-v2-vl:free",
            "meta-llama/llama-3.2-90b-vision-instruct:free",
        ]
        for m in or_models:
            body = _json.dumps({"model": m, "messages": [{"role":"user","content":"Hi"}], "max_tokens": 5}).encode()
            try:
                req = _urllib_req.Request("https://openrouter.ai/api/v1/chat/completions",
                    data=body, headers={'Content-Type':'application/json',
                    'Authorization':f'Bearer {OPENROUTER_API_KEY}'}, method='POST')
                with _urllib_req.urlopen(req, timeout=15) as resp:
                    rb = _json.loads(resp.read())
                    ok = bool(rb.get('choices'))
                    results.append({'provider':'openrouter','model':m,'status':'ok' if ok else 'no_choices','code':200})
            except _urllib_err.HTTPError as e:
                results.append({'provider':'openrouter','model':m,'status':'error','code':e.code,
                                 'detail':e.read().decode('utf-8',errors='replace')[:150]})
            except Exception as e:
                results.append({'provider':'openrouter','model':m,'status':'error','code':0,'detail':str(e)})

    # 测试 Gemini
    if GEMINI_API_KEY:
        gm_models = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.0-flash-lite"]
        test_payload = _json.dumps({"contents":[{"parts":[{"text":"Hello"}]}],"generationConfig":{"maxOutputTokens":5}}).encode()
        for m in gm_models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={GEMINI_API_KEY}"
            try:
                req = _urllib_req.Request(url, data=test_payload, headers={'Content-Type':'application/json'}, method='POST')
                with _urllib_req.urlopen(req, timeout=15) as resp:
                    rb = _json.loads(resp.read())
                    ok = bool(rb.get('candidates'))
                    results.append({'provider':'gemini','model':m,'status':'ok' if ok else 'no_candidates','code':200})
            except _urllib_err.HTTPError as e:
                results.append({'provider':'gemini','model':m,'status':'error','code':e.code,
                                 'detail':e.read().decode('utf-8',errors='replace')[:150]})
            except Exception as e:
                results.append({'provider':'gemini','model':m,'status':'error','code':0,'detail':str(e)})

    configured = []
    if OPENROUTER_API_KEY: configured.append(f'OPENROUTER_API_KEY ✅ (前缀: {OPENROUTER_API_KEY[:12]}...)')
    if GEMINI_API_KEY:     configured.append('GEMINI_API_KEY ✅')
    if not configured:     configured.append('未配置任何 Key ❌')

    return jsonify({'ok': True, 'configured': configured, 'results': results})



def get_source_pages():
    """
    按需返回题目的 Section C 材料页（无需已有答案）。
    参数：session_id, file_idx, q_num
    返回：{source_pages: [{b64,w,h},...] or null}
    """
    sess_id  = request.args.get('session_id', '')
    file_idx = int(request.args.get('file_idx', 0))
    q_num    = int(request.args.get('q_num', 0))

    sess = _get_session(sess_id) if sess_id else None
    if not sess or file_idx >= len(sess):
        return jsonify({'error': 'session不存在'}), 404

    grp = sess[file_idx]
    q_obj = next((q for q in grp.get('questions', []) if q.get('q_num') == q_num), None)
    if not q_obj:
        return jsonify({'error': '题目不存在'}), 404

    source_pages = q_obj.get('source_pages')
    return jsonify({'source_pages': source_pages})


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

        # ── 云端题目（无实体PDF）：直接从 img_bytes_b64 返回 ──
        if not save_path or not os.path.exists(save_path):
            q_obj = next((q for q in questions if q['q_num'] == q_num), None)
            if q_obj and q_obj.get('img_bytes_b64'):
                import base64 as _b64
                img_data = _b64.b64decode(q_obj['img_bytes_b64'])
                return send_file(io.BytesIO(img_data), mimetype='image/jpeg', as_attachment=False)
            return jsonify({'error': '该题目无预览图'}), 404
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



def _build_pdf_merged_worker(task_id, groups_info, dpi, layout, out_path, total_q,
                              cover_title='', ordered_items=None, include_answer=True,
                              answer_visible_map=None):
    """
    后台线程：多文件合并导出 PDF。
    groups_info: [{path, paper_type, questions, q_nums, g_idx}]
    ordered_items: [{gIdx, q_num}] 全局有序列表（来自前端 exportItems，保留 sortOrder 排序）。
                   若提供则按此全局顺序逐题输出；否则按组顺序输出（降级模式）。
    cover_title: 封面标题，非空时在首页插入封面。
    include_answer: 是否在PDF中包含答案页（全局开关，Task3）。
    answer_visible_map: per-question 答案显示状态 {"{gIdx}_{q_num}": bool}，
                        若存在则以此为准（覆盖全局 include_answer）；否则使用全局开关。
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

        # 若提供了 answer_visible_map（per-question 精确控制），优先使用它；
        # 否则回退到全局 include_answer 开关。
        # ★ 重要：这里不能直接修改 ginfo['questions'] 里的原始字典（那是 session 的引用）！
        # 否则"仅题目"导出后，session 里的 answer_b64 被 pop，下次"题目+答案"就没有答案了。
        # 解决方案：把每道需要隐藏答案的题做浅拷贝，替换掉 ginfo['questions'] 里的引用。
        if answer_visible_map:
            # 对每道题根据 answer_visible_map 决定是否保留 answer_b64
            for ginfo in groups_info:
                g_idx = ginfo['g_idx']
                new_qs = []
                for q in ginfo.get('questions', []):
                    uid = f"{g_idx}_{q['q_num']}"
                    visible = answer_visible_map.get(uid, include_answer)
                    if not visible and q.get('answer_b64'):
                        q = dict(q)          # 浅拷贝，不污染 session 原始数据
                        q.pop('answer_b64', None)
                    new_qs.append(q)
                ginfo['questions'] = new_qs
        elif not include_answer:
            # 全局开关：不包含答案，清除所有题目的 answer_b64（同样用拷贝，不污染 session）
            for ginfo in groups_info:
                new_qs = []
                for q in ginfo.get('questions', []):
                    if q.get('answer_b64'):
                        q = dict(q)
                        q.pop('answer_b64', None)
                    new_qs.append(q)
                ginfo['questions'] = new_qs

        # ── 封面页 ──
        if cover_title:
            _generate_cover_page(out_doc, cover_title,
                                  page_w=PAGE_W, page_h=PAGE_H)

        # 构建 gIdx → {src_doc, questions, paper_type} 的映射（延迟打开）
        group_map = {}  # g_idx -> ginfo
        for ginfo in groups_info:
            group_map[ginfo['g_idx']] = ginfo

        def _export_cloud_questions(out_doc, questions, q_items, dpi,
                                     PW, PH, M, HH, GAP, FS, seq_start=0):
            """云端/workbook题目（无实体PDF）：直接将 img_bytes_b64 渲染到输出页。
            q_items: [(q_num, _wb_seq_or_None), ...] — 使用 _wb_seq 精确定位 workbook 题目。
            """
            import base64 as _b64_inner
            AVAIL_W = PW - 2 * M
            # 向后兼容：若传入的是旧格式 [q_num, ...]（纯列表），转为 [(q_num, None), ...]
            if q_items and not isinstance(q_items[0], tuple):
                q_items = [(qn, None) for qn in q_items]

            for done, (q_num, wb_seq) in enumerate(q_items):
                # 优先用 _wb_seq 定位（workbook 多道同 q_num 题目的唯一 key）
                if wb_seq is not None:
                    q_obj = next((q for q in questions if q.get('_wb_seq') == wb_seq), None)
                if wb_seq is None or q_obj is None:
                    q_obj = next((q for q in questions if q['q_num'] == q_num), None)
                if not q_obj:
                    continue
                b64 = q_obj.get('img_bytes_b64', '')
                if not b64:
                    continue
                try:
                    img_data = _b64_inner.b64decode(b64)
                    from PIL import Image as _PILImg2
                    _im = _PILImg2.open(io.BytesIO(img_data))
                    img_w, img_h = _im.size
                    buf = io.BytesIO()
                    _im.convert('RGB').save(buf, format='JPEG', quality=88)
                    jpeg_bytes = buf.getvalue()
                except Exception as _ce:
                    print(f'[cloud_export] decode error q_num={q_num}: {_ce}')
                    continue

                export_seq = seq_start + done + 1
                label = f'第 {export_seq} 题'

                q_meta = None
                diff      = q_obj.get('difficulty')
                topics    = q_obj.get('topics') or []
                exam_date = q_obj.get('exam_date', '')
                if diff is not None or topics or exam_date:
                    q_meta = {'difficulty': diff, 'topics': topics, 'exam_date': exam_date}

                page = out_doc.new_page(width=PW, height=PH)
                _place_jpeg_on_page(page, jpeg_bytes, img_w, img_h,
                                    fitz.Rect(M, M, PW-M, PH-M),
                                    label, HH, GAP, FS, q_meta=q_meta)

                # 答案页（支持多页）
                if q_obj.get('answer_b64') or q_obj.get('answer_pages'):
                    _insert_answer_pages(out_doc, q_obj, PW, PH, M, GAP, label)

        if ordered_items:
            # ── 有序模式：严格按 ordered_items 全局顺序逐题输出 ──
            # 修复：原逻辑先按 gIdx 分组再批量处理，导致跨组排序（如 chapter 排序）时顺序丢失。
            # 新逻辑：逐题遍历 ordered_items，每道题单独查询所在组并渲染一页，保证全局顺序。
            src_docs = {}  # g_idx -> fitz.Document（懒加载，用时再打开）
            is_cloud_group = {}  # g_idx -> bool，缓存判断结果
            done_total = 0

            # 预计算每个 gIdx 是否为 cloud/workbook 组
            for ginfo in groups_info:
                gi  = ginfo['g_idx']
                src = ginfo.get('source', '')
                is_cloud_group[gi] = (
                    src in ('workbook', 'imported', 'cloud')
                    or not ginfo.get('path')
                    or any(q.get('img_bytes_b64') for q in (ginfo.get('questions') or []))
                )

            for item in ordered_items:
                gi    = item.get('gIdx', 0)
                q_num = item.get('q_num')
                _seq  = item.get('_seq', None)
                if q_num is None or gi not in group_map:
                    done_total += 1
                    continue

                ginfo      = group_map[gi]
                questions  = ginfo['questions']
                paper_type = ginfo['paper_type']
                export_seq = done_total  # seq_start 传入（0-based）

                if is_cloud_group.get(gi, False):
                    # ── workbook/云端：直接从 img_bytes_b64 渲染 ──
                    _export_cloud_questions(
                        out_doc, questions, [(q_num, _seq)], dpi,
                        PAGE_W, PAGE_H, MARGIN, HEADER_H, GAP, FS,
                        seq_start=export_seq
                    )
                else:
                    # ── 实体 PDF：懒加载 src_doc，逐题渲染一页 ──
                    if gi not in src_docs:
                        pdf_path = ginfo.get('path', '')
                        if pdf_path and os.path.exists(pdf_path):
                            src_docs[gi] = fitz.open(pdf_path)
                        else:
                            done_total += 1
                            upd(done_total)
                            continue
                    src_doc = src_docs[gi]
                    if layout == 'two_per_page':
                        _export_two_per_page(
                            out_doc, src_doc, questions, [q_num], dpi,
                            paper_type, PAGE_W, PAGE_H, MARGIN,
                            HEADER_H, GAP, FS, seq_start=export_seq
                        )
                    else:
                        _export_one_per_page(
                            out_doc, src_doc, questions, [q_num], dpi,
                            paper_type, PAGE_W, PAGE_H, MARGIN,
                            HEADER_H, GAP, FS, seq_start=export_seq
                        )

                done_total += 1
                upd(done_total)

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

                # ── 云端/workbook 题目：直接从 img_bytes_b64 渲染（不依赖实体 PDF）──
                src = ginfo.get('source', '')
                is_cloud_group = (
                    src in ('workbook', 'imported', 'cloud')
                    or not save_path or not os.path.exists(save_path)
                    or any(q.get('img_bytes_b64') for q in (questions or []))
                )
                if is_cloud_group:
                    _export_cloud_questions(out_doc, questions, q_nums, dpi,
                                            PAGE_W, PAGE_H, MARGIN, HEADER_H, GAP, FS,
                                            seq_start=done_total)
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

        if out_doc.page_count == 0:
            raise ValueError('导出后 PDF 为空，请确保题目图片已正确加载')

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
    dpi       = min(int(data.get('dpi', 150)), 1200)   # 允许最高 1200 dpi
    filename  = (data.get('filename') or 'questions').strip()
    layout    = data.get('layout', 'one_per_page')
    sess_id   = data.get('session_id')
    merged    = data.get('merged', False)
    cover_title    = data.get('cover_title', '').strip()
    include_answer = data.get('include_answer', True)  # Task3: 是否在PDF中包含答案页
    answer_visible_map = data.get('answer_visible_map', {})  # per-question 答案显示状态 {"{gIdx}_{q_num}": bool}

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
            q_nums = item.get('q_nums', [])
            if not q_nums:
                continue
            # ── 支持新格式（item 携带 gIdx + session_id + file_idx）和旧格式 ──
            item_sid   = item.get('session_id', None)
            item_fidx  = item.get('file_idx',   None)
            # g_idx 优先用前端传来的 gIdx（全局唯一 group 索引），
            # 这确保 groups_info 里不同 group 的 g_idx 不重复（追加套题修复）
            g_idx      = item.get('gIdx', item_fidx or 0)

            if item_sid and item_sid != sess_id:
                # 追加套题：使用 item 指定的 session
                item_sess = _get_session(item_sid)
                if not item_sess:
                    app.logger.warning(f'[export_pdf] append session not found: {item_sid}')
                    continue
                fidx = item_fidx if item_fidx is not None else 0
                if fidx >= len(item_sess):
                    app.logger.warning(f'[export_pdf] append file_idx {fidx} out of range for session {item_sid}')
                    continue
                g = item_sess[fidx]
            else:
                # 原始题：使用主 session
                if g_idx >= len(sess):
                    continue
                g = sess[g_idx]

            groups_info.append({
                'path':       g.get('path', ''),
                'paper_type': g.get('paper_type', ''),
                'questions':  g.get('questions', []),
                'q_nums':     q_nums,
                'g_idx':      g_idx,
                'source':     g.get('source', ''),   # 必须传递：_build_pdf_merged_worker 用此判断是否 workbook
            })
            total_q += len(q_nums)

        if total_q == 0:
            return jsonify({'error': '未选择题目'}), 400

        # ── 关键修复：将前端手动替换的图片覆盖到 groups_info 的 questions 里 ──
        # 前端在 _buildExportPayload 中对有 _img_replaced / _answer_replaced 的题目
        # 会在 ordered_items[i] 附带 img_bytes_b64_override / answer_b64_override，
        # 后端必须在这里把覆盖数据写入 questions，否则导出时仍读旧图。
        if ordered_items:
            # 构建 g_idx -> questions 快速索引
            gi_to_questions = {gi['g_idx']: gi['questions'] for gi in groups_info}
            for oi in ordered_items:
                img_override = oi.get('img_bytes_b64_override')
                ans_override = oi.get('answer_b64_override')
                if not img_override and not ans_override:
                    continue  # 该题无替换，跳过
                gi   = oi.get('gIdx', 0)
                qn   = oi.get('q_num')
                _seq = oi.get('_seq')
                qs   = gi_to_questions.get(gi, [])
                # 优先用 _wb_seq 精确定位，fallback 到 q_num
                q_obj = None
                if _seq is not None:
                    q_obj = next((q for q in qs if q.get('_wb_seq') == _seq), None)
                if q_obj is None:
                    q_obj = next((q for q in qs if q.get('q_num') == qn), None)
                if q_obj is None:
                    continue
                if img_override:
                    q_obj['img_bytes_b64'] = img_override
                    if oi.get('img_w'): q_obj['img_w'] = oi['img_w']
                    if oi.get('img_h'): q_obj['img_h'] = oi['img_h']
                if ans_override:
                    q_obj['answer_b64'] = ans_override

        _save_task(task_id, {
            'status': 'running', 'progress': 0,
            'total': total_q, 'out_path': out_path,
            'filename': safe_name, 'error': None,
        })

        threading.Thread(
            target=_build_pdf_merged_worker,
            args=(task_id, groups_info, dpi, layout, out_path, total_q),
            kwargs={'cover_title': cover_title, 'ordered_items': ordered_items,
                    'include_answer': include_answer,
                    'answer_visible_map': answer_visible_map},
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

    若 session_id 以 'wb_' 开头（题库题目），同步持久化修改到 manifest.json。
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

    group     = sess[g_idx]
    questions = group['questions']

    # 题库 session 用 _wb_seq 精确定位；普通 session 用 q_num
    wb_prefix  = group.get('_wb_prefix', '')
    wb_manifest = group.get('_wb_manifest')  # 原始 manifest dict（直接修改后写回）
    is_wb_sess = sess_id.startswith('wb_') and wb_prefix

    if is_wb_sess:
        # 题库题目：前端传来的 q_num 对应 manifest 里的序号；
        # virt_questions 里 '_wb_seq' 是 manifest questions 的 0-based index
        # 用 '_wb_seq' 精确定位，避免多套卷 q_num 重复问题
        _wb_seq_hint = data.get('_wb_seq')  # 前端可选传
        if _wb_seq_hint is not None:
            q_obj = next((q for q in questions if q.get('_wb_seq') == _wb_seq_hint), None)
        else:
            q_obj = next((q for q in questions if q.get('q_num') == q_num), None)
    else:
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

    # ── 题库题目：持久化修改到 manifest.json ──
    if is_wb_sess and wb_manifest:
        try:
            # q_obj['_wb_seq'] 是 manifest questions 的 0-based index
            _seq = q_obj.get('_wb_seq')
            mq_list = wb_manifest.get('questions', [])
            # 找到 manifest 里对应的题目（seq 字段是 1-based，_wb_seq 是 0-based）
            mq = None
            if _seq is not None and 0 <= _seq < len(mq_list):
                mq = mq_list[_seq]
            else:
                # fallback: 用 q_num 匹配
                mq = next((q for q in mq_list if q.get('q_num') == q_num), None)
            if mq is not None:
                mq[field] = q_obj[field]   # 同步到 manifest dict
            # 写回存储
            if storage.is_r2_mode():
                storage.store_json(f'{wb_prefix}/manifest.json', wb_manifest)
            else:
                mf_path = os.path.join(wb_prefix, 'manifest.json')
                with open(mf_path, 'w', encoding='utf-8') as _mf:
                    json.dump(wb_manifest, _mf, ensure_ascii=False, indent=2)
            app.logger.info(f'[update_question_meta] 已持久化 wb={wb_prefix!r} '
                            f'q_num={q_num} field={field!r}')
        except Exception as _pe:
            # 持久化失败不阻断请求，只记录日志
            app.logger.warning(f'[update_question_meta] 持久化失败: {_pe}')

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
    检测页面是否有本工具导出的头栏，并提取题号、难度、知识点ID。

    支持两种导出格式：
    A. 新版（深蓝色色带格式）：
       - 大题目图片 y0 > 60，图片上方有头栏
       - 格式："Q05  |  Medium  |  *P3-4  Differentiation"
    B. 题库册格式（带Logo）：
       - 右上角有小Logo图片（y0≈39，宽度<400）
       - 题目大图 y0≈71（排在Logo后面）
       - 头栏文字格式："· 1 ·  Starter  P4-1.1 Proof by contradiction  June 2020"

    返回 (q_num, difficulty, topic_id, topic_title_hint) 或 None。
    """
    MARGIN = 36

    # ── 收集所有图片的位置和尺寸 ──
    img_list = page.get_images(full=False)
    if not img_list:
        return None

    # 收集所有图片的(xref, y0, y1, width, height)
    img_infos = []
    doc = page.parent
    for xref, *_ in img_list:
        rects = page.get_image_rects(xref)
        if rects:
            try:
                info = doc.extract_image(xref)
                img_w = info.get('width', 9999)
                img_h = info.get('height', 9999)
            except Exception:
                img_w, img_h = 9999, 9999
            for r in rects:
                img_infos.append((xref, r.y0, r.y1, img_w, img_h, r))

    if not img_infos:
        return None

    # 按 y0 排序
    img_infos.sort(key=lambda x: x[1])

    # ── 判断格式 ──
    # 格式B特征：第一张图是小Logo（宽度<400且高度<200且y0<70）
    #           第二张图是大题目图（y0>60）
    first_y0    = img_infos[0][1]
    first_w     = img_infos[0][3]
    first_h     = img_infos[0][4]
    is_logo_fmt = (first_y0 <= 70 and first_w < 400 and first_h < 300)

    if is_logo_fmt:
        # ── 格式B：题库册格式（带Logo）──
        # 找第二张（非Logo）大图的 y0 作为内容起点
        content_y0 = None
        for xref, y0, y1, w, h, rect in img_infos[1:]:
            if w > 400 or h > 200:  # 题目图应该比较大
                content_y0 = y0
                break
        if content_y0 is None and len(img_infos) >= 2:
            content_y0 = img_infos[1][1]

        # 提取页眉文字（在页面顶部到内容图片之间）
        header_top = MARGIN - 8
        header_bot = (content_y0 + 5) if content_y0 else 75
        header_rect = fitz.Rect(0, header_top, page.rect.width, header_bot)
        raw_text = page.get_text('text', clip=header_rect).strip()

        if not raw_text:
            return None

        # 解析格式B的题号：· N · 格式
        q_num_match = re.search(r'·\s*(\d+)\s*·', raw_text)
        if not q_num_match:
            # fallback：Q(数字) 格式
            q_num_match = re.search(r'\bQ\s*(\d+)\b', raw_text, re.IGNORECASE)
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

        # 提取知识点 ID（格式：P4-1.1 → 用连字符+点号形式，兼容两段式和三段式）
        # 先尝试完整格式 P4-1.1（三段）
        topic_id_match = re.search(r'·?\*?([A-Z]\d+-\d+(?:\.\d+)?)', raw_text)
        topic_id = topic_id_match.group(1) if topic_id_match else None
        # 归一化：P4-1.1 → P4-1（只保留两段，与系统内部 ID 一致）
        if topic_id and re.match(r'^[A-Z]\d+-\d+\.\d+$', topic_id):
            topic_id = re.sub(r'\.\d+$', '', topic_id)

        # 提取知识点标题（topic_id 之后的文字）
        topic_title_hint = None
        raw_topic_match = re.search(r'·?\*?[A-Z]\d+-\d+(?:\.\d+)?\s+(.+?)(?:\s+\d{4}|\s*$)',
                                    raw_text, re.DOTALL)
        if raw_topic_match:
            topic_title_hint = raw_topic_match.group(1).strip()
            # 清理多余的 / +2 等辅助标注
            topic_title_hint = re.sub(r'\s*/\s*|\s*\+\d+\s*', ' ', topic_title_hint).strip()

        return q_num, difficulty, topic_id, topic_title_hint

    else:
        # ── 格式A：原版深蓝色色带格式 ──
        first_img_y0 = img_infos[0][1]

        # img.y0 ≤ 60 说明无头栏（图片紧贴页面顶部 margin）
        if first_img_y0 <= 60:
            return None

        # ── Step 2：提取头栏文字（头栏位于图片上方） ──
        header_top = MARGIN - 5
        header_bot = first_img_y0 + 2
        header_rect = fitz.Rect(0, header_top, page.rect.width, header_bot)
        raw_text = page.get_text('text', clip=header_rect).strip()

        if not raw_text:
            # fallback：用 rawdict 颜色法（兼容旧格式）
            return _detect_header_legacy(page, header_rect)

        # ── Step 3：解析头栏文本 ──
        # 典型格式："Q05  |  Medium  |  *P3-4  Differentiation"
        q_num_match = re.search(r'\bQ\s*(\d+)\b', raw_text, re.IGNORECASE)
        if not q_num_match:
            q_num_match = re.search(r'\b(\d+)\b', raw_text)
        if not q_num_match:
            return None
        q_num = int(q_num_match.group(1))

        difficulty = None
        text_lower = raw_text.lower()
        for dname, dval in _DIFF_NAME_MAP.items():
            if dname in text_lower:
                difficulty = dval
                break

        topic_id_match = re.search(r'\*?([A-Z]\d+-\d+)', raw_text)
        topic_id = topic_id_match.group(1) if topic_id_match else None

        topic_title_hint = None
        if topic_id:
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
    if topic_id:
        # Edexcel Economics: U1-1, U2-3, U3-5, U4-2 等格式
        if re.match(r'^U[1-4]-\d+', topic_id):
            return 'edexcel_economics'
        # Edexcel Maths: P1-1, S1-2, M2-3 等格式
        if re.match(r'^[A-Z]\d+-\d+', topic_id):
            return 'edexcel_maths'
    return 'unknown'


def _find_content_img_y0(page, fallback=76):
    """
    找页面上题目内容图片的起始 y 坐标（跳过页眉区域的 Logo 小图）。

    逻辑：
    - 收集所有图片，按 y0 排序
    - 如果第一张是小图（宽度<400 且 高度<300，即 Logo），跳过它
    - 返回第一张"大图"的 y0；若找不到则返回 fallback
    """
    doc = page.parent
    img_infos = []
    for xref, *_ in page.get_images(full=False):
        rects = page.get_image_rects(xref)
        if not rects:
            continue
        try:
            info = doc.extract_image(xref)
            w, h = info.get('width', 9999), info.get('height', 9999)
        except Exception:
            w, h = 9999, 9999
        for r in rects:
            img_infos.append((r.y0, w, h))

    if not img_infos:
        return fallback

    img_infos.sort(key=lambda x: x[0])

    for y0, w, h in img_infos:
        # 跳过 Logo 类小图（y0<70, 宽<400, 高<300）
        if y0 <= 70 and w < 400 and h < 300:
            continue
        return y0

    # 全是小图的极端情况，返回最后一张的 y0
    return img_infos[-1][0]


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
    - 用 P\\d+-\\d+ 格式自动识别为 edexcel_maths 大纲，在 syllabus 中精确定位
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

                # 找本页内容图片的真实起始 y（跳过 Logo 小图）
                content_y0 = _find_content_img_y0(pg_obj,
                             fallback=MARGIN + _EXPORT_HEADER_HEIGHT_PT + 14)

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
                # 自动确认大纲类型（P\\d+-\\d+ → edexcel_maths）
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


def _is_econ_dotted_answer_page(page, ph):
    """
    检测 Edexcel Economics 结构题的答题页（学生写答区）。
    这类页面的特征：含大量 dotted lines "....................."（点线），
    这是学生手写答题区，不是题目内容。

    用于跳过 Section B/C 中介于首页（题干 overview）和末页之间的纯答题页。
    判断条件：页面正文区域内 dotted line 块数量 >= 4
    （Q7等单页题虽然也有 dotted lines，但它们是 pg_start==pg_end，不会触发此判断）
    """
    try:
        blocks = page.get_text('blocks')
        dotted_count = 0
        for b in blocks:
            x0, y0, x1, y1, txt, bno, btype = b
            if btype != 0:
                continue
            if y0 < 40 or y0 > ph - 40:
                continue
            ts = txt.strip()
            # 点线判定：长度 >= 15 且点号占比 > 80%
            if len(ts) >= 15 and ts.count('.') / len(ts) > 0.80:
                dotted_count += 1
        return dotted_count >= 4
    except Exception:
        return False


def _has_question_content_in_range(page, y_min, y_max):
    """
    检测页面在 [y_min, y_max] 范围内是否有属于具体题目的文字内容。
    排除 Section header 类全局指令（SECTION X / Answer ONE question / Write your answer...）。
    用于判断末页（pg_end）上是否有当前题的实质内容，避免把纯 Section header 页误纳入切片。

    返回 True 表示有有效的题目内容；False 表示该区域只有 Section header 或空白。
    """
    SECTION_HEADER_RE = re.compile(
        r'^(SECTION\s+[A-Z]|Answer\s+(ALL|ONE|TWO|THREE)\s+questions?|'
        r'Write\s+your\s+answers?|Study\s+(Figure|Extract)|'
        r'EITHER|OR)\b',
        re.IGNORECASE
    )
    SKIP_RE = re.compile(
        r'^(DO NOT WRITE|Turn over|©|\d{1,4}$|\s*$)',
        re.IGNORECASE
    )
    try:
        blocks = page.get_text('blocks')
        for b in blocks:
            x0, y0, x1, y1, txt, bno, btype = b
            if btype != 0:
                continue
            if y0 < y_min or y0 >= y_max:
                continue
            ts = txt.strip()
            if not ts:
                continue
            if SKIP_RE.match(ts):
                continue
            if SECTION_HEADER_RE.match(ts):
                continue
            # 有非 header、非 skip 的内容 → 有效题目内容
            return True
    except Exception:
        pass
    return False


def _collect_question_slices(src_doc, questions, q_idx, paper_type):
    """
    返回题目所跨的"源页片段"列表，每项：
      (page_obj, clip_rect)  —— clip_rect 单位：PDF pt
    不产生任何像素数据。

    支持格式：
    - Cambridge MCQ / Structured：内容区 x=30~pw-15
    - Edexcel（含 IAL/IGCSE）：内容区 x=38~548（避开两侧装饰条）
    - Edexcel Maths (所有单元 P1-P4/S1/M1 等)：
        * 每题可能跨多页（含 "Question N continued" 续页）
        * 首页从 y_start 截取到页面内容底部
        * 续页从页面顶部内容区截取到内容底部（或到下一题 y_start）
        * bottom 用 _find_edexcel_maths_question_bottom 精确找到 marks 下边界
    """
    # ── Edexcel Maths：支持跨页题目 ──
    if paper_type == 'edexcel_maths':
        q      = questions[q_idx]
        pw_ref = src_doc[q["page_idx"]].rect.width

        # 计算题目占用的页面范围
        pg_start = q["page_idx"]
        if q_idx + 1 < len(questions):
            pg_end = questions[q_idx + 1]["page_idx"]
            # 如果下一题在同一页，当前题只占 pg_start
            if pg_end == pg_start:
                pg_end = pg_start
        else:
            # 最后一题：找最后有内容的页
            pg_end = _find_last_content_page(src_doc, pg_start)

        slices = []
        for pg_i in range(pg_start, pg_end + 1):
            page   = src_doc[pg_i]
            pw, ph = page.rect.width, page.rect.height

            # Task4-edexcel_maths: 跳过答题页（纯横线页/空白页）
            # 首页（pg_start）必须保留（含题目），尾页也保留（含 marks 截断逻辑）
            if pg_i != pg_start and pg_i != pg_end:
                if _is_answer_writing_page(page):
                    continue
            # 尾页若是答题页（当前题 marks 结束后全是横线）
            # → _find_edexcel_maths_question_bottom 会截到 marks，不含横线，无需额外跳过
            # 但若尾页完全是答题页（没有任何 marks）→ 跳过
            if pg_i == pg_end and pg_i != pg_start:
                if _is_answer_writing_page(page):
                    continue

            # 横向裁剪：左避开边框线，右检测内容边界
            left  = 42
            right = min(pw - 36, 560)
            _rl = _detect_right_content_limit(page, pw, ph,
                                              sample_y0=44, sample_y1=ph - 40)
            if _rl < right:
                right = _rl

            # 纵向裁剪
            if pg_i == pg_start:
                y_start = q.get("y_start", 0)
                top = max(0, y_start - 8) if y_start > 10 else 48
            else:
                top = 44    # 续页：从页面内容起始处（跳过页眉）

            if pg_i == pg_end and q_idx + 1 < len(questions):
                nq = questions[q_idx + 1]
                if nq["page_idx"] == pg_end:
                    # 下一题在同一页：截到 min(题干底部, 下一题题号) 确保不含下一题内容
                    y_next = max(top + 20, nq["y_start"] - 8)
                    # ★ 修复：若 y_next 极接近 top（下一题紧接页面顶部），
                    #   说明该 pg_end 页面几乎没有当前题内容（全属下一题），跳过。
                    #   阈值 45pt：top=44 时 y_next≤89 就跳过，避免切入下一题注释区
                    if pg_i != pg_start and y_next <= top + 45:
                        continue
                    stem_b = _find_edexcel_maths_question_bottom(page, ph)
                    bottom = min(stem_b, y_next)
                else:
                    bottom = _find_edexcel_maths_question_bottom(page, ph)
            else:
                bottom = _find_edexcel_maths_question_bottom(page, ph)

            # 退化保护：如果 bottom 异常小，用题干检测兜底（而非直接 ph-25）
            if bottom <= top + 10:
                fallback = _find_edexcel_maths_question_bottom(page, ph)
                bottom = fallback if fallback > top + 10 else ph - 25

            if bottom > top + 10:
                slices.append((page, fitz.Rect(left, top, right, bottom)))

        return slices

    # ── 其他格式（cambridge structured / edexcel 大题）──
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

    # ── edexcel_economics 专项：检测 Source Booklet 开始页，限制 pg_end ──
    # U4 等 QP 将 Source Booklet 内嵌于 PDF 末尾（含图表页），不应纳入题目切割范围
    # 扫描文档，找到 Source Booklet 封面/内容页，pg_end 不超过其前一页
    if paper_type == 'edexcel_economics':
        _sb_start = None
        for _pi in range(src_doc.page_count):
            _pt = src_doc[_pi].get_text()
            if ('Sources for use with Section' in _pt or
                    'Source for use with Section' in _pt):
                _sb_start = _pi
                break
            # Source Booklet 封面特征：同时含 'Source Booklet' 和 'Do not return'
            if 'Source Booklet' in _pt and 'Do not return' in _pt:
                _sb_start = _pi
                break
        if _sb_start is not None and pg_end >= _sb_start:
            pg_end = _sb_start - 1
            y_end  = None  # pg_end 变了，y_end 不再适用（Source Booklet 前的整页）
            if pg_end < pg_start:
                # 题目本身就在 Source Booklet 之后（极端情况），不切割
                return []

    # Edexcel 有左右两侧的 "DO NOT WRITE" 装饰条，裁掉边缘
    is_edexcel = paper_type in ('edexcel', 'edexcel_mcq', 'edexcel_economics')

    slices = []
    for pg_i in range(pg_start, pg_end + 1):
        page = src_doc[pg_i]
        pw, ph = page.rect.width, page.rect.height

        # 跳过中间的纯答题页（空白/横线页），不纳入导出切片
        # 首页（pg_start）永远保留（含题干），尾页（pg_end）也保留（含题目截止位）
        if pg_i != pg_start and pg_i != pg_end:
            if _is_answer_writing_page(page):
                continue
            # edexcel_economics 专项：跳过含大量点线答题区的中间页
            # Section B/C 结构题的 answer pages 含密集 dotted lines "...............",
            # 这些是学生答题区，不是题目内容，应跳过（首页和末页除外）
            if paper_type == 'edexcel_economics':
                if _is_econ_dotted_answer_page(page, ph):
                    continue

        # edexcel_economics 专项：pg_end（非首页）如果是纯答题虚线页，也跳过
        # Section D essay题：Q13/Q14 只截取 page 22 的题干，后续 answer pages 均跳过
        if pg_i == pg_end and pg_i != pg_start and paper_type == 'edexcel_economics':
            if _is_econ_dotted_answer_page(page, ph):
                continue

        if is_edexcel:
            left, right = 36, min(pw - 36, 550)   # 避开 Edexcel 两侧装饰条
        else:
            left, right = 30, pw - 15

        if pg_i == pg_start and pg_i == pg_end:
            # 同一页：先用题干底部检测，再和 y_end 取 min（确保不含下一题）
            # 传入 y_min=y_top，让 _find_question_stem_bottom 忽略 y_top 以上的 answer-zone 标志
            # （Section header 全局指令如 "Write your answer..." 在 y≈116，位于题目起始之上）
            # 传入 y_max=y_end，Phase 3 marks 扫描只看本题范围，不扫下一题的 Total 行
            top = y_top
            stem_bottom = _find_question_stem_bottom(page, ph, paper_type, y_min=y_top, y_max=y_end)
            if y_end is not None:
                if stem_bottom <= y_top:
                    # stem_bottom 仍低于题目起始（极端情况：页面无任何 answer-zone 信号且 fallback 失效）
                    # → 直接用 y_end 作为本题底部
                    bottom = min(ph, y_end)
                else:
                    bottom = min(stem_bottom, min(ph, y_end))
            else:
                if stem_bottom <= y_top:
                    # 无下一题且 stem_bottom 异常 → fallback 到页面内容底部
                    bottom = ph - 25
                else:
                    bottom = stem_bottom
        elif pg_i == pg_start:
            # 首页：从题号到题干底部（不含本页的答题区）
            # 同时：如果下一题也在本页（pg_end==pg_start 已处理），此处 pg_end>pg_start，
            # 首页的 stem_bottom 不受 y_end 约束（y_end 在其他页）
            # 传入 y_min=y_top，排除题目起始以上的 Section header 全局指令
            top    = y_top
            stem_bottom = _find_question_stem_bottom(page, ph, paper_type, y_min=y_top)
            if stem_bottom <= y_top:
                # stem_bottom 异常（排除了 header 指令后 fallback 扫描内容块）
                # 直接用内容块扫描的 fallback 值（_find_question_stem_bottom 已尝试过）
                # 此时 stem_bottom 应该是 ph-25（_find_question_stem_bottom fallback），保留
                bottom = stem_bottom if stem_bottom > y_top + 10 else ph - 25
            else:
                bottom = stem_bottom
        elif pg_i == pg_end:
            # 末页：从页顶到 min(题干底部, 下一题题号)
            top = 50 if is_edexcel else 55
            stem_bottom = _find_question_stem_bottom(page, ph, paper_type)
            if y_end is not None:
                # 末页有 y_end（下一题在此页）：取两者中更小的，防止截入下一题
                bottom = min(stem_bottom, min(ph, y_end))
                # ★ 修复1：若下一题紧接页面顶部（y_end 极小），该页几乎没有当前题内容，跳过
                if pg_i != pg_start and bottom <= top + 45:
                    continue
                # ★ 修复2：对 edexcel_economics，若当前题在此末页上无实质内容（只有 Section header），跳过
                # 判断标准：top 到 y_end 之间不存在属于当前题的文字块（排除 Section header 类）
                if pg_i != pg_start and paper_type == 'edexcel_economics':
                    if not _has_question_content_in_range(page, top, y_end):
                        continue
            else:
                bottom = stem_bottom
        else:
            # 中间页（多页大题的中间部分）：题干底部截止（不含答题区）
            top    = 50 if is_edexcel else 55
            bottom = _find_question_stem_bottom(page, ph, paper_type)

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
                         show_header=False, q_meta=None, draw_logo=True,
                         fit_to_width=True):
    """
    把一段 JPEG 图像放入输出 PDF 页的 area_rect 区域。
    白色背景，干净排版：
      - 顶部信息行：题号 | ★★★ 难度 | 知识点... | 年份 | Logo（右对齐）
      - 无蓝色背景色块，仅用细线和文字颜色区分
      - 图片紧跟信息行下方，左对齐，宽度铺满
    q_meta: dict {difficulty, topics, exam_date} 或 None
    fit_to_width: True（默认）→ 按宽度铺满，不受高度约束（大题一题一页模式）
                  False → min(w_scale, h_scale)，防止超出区域（MCQ 多题共页模式）
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

    if fit_to_width:
        # 大题一题一页模式：始终按宽度铺满，不受高度约束
        # 截掉答题区后图片变矮，不能因为高度小就缩小图片
        scale  = img_area_w / img_w
    else:
        # MCQ 多题共页模式：同时约束宽高，防止单题超出分配区域
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


def _b64_to_jpeg_bytes(b64_str: str) -> bytes | None:
    """将 base64 字符串解码并转换为 JPEG bytes；失败返回 None。"""
    try:
        import base64 as _b64m
        raw = _b64m.b64decode(b64_str)
        from PIL import Image as _PILImg
        im = _PILImg.open(io.BytesIO(raw))
        buf = io.BytesIO()
        im.convert('RGB').save(buf, format='JPEG', quality=88)
        return buf.getvalue()
    except Exception:
        return None


def _place_answer_on_page(out_doc, ans_jpeg, ans_w, ans_h,
                           PW, PH, M, GAP, label, page_no=None, total_pages=None):
    """
    在 out_doc 中新建一页放置答案图片，带"答案 Answer"标签头栏。
    page_no / total_pages: 若多页答案，在头栏显示页码，如 "答案 Answer — Q14 (第2页/共3页)"
    """
    ANS_BAND_H = 22   # 答案头栏高度（pt）
    ANS_GAP    = 4    # 头栏与图片间距
    AVAIL_W    = PW - 2 * M
    AVAIL_H    = PH - 2 * M - ANS_BAND_H - ANS_GAP - GAP

    page = out_doc.new_page(width=PW, height=PH)

    # 答案头栏（浅绿色背景）
    band_rect = fitz.Rect(M, M, PW - M, M + ANS_BAND_H)
    C_ANS_BG  = (0.88, 0.96, 0.90)   # 浅绿
    C_ANS_TXT = (0.10, 0.50, 0.25)   # 深绿
    page.draw_rect(band_rect, color=C_ANS_BG, fill=C_ANS_BG)
    if page_no is not None and total_pages is not None and total_pages > 1:
        hdr_text = f'答案 Answer — {label}  (第{page_no}页/共{total_pages}页)'
    else:
        hdr_text = f'答案 Answer — {label}'
    page.insert_text(
        (M + 8, M + ANS_BAND_H - 7),
        hdr_text,
        fontsize=11, color=C_ANS_TXT, fontname='helv'
    )

    # 答案图片
    img_y0 = M + ANS_BAND_H + ANS_GAP
    scale  = min(AVAIL_W / max(ans_w, 1), AVAIL_H / max(ans_h, 1))
    draw_w = ans_w * scale
    draw_h = ans_h * scale
    img_rect = fitz.Rect(M, img_y0, M + draw_w, img_y0 + draw_h)
    page.insert_image(img_rect, stream=io.BytesIO(ans_jpeg))
    return page


def _insert_source_pages(out_doc, source_pages, PW, PH, M, GAP, label):
    """
    将 source_pages（Section C 材料页）插入 out_doc，每页单独一 PDF 页。
    使用蓝色「材料」头栏样式（区别于绿色答案头栏），归属于题目部分。
    """
    import base64 as _b64src
    from PIL import Image as _PILSrc

    BAND_H  = 22
    SRC_GAP = 4
    AVAIL_W = PW - 2 * M
    AVAIL_H = PH - 2 * M - BAND_H - SRC_GAP - GAP
    # 蓝色系（与答案绿色区分）
    C_SRC_BG  = (0.88, 0.95, 1.00)   # 浅蓝
    C_SRC_TXT = (0.03, 0.42, 0.63)   # 深蓝

    total = len(source_pages)
    for pg_idx, pg in enumerate(source_pages):
        try:
            raw  = _b64src.b64decode(pg['b64'])
            img  = _PILSrc.open(io.BytesIO(raw))
            w, h = img.size
            buf  = io.BytesIO()
            img.convert('RGB').save(buf, format='JPEG', quality=88)
            jpeg = buf.getvalue()

            page = out_doc.new_page(width=PW, height=PH)

            # 蓝色材料头栏
            band_rect = fitz.Rect(M, M, PW - M, M + BAND_H)
            page.draw_rect(band_rect, color=C_SRC_BG, fill=C_SRC_BG)
            if total > 1:
                hdr_text = f'题目材料 Sources — {label}  (第{pg_idx+1}页/共{total}页)'
            else:
                hdr_text = f'题目材料 Sources — {label}'
            page.insert_text(
                (M + 8, M + BAND_H - 7),
                hdr_text,
                fontsize=11, color=C_SRC_TXT, fontname='helv'
            )

            # 材料图片
            img_y0 = M + BAND_H + SRC_GAP
            scale  = min(AVAIL_W / max(w, 1), AVAIL_H / max(h, 1))
            draw_w = w * scale
            draw_h = h * scale
            img_rect = fitz.Rect(M, img_y0, M + draw_w, img_y0 + draw_h)
            page.insert_image(img_rect, stream=io.BytesIO(jpeg))
        except Exception as _se:
            print(f'[pdf_export] source page {pg_idx+1} error: {_se}')


def _insert_answer_pages(out_doc, q_obj, PW, PH, M, GAP, label):
    """
    将 q_obj 的答案插入 out_doc（多页时逐页各占一页，单页同原逻辑）。
    优先使用 q_obj['answer_pages']（多页数组）；若无则退回 answer_b64（第1页）。
    """
    import base64 as _b64x
    from PIL import Image as _PILAns

    ans_pages = q_obj.get('answer_pages')  # None 或 [{b64,w,h},...]

    if ans_pages and isinstance(ans_pages, list) and len(ans_pages) > 1:
        # ── 多页：逐页各占一 PDF 页 ──
        total = len(ans_pages)
        for pg_idx, pg in enumerate(ans_pages):
            try:
                raw  = _b64x.b64decode(pg['b64'])
                img  = _PILAns.open(io.BytesIO(raw))
                w, h = img.size
                # 确保输出 JPEG 字节
                buf = io.BytesIO()
                img.convert('RGB').save(buf, format='JPEG', quality=88)
                jpeg = buf.getvalue()
                _place_answer_on_page(out_doc, jpeg, w, h,
                                      PW, PH, M, GAP, label,
                                      page_no=pg_idx + 1, total_pages=total)
            except Exception as _pe:
                print(f'[pdf_export] answer page {pg_idx+1} error: {_pe}')
    else:
        # ── 单页：原有逻辑 ──
        ans_b64 = q_obj.get('answer_b64', '')
        if not ans_b64:
            return
        ans_jpeg = _b64_to_jpeg_bytes(ans_b64)
        if not ans_jpeg:
            return
        try:
            img     = _PILAns.open(io.BytesIO(ans_jpeg))
            ans_w, ans_h = img.size
            _place_answer_on_page(out_doc, ans_jpeg, ans_w, ans_h,
                                  PW, PH, M, GAP, label)
        except Exception as _ae:
            print(f'[pdf_export] answer page error: {_ae}')



def _export_one_per_page(out_doc, src_doc, questions, q_nums, dpi,
                          paper_type, PW, PH, M, HH, GAP, FS, progress_cb=None,
                          seq_start=0):
    """
    大题模式：每题独占一页（或多页）。
    seq_start: 全局导出序号起始值（0-based），用于头栏题号显示。
    如果题目对象含 answer_b64，则在题目页之后插入答案页。
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

        # ── Task 2: 若题目有 source_pages（题目材料），先插入材料页 ──
        # source_pages 可来自 Q12（Section C 材料）或 Q7（U3 Section B 材料）
        source_pages = q_obj.get('source_pages')
        q_num_cur = q_obj.get('q_num')
        if source_pages and isinstance(source_pages, list) and len(source_pages) > 0 and q_num_cur in (7, 12):
            _insert_source_pages(out_doc, source_pages, PW, PH, M, GAP, label)

        # ── Task 2: 若题目有答案图，在题目页后附加答案页（支持多页）──
        if q_obj.get('answer_b64') or q_obj.get('answer_pages'):
            _insert_answer_pages(out_doc, q_obj, PW, PH, M, GAP, label)

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
                            q_meta=q_meta_mc, fit_to_width=False)
        del jpeg

        cur_y += actual_draw_h + 2 * GAP + ITEM_GAP

        # ── 答案页：若题目有答案，在当前MCQ打包页结束后立即插入独立答案页 ──
        # 每道 MCQ 答案各占一页（不打包），与截图格式一致（每行一题独立展示）
        q_ans_label = f'Q{export_seq:02d}'
        if q_obj_mc.get('answer_b64') or q_obj_mc.get('answer_pages'):
            # 答案页新起一页，本 MCQ 打包页状态重置（答案页后继续新页打包）
            _insert_answer_pages(out_doc, q_obj_mc, PW, PH, M, GAP, q_ans_label)
            # 答案页后需新起一页继续后续题目
            cur_page = None
            cur_y    = M

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
                                    label, HH, GAP, FS, q_meta=q_meta,
                                    fit_to_width=False)
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
    _LIBRARY_DIR = os.path.join(storage.local_root(), 'library')
    os.makedirs(_LIBRARY_DIR, exist_ok=True)
else:
    _LIBRARY_DIR = None  # R2 模式不使用本地 library 目录

_EXAM_BOARDS  = ['Edexcel', 'CAIE', 'AQA', 'OCR', 'IB', 'AP', 'BPHO', '竞赛 Competition']
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
    'OCR':     ['数学 Maths',    '高数 Further Maths', '物理 Physics',
                '化学 Chemistry','生物 Biology',        '经济 Economics',
                '商业 Business', '会计 Accounting'],
    'IB':      ['数学 Maths',    '物理 Physics',        '化学 Chemistry',
                '生物 Biology',  '经济 Economics',      '商业 Business'],
    'AP':      ['数学 Maths',    '物理 Physics',        '化学 Chemistry',
                '生物 Biology',  '经济 Economics',      '商业 Business'],
    'BPHO':    ['竞赛物理 Physics (BPhO)', '其他 Other'],
    '竞赛 Competition': ['物理竞赛 Physics (BPhO)', '数学竞赛 Maths (BMO)',
                         '化学竞赛 Chemistry', '生物竞赛 Biology', '其他 Other'],
}
def _lib_key_prefix(board: str, subject: str, wb_id: str = '') -> str:
    """返回图书馆存储 key 前缀（R2 key 或本地目录路径）。"""
    safe_board   = re.sub(r'[^A-Za-z0-9_\-\u4e00-\u9fff ()]', '', board).strip()
    safe_subject = re.sub(r'[^A-Za-z0-9_\-\u4e00-\u9fff ()]', '', subject).strip()
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
    支持本地模式和 R2 模式。R2 模式使用并发请求加速。
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed

    def _load_manifest_r2(args):
        """并发任务：加载单个 manifest.json，返回 (board, subject, wb_info) 或 None"""
        board, subject, prefix, wb_id = args
        mfest_key = f'{prefix}{wb_id}/manifest.json'
        m = storage.load_json(mfest_key)
        if not m:
            return None
        return (board, subject, {
            'id':           wb_id,
            'title':        m.get('title', wb_id),
            'count':        m.get('count', 0),
            'created_at':   m.get('created_at', ''),
            'sort_order':   m.get('sort_order', 'default'),
            'maths_unit':   m.get('maths_unit', ''),
            'syllabus_type':m.get('syllabus_type', ''),
            'exam_date':    m.get('exam_date', ''),
        })

    tree = []
    if storage.is_r2_mode():
        # ── R2 模式：Step1 并发扫描所有 prefix（新旧路径）──
        # 构建所有需要扫描的 (board, subject, prefix) 组合
        scan_targets = []
        seen_target = set()
        for board in _EXAM_BOARDS:
            for subject in _SUBJECTS_MAP.get(board, []):
                safe_b_new = re.sub(r'[^A-Za-z0-9_\-\u4e00-\u9fff ()]', '', board).strip()
                safe_s_new = re.sub(r'[^A-Za-z0-9_\-\u4e00-\u9fff ()]', '', subject).strip()
                safe_b_old = re.sub(r'[^A-Za-z0-9_\-\u4e00-\u9fff ]', '', board).strip()
                safe_s_old = re.sub(r'[^A-Za-z0-9_\-\u4e00-\u9fff ]', '', subject).strip()
                for sb, ss in [(safe_b_new, safe_s_new), (safe_b_old, safe_s_old)]:
                    pf = f'library/{sb}/{ss}/'
                    key = (board, subject, pf)
                    if key not in seen_target:
                        seen_target.add(key)
                        scan_targets.append((board, subject, pf))

        # 并发执行所有 list_prefix
        def _list_one(args):
            board, subject, prefix = args
            keys = storage.list_prefix(prefix)
            tasks = []
            seen_wb = set()
            for k in keys:
                parts = k.split('/')
                if len(parts) >= 5 and parts[-1] == 'manifest.json':
                    wb_id = parts[3]
                    if wb_id not in seen_wb:
                        seen_wb.add(wb_id)
                        tasks.append((board, subject, prefix, wb_id))
            return tasks

        manifest_tasks = []
        seen_wb_global = set()   # (board, subject, wb_id) 全局去重
        with ThreadPoolExecutor(max_workers=32) as _ex:
            futures = [_ex.submit(_list_one, t) for t in scan_targets]
            for fut in futures:
                for task in fut.result():
                    b, s, pf, wid = task
                    dedup_key = (b, s, wid)
                    if dedup_key not in seen_wb_global:
                        seen_wb_global.add(dedup_key)
                        manifest_tasks.append(task)

        # Step2：并发读取所有 manifest（最多32线程）
        results_map = {}  # (board, subject) -> [wb_info, ...]
        if manifest_tasks:
            with ThreadPoolExecutor(max_workers=32) as _ex:
                futures = {_ex.submit(_load_manifest_r2, t): t for t in manifest_tasks}
                for fut in _as_completed(futures):
                    res = fut.result()
                    if res:
                        b, s, wb_info = res
                        results_map.setdefault((b, s), []).append(wb_info)

        # Step3：按固定顺序组装树（保持board/subject顺序）
        for board in _EXAM_BOARDS:
            subjects_list = []
            for subject in _SUBJECTS_MAP.get(board, []):
                workbooks = results_map.get((board, subject), [])
                workbooks.sort(key=lambda x: x.get('created_at', ''), reverse=True)
                subjects_list.append({'subject': subject, 'workbooks': workbooks})
            tree.append({'board': board, 'subjects': subjects_list})

    else:
        # ── 本地模式：遍历本地目录 ──
        for board in _EXAM_BOARDS:
            subjects_list = []
            for subject in _SUBJECTS_MAP.get(board, []):
                workbooks = []
                safe_board   = re.sub(r'[^A-Za-z0-9_\-\u4e00-\u9fff ()]', '', board).strip()
                safe_subject = re.sub(r'[^A-Za-z0-9_\-\u4e00-\u9fff ()]', '', subject).strip()
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
    try:
        return _library_save_impl()
    except Exception as _top_e:
        import traceback as _tb
        app.logger.error(f'[library_save] 未捕获异常: {_tb.format_exc()}')
        return jsonify({'error': f'服务端错误: {str(_top_e)}'}), 500

def _library_save_impl():
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

    # ── 覆盖模式：若前端传入 wb_id，先删除旧目录再用同一 wb_id 重写 ──
    overwrite_wb_id = data.get('wb_id', '').strip()
    if overwrite_wb_id:
        # 删除旧题册目录（本地模式）或 R2 前缀（R2 模式）
        # 旧 board/subject 可能与新的不同，需要遍历查找
        if not storage.is_r2_mode():
            lib_root = _LIBRARY_DIR
            for old_board in (os.listdir(lib_root) if os.path.isdir(lib_root) else []):
                board_dir = os.path.join(lib_root, old_board)
                if not os.path.isdir(board_dir):
                    continue
                for old_subj in os.listdir(board_dir):
                    subj_dir = os.path.join(board_dir, old_subj)
                    old_wb_dir = os.path.join(subj_dir, overwrite_wb_id)
                    if os.path.isdir(old_wb_dir):
                        import shutil as _shutil
                        _shutil.rmtree(old_wb_dir, ignore_errors=True)
        else:
            # R2 模式：删除旧前缀
            for old_board in _EXAM_BOARDS:
                old_prefix = _lib_key_prefix(old_board, subject, overwrite_wb_id)
                storage.delete_prefix(old_prefix + '/')
        wb_id = overwrite_wb_id
    else:
        wb_id = str(uuid.uuid4())[:8]

    wb_prefix  = _lib_key_prefix(board, subject, wb_id)   # R2 key 前缀 或 本地目录
    # 本地模式需要创建目录
    if not storage.is_r2_mode():
        os.makedirs(wb_prefix, exist_ok=True)

    saved_q = []
    # 按 (session_obj, file_idx) 分组，批量打开 PDF（避免同一文件反复 open/close）
    from collections import defaultdict
    from concurrent.futures import ThreadPoolExecutor as _SaveTPE, as_completed as _save_ac
    import hashlib as _hl_save
    import base64 as _b64

    file_idx_map = defaultdict(list)   # (sess_id_key, file_idx) -> [(list_pos, q_dict, sess_obj)]
    img_results  = {}                  # list_pos -> (img_file, img_w, img_h, ans_file, img_hash)
    q_has_b64    = {}                  # list_pos -> True（步骤2降级时标记，供 late_b64_tasks 使用）

    # 构建 session_id -> sess 映射（支持 session_id_override）
    _sess_cache = {}
    def _get_sess_cached(sid):
        if sid not in _sess_cache:
            _sess_cache[sid] = _get_session(sid) if sid else None
        return _sess_cache[sid]

    # ── 任务分类 ──
    # type-A: 已有 img_bytes_b64（直接压缩写入）
    # type-B: 有 _src_img_file（并发从 R2/本地读图，再压缩写入）
    # type-C: 走 PDF 裁图（session 裁图，保持原有串行逻辑）
    type_a_tasks = []   # [(i, q)]
    type_b_tasks = []   # [(i, q, src_prefix, src_img_file, src_ans_file)]

    for i, q in enumerate(questions):
        b64 = q.get('img_bytes_b64', '')
        if b64:
            type_a_tasks.append((i, q))
        elif q.get('_src_img_file') and not q.get('_img_replaced'):
            # 题库来源且未替换：并发从 R2 读图
            src_wb_id   = q.get('_src_wb_id', '')
            src_board   = q.get('_src_board', '')
            src_subject = q.get('_src_subject', '')
            src_prefix  = _lib_key_prefix(src_board, src_subject, src_wb_id) if src_wb_id else ''
            src_ans_file = q.get('_src_ans_file', '') if not q.get('_answer_replaced') else ''
            type_b_tasks.append((i, q, src_prefix, q['_src_img_file'], src_ans_file))
        else:
            # PDF 裁图路径（type-C）
            q_sid = q.get('session_id_override', '') or session_id
            q_sess = _get_sess_cached(q_sid)
            file_idx = int(q.get('file_idx_override', q.get('file_idx', q.get('gIdx', 0))))
            file_idx_map[(q_sid, file_idx)].append((i, q, q_sess))

    # ── 统一并发任务函数（type-A 和 type-B 共用）──
    def _process_one(task_type, i, q,
                     src_prefix='', src_img_file='', src_ans_file=''):
        """
        读取/解码图片 → 压缩 JPEG → 写入 R2/本地。
        返回 (i, img_fname, w, h, ans_fname, img_hash)
        task_type: 'a'=已有b64  'b'=从R2读取
        """
        img_fname = f'q_{i+1:03d}.jpg'
        ans_fname = ''
        img_hash  = ''
        try:
            # ── 获取题目图片原始字节 ──
            if task_type == 'a':
                raw_img = _b64.b64decode(q.get('img_bytes_b64', ''))
            else:
                # type-b: 从 R2/本地读取
                raw_img = None
                if src_prefix:
                    if storage.is_r2_mode():
                        raw_img = storage.load_bytes(f'{src_prefix}/{src_img_file}')
                    else:
                        _fp = os.path.join(src_prefix, src_img_file)
                        if os.path.isfile(_fp):
                            with open(_fp, 'rb') as _fh:
                                raw_img = _fh.read()
                else:
                    # 没有 src_prefix，遍历已知路径查找
                    for _b2 in _EXAM_BOARDS:
                        for _s2 in _SUBJECTS_MAP.get(_b2, []):
                            src_wb_id2 = q.get('_src_wb_id', '')
                            if not src_wb_id2:
                                continue
                            _pf = _lib_key_prefix(_b2, _s2, src_wb_id2)
                            _raw_try = (storage.load_bytes(f'{_pf}/{src_img_file}')
                                        if storage.is_r2_mode() else
                                        (open(os.path.join(_pf, src_img_file), 'rb').read()
                                         if os.path.isfile(os.path.join(_pf, src_img_file)) else None))
                            if _raw_try:
                                raw_img = _raw_try
                                src_prefix = _pf  # 记录找到的前缀，答案用同一路径
                                break
                        if raw_img:
                            break
                if not raw_img:
                    return (i, '', 0, 0, '', '')

            # ── 压缩 JPEG 并写入 ──
            from PIL import Image as _PIL
            _im = _PIL.open(io.BytesIO(raw_img))
            buf = io.BytesIO()
            _im.convert('RGB').save(buf, format='JPEG', quality=88)
            jpeg_data = buf.getvalue()
            img_hash  = _hl_save.md5(jpeg_data).hexdigest()
            if storage.is_r2_mode():
                storage.store_bytes(f'{wb_prefix}/{img_fname}', jpeg_data)
            else:
                with open(os.path.join(wb_prefix, img_fname), 'wb') as f:
                    f.write(jpeg_data)
            w, h = _im.width, _im.height

            # ── 答案图片 ──
            # type-a: 用 q['answer_b64']；type-b: 从 R2 读 src_ans_file
            if task_type == 'a':
                ans_raw_data = _b64.b64decode(q.get('answer_b64', '')) if q.get('answer_b64') else b''
            else:
                ans_raw_data = b''
                if src_ans_file and src_prefix:
                    if storage.is_r2_mode():
                        _ar = storage.load_bytes(f'{src_prefix}/{src_ans_file}')
                    else:
                        _afp = os.path.join(src_prefix, src_ans_file)
                        _ar = open(_afp, 'rb').read() if os.path.isfile(_afp) else None
                    ans_raw_data = _ar or b''
                elif q.get('answer_b64'):
                    # answer_b64 由前端传来（替换答案或非R2来源）
                    ans_raw_data = _b64.b64decode(q['answer_b64'])

            if ans_raw_data:
                ans_fname = f'q_{i+1:03d}_ans.jpg'
                try:
                    from PIL import Image as _PIL2
                    _aim = _PIL2.open(io.BytesIO(ans_raw_data))
                    abuf = io.BytesIO()
                    _aim.convert('RGB').save(abuf, format='JPEG', quality=88)
                    if storage.is_r2_mode():
                        storage.store_bytes(f'{wb_prefix}/{ans_fname}', abuf.getvalue())
                    else:
                        with open(os.path.join(wb_prefix, ans_fname), 'wb') as f:
                            f.write(abuf.getvalue())
                except Exception:
                    ans_fname = ''

            return (i, img_fname, w, h, ans_fname, img_hash)
        except Exception:
            return (i, '', 0, 0, '', '')

    # ── 并发执行 type-A + type-B（合并进同一线程池，最大并发20）──
    all_concurrent_tasks = (
        [('a', i, q, '', '', '') for (i, q) in type_a_tasks] +
        [(('b', i, q, sp, sif, saf)) for (i, q, sp, sif, saf) in type_b_tasks]
    )
    if all_concurrent_tasks:
        with _SaveTPE(max_workers=20) as _ex:
            futs = {
                _ex.submit(_process_one, tt, i, q, sp, sif, saf): i
                for (tt, i, q, sp, sif, saf) in all_concurrent_tasks
            }
            for fut in _save_ac(futs):
                res_i, img_f, w, h, ans_f, i_hash = fut.result()
                img_results[res_i] = (img_f, w, h, ans_f, i_hash)

    # type-B 失败的题目降级走裁图流程
    for (i, q, sp, sif, saf) in type_b_tasks:
        if img_results.get(i, ('',))[0] == '':
            q_sid = q.get('session_id_override', '') or session_id
            q_sess = _get_sess_cached(q_sid)
            file_idx = int(q.get('file_idx_override', q.get('file_idx', q.get('gIdx', 0))))
            file_idx_map[(q_sid, file_idx)].append((i, q, q_sess))
            app.logger.warning(f'[library_save] type-B 读取失败 i={i}, 降级裁图')

    # ── 步骤2：从 PDF session 裁图（模式A）── 裁图串行，R2上传并发
    _r2_upload_queue = []   # [(key, data)] 待并发上传

    for (q_sid, file_idx), items in file_idx_map.items():
        # items 中每个元素是 (i, q, q_sess)
        # 取第一个的 q_sess（同组 session 相同）
        q_sess = items[0][2] if items else None
        if not q_sess or file_idx >= len(q_sess):
            # session 不可用时：尝试用 fallback img（img_bytes_b64_fallback）
            for (i, q, _qs) in items:
                fb = q.get('img_bytes_b64_fallback', '')
                if fb:
                    q['img_bytes_b64'] = fb  # 降级为 fallback
                    q_has_b64[i] = True
                else:
                    img_results[i] = ('', 0, 0, '', '')
            continue

        group      = q_sess[file_idx]
        save_path  = group['path']
        paper_type = group['paper_type']
        questions_meta = group['questions']

        if not os.path.exists(save_path):
            # PDF 文件丢失：尝试 fallback
            for (i, q, _qs) in items:
                fb = q.get('img_bytes_b64_fallback', '')
                if fb:
                    q['img_bytes_b64'] = fb
                    q_has_b64[i] = True
                else:
                    img_results[i] = ('', 0, 0, '', '')
            continue

        try:
            doc = fitz.open(save_path)
            for (i, q, _qs) in items:
                q_num = int(q.get('q_num', 0))
                q_idx = next((qi for qi, qo in enumerate(questions_meta)
                              if qo['q_num'] == q_num), None)
                if q_idx is None:
                    fb = q.get('img_bytes_b64_fallback', '')
                    if fb:
                        q['img_bytes_b64'] = fb
                        q_has_b64[i] = True
                    else:
                        img_results[i] = ('', 0, 0, '', '')
                    continue
                try:
                    img_bytes, w, h = crop_question_image(
                        doc, questions_meta, q_idx,
                        dpi=300, paper_type=paper_type
                    )
                    img_file = f'q_{i+1:03d}.jpg'
                    from PIL import Image as _PIL
                    _im = _PIL.open(io.BytesIO(img_bytes))
                    buf = io.BytesIO()
                    _im.convert('RGB').save(buf, format='JPEG', quality=88)
                    jpeg_data = buf.getvalue()

                    if storage.is_r2_mode():
                        _r2_upload_queue.append((f'{wb_prefix}/{img_file}', jpeg_data))
                    else:
                        with open(os.path.join(wb_prefix, img_file), 'wb') as f:
                            f.write(jpeg_data)

                    # 答案图片
                    ans_fname = ''
                    q_meta_obj = questions_meta[q_idx]
                    ans_b64 = q_meta_obj.get('answer_b64', '') or q.get('answer_b64', '')
                    if ans_b64:
                        ans_fname = f'q_{i+1:03d}_ans.jpg'
                        try:
                            ans_raw = _b64.b64decode(ans_b64)
                            from PIL import Image as _PIL3
                            _aim = _PIL3.open(io.BytesIO(ans_raw))
                            abuf = io.BytesIO()
                            _aim.convert('RGB').save(abuf, format='JPEG', quality=88)
                            ans_data = abuf.getvalue()
                            if storage.is_r2_mode():
                                _r2_upload_queue.append((f'{wb_prefix}/{ans_fname}', ans_data))
                            else:
                                with open(os.path.join(wb_prefix, ans_fname), 'wb') as f:
                                    f.write(ans_data)
                        except Exception:
                            ans_fname = ''

                    img_results[i] = (img_file, w, h, ans_fname,
                                          _hl_save.md5(jpeg_data).hexdigest())  # ★ 裁图路径也顺便算hash
                except Exception:
                    fb = q.get('img_bytes_b64_fallback', '')
                    if fb:
                        q['img_bytes_b64'] = fb
                        q_has_b64[i] = True
                    else:
                        img_results[i] = ('', 0, 0, '', '')
            doc.close()
        except Exception:
            for (i, q, _qs) in items:
                img_results[i] = ('', 0, 0, '', '')

    # fallback 图片需补充处理（img_bytes_b64_fallback 降级为 b64 任务，用 _process_one type-a 处理）
    late_b64_tasks = [(i, q) for i, q in enumerate(questions)
                      if q_has_b64.get(i) and i not in img_results]
    if late_b64_tasks:
        with _SaveTPE(max_workers=16) as _ex:
            futs_late = {
                _ex.submit(_process_one, 'a', i, q, '', '', ''): i
                for (i, q) in late_b64_tasks
            }
            for fut in _save_ac(futs_late):
                res_i, img_f, w, h, ans_f, i_hash = fut.result()
                img_results[res_i] = (img_f, w, h, ans_f, i_hash)

    # R2 模式：并发上传所有裁图结果
    if storage.is_r2_mode() and _r2_upload_queue:
        def _r2_upload(kv):
            storage.store_bytes(kv[0], kv[1])
        with _SaveTPE(max_workers=16) as _ex:
            list(_ex.map(_r2_upload, _r2_upload_queue))


    # ── 步骤3：构建 manifest（hash 已在写入时计算，无需再读 R2）──
    for i, q in enumerate(questions):
        img_file, img_w, img_h, ans_file, img_hash = img_results.get(i, ('', 0, 0, '', ''))
        saved_q.append({
            'seq':        i + 1,
            'q_num':      q.get('q_num', i + 1),
            'file_idx':   int(q.get('file_idx', q.get('gIdx', 0))),
            'difficulty': q.get('difficulty'),
            'topics':     q.get('topics', []),
            'exam_date':  q.get('exam_date', ''),
            'source':     q.get('source', ''),
            'img_file':   img_file,
            'img_hash':   img_hash,   # 图片内容 MD5，写入时顺便算，无需重读
            'img_w':      img_w,
            'img_h':      img_h,
            'ans_file':   ans_file,
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





def _register_workbook_session(wb_id: str, manifest: dict, questions_out: list,
                               wb_prefix: str = '') -> str:
    """
    将已加载的题册注册为一个虚拟 session，使 export_pdf 等接口可以使用。
    每道题的图片以 img_bytes_b64 存入 questions，
    export_pdf / preview_b64 会优先读取这个字段而不尝试打开 PDF。
    返回新的 session_id（'wb_' + wb_id）。

    wb_prefix: 存储前缀（如 'library/Edexcel/...'），用于 lazy 模式下按需下载图片。
               lazy 模式下 questions_out 里 img_bytes_b64=''，需要从 R2/本地补充下载。
    """
    import base64 as _b64
    from concurrent.futures import ThreadPoolExecutor as _RWS_TPE, as_completed as _rws_ac

    sess_id = f'wb_{wb_id}'

    # ── lazy 模式补充下载：并发从 R2/本地下载所有图片 ──
    # 只对 img_bytes_b64='' 且有 _img_file 的条目下载（非 lazy 条目跳过）
    need_download = []
    for i, q in enumerate(questions_out):
        if not q.get('img_bytes_b64') and wb_prefix:
            img_file = q.get('_img_file', '') or q.get('img_file', '')
            if img_file:
                need_download.append((i, img_file, q.get('_ans_file', '') or q.get('ans_file', '')))

    if need_download:
        def _dl_one(args):
            idx, img_f, ans_f = args
            result = {'idx': idx, 'b64': '', 'ans_b64': ''}
            try:
                raw = storage.load_bytes(f'{wb_prefix}/{img_f}')
                if raw:
                    result['b64'] = _b64.b64encode(raw).decode('ascii')
            except Exception:
                pass
            try:
                if ans_f:
                    raw_ans = storage.load_bytes(f'{wb_prefix}/{ans_f}')
                    if raw_ans:
                        result['ans_b64'] = _b64.b64encode(raw_ans).decode('ascii')
            except Exception:
                pass
            return result

        dl_map = {}
        with _RWS_TPE(max_workers=20) as _ex:
            for fut in _rws_ac([_ex.submit(_dl_one, t) for t in need_download]):
                r = fut.result()
                dl_map[r['idx']] = r
        app.logger.info(f'[_register_workbook_session] lazy 补充下载 {len(need_download)} 张图片，'
                        f'成功 {sum(1 for v in dl_map.values() if v["b64"])} 张 wb_id={wb_id!r}')

        # 将下载结果回填到 questions_out（避免修改原始列表，用 index 映射）
        # 注意：questions_out 可能是共享列表，复制一份防止副作用
        questions_out = list(questions_out)
        for i, q in enumerate(questions_out):
            if i in dl_map and dl_map[i]['b64']:
                q = dict(q)  # 浅复制，不修改原字典
                q['img_bytes_b64'] = dl_map[i]['b64']
                if dl_map[i]['ans_b64']:
                    q['answer_b64'] = dl_map[i]['ans_b64']
                questions_out[i] = q

    # 构建虚拟 group（没有真实 PDF 路径，只有图片 base64）
    virt_questions = []
    for i, q in enumerate(questions_out):
        virt_questions.append({
            'q_num':         q.get('q_num', i + 1),
            '_wb_seq':       i,                          # 唯一序号，用于 export 时精确定位
            'page_idx':      0,
            'difficulty':    q.get('difficulty'),
            'topics':        q.get('topics', []),
            'exam_date':     q.get('exam_date', ''),
            'source':        q.get('source', 'workbook'),
            'img_bytes_b64': q.get('img_bytes_b64', ''),
            'img_w':         q.get('img_w', 0),
            'img_h':         q.get('img_h', 0),
            'answer_b64':    q.get('answer_b64', ''),   # Task1: 答案图片
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
        '_wb_prefix':      wb_prefix,   # ★ 持久化路径，update_question_meta 用来写 manifest
        '_wb_id':          wb_id,        # ★ 题册 ID
        '_wb_manifest':    manifest,     # ★ 原始 manifest 引用（dict，直接修改后写回）
    }
    with _multi_sessions_lock:
        _multi_sessions[sess_id] = [virt_group]
    return sess_id


@app.route('/api/library/load/<wb_id>', methods=['GET'])
def library_load(wb_id):
    """
    读取图书馆中的一个题册，返回题目列表（含图片 base64）。
    前端通过 GET /api/library/load/<wb_id>?board=Edexcel&subject=数学Maths
    board/subject 可选：若不提供，则自动搜索 library 目录查找匹配 wb_id 的题册。
    ?lazy=1：懒加载模式，只返回元数据，不含图片base64（大幅提速首屏）
    """
    try:
        return _library_load_impl(wb_id)
    except Exception as _e:
        import traceback
        app.logger.error(f'[library_load] 未捕获异常: {traceback.format_exc()}')
        return jsonify({'ok': False, 'error': f'服务器内部错误: {str(_e)}'}), 500

def _library_load_impl(wb_id):
    import base64 as _b64
    board   = request.args.get('board', '')
    subject = request.args.get('subject', '')
    lazy    = request.args.get('lazy', '0') == '1'  # 懒加载模式

    # ── 自动搜索：若 board/subject 为空，扫描本地目录或 R2 找 wb_id ──
    if not board or not subject:
        if not storage.is_r2_mode():
            found_prefix = None
            lib_root = _LIBRARY_DIR
            if os.path.isdir(lib_root):
                for b in os.listdir(lib_root):
                    bd = os.path.join(lib_root, b)
                    if not os.path.isdir(bd): continue
                    for s in os.listdir(bd):
                        sd = os.path.join(bd, s, wb_id)
                        if os.path.isdir(sd):
                            found_prefix = sd
                            break
                    if found_prefix: break
            if found_prefix:
                wb_prefix = found_prefix
            else:
                return jsonify({'error': '题册不存在'}), 404
        else:
            # R2 模式：遍历已知 board/subject 组合查找（同时尝试新旧两种路径）
            found_prefix = None
            for b in _EXAM_BOARDS:
                for s in _SUBJECTS_MAP.get(b, []):
                    # 新路径（含括号）
                    pf_new = _lib_key_prefix(b, s, wb_id)
                    if storage.load_json(f'{pf_new}/manifest.json') is not None:
                        found_prefix = pf_new
                        break
                    # 旧路径（无括号）
                    sb_old = re.sub(r'[^A-Za-z0-9_\-\u4e00-\u9fff ]', '', b).strip()
                    ss_old = re.sub(r'[^A-Za-z0-9_\-\u4e00-\u9fff ]', '', s).strip()
                    pf_old = f'library/{sb_old}/{ss_old}/{wb_id}'
                    if pf_old != pf_new and storage.load_json(f'{pf_old}/manifest.json') is not None:
                        found_prefix = pf_old
                        break
                if found_prefix: break
            if found_prefix:
                wb_prefix = found_prefix
            else:
                return jsonify({'error': '题册不存在'}), 404
    else:
        wb_prefix = _lib_key_prefix(board, subject, wb_id)

    app.logger.info(f'[library_load] wb_id={wb_id!r} board={board!r} subject={subject!r} lazy={lazy} prefix={wb_prefix!r}')

    if storage.is_r2_mode():
        manifest = storage.load_json(f'{wb_prefix}/manifest.json')

        # ── 向后兼容：旧版代码保存时括号会被剥离，尝试不含括号的路径 ──
        if manifest is None:
            legacy_board   = re.sub(r'[^A-Za-z0-9_\-\u4e00-\u9fff ]', '', board).strip()
            legacy_subject = re.sub(r'[^A-Za-z0-9_\-\u4e00-\u9fff ]', '', subject).strip()
            legacy_prefix  = f'library/{legacy_board}/{legacy_subject}/{wb_id}' if wb_id else f'library/{legacy_board}/{legacy_subject}'
            if legacy_prefix != wb_prefix:
                app.logger.info(f'[library_load] trying legacy prefix: {legacy_prefix!r}')
                manifest = storage.load_json(f'{legacy_prefix}/manifest.json')
                if manifest is not None:
                    wb_prefix = legacy_prefix
                    app.logger.info(f'[library_load] found via legacy prefix: {wb_prefix!r}')

        if manifest is None:
            app.logger.warning(f'[library_load] manifest not found at {wb_prefix}/manifest.json')
            return jsonify({'ok': False, 'error': f'题册不存在（路径: {wb_prefix}）'}), 404

        from concurrent.futures import ThreadPoolExecutor as _TPE, as_completed as _ac

        q_list = manifest.get('questions', [])

        # ── 诊断日志：打印 manifest 实际内容摘要（排查重复/img_file问题）──
        _diag_total = len(q_list)
        _diag_with_hash  = sum(1 for _q in q_list if _q.get('img_hash','').strip())
        _diag_with_imgf  = sum(1 for _q in q_list if _q.get('img_file','').strip())
        _diag_imgf_vals  = [_q.get('img_file','') for _q in q_list[:5]]  # 前5个
        _diag_dup_imgf   = len(q_list) - len(set(_q.get('img_file','') for _q in q_list))
        _diag_dup_hash   = len(q_list) - len(set(_q.get('img_hash','') for _q in q_list if _q.get('img_hash','')))
        app.logger.info(
            f'[library_load DIAG] wb={wb_id!r} total={_diag_total} '
            f'with_hash={_diag_with_hash} with_imgfile={_diag_with_imgf} '
            f'dup_imgfile={_diag_dup_imgf} dup_hash={_diag_dup_hash} '
            f'first5_imgfile={_diag_imgf_vals!r}'
        )

        # ── 去重：优先按 img_hash（图片内容MD5）去重，无 hash 时降级用 img_file ──
        # img_hash 方案可识别内容相同但文件名不同的重复（如重复追加同套PDF）
        # img_file 降级可兼容旧版 manifest（无 img_hash 字段）
        _seen_hashes   = set()   # 已见 img_hash
        _seen_imgfiles = set()   # 已见 img_file（降级用）
        _deduped = []
        for _q in q_list:
            _hash = _q.get('img_hash', '').strip()
            _imgf = _q.get('img_file', '').strip()
            if _hash:
                # 有哈希：按哈希去重（内容级）
                if _hash in _seen_hashes:
                    app.logger.warning(
                        f'[library_load] 跳过重复 img_hash={_hash!r} '
                        f'img_file={_imgf!r} (exam_date={_q.get("exam_date")!r}, '
                        f'q_num={_q.get("q_num")!r}) in wb_id={wb_id!r}'
                    )
                    continue
                _seen_hashes.add(_hash)
            else:
                # 无哈希：按 img_file 去重（兼容旧 manifest）
                _key = _imgf if _imgf else f'__seq_{_q.get("seq", id(_q))}'
                if _key in _seen_imgfiles:
                    app.logger.warning(
                        f'[library_load] 跳过重复 img_file={_imgf!r} '
                        f'(exam_date={_q.get("exam_date")!r}, q_num={_q.get("q_num")!r}) '
                        f'in wb_id={wb_id!r}'
                    )
                    continue
                _seen_imgfiles.add(_key)
            _deduped.append(_q)
        if len(_deduped) < len(q_list):
            app.logger.warning(f'[library_load] 题册 {wb_id!r} 共去除 {len(q_list)-len(_deduped)} 条重复题目')
        q_list = _deduped

        # ── 懒加载模式：只返回元数据，极速首屏 ──
        if lazy:
            questions_out = []
            for i, q in enumerate(q_list):
                questions_out.append({
                    'seq':        q.get('seq', 0),
                    'q_num':      q.get('q_num', 0),
                    'difficulty': q.get('difficulty'),
                    'topics':     q.get('topics', []),
                    'exam_date':  q.get('exam_date', ''),
                    'source':     q.get('source', ''),
                    'img_bytes_b64': '',
                    'img_w':      q.get('img_w', 0),
                    'img_h':      q.get('img_h', 0),
                    'answer_b64': '',
                    'has_answer': bool(q.get('ans_file', '')),
                    '_wb_id':     wb_id,
                    '_img_file':  q.get('img_file', ''),
                    '_img_hash':  q.get('img_hash', ''),   # 内容MD5，用于前端去重
                    '_ans_file':  q.get('ans_file', ''),
                })
        else:
            # ── 并发下载所有图片（大幅提速，199题从串行~200次R2请求→并发完成）──
            def _fetch_img(args):
                idx, img_file, ans_file = args
                result = {'idx': idx, 'b64': '', 'ans_b64': ''}
                if img_file:
                    raw = storage.load_bytes(f'{wb_prefix}/{img_file}')
                    if raw:
                        result['b64'] = _b64.b64encode(raw).decode('ascii')
                if ans_file:
                    raw_ans = storage.load_bytes(f'{wb_prefix}/{ans_file}')
                    if raw_ans:
                        result['ans_b64'] = _b64.b64encode(raw_ans).decode('ascii')
                return result

            tasks = [(i, q.get('img_file',''), q.get('ans_file','')) for i, q in enumerate(q_list)]
            img_map = {}  # idx -> {b64, ans_b64}
            with _TPE(max_workers=20) as _ex:
                for fut in _ac([_ex.submit(_fetch_img, t) for t in tasks]):
                    r = fut.result()
                    img_map[r['idx']] = r

            questions_out = []
            for i, q in enumerate(q_list):
                imgs = img_map.get(i, {'b64': '', 'ans_b64': ''})
                questions_out.append({
                    'seq':           q.get('seq', 0),
                    'q_num':         q.get('q_num', 0),
                    'difficulty':    q.get('difficulty'),
                    'topics':        q.get('topics', []),
                    'exam_date':     q.get('exam_date', ''),
                    'source':        q.get('source', ''),
                    'img_bytes_b64': imgs['b64'],
                    'img_w':         q.get('img_w', 0),
                    'img_h':         q.get('img_h', 0),
                    'answer_b64':    imgs['ans_b64'],
                    'has_answer':    bool(imgs['ans_b64']),
                    '_img_file':     q.get('img_file', ''),   # 全局唯一，用于前端去重
                    '_img_hash':     q.get('img_hash', ''),   # 内容MD5，用于前端去重
                    '_ans_file':     q.get('ans_file', ''),   # 答案文件名，用于精确定位
                })
    else:
        mfest = os.path.join(wb_prefix, 'manifest.json')
        if not os.path.isfile(mfest):
            return jsonify({'error': '题册不存在'}), 404
        with open(mfest, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        # ── 去重：优先按 img_hash（内容MD5）去重，无 hash 时降级用 img_file（本地模式）──
        _local_qs = manifest.get('questions', [])
        _seen_local_hashes   = set()
        _seen_local_imgfiles = set()
        _deduped_local = []
        for _lq in _local_qs:
            _lhash = _lq.get('img_hash', '').strip()
            _limgf = _lq.get('img_file', '').strip()
            if _lhash:
                if _lhash in _seen_local_hashes:
                    continue
                _seen_local_hashes.add(_lhash)
            else:
                _lkey = _limgf if _limgf else f'__seq_{_lq.get("seq", id(_lq))}'
                if _lkey in _seen_local_imgfiles:
                    continue
                _seen_local_imgfiles.add(_lkey)
            _deduped_local.append(_lq)
        questions_out = []
        for q in _deduped_local:
            img_file = q.get('img_file', '')
            b64 = ''
            if img_file:
                img_path = os.path.join(wb_prefix, img_file)
                if os.path.isfile(img_path):
                    with open(img_path, 'rb') as f:
                        b64 = _b64.b64encode(f.read()).decode('ascii')
            # Task1: 读取答案图片
            ans_b64 = ''
            ans_file = q.get('ans_file', '')
            if ans_file:
                ans_path = os.path.join(wb_prefix, ans_file)
                if os.path.isfile(ans_path):
                    with open(ans_path, 'rb') as f:
                        ans_b64 = _b64.b64encode(f.read()).decode('ascii')
            questions_out.append({
                'seq':            q.get('seq', 0),
                'q_num':          q.get('q_num', 0),
                'difficulty':     q.get('difficulty'),
                'topics':         q.get('topics', []),
                'exam_date':      q.get('exam_date', ''),
                'source':         q.get('source', ''),
                'img_bytes_b64':  b64,
                'img_w':          q.get('img_w', 0),
                'img_h':          q.get('img_h', 0),
                'answer_b64':     ans_b64,
                'ai_text_answer': q.get('ai_text_answer', ''),   # AI解析文字答案
                '_img_file':      q.get('img_file', ''),         # 全局唯一，用于前端去重
                '_img_hash':      q.get('img_hash', ''),         # 内容MD5，用于前端去重
                '_ans_file':      q.get('ans_file', ''),         # 答案文件名，用于精确定位
            })

    return jsonify({
        'ok':           True,
        'id':           wb_id,
        'session_id':   _register_workbook_session(wb_id, manifest, questions_out, wb_prefix=wb_prefix),
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


# ─────────────────────────────── 懒加载单题图片 ───────────────────────────────
@app.route('/api/library/img/<wb_id>/<q_num_str>', methods=['GET'])
def library_img_lazy(wb_id, q_num_str):
    """
    懒加载单题图片。GET /api/library/img/<wb_id>/<q_num>?board=&subject=&type=q|a&img_file=xxx
    type=q: 返回题目图片; type=a: 返回答案图片
    img_file: 可选，manifest 里的唯一文件名（如 q_005.jpg）。
              ★ 优先用 img_file 精确定位，避免多套卷同 q_num 冲突导致图片张冠李戴。
              无 img_file 时降级用 q_num（兼容旧调用）。
    返回: {ok, b64, w, h}
    """
    import base64 as _b64
    board    = request.args.get('board', '')
    subject  = request.args.get('subject', '')
    img_type = request.args.get('type', 'q')      # 'q' or 'a'
    img_file_hint = request.args.get('img_file', '').strip()  # ★ 精确文件名（首选）
    q_seq_hint    = request.args.get('q_seq', '').strip()     # ★ manifest 内序号（次选）
    try:
        q_num = int(q_num_str)
    except ValueError:
        return jsonify({'ok': False, 'error': 'invalid q_num'}), 400
    try:
        q_seq_int = int(q_seq_hint) if q_seq_hint else None
    except ValueError:
        q_seq_int = None

    wb_prefix = _lib_key_prefix(board, subject, wb_id)

    def _find_q_obj(q_list):
        """按优先级定位题目：① img_file精确匹配 → ② q_seq序号 → ③ q_num模糊（旧）"""
        if img_file_hint:
            # img_file 是 manifest 内全局唯一文件名，精确匹配不会冲突
            target_field = 'ans_file' if img_type == 'a' else 'img_file'
            obj = next((q for q in q_list if q.get(target_field, '') == img_file_hint), None)
            if obj:
                return obj
            # img_file_hint 传的是题目图片名但 type=a，尝试用 img_file 锁定行再取 ans_file
            obj = next((q for q in q_list if q.get('img_file', '') == img_file_hint), None)
            if obj:
                return obj
        # 次选：按 seq 序号精确匹配（manifest 中第几条，全局唯一）
        if q_seq_int is not None:
            obj = next((q for q in q_list if q.get('seq') == q_seq_int), None)
            if obj:
                return obj
        # 降级：按 q_num（旧逻辑，多套卷可能冲突）
        return next((q for q in q_list if q.get('q_num') == q_num), None)

    # 从 manifest 找到对应题目的文件名
    if storage.is_r2_mode():
        manifest = storage.load_json(f'{wb_prefix}/manifest.json')
        # 向后兼容：旧版路径无括号
        if not manifest:
            legacy_board   = re.sub(r'[^A-Za-z0-9_\-\u4e00-\u9fff ]', '', board).strip()
            legacy_subject = re.sub(r'[^A-Za-z0-9_\-\u4e00-\u9fff ]', '', subject).strip()
            legacy_prefix  = f'library/{legacy_board}/{legacy_subject}/{wb_id}'
            if legacy_prefix != wb_prefix:
                manifest = storage.load_json(f'{legacy_prefix}/manifest.json')
                if manifest is not None:
                    wb_prefix = legacy_prefix
        if not manifest:
            return jsonify({'ok': False, 'error': '题册不存在'}), 404
        q_list = manifest.get('questions', [])
        q_obj  = _find_q_obj(q_list)
        if not q_obj:
            return jsonify({'ok': False, 'error': '题目不存在'}), 404

        file_key = q_obj.get('ans_file' if img_type == 'a' else 'img_file', '')
        if not file_key:
            return jsonify({'ok': False, 'error': '无图片文件', 'b64': ''}), 200

        raw = storage.load_bytes(f'{wb_prefix}/{file_key}')
        if not raw:
            return jsonify({'ok': False, 'error': '图片读取失败', 'b64': ''}), 200

        b64 = _b64.b64encode(raw).decode('ascii')
        return jsonify({
            'ok': True,
            'b64': b64,
            'w': q_obj.get('img_w', 0),
            'h': q_obj.get('img_h', 0),
        })
    else:
        # 本地模式
        mfest = os.path.join(wb_prefix, 'manifest.json')
        if not os.path.isfile(mfest):
            return jsonify({'ok': False, 'error': '题册不存在'}), 404
        with open(mfest, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        q_obj = _find_q_obj(manifest.get('questions', []))
        if not q_obj:
            return jsonify({'ok': False, 'error': '题目不存在'}), 404
        file_name = q_obj.get('ans_file' if img_type == 'a' else 'img_file', '')
        if not file_name:
            return jsonify({'ok': False, 'error': '无图片文件', 'b64': ''}), 200
        img_path = os.path.join(wb_prefix, file_name)
        if not os.path.isfile(img_path):
            return jsonify({'ok': False, 'error': '图片不存在', 'b64': ''}), 200
        with open(img_path, 'rb') as f:
            b64 = _b64.b64encode(f.read()).decode('ascii')
        return jsonify({'ok': True, 'b64': b64, 'w': q_obj.get('img_w', 0), 'h': q_obj.get('img_h', 0)})



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


@app.route('/api/library/append/<wb_id>', methods=['POST'])
def library_append(wb_id):
    """
    将新一批题目（来自 upload_multi session）追加到已有题册，不覆盖原有题目。

    前端传 JSON：
    {
      session_id,          # 本次新上传的 session
      board, subject,      # 题册所在分类（用于定位题册目录）
      questions: [{        # 仅传新增的题目
        q_num, file_idx,
        difficulty, topics, exam_date, source,
        img_bytes_b64      # 可选，已有 b64 时跳过裁图
      }]
    }

    返回：{ok, id, title, added, total, msg}
    """
    import base64 as _b64
    data       = request.json or {}
    session_id = data.get('session_id', '')
    board      = data.get('board', 'Edexcel')
    subject    = data.get('subject', '经济 Economics')
    questions  = data.get('questions', [])

    if not questions:
        return jsonify({'error': '没有新题目数据'}), 400

    wb_prefix = _lib_key_prefix(board, subject, wb_id)

    # ── 读取已有 manifest ──
    if storage.is_r2_mode():
        manifest = storage.load_json(f'{wb_prefix}/manifest.json')
    else:
        mfest_path = os.path.join(wb_prefix, 'manifest.json')
        if not os.path.isfile(mfest_path):
            return jsonify({'error': '题册不存在'}), 404
        with open(mfest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)

    if manifest is None:
        return jsonify({'error': '题册不存在'}), 404

    # ── 获取 session（用于裁图）──
    sess = None
    if session_id:
        sess = _get_session(session_id)

    # ── 计算新图片的起始序号（在旧题目后面接续编号）──
    existing_qs = manifest.get('questions', [])

    # ── 去重：过滤掉已存在于题册中的题目（按 img_file 内容去重）──
    # 防止用户重复追加同一套 PDF 导致题目翻倍。
    # 策略：对 existing_qs 里所有 img_file 建立集合；新题目的 img_bytes_b64
    # 做 MD5 哈希，与已有图片哈希比对，相同则跳过。
    # 降级方案（无 img_bytes_b64 时）：用 (exam_date, q_num) 作辅助判断。
    import hashlib as _hashlib
    import base64 as _b64_dedup

    # 构建已有图片哈希集合（需下载 R2 图片才能比较，代价较高）
    # 改用轻量方案：对已有 manifest 里的 (exam_date, file_idx, q_num) 三元组去重
    # 比 (exam_date, q_num) 更精准：加上 file_idx 区分同一套卷内的不同子题
    existing_keys = set()
    for _eq in existing_qs:
        _edate  = str(_eq.get('exam_date')  or '').strip()
        _eqnum  = str(_eq.get('q_num')      or '').strip()
        _efidx  = str(_eq.get('file_idx')   or '').strip()
        # 三元组必须全部相同才认定为重复（防止不同套卷的同 q_num 被误删）
        if _edate and _eqnum:
            existing_keys.add((_edate, _efidx, _eqnum))

    if existing_keys:
        orig_count = len(questions)
        questions = [
            q for q in questions
            if not (
                str(q.get('exam_date') or '').strip()
                and (
                    str(q.get('exam_date') or '').strip(),
                    str(q.get('file_idx',  q.get('gIdx', '')) or '').strip(),
                    str(q.get('q_num')     or '').strip()
                ) in existing_keys
            )
        ]
        skipped = orig_count - len(questions)
        if skipped:
            app.logger.warning(
                f'[library/append] wb_id={wb_id!r} 跳过 {skipped} 道已存在的重复题目'
                f'（按 exam_date+file_idx+q_num 判断），剩余 {len(questions)} 道新题'
            )
        if not questions:
            return jsonify({
                'ok':    True,
                'id':    wb_id,
                'title': manifest.get('title', wb_id),
                'added': 0,
                'total': len(existing_qs),
                'msg':   f'所有题目在题册中已存在（共跳过 {skipped} 题），无需追加',
            })

    start_idx   = len(existing_qs)   # 旧题目数量，新题从 start_idx+1 开始编号

    # ── 按 file_idx 分组，批量裁图（复用 library/save 逻辑）──
    from collections import defaultdict
    file_idx_map = defaultdict(list)
    q_has_b64    = {}

    for i, q in enumerate(questions):
        if q.get('img_bytes_b64'):
            q_has_b64[i] = True
        else:
            file_idx = int(q.get('file_idx', q.get('gIdx', 0)))
            file_idx_map[file_idx].append((i, q))

    img_results = {}

    # 处理已有 b64 的题目
    for i, q in enumerate(questions):
        if not q_has_b64.get(i):
            continue
        b64       = q.get('img_bytes_b64', '')
        img_fname = f'q_{start_idx + i + 1:03d}.jpg'
        try:
            raw = _b64.b64decode(b64)
            from PIL import Image as _PIL
            _im = _PIL.open(io.BytesIO(raw))
            buf = io.BytesIO()
            _im.convert('RGB').save(buf, format='JPEG', quality=88)
            img_bytes = buf.getvalue()
            w, h = _im.width, _im.height
            if storage.is_r2_mode():
                storage.store_bytes(f'{wb_prefix}/{img_fname}', img_bytes)
            else:
                with open(os.path.join(wb_prefix, img_fname), 'wb') as f:
                    f.write(img_bytes)
            # 答案图片
            ans_fname = ''
            ans_b64   = q.get('answer_b64', '')
            if ans_b64:
                ans_fname = f'q_{start_idx + i + 1:03d}_ans.jpg'
                try:
                    ans_raw = _b64.b64decode(ans_b64)
                    from PIL import Image as _PIL2
                    _aim = _PIL2.open(io.BytesIO(ans_raw))
                    abuf = io.BytesIO()
                    _aim.convert('RGB').save(abuf, format='JPEG', quality=88)
                    if storage.is_r2_mode():
                        storage.store_bytes(f'{wb_prefix}/{ans_fname}', abuf.getvalue())
                    else:
                        with open(os.path.join(wb_prefix, ans_fname), 'wb') as f:
                            f.write(abuf.getvalue())
                except Exception:
                    ans_fname = ''
            img_results[i] = (img_fname, w, h, ans_fname)
        except Exception:
            img_results[i] = ('', 0, 0, '', '')

    # 裁图（从原始 PDF session 中裁取）
    if sess and file_idx_map:
        for file_idx, items in file_idx_map.items():
            if file_idx >= len(sess):
                for (i, q) in items:
                    img_results[i] = ('', 0, 0, '', '')
                continue
            grp = sess[file_idx]
            pdf_path = grp.get('path', '')
            try:
                if pdf_path and os.path.isfile(pdf_path):
                    doc = fitz.open(pdf_path)
                else:
                    doc = None

                for (i, q) in items:
                    img_fname = f'q_{start_idx + i + 1:03d}.jpg'
                    img_bytes = b''
                    w = h = 0
                    # 优先从 grp.questions 取已缓存的 img_bytes_b64
                    grp_qs    = grp.get('questions', [])
                    q_num     = q.get('q_num')
                    cached_b64 = ''
                    for gq in grp_qs:
                        if gq.get('q_num') == q_num:
                            cached_b64 = gq.get('img_bytes_b64', '')
                            break
                    if cached_b64:
                        raw = _b64.b64decode(cached_b64)
                        from PIL import Image as _PIL3
                        _im = _PIL3.open(io.BytesIO(raw))
                        buf = io.BytesIO()
                        _im.convert('RGB').save(buf, format='JPEG', quality=88)
                        img_bytes = buf.getvalue()
                        w, h = _im.width, _im.height
                    elif doc:
                        # 从 PDF 裁图（fallback）
                        try:
                            grp_qs_list = grp.get('questions', [])
                            q_idx_in_grp = next(
                                (idx for idx, gq in enumerate(grp_qs_list) if gq.get('q_num') == q_num),
                                None
                            )
                            if q_idx_in_grp is not None:
                                paper_type = grp.get('paper_type', 'edexcel_economics')
                                img_bytes_raw, w, h = crop_question_image(
                                    doc, grp_qs_list, q_idx_in_grp, dpi=300, paper_type=paper_type
                                )
                                if img_bytes_raw:
                                    from PIL import Image as _PIL4
                                    _im4 = _PIL4.open(io.BytesIO(img_bytes_raw))
                                    buf2 = io.BytesIO()
                                    _im4.convert('RGB').save(buf2, format='JPEG', quality=88)
                                    img_bytes = buf2.getvalue()
                        except Exception:
                            pass

                    if img_bytes:
                        if storage.is_r2_mode():
                            storage.store_bytes(f'{wb_prefix}/{img_fname}', img_bytes)
                        else:
                            with open(os.path.join(wb_prefix, img_fname), 'wb') as f:
                                f.write(img_bytes)
                        # 答案图片
                        ans_fname = ''
                        for gq in grp_qs:
                            if gq.get('q_num') == q_num:
                                ans_b64 = gq.get('answer_b64', '')
                                if ans_b64:
                                    ans_fname = f'q_{start_idx + i + 1:03d}_ans.jpg'
                                    try:
                                        ans_raw = _b64.b64decode(ans_b64)
                                        from PIL import Image as _PIL5
                                        _aim = _PIL5.open(io.BytesIO(ans_raw))
                                        abuf = io.BytesIO()
                                        _aim.convert('RGB').save(abuf, format='JPEG', quality=88)
                                        if storage.is_r2_mode():
                                            storage.store_bytes(f'{wb_prefix}/{ans_fname}', abuf.getvalue())
                                        else:
                                            with open(os.path.join(wb_prefix, ans_fname), 'wb') as f:
                                                f.write(abuf.getvalue())
                                    except Exception:
                                        ans_fname = ''
                                break
                        img_results[i] = (img_fname, w, h, ans_fname)
                    else:
                        img_results[i] = ('', 0, 0, '', '')

                if doc:
                    doc.close()
            except Exception as e:
                print(f'[library/append] 裁图异常 file_idx={file_idx}: {e}')
                for (i, q) in items:
                    img_results[i] = ('', 0, 0, '', '')

    # ── 构建新增题目条目，并计算 img_hash 用于内容级去重 ──
    import hashlib as _hl_append
    # 收集已有题目的 img_hash（有哈希值的），用于去重新题目
    existing_hashes = set(
        _eq['img_hash'] for _eq in existing_qs
        if _eq.get('img_hash')
    )

    new_qs = []
    for i, q in enumerate(questions):
        img_file, img_w, img_h, ans_file = img_results.get(i, ('', 0, 0, ''))
        # 计算新题目图片的 MD5 哈希
        img_hash = ''
        if img_file:
            try:
                if storage.is_r2_mode():
                    _raw_h = storage.load_bytes(f'{wb_prefix}/{img_file}')
                else:
                    _img_hp = os.path.join(wb_prefix, img_file)
                    with open(_img_hp, 'rb') as _fh2:
                        _raw_h = _fh2.read()
                if _raw_h:
                    img_hash = _hl_append.md5(_raw_h).hexdigest()
            except Exception:
                img_hash = ''
        # 哈希去重：如果新题目图片内容与已有题目完全相同，跳过
        if img_hash and img_hash in existing_hashes:
            app.logger.warning(
                f'[library/append] 跳过内容重复题目 img_file={img_file!r} '
                f'img_hash={img_hash!r} (q_num={q.get("q_num")!r})'
            )
            continue
        if img_hash:
            existing_hashes.add(img_hash)  # 避免同批次内重复
        new_qs.append({
            'seq':            start_idx + len(new_qs) + 1,
            'q_num':          q.get('q_num', start_idx + len(new_qs) + 1),
            'file_idx':       int(q.get('file_idx', q.get('gIdx', 0))),
            'difficulty':     q.get('difficulty'),
            'topics':         q.get('topics', []),
            'exam_date':      q.get('exam_date', ''),
            'source':         q.get('source', ''),
            'img_file':       img_file,
            'img_hash':       img_hash,   # 图片内容 MD5，用于内容级去重
            'img_w':          img_w,
            'img_h':          img_h,
            'ans_file':       ans_file,
            'ai_text_answer': q.get('ai_text_answer', ''),   # AI解析文字答案
        })

    added_count   = sum(1 for q in new_qs if q.get('img_file'))
    all_questions = existing_qs + new_qs

    # ── 更新 manifest ──
    manifest['questions']     = all_questions
    manifest['count']         = len(all_questions)
    manifest['success_count'] = sum(1 for q in all_questions if q.get('img_file'))
    manifest['updated_at']    = time.strftime('%Y-%m-%d %H:%M', time.localtime())

    if storage.is_r2_mode():
        storage.store_json(f'{wb_prefix}/manifest.json', manifest)
    else:
        with open(os.path.join(wb_prefix, 'manifest.json'), 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    msg = f'已追加 {added_count} 题，题册共 {len(all_questions)} 题'
    if added_count < len(questions):
        msg += f'（{len(questions) - added_count} 题图片获取失败）'

    print(f'[library/append] wb_id={wb_id} 追加 {added_count}/{len(questions)} 题，'
          f'题册合计 {len(all_questions)} 题')

    return jsonify({
        'ok':    True,
        'id':    wb_id,
        'title': manifest.get('title', wb_id),
        'added': added_count,
        'total': len(all_questions),
        'msg':   msg,
    })


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
        mat     = fitz.Matrix(2.0, 2.0)

        page_info = []
        start_page = 1 if (meta_obj and meta_obj.get('marker') == _WORKBOOK_MARKER) else 0

        for page_idx in range(start_page, len(doc)):
            page   = doc[page_idx]
            result = _detect_exported_pdf_header(page)
            if result is None:
                if page_info:
                    page_info.append({
                        'q_num':      page_info[-1]['q_num'],
                        'difficulty': page_info[-1]['difficulty'],
                        'topic_id':   page_info[-1]['topic_id'],
                        'topic_hint': page_info[-1]['topic_hint'],
                        'page_obj':   page,
                    })
            else:
                q_num, difficulty, topic_id, topic_hint = result
                page_info.append({
                    'q_num':      q_num,
                    'difficulty': difficulty,
                    'topic_id':   topic_id,
                    'topic_hint': topic_hint,
                    'page_obj':   page,
                })

        from collections import OrderedDict
        q_groups = OrderedDict()
        for pi in page_info:
            qn = pi['q_num']
            if qn not in q_groups:
                q_groups[qn] = {
                    'difficulty': pi['difficulty'],
                    'topic_id':   pi['topic_id'],
                    'topic_hint': pi['topic_hint'],
                    'pages':      [],
                }
            q_groups[qn]['pages'].append(pi['page_obj'])

        edx_syllabus = _load_edexcel_maths_syllabus()
        questions = []
        for q_num, gdata in q_groups.items():
            pixmaps = []
            for pg in gdata['pages']:
                PW = pg.rect.width
                PH = pg.rect.height
                # 动态找内容图片起始 y（跳过 Logo 小图），而不是使用固定偏移
                content_y0 = _find_content_img_y0(pg,
                             fallback=MARGIN + _EXPORT_HEADER_HEIGHT_PT + 12)
                img_rect = fitz.Rect(MARGIN, content_y0, PW - MARGIN, PH - MARGIN)
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

            # 优先使用封面元数据中的 difficulty/topics，否则用页眉解析的数据
            saved_meta = q_meta_map.get(q_num, {})
            difficulty = saved_meta.get('difficulty')
            topics     = saved_meta.get('topics', [])

            # 如果封面元数据没有，尝试从页眉解析数据中补充
            if difficulty is None:
                difficulty = gdata.get('difficulty')
            if not topics:
                topic_id   = gdata.get('topic_id')
                topic_hint = gdata.get('topic_hint', '')
                if topic_id:
                    topic_info = _lookup_topic_in_syllabus(
                        _load_edexcel_maths_syllabus(), topic_id, topic_hint)
                    if topic_info:
                        topics = [{'id': topic_id, 'title': topic_info['title']}]

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
                   '商业 Business', '会计 Accounting', '竞赛物理 BPHO']
_CLOUD_BOARDS   = ['CAIE', 'Edexcel', 'AQA', 'OCR', 'IB', 'AP', 'BPHO']

# 知识点树：从现有 syllabus JSON 中加载（cambridge / edexcel_maths），
# 运行时根据 board+subject 动态确定使用哪套知识点

def _cloud_prefix(subject: str = '', board: str = '', topic1: str = '',
                   topic2: str = '', qid: str = '', maths_unit: str = '') -> str:
    """生成云端题库的存储 key（R2 key 或本地路径）。
    路径结构：cloud_db/{subject}/{board}/{maths_unit}/{topic1}/{topic2}/{qid}
    maths_unit 为空时跳过该层级（向后兼容旧数据）。
    """
    def _safe(s):
        return re.sub(r'[^A-Za-z0-9_\-\u4e00-\u9fff. ]', '_', s).strip()

    parts = ['cloud_db']
    if subject:     parts.append(_safe(subject))
    if board:       parts.append(_safe(board))
    if maths_unit:  parts.append(_safe(maths_unit))   # Task 3: 新增 paper 分级
    if topic1:      parts.append(_safe(topic1))
    if topic2:      parts.append(_safe(topic2))
    if qid:         parts.append(qid)

    if storage.is_r2_mode():
        return '/'.join(parts)
    else:
        base = storage.local_root()
        return os.path.join(base, *parts)


def _cloud_img_key(qid: str) -> str:
    """题目图片存储 key（question 子目录）"""
    if storage.is_r2_mode():
        return f'cloud_db/images/question/{qid}.jpg'
    else:
        base = os.path.join(storage.local_root(), 'cloud_db', 'images', 'question')
        os.makedirs(base, exist_ok=True)
        return os.path.join(base, f'{qid}.jpg')


def _cloud_ans_img_key(qid: str) -> str:
    """答案图片存储 key（scheme 子目录）"""
    if storage.is_r2_mode():
        return f'cloud_db/images/scheme/{qid}.jpg'
    else:
        base = os.path.join(storage.local_root(), 'cloud_db', 'images', 'scheme')
        os.makedirs(base, exist_ok=True)
        return os.path.join(base, f'{qid}.jpg')


def _cloud_stats_key() -> str:
    if storage.is_r2_mode():
        return 'cloud_db/_stats.json'
    else:
        base = os.path.join(storage.local_root(), 'cloud_db')
        os.makedirs(base, exist_ok=True)
        return os.path.join(base, '_stats.json')


def _list_cloud_questions(subject: str, board: str,
                          topic1: str = '', topic2: str = '') -> list:
    """列出指定路径下所有题目元数据（JSON文件）的 key 列表。
    修复：当 topic1/topic2 存在时，需要穿透 maths_unit 层级进行搜索，
    因为实际存储路径为 cloud_db/{subject}/{board}/{maths_unit}/{topic1}/{topic2}/。
    策略：先获取 board 级别的全部 keys，再用元数据字段过滤。
    """
    if storage.is_r2_mode():
        if topic1 or topic2:
            # 需要穿透 maths_unit 层：先拿 board 下所有 json
            base_prefix = _cloud_prefix(subject, board) + '/'
            all_keys = storage.list_prefix(base_prefix)
            candidates = [k for k in all_keys if k.endswith('.json') and not k.endswith('_meta.json')]
            # 再按元数据过滤
            results = []
            for key in candidates:
                q = _load_cloud_question(key)
                if not q:
                    continue
                if topic1 and q.get('topic1', '') != topic1:
                    continue
                if topic2 and q.get('topic2', '') != topic2:
                    continue
                results.append(key)
            return results
        else:
            prefix = _cloud_prefix(subject, board, topic1, topic2)
            prefix_key = prefix + '/'
            all_keys   = storage.list_prefix(prefix_key)
            return [k for k in all_keys if k.endswith('.json') and not k.endswith('_meta.json')]
    else:
        import glob as _glob
        if topic1 or topic2:
            # 穿透 maths_unit 层：从 board 目录扫全部
            base_prefix = _cloud_prefix(subject, board)
            results = []
            if not os.path.isdir(base_prefix):
                return results
            pattern = os.path.join(base_prefix, '**', '*.json')
            for fp in _glob.glob(pattern, recursive=True):
                if os.path.basename(fp).startswith('_'):
                    continue
                q = _load_cloud_question(fp)
                if not q:
                    continue
                if topic1 and q.get('topic1', '') != topic1:
                    continue
                if topic2 and q.get('topic2', '') != topic2:
                    continue
                results.append(fp)
            return results
        else:
            prefix = _cloud_prefix(subject, board, topic1, topic2)
            results = []
            if not os.path.isdir(prefix):
                return results
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
      by_subject: {subject: {total, by_board: {board: {total, by_paper: {paper: {total, by_topic1: {t1: {total, by_topic2: {t2: int}}}}}}}}}
    }
    5层结构：学科 → 考试局 → paper类型(maths_unit) → 一级知识点 → 二级知识点
    """
    stats = {'total': 0, 'by_subject': {}}

    for subj in _CLOUD_SUBJECTS:
        for board in _CLOUD_BOARDS:
            keys = _list_cloud_questions(subj, board)
            if not keys:
                continue
            subj_stats  = stats['by_subject'].setdefault(subj, {'total': 0, 'by_board': {}})
            board_stats = subj_stats['by_board'].setdefault(board, {'total': 0, 'by_paper': {}})

            for key in keys:
                q = _load_cloud_question(key)
                if not q:
                    continue
                paper = q.get('maths_unit', '') or '未分类'
                t1    = q.get('topic1', '未分类')
                t2    = q.get('topic2', '未分类')
                paper_stats = board_stats['by_paper'].setdefault(paper, {'total': 0, 'by_topic1': {}})
                t1_stats    = paper_stats['by_topic1'].setdefault(t1, {'total': 0, 'by_topic2': {}})
                t1_stats['by_topic2'][t2] = t1_stats['by_topic2'].get(t2, 0) + 1
                t1_stats['total']  += 1
                paper_stats['total'] += 1
                board_stats['total'] += 1
                subj_stats['total']  += 1
                stats['total']       += 1

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
        # 优先用前端传来的图片 base64（字段名可能是 img_bytes_b64 或 _img_b64）
        b64_from_frontend = q.get('img_bytes_b64') or q.get('_img_b64') or ''
        if b64_from_frontend:
            b64_cache[i] = b64_from_frontend
        else:
            file_groups[int(q.get('file_idx', q.get('gIdx', 0)))].append((i, q))

    # 从 PDF session 裁图（仅当前端没有传图片时才走这里）
    for file_idx, items in file_groups.items():
        if not sess or file_idx >= len(sess):
            # 没有 PDF session（workbook/cloud 来源），跳过裁图
            # 这些题目的图片应已由前端通过 img_bytes_b64 传过来
            print(f'[cloud_save] no sess for file_idx={file_idx}, skip crop ({len(items)} items)')
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
        topic1   = (q.get('topic1') or '').strip()
        topic2   = (q.get('topic2') or '').strip()
        maths_unit = (q.get('maths_unit') or '').strip()   # Task 3: paper 分级

        # 如果 topic1/topic2 未设置，从 topics 列表自动推断
        if not topic1 or topic1 == '未分类':
            qtopics = q.get('topics') or []
            if qtopics:
                # 优先选择有 parent_id（subtopic级）且不是纯通用建模章节的条目
                # 已知通用章节（代表性差，几乎每道题都匹配）
                GENERIC_CHAPTERS = {'M1-1', 'M2-1', 'S1-1', 'FP1-1'}
                t0 = (
                    next((t for t in qtopics if t.get('parent_id') and t.get('parent_id') not in GENERIC_CHAPTERS), None) or
                    next((t for t in qtopics if t.get('parent_id')), None) or
                    qtopics[0]
                )
                t0_id       = t0.get('id', '')
                t0_title    = t0.get('title', '')
                parent_id   = t0.get('parent_id', '')
                parent_title = t0.get('parent_title', '')  # 新格式：edexcel_maths subtopic 带 parent_title

                if parent_title:
                    # 新版 Edexcel Maths subtopic 格式：parent_title 直接可用
                    # topic1 = 章节 title，topic2 = subtopic title
                    topic1 = parent_title
                    topic2 = t0_title or topic1
                elif parent_id:
                    # Cambridge 结构：parent_id 是数字编号 → 查 syllabus 得到人类可读 title
                    cam_syl = _load_syllabus()
                    cam_title_map = {}
                    if cam_syl:
                        for _t in cam_syl.get('topics', []):
                            cam_title_map[str(_t.get('id', ''))] = _t.get('title', '')
                    parent_title = cam_title_map.get(str(parent_id), str(parent_id))
                    topic1 = parent_title
                    topic2 = t0_title or topic1
                elif re.match(r'^[A-Z0-9]+-\d+$', t0_id):
                    # Edexcel 章节 ID（P3-1, M1-2 等）→ 用 title 做 topic1（title 已从 JSON 反查）
                    topic1 = t0_title or t0_id
                    topic2 = t0_title or t0_id
                else:
                    topic1 = t0_title or '未分类'
                    topic2 = t0_title or '通用'
        if not topic2 or topic2 == '通用':
            topic2 = topic1  # 没有二级则和一级一样
        if not topic1:
            topic1 = '未分类'
        if not topic2:
            topic2 = topic1

        # 生成唯一题目ID
        qid = str(uuid.uuid4())[:12]

        # Task 4: 保存题目图片到 question/ 子目录
        if b64:
            try:
                img_data = _b64.b64decode(b64)
                img_key  = _cloud_img_key(qid)   # -> cloud_db/images/question/{qid}.jpg
                if storage.is_r2_mode():
                    storage.store_bytes(img_key, img_data)
                else:
                    with open(img_key, 'wb') as f:
                        f.write(img_data)
            except Exception as e:
                print(f'[cloud_save] img save error qid={qid}: {e}')
                failed += 1
                continue

        # Task 4: 保存答案图片到 scheme/ 子目录（如果有）
        ans_b64 = q.get('answer_b64', '')
        has_answer_image = False
        if ans_b64:
            try:
                ans_data = _b64.b64decode(ans_b64)
                ans_key  = _cloud_ans_img_key(qid)  # -> cloud_db/images/scheme/{qid}.jpg
                if storage.is_r2_mode():
                    storage.store_bytes(ans_key, ans_data)
                else:
                    with open(ans_key, 'wb') as f:
                        f.write(ans_data)
                has_answer_image = True
            except Exception as e:
                print(f'[cloud_save] answer img save error qid={qid}: {e}')

        # 构建元数据（不含图片 base64，图片单独存）
        q_meta_save = {
            'qid':              qid,
            'q_num':            q.get('q_num'),
            'subject':          subject,
            'board':            board,
            'topic1':           topic1,
            'topic2':           topic2,
            'difficulty':       q.get('difficulty'),
            'topics':           q.get('topics', []),
            'exam_date':        q.get('exam_date', ''),
            'source':           q.get('source', ''),
            'paper_type':       q.get('paper_type', ''),
            'maths_unit':       maths_unit,
            'has_image':        bool(b64),
            'has_answer_image': has_answer_image,   # Task 4: 记录是否有答案图
            'saved_at':         __import__('datetime').datetime.utcnow().isoformat(),
        }

        # Task 3: 写元数据 JSON（路径含 maths_unit 层级）
        meta_key = _cloud_prefix(subject, board, topic1, topic2, maths_unit=maths_unit) + \
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
    参数：subject, board, maths_unit(可选), topic1(可选), topic2(可选), page(默认1), per_page(默认30)
    返回：{questions: [...], total: int, page: int, pages: int}
    """
    subject    = request.args.get('subject', '')
    board      = request.args.get('board', '')
    maths_unit = request.args.get('maths_unit', '')  # paper 层过滤
    topic1     = request.args.get('topic1', '')
    topic2     = request.args.get('topic2', '')
    page       = max(1, int(request.args.get('page', 1)))
    per_page   = min(100, int(request.args.get('per_page', 30)))

    # 若未指定 subject，则遍历所有 subject 合并结果
    subjects_to_query = [subject] if subject else _CLOUD_SUBJECTS
    # 若未指定 board，则遍历所有 board
    boards_to_query = [board] if board else _CLOUD_BOARDS

    keys = []
    for subj in subjects_to_query:
        for brd in boards_to_query:
            keys.extend(_list_cloud_questions(subj, brd, topic1, topic2))

    # 如果指定了 maths_unit，则按 maths_unit 过滤（元数据字段）
    if maths_unit:
        filtered_keys = []
        for key in keys:
            q = _load_cloud_question(key)
            if q and (q.get('maths_unit', '') or '未分类') == maths_unit:
                filtered_keys.append(key)
        keys = filtered_keys

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
            # 附上答案图片URL
            qid = q.get('qid', '')
            if qid and q.get('has_answer_image'):
                q['ans_url'] = f'/api/cloud_library/answer/{qid}'
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


# ── API: 获取题目答案图片 ──
@app.route('/api/cloud_library/answer/<qid>', methods=['GET'])
def cloud_library_answer_image(qid):
    """返回云端题库中指定题目的答案图片（JPEG）。"""
    qid = re.sub(r'[^A-Za-z0-9\-_]', '', qid)
    ans_key = _cloud_ans_img_key(qid)
    if storage.is_r2_mode():
        data = storage.load_bytes(ans_key)
        if not data:
            return jsonify({'error': '答案图片不存在'}), 404
        return send_file(io.BytesIO(data), mimetype='image/jpeg')
    else:
        if not os.path.isfile(ans_key):
            return jsonify({'error': '答案图片不存在'}), 404
        return send_file(ans_key, mimetype='image/jpeg')


# ── API: 更新云端题目元数据（知识点/难度等）──
@app.route('/api/cloud_library/update_question/<qid>', methods=['POST'])
def cloud_library_update_question(qid):
    """
    更新云端题目的元数据（knowledge points, difficulty 等）。
    请求体：{
      topic1: str (可选),
      topic2: str (可选),
      difficulty: int (可选),
      subject: str (可选, 旧值用于定位文件),
      board: str (可选, 旧值用于定位文件)
    }
    """
    qid = re.sub(r'[^A-Za-z0-9\-_]', '', qid)
    data = request.json or {}

    # 先找到这道题（遍历所有路径找 qid 匹配的 JSON）
    found_key = None
    found_q   = None

    old_subject = data.get('subject', '')
    old_board   = data.get('board', '')

    subjs = [old_subject] if old_subject else _CLOUD_SUBJECTS
    brds  = [old_board]   if old_board   else _CLOUD_BOARDS

    for subj in subjs:
        for brd in brds:
            keys = _list_cloud_questions(subj, brd, '', '')
            for key in keys:
                # 从 key 路径中提取 qid（文件名去掉 .json）
                fname = key.replace('\\', '/').rstrip('/')
                fname = fname.rsplit('/', 1)[-1] if '/' in fname else fname
                if fname == f'{qid}.json':
                    found_key = key
                    found_q   = _load_cloud_question(key)
                    break
            if found_key:
                break
        if found_key:
            break

    if not found_q:
        return jsonify({'error': f'未找到题目 {qid}'}), 404

    # 更新字段
    if 'topic1' in data and data['topic1']:
        found_q['topic1'] = data['topic1'].strip()
    if 'topic2' in data:
        found_q['topic2'] = data['topic2'].strip()
    if 'difficulty' in data and data['difficulty'] is not None:
        found_q['difficulty'] = int(data['difficulty'])
    if 'maths_unit' in data:
        found_q['maths_unit'] = data['maths_unit'].strip()

    # 检查是否需要移动到新路径（topic1/topic2/subject/board 变化）
    new_subject    = data.get('new_subject', found_q.get('subject', ''))
    new_board      = data.get('new_board', found_q.get('board', ''))
    new_topic1     = found_q.get('topic1', '未分类')
    new_topic2     = found_q.get('topic2', '')
    new_maths_unit = found_q.get('maths_unit', '')

    found_q['subject'] = new_subject
    found_q['board']   = new_board

    # 计算新 key
    new_key = _cloud_prefix(new_subject, new_board, new_topic1, new_topic2,
                             qid=qid, maths_unit=new_maths_unit)
    if not new_key.endswith('.json'):
        sep = '/' if storage.is_r2_mode() else os.sep
        new_key = new_key + sep + f'{qid}.json'

    # 保存到新位置
    _save_cloud_question(new_key, found_q)

    # 若位置变了，删除旧文件
    if found_key != new_key:
        if storage.is_r2_mode():
            try: storage.delete_object(found_key)
            except Exception: pass
        else:
            try: os.remove(found_key)
            except Exception: pass

    _invalidate_stats()
    return jsonify({'ok': True, 'qid': qid})


# ── API: 替换题目图片 ──
@app.route('/api/cloud_library/replace_image/<qid>', methods=['POST'])
def cloud_library_replace_image(qid):
    """
    替换云端题目的题目图片（上传新的图片文件或 base64）。
    请求体支持：
      - multipart/form-data: 文件字段 'image'
      - application/json: {'img_b64': '...'}
    """
    import base64 as _b64
    qid = re.sub(r'[^A-Za-z0-9\-_]', '', qid)

    img_bytes = None
    if request.content_type and 'multipart' in request.content_type:
        f = request.files.get('image')
        if not f:
            return jsonify({'error': '未提供图片'}), 400
        img_bytes = f.read()
    else:
        body = request.json or {}
        b64 = body.get('img_b64', '')
        if not b64:
            return jsonify({'error': '未提供图片数据'}), 400
        try:
            img_bytes = _b64.b64decode(b64)
        except Exception:
            return jsonify({'error': 'base64 解码失败'}), 400

    # 转换为 JPEG
    try:
        from PIL import Image as _PIL
        im = _PIL.open(io.BytesIO(img_bytes))
        buf = io.BytesIO()
        im.convert('RGB').save(buf, format='JPEG', quality=88)
        img_bytes = buf.getvalue()
    except Exception as e:
        return jsonify({'error': f'图片处理失败: {e}'}), 400

    # 保存到 R2 / 本地
    img_key = _cloud_img_key(qid)
    try:
        if storage.is_r2_mode():
            storage.store_bytes(img_key, img_bytes)
        else:
            os.makedirs(os.path.dirname(img_key), exist_ok=True)
            with open(img_key, 'wb') as f:
                f.write(img_bytes)
    except Exception as e:
        return jsonify({'error': f'保存失败: {e}'}), 500

    # ── 同步更新内存中所有 session 里对应题目的 img_bytes_b64 ──
    # 避免导出PDF时仍使用替换前的旧图片
    import base64 as _b64_sync
    new_b64 = _b64_sync.b64encode(img_bytes).decode()
    with _multi_sessions_lock:
        for sess_groups in _multi_sessions.values():
            for group in sess_groups:
                for q_obj in group.get('questions', []):
                    if q_obj.get('_cloud_qid') == qid:
                        q_obj['img_bytes_b64'] = new_b64

    return jsonify({'ok': True, 'qid': qid})


# ── API: 替换题目答案图片 ──
@app.route('/api/cloud_library/replace_answer/<qid>', methods=['POST'])
def cloud_library_replace_answer(qid):
    """
    替换云端题目的答案图片。
    请求体支持：
      - multipart/form-data: 文件字段 'image'
      - application/json: {'img_b64': '...'}
    同时更新元数据中的 has_answer_image 字段。
    """
    import base64 as _b64
    qid = re.sub(r'[^A-Za-z0-9\-_]', '', qid)

    img_bytes = None
    if request.content_type and 'multipart' in request.content_type:
        f = request.files.get('image')
        if not f:
            return jsonify({'error': '未提供图片'}), 400
        img_bytes = f.read()
    else:
        body = request.json or {}
        b64 = body.get('img_b64', '')
        if not b64:
            return jsonify({'error': '未提供图片数据'}), 400
        try:
            img_bytes = _b64.b64decode(b64)
        except Exception:
            return jsonify({'error': 'base64 解码失败'}), 400

    # 转换为 JPEG
    try:
        from PIL import Image as _PIL
        im = _PIL.open(io.BytesIO(img_bytes))
        buf = io.BytesIO()
        im.convert('RGB').save(buf, format='JPEG', quality=88)
        img_bytes = buf.getvalue()
    except Exception as e:
        return jsonify({'error': f'图片处理失败: {e}'}), 400

    # 保存答案图片到 R2 / 本地
    ans_key = _cloud_ans_img_key(qid)
    try:
        if storage.is_r2_mode():
            storage.store_bytes(ans_key, img_bytes)
        else:
            os.makedirs(os.path.dirname(ans_key), exist_ok=True)
            with open(ans_key, 'wb') as f:
                f.write(img_bytes)
    except Exception as e:
        return jsonify({'error': f'保存失败: {e}'}), 500

    # ── 同步更新内存中所有 session 里对应题目的 answer_b64 ──
    import base64 as _b64_sync_ans
    new_ans_b64 = _b64_sync_ans.b64encode(img_bytes).decode()
    with _multi_sessions_lock:
        for sess_groups in _multi_sessions.values():
            for group in sess_groups:
                for q_obj in group.get('questions', []):
                    if q_obj.get('_cloud_qid') == qid:
                        q_obj['answer_b64'] = new_ans_b64
                        q_obj['has_answer_image'] = True

    # 更新元数据 has_answer_image 标记
    for subj in _CLOUD_SUBJECTS:
        for brd in _CLOUD_BOARDS:
            keys = _list_cloud_questions(subj, brd, '', '')
            for key in keys:
                fname = key.replace('\\', '/').rstrip('/')
                fname = fname.rsplit('/', 1)[-1] if '/' in fname else fname
                if fname == f'{qid}.json':
                    q = _load_cloud_question(key)
                    if q:
                        q['has_answer_image'] = True
                        _save_cloud_question(key, q)
                    return jsonify({'ok': True, 'qid': qid})

    return jsonify({'ok': True, 'qid': qid, 'note': '元数据未找到，但图片已保存'})
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
        base = os.path.join(storage.local_root(), 'cloud_db')
        shutil.rmtree(base, ignore_errors=True)
        os.makedirs(base, exist_ok=True)

    _invalidate_stats()
    return jsonify({'ok': True, 'message': '已清空云端题库'})


# ── API: 获取四级目录树（含各层级题目数量） ──
@app.route('/api/cloud_library/tree', methods=['GET'])
def cloud_library_tree():
    """
    返回完整的五级目录树：
    学科 → 考试局 → paper类型(maths_unit) → 知识点一级 → 知识点二级，每级带题目数量。
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

            papers_out = []
            for paper, paper_data in sorted(board_data.get('by_paper', {}).items()):
                paper_total = paper_data.get('total', 0)

                topics1_out = []
                for t1, t1_data in sorted(paper_data.get('by_topic1', {}).items()):
                    t1_total = t1_data.get('total', 0)
                    topics2_out = []
                    for t2, t2_cnt in sorted(t1_data.get('by_topic2', {}).items()):
                        topics2_out.append({'name': t2, 'count': t2_cnt})
                    topics1_out.append({'name': t1, 'count': t1_total, 'subtopics': topics2_out})

                papers_out.append({
                    'name':   paper,
                    'count':  paper_total,
                    'topics': topics1_out,
                })

            boards_out.append({
                'name':   board,
                'count':  board_total,
                'papers': papers_out,
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


# ── API: 修复云端已存储题目的知识点分类 ──
@app.route('/api/cloud_library/repair_topics', methods=['POST'])
def cloud_library_repair_topics():
    """
    重新推导云端所有题目的 topic1/topic2，修复旧版 M1-1 过匹配导致的分类错误。
    对于每道有 topics[] 的题，用新逻辑重新推导 topic1/topic2：
      - 优先选有 parent_id 且不是通用章节（M1-1/M2-1/S1-1/FP1-1）的 subtopic
      - 其次选任意有 parent_id 的条目
      - 最后 fallback 到 topics[0]
    如果推导结果与存储值不同，则：
      1. 在新路径写新元数据
      2. 删除旧路径元数据（旧图片 key 不变，图片不需要移动）
    返回：{checked: int, repaired: int, failed: int}
    """
    data    = request.json or {}
    subject = data.get('subject', '')
    board   = data.get('board', '')
    dry_run = data.get('dry_run', False)   # dry_run=true 仅预览，不写盘

    GENERIC_CHAPTERS = {'M1-1', 'M2-1', 'S1-1', 'FP1-1'}

    # 加载 edexcel_maths 考纲，用于通过章节 id 反查 title
    maths_syllabus = _load_edexcel_maths_syllabus()
    maths_title_map: dict[str, str] = {}
    if maths_syllabus:
        for _t in maths_syllabus.get('topics', []):
            maths_title_map[str(_t['id'])] = _t.get('title', '')
            for _s in _t.get('subtopics', []):
                maths_title_map[str(_s['id'])] = _s.get('title', '')

    checked = 0
    repaired = 0
    failed = 0
    preview = []

    # 遍历要修复的学科/考试局组合
    subjects_to_check = [subject] if subject else _CLOUD_SUBJECTS
    boards_to_check   = [board]   if board   else _CLOUD_BOARDS

    for subj in subjects_to_check:
        for brd in boards_to_check:
            keys = _list_cloud_questions(subj, brd)
            for key in keys:
                q = _load_cloud_question(key)
                if not q:
                    continue
                checked += 1
                old_t1 = q.get('topic1', '')
                old_t2 = q.get('topic2', '')
                qtopics = q.get('topics') or []

                if not qtopics:
                    continue   # 无 topics 数组，无法推导，跳过

                # ── 用新逻辑推导 topic1/topic2 ──
                best = (
                    next((t for t in qtopics if t.get('parent_id') and
                          str(t.get('parent_id', '')) not in GENERIC_CHAPTERS), None) or
                    next((t for t in qtopics if t.get('parent_id')), None) or
                    qtopics[0]
                )
                pt      = best.get('parent_title', '')
                pid     = str(best.get('parent_id', ''))
                bt      = best.get('title', '')
                bid     = str(best.get('id', ''))

                if pt:
                    new_t1 = pt
                    new_t2 = bt or new_t1
                elif pid:
                    # 查 maths 考纲 title；Cambridge 结构
                    cam_syl = _load_syllabus()
                    cam_map: dict[str, str] = {}
                    if cam_syl:
                        for _t in cam_syl.get('topics', []):
                            cam_map[str(_t.get('id', ''))] = _t.get('title', '')
                    pt_resolved = cam_map.get(pid) or maths_title_map.get(pid) or pid
                    new_t1 = pt_resolved
                    new_t2 = bt or new_t1
                elif re.match(r'^[A-Z0-9]+-\d+(\.\d+)?$', bid):
                    new_t1 = bt or bid
                    new_t2 = bt or bid
                else:
                    new_t1 = bt or '未分类'
                    new_t2 = bt or '通用'

                if not new_t2 or new_t2 == '通用':
                    new_t2 = new_t1
                if not new_t1:
                    new_t1 = '未分类'

                # 与当前存储值相同 → 跳过
                if new_t1 == old_t1 and new_t2 == old_t2:
                    continue

                preview.append({'qid': q.get('qid', ''), 'old_t1': old_t1, 'old_t2': old_t2,
                                 'new_t1': new_t1, 'new_t2': new_t2})
                if dry_run:
                    repaired += 1
                    continue

                # ── 写到新路径 ──
                try:
                    q['topic1'] = new_t1
                    q['topic2'] = new_t2
                    maths_unit  = q.get('maths_unit', '')
                    new_key = _cloud_prefix(subj, brd, new_t1, new_t2, maths_unit=maths_unit) + \
                              (f'/{q["qid"]}.json' if storage.is_r2_mode()
                               else f'{os.sep}{q["qid"]}.json')
                    _save_cloud_question(new_key, q)

                    # ── 删除旧路径（仅当新旧路径不同时）──
                    if new_key != key:
                        if storage.is_r2_mode():
                            try: storage.delete_object(key)
                            except Exception: pass
                        else:
                            try: os.remove(key)
                            except Exception: pass
                    repaired += 1
                except Exception as e:
                    print(f'[repair_topics] error qid={q.get("qid","?")}: {e}')
                    failed += 1

    if not dry_run:
        _invalidate_stats()

    return jsonify({
        'checked':  checked,
        'repaired': repaired,
        'failed':   failed,
        'dry_run':  dry_run,
        'preview':  preview[:50],   # 最多返回前50条预览
    })


# ── API: 获取云端题库可用年份列表 ──
@app.route('/api/cloud_library/papers', methods=['GET'])
def cloud_library_papers():
    """
    返回云端题库中指定学科+考试局下所有 paper（maths_unit）的列表。
    可选参数：subject, board
    返回：{papers: ['P1','P2','P3',...]} 按字母排序
    """
    subject = request.args.get('subject', '')
    board   = request.args.get('board', '')

    all_keys = _list_cloud_questions(subject, board, '', '')
    papers = set()
    for key in all_keys:
        q = _load_cloud_question(key)
        if q:
            mu = q.get('maths_unit', '')
            if mu:
                papers.add(mu)
    return jsonify({'papers': sorted(papers)})


@app.route('/api/cloud_library/years', methods=['GET'])
def cloud_library_years():
    """
    返回云端题库中所有题目涉及的年份列表（从 exam_date 中提取）。
    可选参数：subject, board, maths_unit
    """
    subject    = request.args.get('subject', '')
    board      = request.args.get('board', '')
    maths_unit = request.args.get('maths_unit', '')

    all_keys = _list_cloud_questions(subject, board, '', '')
    years = set()
    for key in all_keys:
        q = _load_cloud_question(key)
        if q:
            # Task 3: 按 maths_unit 过滤
            if maths_unit and q.get('maths_unit', '') != maths_unit:
                continue
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
    按学科、考试局、paper(maths_unit)、年份筛选云端题目，返回一个新 session_id 供后续操作。
    请求体：{subject, board, maths_unit (可选), years: [str] (空=全部)}
    返回：{session_id, groups, total_questions}
    """
    import base64 as _b64
    data       = request.json or {}
    subject    = data.get('subject', '')
    board      = data.get('board', '')
    maths_unit = data.get('maths_unit', '')   # Task 3: paper 过滤
    years      = data.get('years', [])         # 空列表 = 全部年份

    if not subject or not board:
        return jsonify({'error': '必须提供 subject 和 board'}), 400

    all_keys = _list_cloud_questions(subject, board, '', '')
    if not all_keys:
        return jsonify({'error': f'云端题库中没有 {subject} / {board} 的题目'}), 404

    # 过滤 maths_unit + 年份
    selected_qs = []
    for key in all_keys:
        q = _load_cloud_question(key)
        if not q:
            continue
        # Task 3: 按 paper 过滤
        if maths_unit and q.get('maths_unit', '') != maths_unit:
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
        ans_b64 = ''

        if qid:
            # Task 5: 加载题目图片（question 子目录）
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

            # Task 5: 加载答案图片（scheme 子目录），放入 answer_b64
            if q.get('has_answer_image'):
                ans_key = _cloud_ans_img_key(qid)
                try:
                    if storage.is_r2_mode():
                        raw_ans = storage.load_bytes(ans_key)
                        if raw_ans:
                            ans_b64 = _b64.b64encode(raw_ans).decode()
                    else:
                        if os.path.isfile(ans_key):
                            with open(ans_key, 'rb') as f:
                                ans_b64 = _b64.b64encode(f.read()).decode()
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
            'answer_b64':    ans_b64,    # Task 5: 导入时携带答案，前端 toggleAnswer 可直接使用
            '_cloud_qid':    qid,        # 保留原始云端 ID
        })

    # 注册为虚拟 session
    # 推导 maths_unit：若题目都有相同的 maths_unit，用该值；否则用 maths_unit 参数
    q_mu_vals = [q.get('maths_unit', '') for q in selected_qs if q.get('maths_unit')]
    inferred_maths_unit = maths_unit or (q_mu_vals[0] if len(set(q_mu_vals)) == 1 else None)
    # 推导 paper_type：优先用 maths_unit 判定为 edexcel_maths，否则用题目里的值
    inferred_paper_type = 'edexcel_maths' if inferred_maths_unit else (
        selected_qs[0].get('paper_type', 'structured') if selected_qs else 'structured'
    )

    session_id = str(uuid.uuid4())
    virt_group = {
        'filename':        f'{subject}_{board}_云端导入.pdf',
        'path':            '',
        'r2_key':          '',
        'source':          'cloud',
        'paper_type':      inferred_paper_type,
        'maths_unit':      inferred_maths_unit,
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
            'maths_unit':      virt_group['maths_unit'],
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
