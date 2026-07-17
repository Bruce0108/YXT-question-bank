"""
storage.py — 统一文件存储抽象层
===================================
支持两种模式，通过环境变量自动切换：

  本地模式（开发/沙盒）: R2_BUCKET_NAME 未设置  → 读写本地磁盘
  云端模式（Railway）:   R2_BUCKET_NAME 已设置  → 读写 Cloudflare R2

环境变量（Railway 中设置）：
  R2_BUCKET_NAME      必须，Cloudflare R2 存储桶名称
  R2_ACCOUNT_ID       必须，Cloudflare Account ID
  R2_ACCESS_KEY_ID    必须，R2 API Token 的 Access Key ID
  R2_SECRET_ACCESS_KEY 必须，R2 API Token 的 Secret Access Key
  LOCAL_TMP_DIR       可选，本地临时目录（默认 /tmp/pdf_uploads）

文件布局（R2 Key 前缀）：
  multi/{session_id}_{filename}.pdf      上传的 PDF 原文件
  sessions/{session_id}.json             Session 元数据
  library/{board}/{subject}/{wb_id}/     图库题册
    manifest.json
    q_001.jpg, q_002.jpg, ...
  output/{task_id}.pdf                   导出的 PDF

对外接口（本地 / R2 统一）：
  store_bytes(key, data: bytes)
  load_bytes(key) -> bytes | None
  store_text(key, text: str)
  load_text(key) -> str | None
  delete_object(key)
  list_prefix(prefix) -> [key, ...]
  exists(key) -> bool
  make_local_copy(key, local_path)       下载到本地临时文件（供 fitz.open 使用）
  upload_from_local(local_path, key)     把本地文件上传到存储
  local_tmp_path(filename) -> str        获取临时文件路径（本地模式直接返回，云端返回 /tmp/...）
"""

import os
import io
import json
import threading

# ── 环境变量读取 ──
_R2_BUCKET      = os.environ.get('R2_BUCKET_NAME', '')
_R2_ACCOUNT_ID  = os.environ.get('R2_ACCOUNT_ID', '')
_R2_ACCESS_KEY  = os.environ.get('R2_ACCESS_KEY_ID', '')
_R2_SECRET_KEY  = os.environ.get('R2_SECRET_ACCESS_KEY', '')
_LOCAL_TMP      = os.environ.get('LOCAL_TMP_DIR', '/tmp/pdf_uploads')
_USE_R2         = bool(_R2_BUCKET and _R2_ACCOUNT_ID and _R2_ACCESS_KEY and _R2_SECRET_KEY)

# 本地模式根目录（沙盒/开发时使用）
_LOCAL_ROOT = os.environ.get('LOCAL_STORAGE_ROOT',
              os.path.join(os.path.dirname(__file__), 'uploads'))

os.makedirs(_LOCAL_TMP, exist_ok=True)
if not _USE_R2:
    os.makedirs(_LOCAL_ROOT, exist_ok=True)

# ── R2 客户端（懒初始化，仅云端模式）──
_r2_client = None
_r2_lock   = threading.Lock()

def _get_r2():
    global _r2_client
    if _r2_client is not None:
        return _r2_client
    with _r2_lock:
        if _r2_client is not None:
            return _r2_client
        import boto3
        endpoint = f'https://{_R2_ACCOUNT_ID}.r2.cloudflarestorage.com'
        _r2_client = boto3.client(
            's3',
            endpoint_url=endpoint,
            aws_access_key_id=_R2_ACCESS_KEY,
            aws_secret_access_key=_R2_SECRET_KEY,
            region_name='auto',
        )
    return _r2_client


def _local_key_path(key: str) -> str:
    """将 R2 key 转换为本地文件路径"""
    return os.path.join(_LOCAL_ROOT, key.replace('/', os.sep))


# ════════════════════════════════════════════════════════════
# 核心读写接口
# ════════════════════════════════════════════════════════════

def store_bytes(key: str, data: bytes) -> bool:
    """上传二进制数据到存储"""
    if _USE_R2:
        try:
            _get_r2().put_object(Bucket=_R2_BUCKET, Key=key, Body=data)
            return True
        except Exception as e:
            print(f'[storage] R2 put_object failed: {key} — {e}')
            return False
    else:
        path = _local_key_path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + '.tmp'
        try:
            with open(tmp, 'wb') as f:
                f.write(data)
            os.replace(tmp, path)
            return True
        except Exception as e:
            print(f'[storage] local write failed: {path} — {e}')
            return False


def load_bytes(key: str):
    """下载二进制数据，找不到返回 None"""
    if _USE_R2:
        try:
            resp = _get_r2().get_object(Bucket=_R2_BUCKET, Key=key)
            return resp['Body'].read()
        except Exception:
            return None
    else:
        path = _local_key_path(key)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, 'rb') as f:
                return f.read()
        except Exception:
            return None


def store_text(key: str, text: str) -> bool:
    """上传 UTF-8 文本"""
    return store_bytes(key, text.encode('utf-8'))


def load_text(key: str):
    """下载文本，找不到返回 None"""
    data = load_bytes(key)
    if data is None:
        return None
    return data.decode('utf-8')


def store_json(key: str, obj) -> bool:
    """上传 JSON 对象"""
    return store_text(key, json.dumps(obj, ensure_ascii=False))


def load_json(key: str):
    """下载 JSON，找不到返回 None"""
    text = load_text(key)
    if text is None:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def delete_object(key: str):
    """删除单个对象"""
    if _USE_R2:
        try:
            _get_r2().delete_object(Bucket=_R2_BUCKET, Key=key)
        except Exception:
            pass
    else:
        path = _local_key_path(key)
        try:
            os.remove(path)
        except Exception:
            pass


def delete_prefix(prefix: str):
    """删除某前缀下的所有对象（R2 批量删除 / 本地 rmtree）"""
    if _USE_R2:
        keys = list_prefix(prefix)
        if not keys:
            return
        try:
            _get_r2().delete_objects(
                Bucket=_R2_BUCKET,
                Delete={'Objects': [{'Key': k} for k in keys]}
            )
        except Exception as e:
            print(f'[storage] R2 batch delete failed: {prefix} — {e}')
    else:
        import shutil
        path = _local_key_path(prefix)
        shutil.rmtree(path, ignore_errors=True)


def list_prefix(prefix: str) -> list:
    """列出某前缀下所有 key（不递归分页，最多 1000 个）"""
    if _USE_R2:
        try:
            paginator = _get_r2().get_paginator('list_objects_v2')
            keys = []
            for page in paginator.paginate(Bucket=_R2_BUCKET, Prefix=prefix):
                for obj in page.get('Contents', []):
                    keys.append(obj['Key'])
            return keys
        except Exception:
            return []
    else:
        base = _local_key_path(prefix)
        if not os.path.isdir(base):
            # 也许是文件前缀，列同目录下匹配的文件
            parent = os.path.dirname(base)
            name   = os.path.basename(base)
            if not os.path.isdir(parent):
                return []
            results = []
            for f in os.listdir(parent):
                if f.startswith(name):
                    rel = os.path.relpath(os.path.join(parent, f), _LOCAL_ROOT)
                    results.append(rel.replace(os.sep, '/'))
            return results
        results = []
        for root, dirs, files in os.walk(base):
            for fname in files:
                full = os.path.join(root, fname)
                rel  = os.path.relpath(full, _LOCAL_ROOT)
                results.append(rel.replace(os.sep, '/'))
        return results


def exists(key: str) -> bool:
    """检查对象是否存在"""
    if _USE_R2:
        try:
            _get_r2().head_object(Bucket=_R2_BUCKET, Key=key)
            return True
        except Exception:
            return False
    else:
        return os.path.isfile(_local_key_path(key))


# ════════════════════════════════════════════════════════════
# 本地临时文件辅助（供 fitz.open / send_file 使用）
# ════════════════════════════════════════════════════════════

def local_tmp_path(filename: str) -> str:
    """返回一个临时文件的完整路径（在 /tmp/pdf_uploads 下）"""
    return os.path.join(_LOCAL_TMP, filename)


def make_local_copy(key: str, local_path: str) -> bool:
    """
    将存储中的对象复制到本地临时文件。
    本地模式：直接返回本地路径（无需复制，原文件就在磁盘上）。
    R2 模式：下载到 local_path。
    返回 True 表示成功。
    """
    if not _USE_R2:
        # 本地模式：源文件就是本地文件，检查存在即可
        src = _local_key_path(key)
        if os.path.isfile(src):
            if src != local_path:
                import shutil
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                shutil.copy2(src, local_path)
            return True
        return False
    else:
        data = load_bytes(key)
        if data is None:
            return False
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, 'wb') as f:
            f.write(data)
        return True


def upload_from_local(local_path: str, key: str) -> bool:
    """
    将本地文件上传到存储。
    本地模式：把文件移动/链接到正确位置。
    R2 模式：上传文件内容。
    """
    if not _USE_R2:
        dest = _local_key_path(key)
        if os.path.abspath(local_path) == os.path.abspath(dest):
            return True  # 已在正确位置
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        import shutil
        shutil.copy2(local_path, dest)
        return True
    else:
        try:
            with open(local_path, 'rb') as f:
                _get_r2().put_object(Bucket=_R2_BUCKET, Key=key, Body=f)
            return True
        except Exception as e:
            print(f'[storage] upload_from_local failed: {local_path} -> {key} — {e}')
            return False


def get_local_path_for_key(key: str) -> str:
    """
    本地模式：直接返回本地文件路径。
    R2 模式：下载到 /tmp 并返回本地临时路径。
    调用者负责在用完后删除临时文件（R2 模式）。
    """
    if not _USE_R2:
        return _local_key_path(key)
    # R2: 下载到临时目录
    filename = key.replace('/', '_')
    tmp_path = local_tmp_path(filename)
    if make_local_copy(key, tmp_path):
        return tmp_path
    return ''


def is_r2_mode() -> bool:
    """当前是否在 R2 云端模式"""
    return _USE_R2
