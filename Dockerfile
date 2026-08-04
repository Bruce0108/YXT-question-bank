# 使用官方 Python 镜像
FROM python:3.13-slim

# 安装系统依赖（PyMuPDF 需要）
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 先复制 requirements.txt（利用 Docker 缓存层）
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制所有代码
COPY . .

# 创建临时目录
RUN mkdir -p /tmp/pdf_uploads

# 用 shell 形式的 CMD，确保 $PORT 环境变量能被正确展开
CMD ["sh", "-c", "gunicorn app:app --workers 1 --threads 8 --bind 0.0.0.0:${PORT:-8080} --timeout 300 --worker-class gthread"]
