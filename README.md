# 教学教研题库系统 (PDF Tool)

## 项目概述
- **名称**: 教学教研题库
- **功能**: 将 Cambridge / Edexcel 试卷 PDF 自动解析，按题切割成图片，存储到图库，支持按知识点筛选、组卷导出 PDF
- **运行方式**: Python Flask 在 Sandbox 本地运行；图片和题库数据存储到 Cloudflare R2 云端

## 访问地址
- **本地开发**: http://localhost:3000
- **Sandbox 预览**: https://3000-iusesdzresna0c72onrm4-583b4d74.sandbox.novita.ai

## 技术栈
- **后端**: Python 3 + Flask + gunicorn
- **PDF解析**: PyMuPDF (fitz)
- **图片处理**: Pillow
- **云端存储**: Cloudflare R2 (via boto3/S3 兼容 API)
- **进程管理**: PM2 + gunicorn

## 支持的试卷格式

| 格式 | 说明 |
|------|------|
| `mcq` | Cambridge 纯选择题 |
| `structured` | Cambridge 大题（结构化问答）|
| `edexcel` | Edexcel 大题/混合题型 |
| `edexcel_mcq` | Edexcel 纯选择题 |
| `edexcel_maths` | Edexcel IAL 纯数 (P1–P4) |

## 主要功能

1. **PDF上传解析** - 支持多文件同时上传，自动识别试卷来源和题型
2. **题目图片切割** - 精确截取每道题目为独立图片
3. **知识点标注** - 自动匹配 Cambridge A-Level 物理 / Edexcel IAL 数学知识点
4. **难度评级** - Edexcel P3 试卷支持 1-5 星难度标注（基于历年考官报告）
5. **图库管理** - 保存题目图片到云端图库，支持按考试局/科目/题册分类
6. **筛选导出** - 按知识点/难度筛选题目，批量导出 PDF 或 ZIP

## API 路由

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 主页 |
| POST | `/api/upload_multi` | 上传多个PDF文件 |
| GET | `/api/preview/<q_num>` | 预览题目图片 |
| GET | `/api/preview_b64/<q_num>` | 获取题目图片(base64) |
| POST | `/api/preview_b64_batch` | 批量获取题目图片 |
| GET | `/api/download/<q_num>` | 下载单题图片 |
| POST | `/api/download_batch` | 批量下载题目(ZIP) |
| POST | `/api/export_pdf` | 导出选中题目为PDF |
| GET | `/api/export_pdf/progress/<task_id>` | 查询PDF导出进度 |
| GET | `/api/export_pdf/download/<task_id>` | 下载导出的PDF |
| POST | `/api/library/save` | 保存题目到图库 |
| GET | `/api/library/list` | 列出图库中的题册 |
| GET | `/api/library/load/<wb_id>` | 加载题册 |
| DELETE | `/api/library/delete/<wb_id>` | 删除题册 |
| GET | `/api/syllabus` | 获取Cambridge 9702知识库 |
| GET | `/api/syllabus/edexcel_maths` | 获取Edexcel Maths知识库 |

## 数据架构

### 存储结构（R2 Key / 本地路径）
```
multi/{session_id}_{filename}.pdf      # 上传的 PDF 原文件
sessions/{session_id}.json             # Session 元数据
library/{board}/{subject}/{wb_id}/     # 图库题册
  manifest.json                          题册元信息
  q_001.jpg, q_002.jpg, ...             题目图片
output/{task_id}.pdf                   # 导出的 PDF
```

### 存储模式切换
- **本地模式**（默认）: 不设置 R2 环境变量，数据存储在 `uploads/` 目录
- **云端模式**: 设置以下环境变量，数据自动存储到 Cloudflare R2

## 配置云端存储（Cloudflare R2）

复制 `.env.example` 为 `.env`，填写 R2 配置：

```env
R2_BUCKET_NAME=pdf-tool-library        # R2 存储桶名称
R2_ACCOUNT_ID=your-account-id          # Cloudflare Account ID
R2_ACCESS_KEY_ID=your-access-key-id    # R2 API Access Key
R2_SECRET_ACCESS_KEY=your-secret-key   # R2 API Secret Key
```

### Cloudflare R2 创建步骤
1. 登录 [Cloudflare Dashboard](https://dash.cloudflare.com)
2. 进入 **R2 Object Storage** → **Create bucket**，创建名为 `pdf-tool-library` 的存储桶
3. 进入 **R2 Settings** → **Manage API tokens**，创建 API Token（赋予 Object Read/Write 权限）
4. 复制 Account ID、Access Key ID 和 Secret Key 到 `.env` 文件
5. 重启服务

## 本地运行

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务（开发模式）
python app.py

# 或使用 PM2 + gunicorn（生产模式）
pm2 start ecosystem.config.cjs

# 查看日志
pm2 logs pdf-tool --nostream
```

## 部署状态
- **平台**: Sandbox (Novita AI)
- **进程管理**: PM2 + gunicorn
- **状态**: ✅ 运行中（端口 3000）
- **最后更新**: 2026-07-17
