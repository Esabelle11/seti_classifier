# 假设你的项目结构是：
# seti_classifier/
# ├── weight/
# │   └── model_weights_vgg16pre.pth
# ├── app/
# ├── src/
# └── Dockerfile

# Dockerfile 示例
FROM python:3.11

# 系统依赖（OpenCV 需要）
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# ⚡ 这里保证权重文件在容器里
COPY weight /app/weight

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
