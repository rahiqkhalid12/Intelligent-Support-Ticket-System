FROM python:3.10-slim

# منع إنشاء pyc وتحسين الـ logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# انسخ ملف المتطلبات أولًا للاستفادة من Docker Cache
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# انسخ باقي ملفات المشروع
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]