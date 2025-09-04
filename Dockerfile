# استخدم نسخة slim من Python لتقليل الحجم
FROM python:3.11-slim

# تثبيت أدوات أساسية
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# تحديد مجلد العمل
WORKDIR /app

# نسخ ملفات الكود و requirements
COPY . .

# تثبيت المتطلبات بدون cache لتقليل الحجم
RUN pip install --no-cache-dir -r requirements.txt

# كشف البورت
EXPOSE 8000

# تشغيل FastAPI باستخدام uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
