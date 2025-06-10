FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8080
ENV PYTHONPATH=/app
ENV DJANGO_SETTINGS_MODULE=config.settings
ENV DJANGO_COLLECTSTATIC=0

# 静的ファイル対応しないなら以下でOK
CMD ["gunicorn", "config.wsgi:application", "--bind", "0.0.0.0:8080"]
