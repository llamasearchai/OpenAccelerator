# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app/src

CMD ["uvicorn", "src.open_accelerator.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
