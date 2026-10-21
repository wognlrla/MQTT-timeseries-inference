FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 TZ=Asia/Seoul
WORKDIR /app
RUN pip install --no-cache-dir pandas==2.2.3 paho-mqtt==2.1.0
COPY simfeed.py ./
CMD ["python", "simfeed.py"]
