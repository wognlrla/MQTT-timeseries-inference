FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 TZ=Asia/Seoul
WORKDIR /app
RUN pip install --no-cache-dir paho-mqtt==2.1.0 influxdb-client==1.49.0 pyyaml==6.0.2
COPY collector.py coverage.py ./
CMD ["python", "collector.py"]
