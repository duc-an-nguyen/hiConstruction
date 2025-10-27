FROM python:3.12-slim

RUN pip install --no-cache-dir paho-mqtt

WORKDIR /app

COPY mqttServer.py .

CMD ["python", "mqttServer.py"]
