FROM python:3.12-slim
RUN pip install --no-cache-dir paho-mqtt
WORKDIR /app
COPY mqttEdge.py .
CMD [ "python", "mqttEdge.py"]

