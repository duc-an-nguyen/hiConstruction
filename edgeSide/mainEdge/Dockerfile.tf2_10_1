FROM tensorflow/tensorflow:2.10.1-gpu

RUN pip install --no-cache-dir paho-mqtt
RUN pip install pandas==1.5.2
RUN pip install matplotlib==3.5.3
RUN pip install py-cpuinfo==6.0.0 Cython==0.29.26 blosc2==2.0.0 numexpr==2.8.0 
RUN pip install scipy==1.7.2
RUN pip install scikit-learn==1.0.2
RUN pip install nbformat==5.1.3
RUN pip install tqdm==4.61.2
RUN pip install numba==0.55.0
RUN pip install tables==3.8.0


WORKDIR /app
COPY main.py .
COPY tepEdgeDevice.py .
COPY preProcessingDataTEP.py .
COPY ai_handling.py .
COPY input_signal2_ai.py .
COPY model/ /app/model/
COPY utility/ /app/utility/


CMD [ "python", "main.py"]
