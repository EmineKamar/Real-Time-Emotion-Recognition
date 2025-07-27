FROM python:3.10

WORKDIR /app

COPY . /app

RUN apt-get update && \
    apt-get install -y ffmpeg libsndfile1 portaudio19-dev && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

CMD ["python", "real_time_emotion.py"]
