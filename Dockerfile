FROM docker.io/library/python:3.9.12-slim-buster

LABEL purpose="cerebrum"

WORKDIR /home/hippocampus
COPY . /home/hippocampus/

RUN cp /home/hippocampus/start.sh /usr/local/bin/start.sh \
    && mkdir -p /root/.config/Ultralytics/ \
    && apt-get update \
    && apt-get install -y wget curl ffmpeg \
    && /usr/local/bin/python -m pip install --upgrade pip \
    && pip install -r /home/hippocampus/requirements.txt \
    && rm -rf ~/.cache/pip


CMD ["sh", "/usr/local/bin/start.sh"]