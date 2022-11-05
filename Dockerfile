# FROM docker.io/library/python:3.9.12-slim-buster
FROM docker.io/openv2x/hippocampus-base

LABEL purpose="hippocampus"

WORKDIR /home/hippocampus
COPY . /home/hippocampus/

RUN cp /home/hippocampus/start.sh /usr/local/bin/start.sh \
    && mkdir -p /root/.config/Ultralytics/ \
    && cp /home/hippocampus/Arial.ttf /root/.config/Ultralytics/ \
    # && apt-get update \
    # && apt-get install -y wget curl ffmpeg vim procps\
    && /usr/local/bin/python -m pip install --upgrade pip \
    && pip install -r /home/hippocampus/requirements.txt \
    && rm -rf ~/.cache/pip

CMD ["sh", "/usr/local/bin/start.sh"]
