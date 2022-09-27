FROM docker.io/library/python:3.9.12-slim-buster

LABEL purpose="cerebrum"

WORKDIR /home/hippocampus
COPY . /home/hippocampus/

RUN cp /home/hippocampus/start.sh /usr/local/bin/start.sh \
    && mkdir -p /root/.config/Ultralytics/ \
    && cp /home/hippocampus/Arial.ttf /root/.config/Ultralytics/Arial.ttf \
    && mv /etc/apt/sources.list /etc/apt/sources.list.bak \
    && cp /home/hippocampus/sources.list /etc/apt/ \
    && apt-get update \
    && apt-get install -y wget curl ffmpeg \
    && /usr/local/bin/python -m pip install --upgrade pip \
    && pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r /home/hippocampus/requirements.txt


CMD ["sh", "/usr/local/bin/start.sh"]