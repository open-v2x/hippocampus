# FROM docker.io/library/python:3.9.12-slim-buster
FROM docker.io/openv2x/hippocampus-base

LABEL purpose="hippocampus"

WORKDIR /home/hippocampus
COPY . /home/hippocampus/

RUN pip install --upgrade pip \
    && pip install -r /home/hippocampus/requirements.txt \
    && cp /home/hippocampus/start.sh /usr/local/bin/start.sh \
    && mkdir -p /root/.config/Ultralytics/ \
    && cp /home/hippocampus/Arial.ttf /root/.config/Ultralytics/ \
    && rm -rf ~/.cache/pip

CMD ["sh", "/usr/local/bin/start.sh"]
