# FROM docker.io/library/python:3.9.12-slim-buster
FROM docker.io/openv2x/hippocampus-base

ARG GIT_BRANCH
ARG GIT_COMMIT
ARG RELEASE_VERSION
ARG REPO_URL

LABEL hippocampus.build_branch=${GIT_BRANCH} \
      hippocampus.build_commit=${GIT_COMMIT} \
      hippocampus.release_version=${RELEASE_VERSION} \
      hippocampus.repo_url=${REPO_URL}

WORKDIR /home/hippocampus
COPY . /home/hippocampus/

RUN pip install --upgrade pip \
    && pip install -r /home/hippocampus/requirements.txt \
    && cp /home/hippocampus/start.sh /usr/local/bin/start.sh \
    && mkdir -p /root/.config/Ultralytics/ \
    && cp /home/hippocampus/Arial.ttf /root/.config/Ultralytics/ \
    && rm -rf ~/.cache/pip

CMD ["sh", "/usr/local/bin/start.sh"]
