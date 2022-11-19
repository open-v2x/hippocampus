FROM golang:1.17.12-alpine AS BUILD

RUN apk --update add git
ENV GOPROXY=https://goproxy.cn,direct

#RTSP SIMPLE SERVER
WORKDIR /tmp
RUN git clone https://gitclone.com/github.com/aler9/rtsp-simple-server.git
WORKDIR /tmp/rtsp-simple-server

RUN go mod download
RUN go build -o /go/bin/rtsp-simple-server .


FROM bitr0t/ffmpeg-alpine:latest

ENV SOURCE_URL ''
ENV STREAM_NAME 'mystream'
ENV RTSP_PROXY_SOURCE_TCP '-rtsp_transport tcp'
ENV FORCE_FFMPEG 'true'
ENV FFMPEG_INPUT_ARGS ''
ENV FFMPEG_OUTPUT_ARGS '-c:v libx264'

RUN apk --update add gettext bash

COPY --from=BUILD /go/bin/rtsp-simple-server /bin/rtsp-simple-server

ADD start.sh /
ADD rtsp-simple-server.yml /

ENTRYPOINT [ "/bin/bash" ]

CMD ["/start.sh"]