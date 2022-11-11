# RTSP-Simulator

rtsp-simulator provides a Dockerfile that combines ffmpeg and rtsp-simple server. default rtsp url
host is localhost. RTSP UDP/TCP listener: 8554

## Usage

### build image

```bash
docker build -t rtsp-simulator:latest .
```

### run container

```bash
docker run -d -e SOURCE_URL=<your-video-source-file> -v {source-file-mount} --restart=always --name=rtsp_simulator --net=host rtsp_simulator:latest
```

## Envs

- SOURCE_URL - source media URL to be used as input for RTSP stream. Cannot be empty (etc.
  SOURCE_URL=/tmp/workdir/test.mp4)
- RTSP_PROXY_SOURCE_TCP - whetever source RTSP stream is UDP (none) or TCP ('-rtsp_transport tcp').
  defaults to '-rtsp_transport tcp'
- STREAM_NAME - path for "rtsp://[host]:[port]/[STREAM_NAME]. default to 'mystream'
- FFMPEG_ARGS - additional arguments to ffmpeg publisher. ex: "-err_detect aggressive -fflags
  discardcorrupt"
- FFMPEG_INPUT_ARGS - ffmpeg args applied near input params. ex.: '-v 10'. defaults to ''
- FFMPEG_OUTPUT_ARGS - ffmpeg args applied near output params. ex.: '-vcodec h264'. defaults to
  '-c:v libx264' .
