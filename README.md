# hippocampus
OpenV2X Video Perception Module, it supports Video Object Detection (VID) and Multiple Object Tracking (MOT)

## Prerequisites
If you want to stream your video analyze results via RTMP, you need to have the prerequisites in your environments.
 - ffmpeg (https://ffmpeg.org/), and
 - live_go(https://github.com/gwuhaolin/livego)
Here we recommend use docker to install live_go and use apt install for ffmpeg(ubuntu env):
```bash
docker pull gwuhaolin/livego

apt install ffmpeg

docker run -p 1935:1935 -p 7001:7001 -p 7002:7002 -p 8090:8090 -d gwuhaolin/livego
```

Live_go acts as a RTMP publisher to publish. By default, port 1935 is used.

## How to use
Get channelkey by opening (http://localhost:8090/control/get?room=result) in your browser, then copy `data` field as channelkey. The RTMP url will be look like rtmp://localhost:1935/{appname}/{channelkey} ('live' as default appname).Then modify main.py line 13 with the RTMP url.
Finally start the service by running main.py through command:
```bash
python main.py
```

Once the inference is started, "main.py" will pipe every infered frame to ffmpeg. FFmpeg then live transcodes and outputs to rtmp://<your-machine's-ip>:1935/{appname}/{channelkey}

Open browser or streaming software like VLC and stream from the url:
http://localhost:7001/live/result.flv