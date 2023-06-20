import datetime
import os
import subprocess
import threading
import time
from collections import deque

import cv2
import torch

from detector import Detector
from tracker import plot_bboxes

deq = deque(maxlen=20)
RECEIVE_SLEEP_INTERVAL_SECOND = 1
PROCESS_SLEEP_INTERVAL_SECOND = 0.1

cap = cv2.VideoCapture(os.getenv("rtsp"))

# 检查视频流是否成功打开
if not cap.isOpened():
    print("无法打开视频流")
    exit()

# 获取视频的宽度和高度
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
res_ratio = f"{width}x{height}"


# 释放资源
cap.release()
cv2.destroyAllWindows()


def Receive():
    print("start Reveive ...")
    while True:
        cap = cv2.VideoCapture(os.getenv("rtsp"))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            deq.append(frame)
        print("[{} UTC] Can't read rtsp".format(datetime.datetime.utcnow()))
        time.sleep(RECEIVE_SLEEP_INTERVAL_SECOND)


def Stream():
    print("Start Streaming ...")
    det = Detector()
    rtmp = "rtmp://localhost:1935/live/" + os.getenv("camera_id") + ".flv"
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        res_ratio,
        "-r",
        "25",
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-bufsize",
        "64M",
        "-preset",
        "ultrafast",
        "-f",
        "flv",
        rtmp,
    ]
    pipe = subprocess.Popen(command, stdin=subprocess.PIPE)

    i = 1
    count = 5
    bboxes = []
    if not torch.cuda.is_available():
        while True:
            if deq:
                im = deq.popleft()
                pipe.stdin.write(im.tobytes())
            else:
                time.sleep(PROCESS_SLEEP_INTERVAL_SECOND)
    while True:
        if deq:
            im = deq.popleft()
            if i == 0:
                bboxes.clear()
                result = det.feedCap(im)
                frame = result["frame"]
                if result["bboxes2draw"]:
                    bboxes.append(result["bboxes2draw"])
            else:
                if bboxes:
                    frame = plot_bboxes(im, bboxes[0])
                else:
                    frame = im
            pipe.stdin.write(frame.tobytes())
            i = (i + 1) % count
        else:
            time.sleep(PROCESS_SLEEP_INTERVAL_SECOND)


if __name__ == "__main__":
    p1 = threading.Thread(target=Receive, daemon=True)
    p2 = threading.Thread(target=Stream)
    p1.start()
    p2.start()
