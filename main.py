import os
import subprocess
import threading
import time
from collections import deque

import cv2

from detector import Detector
from tracker import plot_bboxes

deq = deque(maxlen=50)


def Receive():
    print("start Reveive ...")
    cap = cv2.VideoCapture(os.getenv("rtsp"))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = 50 / fps
    while cap.isOpened():
        _, frame = cap.read()
        deq.append(frame)
        if len(deq) >= 50:
            time.sleep(delay)


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
        "1920x1080",
        "-r",
        "25",
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
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


if __name__ == "__main__":
    p1 = threading.Thread(target=Receive)
    p2 = threading.Thread(target=Stream)
    p1.start()
    p2.start()
