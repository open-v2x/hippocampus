import os
import subprocess
import time

import cv2

from detector import Detector
from tracker import plot_bboxes


def main():

    det = Detector()
    cap = cv2.VideoCapture(os.getenv("rtsp"))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    sizeStr = str(size[0]) + "x" + str(size[1])
    fps = str(cap.get(cv2.CAP_PROP_FPS))
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
        sizeStr,
        "-r",
        fps,
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
        ret, im = cap.read()
        if not ret:
            time.sleep(1)
            continue
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

    # cap.release()


if __name__ == "__main__":
    main()
