from detector import Detector
from utils.requests_utils import post
from tracker import plot_bboxes
import cv2
import os
import subprocess
import yaml
import multiprocessing


def func(camera):

    det = Detector()
    cap = cv2.VideoCapture(os.getenv('rtsp') or camera["rtsp"])
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    sizeStr = str(size[0]) + 'x' + str(size[1])
    fps = str(cap.get(cv2.CAP_PROP_FPS))
    rtmp="rtmp://localhost:1935/live/" + camera.get("camera_id") + ".flv"
    command = ['ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', sizeStr,
            '-r', fps,
            '-i', '-',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'ultrafast',
            '-f', 'flv',
            rtmp]
    pipe = subprocess.Popen(command, stdin=subprocess.PIPE)

    i = 1
    count = 3
    bboxes = []
    while True:
        ret, im = cap.read()
        if not ret:
            break
        if i == 0:
            bboxes.clear()
            result = det.feedCap(im)
            frame = result['frame']
            if result['bboxes2draw']:
                bboxes.append(result['bboxes2draw'])
        else:
            if bboxes: 
                frame = plot_bboxes(im, bboxes[0])
            else:
                frame = im
        pipe.stdin.write(frame.tobytes())
        i = i + 1
        i = i % count

    cap.release()


def main():
    """
    1 配置 Config 文件
    2 读取 yaml 文件，获取需要接入的 RSU
    3 开启进程 调用 fun 函数，每个线程都传入一个 CameraID
    4 每个线程中开启多进程进行 ffmpeg 推流

    Returns
    -------

    """
    yml_path = "config/default.yml"

    with open(yml_path, encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.Loader)

    processes = []
    multiprocessing.set_start_method("spawn")
    for key in data:
        for camera in data[key]:
            # 开启进程去执行任务
            p=multiprocessing.Process(target = func,args=(camera,))
            p.start()
            processes.append(p)
    for p in processes:
        p.join()


if __name__ == '__main__':
    main()