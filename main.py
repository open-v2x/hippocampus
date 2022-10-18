from detector import Detector
from utils.requests_utils import post
import cv2
import os
import subprocess
import yaml
import multiprocessing

def func(camera):

    # 创建livego channel,获得 channelkey
    url = "http://localhost:8090/control/get?room="+camera.get("channel_name","")
    channlkey = post(url)
    if channlkey == "":
        print("Did not get channelkey")
        return

    while True:
        det = Detector()
        cap = cv2.VideoCapture(os.getenv('rtsp') or camera["camera_id"])
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        sizeStr = str(size[0]) + 'x' + str(size[1])
        rtmp="rtmp://localhost:1935/live/" + channlkey
        command = ['ffmpeg',
                '-y',
                '-stream_loop', '-1',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', '720x480',
                '-r', '25',
                '-i', '-',
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-preset', 'ultrafast',
                '-f', 'flv',
                rtmp]
        pipe = subprocess.Popen(command, stdin=subprocess.PIPE)

        while True:
            ret, im = cap.read()
            if not ret:
                break
            im = cv2.resize(im_rgb, (720, 480), interpolation=cv2.INTER_LINEAR)
            result = det.feedCap(im)
            result = result['frame']
            pipe.stdin.write(result.tobytes())

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

    # 获取 hippocampus接入的RSU
    key = os.getenv("RSU") or "R328328"
    with open(yml_path, encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.Loader)

    processes = []
    multiprocessing.set_start_method("spawn")
    for camera in data[key]:
        # 开启进程去执行任务
        p=multiprocessing.Process(target = func,args=(camera,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


if __name__ == '__main__':
    main()
