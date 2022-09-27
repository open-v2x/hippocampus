from detector import Detector
import cv2
import os
import subprocess

def main():

    det = Detector()
    cap = cv2.VideoCapture(os.getenv('rtsp') or 'video/test.mp4')
    # size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # sizeStr = str(size[0]) + 'x' + str(size[1])
    rtmp = os.getenv('rtmp') or 'rtmp://localhost:1935/live/rfBd56ti2SMtYvSgD5xAV0YU99zampta7Z7S575KLkIZ9PYk'
    command = ['ffmpeg',
               '-y',
               '-stream_loop', '-1', 
               '-f', 'rawvideo',
               '-vcodec', 'rawvideo',
               '-pix_fmt', 'bgr24',
               '-s','480x270',
               '-r', '25',
               '-i', '-',
               '-c:v', 'libx264',
               '-pix_fmt', 'yuv420p',
               '-preset', 'ultrafast',
               '-f', 'flv',
               rtmp]
    ffrtsp = subprocess.Popen(command, stdin=subprocess.PIPE)


    while True:

        ret, im = cap.read()
        if not ret:
            break
        im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im_rgb, (480, 270), interpolation=cv2.INTER_LINEAR)
        result = det.feedCap(im)
        result = result['frame']
        ffrtsp.stdin.write(result.tobytes())

    cap.release()

if __name__ == '__main__':
    
    main()