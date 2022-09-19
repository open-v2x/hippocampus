from detector import Detector
from camera import LoadStreams, LoadImages
from utils.general import non_max_suppression, scale_coords, letterbox, check_imshow
from flask import Response
from flask import Flask
from flask import render_template
import tracker
import imutils
import time
import torch
import json
import cv2
import os


# initialize a flask object
app = Flask(__name__)

# initialize the video stream and allow the camera sensor to warmup
with open('config.json', 'r', encoding='utf8') as fp:
    opt = json.load(fp)
    print('[INFO] Object Detection and Tracking Config:', opt)

det = Detector(opt)
if det.webcam:
    # cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams(det.source, img_size=opt["imgsz"], stride=det.stride)
else:
    dataset = LoadImages(det.source, img_size=opt["imgsz"], stride=det.stride)
time.sleep(2.0)

@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")

def detect_gen(dataset, feed_type):
    view_img = check_imshow()
    t0 = time.time()
    for path, img, img0s, vid_cap in dataset:
        img = Detector.preprocess(img)

        t1 = time.time()
        pred = Detector.model(img, augment=Detector.opt["augment"])[0]  # 0.22s
        pred = pred.float()
        pred = non_max_suppression(pred, Detector.opt["conf_thres"], Detector.opt["iou_thres"])
        t2 = time.time()

        pred_boxes = []
        for i, det in enumerate(pred):
            if Detector.webcam:  # batch_size >= 1
                feed_type_curr, p, s, im0, frame = "Camera_%s" % str(i), path[i], '%g: ' % i, img0s[i].copy(), dataset.count
            else:
                feed_type_curr, p, s, im0, frame = "Camera", path, '', img0s, getattr(dataset, 'frame', 0)

            s += '%gx%g ' % img.shape[2:]  # print string
            if det is not None and len(det):
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {Detector.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                for *xyxy, conf, cls_id in det:
                    lbl = Detector.names[int(cls_id)]
                    if not lbl in ['person', 'car', 'truck']:
                        continue
                    xyxy = torch.tensor(xyxy).view(1, 4).view(-1).tolist()
                    score = round(conf.tolist(), 3)
                    label = "{}: {}".format(lbl, score)
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    pred_boxes.append((x1, y1, x2, y2, lbl, score))
                    if view_img:
                        Detector.plot_one_box(xyxy, im0, color=(255, 0, 0), label=label)

            print(f'{s}Done. ({t2 - t1:.3f}s)')
            if feed_type_curr == feed_type:
                frame = cv2.imencode('.jpg', im0)[1].tobytes()
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed/<feed_type>')
def video_feed(feed_type):
    """Video streaming route. Put this in the src attribute of an img tag."""
    if feed_type == 'Camera_0':
        return Response(detect_gen(dataset=dataset, feed_type=feed_type),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    elif feed_type == 'Camera_1':
        return Response(detect_gen(dataset=dataset, feed_type=feed_type),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

def main():
    name = 'demo'
    b_boxes_list = []
    det = Detector()
    cap = cv2.VideoCapture('rtsp://0.0.0.0:1554/test.mp4')
    fps = 60
    print('fps:', fps)
    t = int(1000/fps)

    videoWriter = None

    while True:

        # try:
        _, im = cap.read()
        if im is None:
            break
        
        result = det.feedCap(im) 
        _, b_boxes = det.detect(im)
        result = result['frame']
        print("bouding_boxes=", b_boxes)
        b_boxes_list.append(b_boxes)
        result = imutils.resize(result, height=500)
        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc(
                'm', 'p', '4', 'v')  # opencv3.0
            videoWriter = cv2.VideoWriter(
                'result.mp4', fourcc, fps, (result.shape[1], result.shape[0]))

        videoWriter.write(result)
        cv2.imshow(name, result)
        cv2.waitKey(t)

        if cv2.getWindowProperty(name, cv2.WND_PROP_AUTOSIZE) < 1:
            # 点x退出
            break
        # except Exception as e:
        #     print(e)
        #     break
    
    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    
    main()