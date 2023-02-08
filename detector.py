import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml  # type:ignore

from BaseDetector import baseDet
from models.common import DetectMultiBackend
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.get_from_grpc import get_data
from utils.torch_utils import select_device

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


class Detector(baseDet):
    def __init__(self):
        super(Detector, self).__init__()
        with open("config/bankend.yaml", errors="ignore") as f:
            self.bankend = yaml.safe_load(f)["bankend"]
        self.imgsz = [640, 640]
        self.stride = 32
        with open("data/coco128.yaml", errors="ignore") as f:
            self.names = yaml.safe_load(f)["names"]
        if self.bankend != "tfdl":
            self.init_model()
        self.build_config()

    def init_model(self):
        self.weights = "weights/yolov5s.pt"
        self.device = "0" if torch.cuda.is_available() else "cpu"
        # Load model
        self.device = select_device(self.device)
        self.model = DetectMultiBackend(
            self.weights, device=self.device, dnn=False, data=ROOT / "data/coco128.yaml"
        )
        stride, names = self.model.stride, self.model.names
        check_img_size(self.imgsz, s=stride)  # check image size

        self.model.model.float()
        self.names = self.model.module.names if hasattr(self.model, "module") else names

    def preprocess(self, img):
        img0 = img.copy()
        img = letterbox(img0, self.imgsz, stride=self.stride, auto=False)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.tensor(img).float()
        img /= 255  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        return img0, img

    def detect(self, im):
        im0, img = self.preprocess(im)
        pred = self.get_pred(img=img)
        pred = non_max_suppression(
            pred,
            self.threshold,
            iou_thres=0.45,
            classes=None,
            agnostic=False,
            multi_label=False,
            labels=(),
            max_det=300,
        )

        pred_boxes = []
        for det in pred:
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for *x, conf, cls_id in det:
                    lbl = self.names[int(cls_id)]
                    if lbl not in ["person", "car", "truck"]:
                        continue
                    x1, y1 = int(x[0]), int(x[1])
                    x2, y2 = int(x[2]), int(x[3])
                    pred_boxes.append((x1, y1, x2, y2, lbl, conf))

        return im, pred_boxes

    def get_pred(self, img):
        if self.bankend == "tfdl":
            input = (
                img.cpu()
                .numpy()
                .reshape(
                    [
                        1228800,
                    ]
                )
            )
            reply = get_data(model="yolov5s_int8", shape=[1, 3, 640, 640], input=input)
            pred = reply.output[0].data
            pred = torch.tensor(np.reshape(pred, [1, 25200, 85]))
        else:
            pred = self.model(img, augment=False)
        return pred


if __name__ == "__main__":
    img = cv2.imread("test.jpg")
    img = cv2.resize(img, (640, 640))
    det = Detector()
    im, pred_boxes = det.detect(img)
    print(pred_boxes)
