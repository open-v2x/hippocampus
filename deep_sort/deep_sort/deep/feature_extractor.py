import logging
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import sys
import os
import yaml  # type:ignore
from .model import Net
from pathlib import Path

path = Path.cwd().parent.parent.parent
if path not in sys.path:
    sys.path.append(str(path))
from utils.get_from_grpc import get_data

config_path = "config/bankend.yaml"


class Extractor(object):
    def __init__(self, model_path, use_cuda=True):
        with open(config_path, errors="ignore") as f:
            self.bankend = yaml.safe_load(f)["bankend"]
        self.size = (64, 128)
        self.norm = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        if self.bankend != "tfdl":
            self.init_model(model_path, use_cuda)

    def init_model(self, model_path, use_cuda):
        self.net = Net(reid=True)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)["net_dict"]
        self.net.load_state_dict(state_dict)
        logger = logging.getLogger("root.tracker")
        logger.info("Loading weights from {}... Done!".format(model_path))
        self.net.to(self.device)

    def _preprocess(self, im_crops):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """

        def _resize(im, size):
            return cv2.resize(im.astype(np.float32) / 255.0, size)

        im_batch = torch.cat(
            [self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0
        ).float()
        return im_batch

    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        if self.bankend == "tfdl":
            features = []
            for im in im_batch:
                input = im.reshape(
                    [
                        24576,
                    ]
                )
                reply = get_data(model="deepsort_int8", shape=[3, 128, 64], input=input)
                features.append(reply.output[0].data)
            return np.reshape(features, [len(im_crops), 512])
        else:
            with torch.no_grad():
                im_batch = im_batch.to(self.device)
                features = self.net(im_batch)
        return features.cpu().numpy()


if __name__ == "__main__":
    import os

    img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "car2.png")
    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoint/ckpt.t7")

    img = cv2.imread(img_path)[:, :, (2, 1, 0)]
    extr = Extractor(ckpt_path)
    feature = extr([img])
    print(feature.shape)
