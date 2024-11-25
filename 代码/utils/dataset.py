import os
import os.path as osp
import logging

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
import albumentations as A
import torchvision.transforms as transforms
from config import UNetConfig
import random
import cv2

transform = A.Compose(
    [
        A.HorizontalFlip(),
        A.VerticalFlip(),
        # A.InvertImg(),
        A.Flip(),
        A.RandomRotate90(),
    ]
)
cfg = UNetConfig()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


## ap: aug image
def aug_img(img, mask):
    transformed = transform(image=img, mask=mask)
    img = transformed["image"]
    mask = transformed["mask"]
    return img, mask


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, img_names=None, scale=1, traing=False):
        self.img_names = []
        self.top_img_dir = imgs_dir
        if img_names == None:
            img_dirs = os.listdir(self.top_img_dir)
            for dir in img_dirs:
                for name in os.listdir(os.path.join(self.top_img_dir, dir)):
                    self.img_names.append(os.path.join(self.top_img_dir, dir, name))
        else:
            img_dirs = os.listdir(self.top_img_dir)
            names = os.listdir(os.path.join(self.top_img_dir, img_dirs[0]))
            for x in img_names:
                x = x.split(".")[0].split("/")[-1] + ".jpg"
                if x in names:
                    self.img_names.append(
                        os.path.join(self.top_img_dir, img_dirs[0], x)
                    )

        np.random.shuffle(self.img_names)
        self.top_mask_dir = masks_dir
        self.scale = scale
        self.traing = traing
        assert 0 < scale <= 1, "Scale must be between 0 and 1"

        logging.info(f"Creating dataset with {len(self.img_names)} examples")

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, i):
        img_path = self.img_names[i]
        dir_name = img_path.split(".")[0].split("/")[-2]
        img_name = img_path.split(".")[0].split("/")[-1]
        img_path = os.path.join(self.top_img_dir, dir_name, img_name + ".jpg")
        mask_path = os.path.join(self.top_mask_dir, dir_name, img_name + ".png")

        img = cv2.imread(img_path)
        mask = Image.open(mask_path)
        # mask = cv2.resize(np.array(mask),img.shape[0:2])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.traing:
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(
                np.array(mask), (256, 256), interpolation=cv2.INTER_NEAREST
            )
            img, mask = aug_img(img, mask)
        else:
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(
                np.array(mask), (256, 256), interpolation=cv2.INTER_NEAREST
            )
            img = np.array(img)
            mask = np.array(mask)
        img = img / 255
        # mask = np.where(mask != 0, mask - 1, 0)
        edge = np.where(mask != 0, 1, 0)

        assert (
            img.size == mask.size * 3
        ), f"Image and mask {img_name} should be the same size, but are {img.size} and {mask.size}"

        categorical = torch.unique(torch.tensor(np.array(mask)))
        categorical = F.one_hot(categorical.unsqueeze(0).to(torch.int64), cfg.n_classes)
        categorical = torch.sum(categorical, dim=1)
        categorical = torch.as_tensor(categorical, dtype=torch.float64)

        return {
            "image": torch.from_numpy(img.transpose(2, 0, 1)),
            "mask": torch.from_numpy(mask).unsqueeze(0),
            "categorical": categorical,
            "edge": torch.from_numpy(edge).unsqueeze(0),
        }
