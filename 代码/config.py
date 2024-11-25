# -*- coding: utf-8 -*-
# @Time    : 2020-02-26 17:54
# @Author  : Zonas
# @Email   : zonas.wang@gmail.com
# @File    : config.py
"""

"""
import os


class UNetConfig:

    def __init__(
        self,
        epochs=180,  # Number of epochs
        batch_size=32,  # Batch size
        # Percent of the data that is used as validation (0-100)
        validation=0.1,
        out_threshold=0.5,
        optimizer="AdamW",
        lr=4e-4,  # learning rate
        lr_decay_milestones=[
            10,
            20,
            30,
            40,
            50,
            60,
            65,
            70,
            80,
            85,
            90,
            160,
            170,
            180,
            230,
            240,
            250,
            260,
            270,
        ],
        lr_decay_gamma=0.9,
        weight_decay=5e-4,
        momentum=0.9,
        nesterov=True,
        n_channels=3,  # Number of channels in input images
        n_classes=5,  # Number of classes in the segmentation
        scale=1,  # Downscaling factor of the images
        load=False,  # Load model from a .pth file
        save_cp=True,
        model="PFNet",
        bilinear=True,
        deepsupervision=False,
    ):
        super(UNetConfig, self).__init__()

        self.images_dir = "data/1015/Images/train"
        self.masks_dir = "data/1015/Mask/train"

        self.val_images_dir = "data/1015/Images/val"
        self.val_masks_dir = "data/1015/Mask/val"

        self.checkpoints_dir = "./check/1015pf"

        self.epochs = epochs
        self.batch_size = batch_size
        self.validation = validation
        self.out_threshold = out_threshold

        self.optimizer = optimizer
        self.lr = lr
        self.lr_decay_milestones = lr_decay_milestones
        self.lr_decay_gamma = lr_decay_gamma
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.nesterov = nesterov

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.scale = scale

        self.load = load
        self.save_cp = save_cp

        self.model = model
        self.bilinear = bilinear
        self.deepsupervision = deepsupervision

        os.makedirs(self.checkpoints_dir, exist_ok=True)
