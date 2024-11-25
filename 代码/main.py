import argparse
from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    DeviceStatsMonitor,
)
from pytorch_lightning.tuner import Tuner
from torch.utils.data import random_split, DataLoader

from config import UNetConfig
from losses import LovaszLossSoftmax, structure_loss, ProbOhemCrossEntropy2d
from mynet.pfnet import PFNet
from utils.dataset import BasicDataset, setup_seed
from utils.metric import SegmentationMetric
from utils.AdamW import AdamW
from utils.lr_scheduler import WarmupMultiStepLR
from spiltDataSet import splitDataSet

import segmentation_models_pytorch as smp

# setup_seed(20)
pl.seed_everything(20)


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
cfg = UNetConfig()


class Pixcel_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterionPixel = LovaszLossSoftmax()
        self.criterionPixel1 = ProbOhemCrossEntropy2d()

    def forward(self, predict, mask):
        criterionPixel = self.criterionPixel(predict, mask)
        criterionPixel1 = self.criterionPixel1(predict, mask)
        lossLog = criterionPixel + criterionPixel1
        loss = (
            criterionPixel / criterionPixel.detach()
            + criterionPixel1 / criterionPixel1.detach()
        )
        return loss, lossLog


class Categorical_loss(nn.Module):
    def __init__(self):
        super(Categorical_loss, self).__init__()
        self.criterionCategorical = torch.nn.BCEWithLogitsLoss()

    def forward(self, predict, mask):
        return self.criterionCategorical(predict, mask)


class Model(pl.LightningModule):
    def __init__(self, model=None, pixcel_loss=None, categorical_loss=None):
        super(Model, self).__init__()
        if model != None:
            self.model = model
        else:
            self.model = PFNet(cfg)
        if pixcel_loss != None:
            self.pixcel_loss = pixcel_loss
        else:
            self.pixcel_loss = Pixcel_loss()
        if categorical_loss != None:
            self.categorical_loss = categorical_loss
        else:
            self.categorical_loss = Categorical_loss()
        self.edgeLoss = structure_loss()
        self.metric_test = SegmentationMetric(cfg.n_classes, True)
        self.metric = SegmentationMetric(cfg.n_classes)
        self.lr = cfg.lr
        self.iou = torch.zeros(cfg.n_classes - 1)
        self.cnt = 0

    def forward(self, x) -> Any:
        (
            predict4,
            predict3,
            predict2,
            predict1,
            categorical1,
            categorical2,
            categorical3,
            categorical4,
        ) = self.model(x)
        return predict1

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=5e-4)

        scheduler = WarmupMultiStepLR(
            optimizer=optimizer,
            milestones=cfg.lr_decay_milestones,
            gamma=cfg.lr_decay_gamma,
            warmup_iters=1,
        )

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        batch_imgs = batch["image"]
        batch_masks = batch["mask"]
        batch_categorical = batch["categorical"]
        batch_edge = batch["edge"]

        batch_imgs = batch_imgs.to(dtype=torch.float32)
        batch_edge = batch_edge.to(dtype=torch.float32)
        mask_type = torch.long
        batch_masks = batch_masks.to(dtype=mask_type)
        batch_categorical = batch_categorical.to(dtype=mask_type)

        (
            predict4,
            predict3,
            predict2,
            predict1,
            categorical1,
            categorical2,
            categorical3,
            categorical4,
            edge1,
            edge2,
            edge3,
            edge4,
        ) = self.model(batch_imgs)

        pixcel1, pixcel1Log = self.pixcel_loss(predict1, batch_masks.squeeze(1))
        pixcel2, pixcel2Log = self.pixcel_loss(predict2, batch_masks.squeeze(1))
        pixcel3, pixcel3Log = self.pixcel_loss(predict3, batch_masks.squeeze(1))
        pixcel4, pixcel4Log = self.pixcel_loss(predict4, batch_masks.squeeze(1))

        categorical_loss1 = self.categorical_loss(
            categorical1, batch_categorical.squeeze(1).float()
        )
        categorical_loss2 = self.categorical_loss(
            categorical2, batch_categorical.squeeze(1).float()
        )
        categorical_loss3 = self.categorical_loss(
            categorical3, batch_categorical.squeeze(1).float()
        )
        categorical_loss4 = self.categorical_loss(
            categorical4, batch_categorical.squeeze(1).float()
        )

        edgeLoss1 = self.edgeLoss(edge1, batch_edge)
        edgeLoss2 = self.edgeLoss(edge2, batch_edge)
        edgeLoss3 = self.edgeLoss(edge3, batch_edge)
        edgeLoss4 = self.edgeLoss(edge4, batch_edge)

        l1 = (
            pixcel1
            + categorical_loss1 / categorical_loss1.detach() * 0.5
            + edgeLoss1 / edgeLoss1.detach() * 0.5
        )
        l2 = (
            pixcel2
            + categorical_loss2 / categorical_loss2.detach() * 0.5
            + edgeLoss2 / edgeLoss1.detach() * 0.5
        )
        l3 = (
            pixcel3
            + categorical_loss3 / categorical_loss3.detach() * 0.5
            + edgeLoss3 / edgeLoss1.detach() * 0.5
        )
        l4 = (
            pixcel4
            + categorical_loss4 / categorical_loss4.detach() * 0.5
            + edgeLoss4 / edgeLoss1.detach() * 0.5
        )

        l1Log = pixcel1Log + categorical_loss1 + edgeLoss1
        l2Log = pixcel2Log + categorical_loss2 + edgeLoss2
        l3Log = pixcel3Log + categorical_loss3 + edgeLoss3
        l4Log = pixcel4Log + categorical_loss4 + edgeLoss4
        # class_wise = self.pixcel_loss(class_wise_mask, batch_masks.squeeze(1))
        # loss = pixcel1 + pixcel2 + pixcel3 + pixcel4 + categorical_loss + class_wise
        lossLog = l1Log * 4 + l2Log * 3 + l3Log * 2 + l4Log
        loss = l1 * 4 + l2 * 3 + l3 * 2 + l4
        # loss = self.pixcel_loss(predict1, batch_masks.squeeze(1))

        self.log("train_loss", lossLog, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_imgs = batch["image"]
        batch_masks = batch["mask"]
        batch_categorical = batch["categorical"]

        batch_imgs = batch_imgs.to(dtype=torch.float32)
        mask_type = torch.long
        batch_masks = batch_masks.to(dtype=mask_type)
        batch_categorical = batch_categorical.to(dtype=mask_type)

        (
            predict4,
            predict3,
            predict2,
            predict1,
            categorical1,
            categorical2,
            categorical3,
            categorical4,
        ) = self.model(batch_imgs)
        # predict1 = self.model(batch_imgs)

        self.metric.update(predict1, batch_masks.squeeze(1))
        acc, miou = self.metric.get()
        self.log("val_acc", acc, prog_bar=True, on_epoch=True)
        self.log("val_miou", miou, prog_bar=True, on_epoch=True)

    def on_validation_epoch_start(self) -> None:
        self.metric.reset()

    def test_step(self, batch, batch_idx):
        batch_imgs = batch["image"]
        batch_masks = batch["mask"]
        batch_categorical = batch["categorical"]

        batch_imgs = batch_imgs.to(dtype=torch.float32)
        mask_type = torch.long
        batch_masks = batch_masks.to(dtype=mask_type)
        batch_categorical = batch_categorical.to(dtype=mask_type)

        (
            predict4,
            predict3,
            predict2,
            predict1,
            categorical1,
            categorical2,
            categorical3,
            categorical4,
        ) = self.model(batch_imgs)
        # predict1 = self.model(batch_imgs)

        self.metric_test.update(predict1, batch_masks.squeeze(1))
        acc, miou, iou = self.metric_test.get()
        self.iou += iou
        self.cnt += 1
        self.log("test_acc", acc, prog_bar=True, on_epoch=True)
        self.log("test_miou", miou, prog_bar=True, on_epoch=True)

    def on_test_epoch_start(self) -> None:
        self.metric_test.reset()

    def on_test_epoch_end(self) -> None:
        print(self.iou / self.cnt)


def get_dataloader(cfg):
    trainImgName, valImgName = splitDataSet()
    train_dataset = BasicDataset(
        cfg.images_dir, cfg.masks_dir, None, cfg.scale, traing=True
    )
    val_dataset = BasicDataset(cfg.images_dir, cfg.masks_dir, None, cfg.scale)
    test_dataset = BasicDataset(cfg.val_images_dir, cfg.val_masks_dir, None, cfg.scale)
    # val_percent = cfg.validation
    # n_val = int(len(dataset) * val_percent)
    # n_train = len(dataset) - n_val
    # train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, num_workers=8, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, num_workers=8, pin_memory=True
    )
    return train_loader, val_loader, test_loader


class CmpModel(pl.LightningModule):
    def __init__(self, model=None, pixcel_loss=None, categorical_loss=None):
        super(CmpModel, self).__init__()
        if model != None:
            self.model = model
        else:
            self.model = smp.UnetPlusPlus(
                encoder_name="resnet50",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                # encoder_weights=None,  # use `imagenet` pre-trained weights for encoder initialization
                in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=5,  # model output channels (number of classes in your dataset)
            )
        if pixcel_loss != None:
            self.pixcel_loss = pixcel_loss
        else:
            self.pixcel_loss = Pixcel_loss()
        self.metric_test = SegmentationMetric(cfg.n_classes, True)
        self.metric = SegmentationMetric(cfg.n_classes)
        self.lr = cfg.lr
        self.iou = torch.zeros(4)
        self.cnt = 0

    def forward(self, x) -> Any:
        predict = self.model(x)
        return predict

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=5e-4)

        scheduler = WarmupMultiStepLR(
            optimizer=optimizer,
            milestones=cfg.lr_decay_milestones,
            gamma=cfg.lr_decay_gamma,
            warmup_iters=1,
        )

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        batch_imgs = batch["image"]
        batch_masks = batch["mask"]
        batch_categorical = batch["categorical"]
        batch_edge = batch["edge"]

        batch_imgs = batch_imgs.to(dtype=torch.float32)
        batch_edge = batch_edge.to(dtype=torch.float32)
        mask_type = torch.long
        batch_masks = batch_masks.to(dtype=mask_type)
        batch_categorical = batch_categorical.to(dtype=mask_type)

        predict = self.model(batch_imgs)

        pixcel1, pixcel1Log = self.pixcel_loss(predict, batch_masks.squeeze(1))

        loss = pixcel1

        self.log("train_loss", pixcel1Log, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_imgs = batch["image"]
        batch_masks = batch["mask"]
        batch_categorical = batch["categorical"]

        batch_imgs = batch_imgs.to(dtype=torch.float32)
        mask_type = torch.long
        batch_masks = batch_masks.to(dtype=mask_type)
        batch_categorical = batch_categorical.to(dtype=mask_type)

        predict1 = self.model(batch_imgs)

        self.metric.update(predict1, batch_masks.squeeze(1))
        acc, miou = self.metric.get()
        self.log("val_acc", acc, prog_bar=True, on_epoch=True)
        self.log("val_miou", miou, prog_bar=True, on_epoch=True)

    def on_validation_epoch_start(self) -> None:
        self.metric.reset()

    def test_step(self, batch, batch_idx):
        batch_imgs = batch["image"]
        batch_masks = batch["mask"]
        batch_categorical = batch["categorical"]

        batch_imgs = batch_imgs.to(dtype=torch.float32)
        mask_type = torch.long
        batch_masks = batch_masks.to(dtype=mask_type)
        batch_categorical = batch_categorical.to(dtype=mask_type)

        predict1 = self.model(batch_imgs)

        self.metric_test.update(predict1, batch_masks.squeeze(1))
        acc, miou, iou = self.metric_test.get()
        self.iou += iou
        self.cnt += 1
        self.log("test_acc", acc, prog_bar=True, on_epoch=True)
        self.log("test_miou", miou, prog_bar=True, on_epoch=True)

    def on_test_epoch_start(self) -> None:
        self.metric_test.reset()

    def on_test_epoch_end(self) -> None:
        print(self.iou / self.cnt)


def train():
    net = PFNet(cfg)
    # net = smp.UnetPlusPlus(
    #    encoder_name="resnet50",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    #    # encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
    #    in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    #    classes=5,  # model output channels (number of classes in your dataset)
    # )
    pixcel_loss = Pixcel_loss()
    categorical_loss = Categorical_loss()
    model = Model(model=net, pixcel_loss=pixcel_loss, categorical_loss=categorical_loss)

    train_loader, val_loader, test_loader = get_dataloader(cfg)

    checkpoint = ModelCheckpoint(
        dirpath=cfg.checkpoints_dir,
        filename="{epoch:02d}-{val_miou:.4f}",
        save_weights_only=False,
        save_on_train_epoch_end=False,
        monitor="val_miou",
        mode="max",
        save_top_k=3,
        save_last=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        check_val_every_n_epoch=1,
        devices=[0],
        enable_progress_bar=1,
        log_every_n_steps=1,
        callbacks=[checkpoint, lr_monitor],
        precision="16-mixed",
    )
    # tunner = Tuner(trainer=trainer)
    # tunner.lr_find(model,train_dataloaders=train_loader,max_lr=1e-3,num_training=1060)
    trainer.fit(model, train_loader, val_loader)
    # trainer.test(model, test_loader)


def evalWithTest():
    net = Model.load_from_checkpoint("check/1015pf/epoch=178-val_miou=0.8348.ckpt")
    train_loader, val_loader, test_loader = get_dataloader(cfg)

    checkpoint = ModelCheckpoint(
        dirpath=cfg.checkpoints_dir,
        filename="{epoch:02d}-{val_miou:.4f}",
        save_weights_only=False,
        save_on_train_epoch_end=False,
        monitor="val_miou",
        mode="max",
        save_top_k=1,
        save_last=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        check_val_every_n_epoch=0,
        devices=[0],
        enable_progress_bar=1,
        log_every_n_steps=1,
        callbacks=[checkpoint, lr_monitor],
        precision=16,
    )
    trainer.test(net, test_loader)


if __name__ == "__main__":
    # train()
    evalWithTest()
