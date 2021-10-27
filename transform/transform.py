from .randAugment import RandAugment
import ttach as tta
import albumentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

# set_tuple = lambda x : tuple([float(z) for z in x.split(",")])


class NO_resize_Base_C348TransForm(object):
    def __init__(self, mean, std, resize, use_rand_aug=False):
        self.mean = mean
        self.std = std
        self.x, self.y = resize  # notuse
        self.use_rand_aug = use_rand_aug
        self.get_transforms()

    def get_transforms(self, need=("train", "val", "eavl")):
        self.transformations = {}
        if "train" in need:
            self.transformations["train"] = transforms.Compose(
                [
                    transforms.CenterCrop((348, 348)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.mean, std=self.std),
                ]
            )
            if self.use_rand_aug:
                self.transformations["train"].transforms.insert(1, RandAugment())
        if "val" in need:
            self.transformations["val"] = transforms.Compose(
                [
                    transforms.CenterCrop((348, 348)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.mean, std=self.std),
                ]
            )
        if "eval" in need:
            self.transformations["eval"] = tta.Compose(
                [
                    tta.HorizontalFlip(),
                    # tta.Rotate90(angles=[0, 90]),
                    # tta.Scale(scales=[1, 2]),
                    tta.FiveCrops(320, 160),
                    # tta.Multiply(factors=[0.7, 1]),
                ]
            )
        return self.transformations


class CustomTransForm_seg(object):
    def __init__(self, mean, std, resize, use_rand_aug=False):
        self.mean = mean
        self.std = std
        self.resize = resize # Not use
        self.use_rand_aug = use_rand_aug
        self.get_transforms()

    def get_transforms(self, need=("train", "val", "eavl")):
        self.transforms = {}
        if "train" in need:
            self.transforms["train"] = A.Compose(
                [
                    # A.GridDropout(holes_number_x=30, holes_number_y=30, p=1.0),
                    # A.Normalize(mean=self.mean, std=self.std, p=1.0),
                    ToTensorV2(p=1.0),
                ]
            )

        if "val" in need:
            self.transforms["val"] = albumentations.Compose(
                [
                    # A.Normalize(mean=self.mean, std=self.std, p=1.0),
                    ToTensorV2(p=1.0),
                ]
            )
