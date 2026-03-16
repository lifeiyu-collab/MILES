# coding=utf-8
import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
# import cv2
import numpy as np
from torchvision import transforms


# 自定义中心裁剪函数
# class CenterCropImg(object):
#     def __init__(self, size: int):
#         self.size = size
#
#     def __call__(self, img: Image.Image) -> Image.Image:
#         img = np.array(img)  # 将 PIL 图像转换为 numpy 数组
#         img = self.center_crop_img(img, self.size)  # 执行裁剪
#         img = Image.fromarray(img)  # 转回 PIL 图像
#         return img
#
#     def center_crop_img(self, img: np.ndarray, size: int) -> np.ndarray:
#         h, w, c = img.shape
#
#         if w == h == size:
#             return img
#
#         if w < h:
#             new_w = size
#             new_h = int(h * size / w)
#         else:
#             new_h = size
#             new_w = int(w * size / h)
#
#         img = cv2.resize(img, dsize=(new_w, new_h))
#
#         if new_w == size:
#             h = (new_h - size) // 2
#             img = img[h: h + size]
#         else:
#             w = (new_w - size) // 2
#             img = img[:, w: w + size]
#         try:
#             assert img.shape[0] == img.shape[
#                 1] == self.size, f"Assertion failed: x ({img.shape[0]}) is not greater than y ({img.shape[1]})"
#         except AssertionError as e:
#             print(f"Debug info: x = {img.shape[0]}, y = {img.shape[1]}")
#             raise  # 再次抛出异常
#         return img


def image_train(args, resize_size=256, crop_size=224):
    normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    if args.dataset == 'digits_dg' or args.dataset == 'dg5':
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    if args.dataset in ['office-home']:
        return transforms.Compose([
            transforms.RandomResizedCrop(224,scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3,
                                   contrast=0.3,
                                   saturation=0.3,
                                   hue=min(0.5, 0.3)),
            transforms.ToTensor(),
            normalize
        ])

    elif args.dataset == 'VLCS':
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3,
                                   contrast=0.3,
                                   saturation=0.3,
                                   hue=min(0.5, 0.3)),
            transforms.ToTensor(),
            normalize
        ])
    elif args.dataset == 'terra_incognita':
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

    elif args.dataset == 'cifar100':
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3,
                                   contrast=0.3,
                                   saturation=0.3,
                                   hue=min(0.5, 0.3)),
            transforms.ToTensor(),
            normalize
        ])

    elif args.dataset == 'domainnet':
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])


    elif args.dataset == 'ImageNet':
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    else:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])


def image_test(args, resize_size=256, crop_size=224):
    if args.dataset == 'digits_dg' or args.dataset == 'dg5':
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    return args.preprocess


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def l_loader(path):  # convert gray color image
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')
