import numpy as np
import random
import torchvision.transforms.functional as TF
from PIL import Image


def get_composed_augmentations(aug_dict):
    augmentations = []
    for aug_key, aug_param in aug_dict.items():
        augmentations.append(key2aug[aug_key](**aug_param))
    return Compose(augmentations)


class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, img, label):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img.astype('uint8'), mode='RGB')
            label = Image.fromarray(label.astype('uint8'), mode='L')
        for a in self.augmentations:
            img, label = a(img, label)
        img, label = np.array(img), np.array(label, dtype=np.uint8)
        return img, label


class RandomHorizontalFlip(object):
    def __init__(self, probability):
        self.probability = probability

    def __call__(self, img, label):
        if random.random() < self.probability:
            img = TF.hflip(img)
            label = TF.hflip(label)
        return img, label


class RandomCrop(object):
    def __init__(self, img_size, padding):
        self.img_size = img_size
        self.padding = padding

    def __call__(self, img, label):
        if self.padding > 0:
            img = TF.pad(img, self.padding)
            label = TF.pad(label, self.padding, fill=255)

        w, h = img.size
        th, tw = self.img_size
        if w == tw and h == th:
            return img, label

        new_x = random.randint(0, h - th)
        new_y = random.randint(0, w - tw)
        return (
            TF.crop(img, new_x, new_y, th, tw),
            TF.crop(label,  new_x, new_y, th, tw),
        )

key2aug = {
            'RandomCrop': RandomCrop,       
            'RandomHorizontalFlip': RandomHorizontalFlip,
            }
