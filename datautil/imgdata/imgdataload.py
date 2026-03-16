# coding=utf-8
import os
import json
import numpy as np
from datautil.util import Nmax
from datautil.imgdata.util import rgb_loader, l_loader
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader


class ImageDataset(object):
    def __init__(self, dataset, task, root_dir, domain_name, domain_label=-1, transform=None,
                 target_transform=None, indices=None, test_envs=[], mode='Default', shots_per_class=None,
                 class_index_path=None):
        IF = ImageFolder(root_dir + domain_name)
        self.imgs = IF.imgs
        self.domain_num = 0
        self.task = task
        self.dataset = dataset
        imgs = [item[0] for item in self.imgs]
        labels = [item[1] for item in self.imgs]
        self.labels = np.array(labels)
        self.x = imgs
        self.transform = transform
        self.target_transform = target_transform

        if indices is None:
            self.indices = np.arange(len(imgs))
        else:
            self.indices = indices

        if shots_per_class is not None:
            self._limit_samples_per_class(shots_per_class)

        if mode == 'Default':
            self.loader = default_loader
        elif mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader
        self.dlabels = np.ones(self.labels.shape) * \
                       (domain_label - Nmax(test_envs, domain_label))

        if class_index_path:
            if os.path.exists(class_index_path):
                self.classes = self._load_classnames_from_json(class_index_path, IF.classes)
                # self.classes = self._load_classnames_from_txt(class_index_path, IF.classes)
            else:
                print("No such file: ", class_index_path)
        else:
            self.classes = [c for c in IF.classes]

    def _load_classnames_from_txt(self, classnames_path, dir_classes):
        with open(classnames_path, "r", encoding="utf-8") as f:
            # 解析每一行，创建目录名到类名的映射
            dir_to_class = {}
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    dir_to_class[parts[0]] = parts[1]

        # 根据目录名列表获取对应的类名
        class_list = [dir_to_class.get(dir_name, dir_name) for dir_name in dir_classes]

        if len(class_list) != len(dir_classes):
            raise ValueError("类名文件中的类数量与目录中的类数量不匹配")

        return class_list

    def _load_classnames_from_json(self, classnames_path, dir_classes):
        with open(classnames_path, "r") as f:
            class_map = json.load(f)

        dir_to_class = {v[0]: v[1] for v in class_map.values()}
        return [dir_to_class.get(dir_name, dir_name) for dir_name in dir_classes]

    def _limit_samples_per_class(self, shots_per_class):
        filtered_labels = self.labels[self.indices]
        filtered_indices = self.indices

        # Create a mapping of class labels to their indices
        class_indices = {}
        for idx, label in zip(filtered_indices, filtered_labels):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)

        # Limit the number of samples per class
        limited_indices = []
        for label, indices in class_indices.items():
            limited_indices.extend(indices[:shots_per_class])

        self.indices = np.array(limited_indices)

    def set_labels(self, tlabels=None, label_type='domain_label'):
        assert len(tlabels) == len(self.x)
        if label_type == 'domain_label':
            self.dlabels = tlabels
        elif label_type == 'class_label':
            self.labels = tlabels

    def target_trans(self, y):
        if self.target_transform is not None:
            return self.target_transform(y)
        else:
            return y

    def input_trans(self, x):
        if self.transform is not None:
            return self.transform(x)
        else:
            return x

    def __getitem__(self, index):
        index = self.indices[index]
        img = self.input_trans(self.loader(self.x[index]))
        ctarget = self.target_trans(self.labels[index])
        dtarget = self.target_trans(self.dlabels[index])
        return img, ctarget, dtarget

    def __len__(self):
        return len(self.indices)
