import os
import torch.utils.data as data
import csv
from PIL import Image


def default_loader(path):
    return Image.open(path).convert('RGB')


class myImageFloder(data.Dataset):
    def __init__(self, root, label, transform=None, target_transform=None, loader=default_loader):
        imgs = []
        class_names = []
        row_number = 0
        with open(label, newline='') as csvfile:
            rows = csv.reader(csvfile)
            for row in rows:
                if row_number == 0:  # 'id', 'label' in training_labels.csv
                    row_number = 1
                else:
                    if os.path.isfile(os.path.join(root, row[0] + '.jpg')):
                        if row[1] not in class_names:
                            class_names.append(row[1])  # make class matrix
                        imgs.append(row)
        self.root = root
        self.imgs = imgs
        self.classes = class_names
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        label_idx = self.classes.index(label)
        img = self.loader(os.path.join(self.root, fn + '.jpg'))
        if self.transform is not None:
            img = self.transform(img)
        return img, label_idx

    def __len__(self):
        return len(self.imgs)

    def get_name(self, index):
        fn, label = self.imgs[index]
        return fn
