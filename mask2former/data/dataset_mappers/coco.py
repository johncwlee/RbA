"""
Adapted from https://github.com/tianyu0207/PEBAL/blob/main/code/dataset/data_loader.py
"""

import os
import random
from typing import Optional, Callable
from torch.utils.data import Dataset
from PIL import Image
import json
from pathlib import Path

class COCO(Dataset):
    train_id_in = 0
    train_id_out = 254
    min_image_size = 480

    COCO_OVERLAP_CLASSES = [
        "microwave", "oven", "toaster", "refrigerator", "scissors", "hair drier"
    ]

    def __init__(self, root: str, proxy_size: int, split: str = "train",
                 transform: Optional[Callable] = None, shuffle=True,
                 filter: bool = False) -> None:
        """
        COCO dataset loader
        """
        self.root = root
        self.coco_year = '2017'
        self.split = split + self.coco_year
        self.images = []
        self.targets = []
        self.transform = transform
        
        #? Filter images with class overlap according to the filter mode
        annotations = json.load(open(Path(self.root) / "annotations" / f"instances_{self.split}.json"))
        cat2id = {cat['name']: cat['id'] for cat in annotations['categories']}
        overlap_cat_ids = [cat2id[cat] for cat in self.COCO_OVERLAP_CLASSES]

        imgid2catid = {}
        #? Create a mapping from image id to category id for categories in each image
        for ann in annotations['annotations']:
            if ann['image_id'] not in imgid2catid:
                imgid2catid[ann['image_id']] = [ann['category_id']]
            else:
                imgid2catid[ann['image_id']].append(ann['category_id'])
        
        for root, _, filenames in os.walk(os.path.join(self.root, "annotations", "ood_seg_" + self.split)):
            assert self.split in ['train' + self.coco_year, 'val' + self.coco_year]
            for filename in filenames:
                if os.path.splitext(filename)[-1] == '.png':
                    file_id = int(Path(filename).stem.lstrip('0'))
                    cats = imgid2catid[file_id] #* category ids of the image
                    
                    #? Skip image if at least one class overlaps
                    #* there are many coco images so we can afford to skip images entirely
                    if filter and any(cat in overlap_cat_ids for cat in cats):
                        continue
                    self.targets.append(os.path.join(root, filename))
                    self.images.append(os.path.join(self.root, self.split, filename.split(".")[0] + ".jpg"))

        """
        shuffle data and subsample
        """

        if shuffle:
            zipped = list(zip(self.images, self.targets))
            random.shuffle(zipped)
            self.images, self.targets = zip(*zipped)

        if proxy_size is not None:
            self.images = list(self.images[:int(proxy_size)])
            self.targets = list(self.targets[:int(proxy_size)])
        else:
            self.images = list(self.images[:5000])
            self.targets = list(self.targets[:5000])

    def __len__(self):
        """Return total number of images in the whole dataset."""
        return len(self.images)

    def __getitem__(self, i):
        """Return raw image and ground truth in PIL format or as torch tensor"""
        image = Image.open(self.images[i]).convert('RGB')
        target = Image.open(self.targets[i]).convert('L')
        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target

    def __repr__(self):
        """Return number of images in each dataset."""

        fmt_str = 'Number of COCO Images: %d\n' % len(self.images)
        return fmt_str.strip()