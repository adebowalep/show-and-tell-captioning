import os
import random

import torch
import torch.utils.data as data
from PIL import Image
from pycocotools.coco import COCO


class CoCoValDataset(data.Dataset):
    """Validation images with their five reference captions each (COCO val2014)."""

    def __init__(self, transform, annotations_file, img_folder, subset_size=None, seed=0):
        self.transform = transform
        self.img_folder = img_folder
        self.coco = COCO(annotations_file)
        self.img_ids = sorted(self.coco.imgs.keys())
        if subset_size is not None and subset_size < len(self.img_ids):
            random.Random(seed).shuffle(self.img_ids)
            self.img_ids = self.img_ids[:subset_size]

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        file_name = self.coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(self.img_folder, file_name)).convert('RGB')
        return self.transform(image), img_id

    def references(self, img_id):
        """Raw reference caption strings for one image."""
        return [ann['caption'] for ann in self.coco.imgToAnns[img_id]]

    def __len__(self):
        return len(self.img_ids)


def get_loader_val(transform, cocoapi_loc='/opt', subset_size=1000, num_workers=4):
    """Returns a validation data loader yielding (image, image_id), one image at a time."""
    img_folder = os.path.join(cocoapi_loc, 'cocoapi/images/val2014/')
    annotations_file = os.path.join(cocoapi_loc, 'cocoapi/annotations/captions_val2014.json')
    dataset = CoCoValDataset(transform, annotations_file, img_folder, subset_size)
    return data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)
