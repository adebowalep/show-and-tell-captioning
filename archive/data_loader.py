import nltk
import os
import torch
import torch.utils.data as data
from vocabulary import Vocabulary
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
from tqdm import tqdm
import random
import json

def get_loader(transform,
               mode='train',
               batch_size=1,
               vocab_threshold=None,
               vocab_file='./vocab.pkl',
               start_word="<start>",
               end_word="<end>",
               unk_word="<unk>",
               vocab_from_file=True,
               num_workers=0,
               cocoapi_loc='/opt'):
    """Returns the data loader for the desired dataset."""
    assert mode in ['train', 'val']

    if vocab_from_file==False:
        assert mode=='train', "If starting from scratch, then sorted training file must be used, to cope with annotations."

    img_folder = os.path.join(cocoapi_loc, 'cocoapi/images/train2014' if mode=='train' else 'cocoapi/images/val2014/')
    annotations_file = os.path.join(cocoapi_loc, 'cocoapi/annotations/captions_train2014.json' if mode=='train' else 'cocoapi/annotations/captions_val2014.json')

    v_threshold = vocab_threshold if vocab_from_file else vocab_threshold
    dataset = CoCoDataset(transform=transform,
                          mode=mode,
                          batch_size=batch_size,
                          vocab_threshold=v_threshold,
                          vocab_file=vocab_file,
                          start_word=start_word,
                          end_word=end_word,
                          unk_word=unk_word,
                          annotations_file=annotations_file,
                          vocab_from_file=vocab_from_file,
                          img_folder=img_folder)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers,
                                  collate_fn=dataset.get_batches)

    return data_loader


class CoCoDataset(data.Dataset):

    def __init__(self,
                 transform,
                 mode,
                 batch_size,
                 vocab_threshold,
                 vocab_file,
                 start_word,
                 end_word,
                 unk_word,
                 annotations_file,
                 vocab_from_file,
                 img_folder):
        self.transform = transform
        self.mode = mode
        self.batch_size = batch_size
        self.vocab_threshold = vocab_threshold
        self.vocab_file = vocab_file
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = unk_word
        self.vocab_from_file = vocab_from_file
        self.img_folder = img_folder

        self.coco = COCO(annotations_file)
        self.ids = list(self.coco.anns.keys())

        print('Obtaining caption lengths...')
        self.caption_lengths = []
        for idx in tqdm(np.arange(len(self.ids))):
            caption = str(self.coco.anns[self.ids[idx]]['caption'])
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            self.caption_lengths.append(len(tokens))

        print('Initializing vocabulary object...')
        self.vocab = Vocabulary(vocab_threshold=self.vocab_threshold,
                               vocab_file=self.vocab_file,
                               start_word=self.start_word,
                               end_word=self.end_word,
                               unk_word=self.unk_word,
                               annotations_file=annotations_file,
                               vocab_from_file=self.vocab_from_file)

        print('Vocabulary initialized.')
        print("Total vocabulary size: %d" %len(self.vocab))

    def __getitem__(self, index):
        ann_id = self.ids[index]
        caption = str(self.coco.anns[ann_id]['caption'])
        img_id = self.coco.anns[ann_id]['image_id']
        path = self.coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        tokens = nltk.tokenize.word_tokenize(caption.lower())
        caption = []
        caption.append(self.vocab(self.start_word))
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab(self.end_word))
        caption = torch.Tensor(caption).long()

        return image, caption

    def __len__(self):
        return len(self.ids)

    def get_batches(self, batch_info):
        images, captions = [], []
        for image, caption in batch_info:
            images.append(image)
            captions.append(caption)
        images = torch.stack(images, dim=0)
        lengths = torch.tensor([len(cap) for cap in captions])
        captions = torch.nn.utils.rnn.pad_sequence(captions, batch_first=True)
        return images, captions, lengths
