import os
import pickle
import numpy as np
from PIL import Image
import h5py
import json
from glob import glob

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transforms import Scale


def get_dataloader(cfg, mode="clevr"):
    if mode == "clevr":
        train_clevr = CLEVR(cfg.DATALOADER.FEATURES_PATH, train_percent=cfg.DATALOADER.DATA_PERCENT)
        train_dataloader = DataLoader(
            train_clevr, batch_size=cfg.DATALOADER.BATCH_SIZE, num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=collate_data, drop_last=True, shuffle=True,
        )
        val_clevr = CLEVR(cfg.DATALOADER.FEATURES_PATH, 'val')
        val_dataloader = DataLoader(
            val_clevr, batch_size=cfg.DATALOADER.BATCH_SIZE, num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=collate_data, drop_last=True
        )
        return train_dataloader, val_dataloader
    elif mode == "gqa":
        train_gqa = GQA(cfg.DATALOADER.FEATURES_PATH, 'train', cfg.DATALOADER.TRAIN_SPLIT, train_percent=cfg.DATALOADER.DATA_PERCENT)
        train_dataloader = DataLoader(
            train_gqa, batch_size=cfg.DATALOADER.BATCH_SIZE, num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=collate_data_GQA, drop_last=True, shuffle=True,
        )
        val_gqa = GQA(cfg.DATALOADER.FEATURES_PATH, 'testdev', cfg.DATALOADER.VAL_SPLIT)
        val_dataloader = DataLoader(
            val_gqa, batch_size=cfg.DATALOADER.BATCH_SIZE, num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=collate_data_GQA, drop_last=True
        )
    else:
        raise NotImplementedError()
    return train_dataloader, val_dataloader


class CLEVR(Dataset):
    def __init__(self, root, split='train', train_percent=1.0):
        features_path = os.path.join(root, 'features')
        with open('{}/{}.pkl'.format(features_path, split), 'rb') as f:
            self.data = pickle.load(f)

        self.root = root
        self.split = split
        self.train_percent = train_percent

        self.h = h5py.File('{}/{}_features.hdf5'.format(features_path, split), 'r')
        self.img = self.h['data']

    def close(self):
        self.h.close()

    def __getitem__(self, index):
        imgfile, question, answer, family = self.data[int(index/self.train_percent)]

        id = int(imgfile.rsplit('_', 1)[1][:-4])
        img = torch.from_numpy(self.img[id])

        return img, question, len(question), answer, family, index

    def __len__(self):
        return int(len(self.data) * self.train_percent)


class GQA(Dataset):
    def __init__(self, root, split='train', variant='all', train_percent=1.0):
        features_path = os.path.join(root, 'features')

        self.root = root
        self.split = split
        self.train_percent = train_percent

        # load tokenized questions
        with open('{}/{}_{}.pkl'.format(features_path, variant, split), 'rb') as f:
            self.data = pickle.load(f)

        # load object features
        objects_path = os.path.join(features_path, 'objects')

        with open(os.path.join(objects_path, 'gqa_objects_info.json')) as f:
            self.objects_info = json.load(f)

        sorted_h5_paths = sorted(
            glob('{}/gqa_objects_*.h5'.format(objects_path)),
            key=lambda path: int(os.path.basename(path).split('_')[-1].split('.')[0])
        )

        self.h5s = [h5py.File(h5_path, 'r') for h5_path in sorted_h5_paths]
        self.feature_h5s = [h5_file['features'] for h5_file in self.h5s]

    def close(self):
        for file in self.h5s:
            file.close()

    def __getitem__(self, q_index):
        imgfile, question, answer = self.data[int(q_index/self.train_percent)]

        object_info = self.objects_info[imgfile]
        file = object_info['file']
        img_idx = object_info['idx']
        num_objects = object_info['objectsNum']

        img_objects = torch.from_numpy(self.feature_h5s[file][img_idx])   # (100, 2048)
        img_objects = img_objects.t().unsqueeze(1)   # (2048, 1, 100)

        return img_objects, num_objects, question, len(question), answer, q_index

    def __len__(self):
        return int(len(self.data) * self.train_percent)
