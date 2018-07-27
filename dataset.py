from config import *
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

class_list = {'airplane':0, 'bathtub':1, 'bed':2, 'bench':3, 'bookshelf':4, 'bottle':5, 'bowl':6, 'car':7, 'chair':8, 'cone':9,
              'cup':10, 'curtain':11, 'desk':12, 'door':13, 'dresser':14, 'flower_pot':15, 'glass_box':16, 'guitar':17, 'keyboard':18,'lamp':19,
              'laptop':20, 'mantel':21, 'monitor':22, 'night_stand':23, 'person':24, 'piano':25, 'plant':26, 'radio':27, 'range_hood':28, 'sink':29,
              'sofa':30, 'stairs':31, 'stool':32, 'table':33, 'tent':34, 'toilet':35, 'tv_stand':36, 'vase':37, 'wardrobe':38, 'xbox':39 }

class Load_Paths(object):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir

    def classes(self,label):
        for name,clas in class_list.items():
            if name == label:
               return clas

    def load(self, phase = 'train'):
        train_path=[]
        train_lbl=[]

        for imag in os.listdir(self.dataset_dir):
            lbl_name = imag
            lbl = self.classes(lbl_name)
            path = os.path.join(self.dataset_dir, imag)

            trn_tst = phase
            tt_path = os.path.join(path,phase)
            for img in os.listdir(tt_path):
                full_path = os.path.join(tt_path,img)

                train_path.append(full_path)
                train_lbl.append(lbl)

        df = pd.DataFrame(data={"images":train_path, "labels": train_lbl})

        return df

class ModelNetDataset(Dataset):
    """ModelNet 3D dataset represented as Multi-Layered Height maps."""

    #root_dir: root dir to MLH descriptors,
    #phase: train or test
    def __init__(self, root_dir, phase = 'train', transform=None):

        loadpaths = Load_Paths(root_dir)
        self.model_data = loadpaths.load(phase)

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.model_data)

    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.model_data.ix[idx, 0])
        features = np.load(file_path)
        lbl = self.model_data.ix[idx, 1].astype('float')
        sample = {'features': features, 'lables': lbl}
        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        features, lbl = sample['features'], sample['lables']

        features = features.transpose((0, 3, 1, 2))
        features = torch.from_numpy(features)
        return features,lbl