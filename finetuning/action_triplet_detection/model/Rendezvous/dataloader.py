#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CODE RELEASE TO SUPPORT RESEARCH.
COMMERCIAL USE IS NOT PERMITTED.
#==============================================================================
An implementation based on:
***
    C.I. Nwoye, T. Yu, C. Gonzalez, B. Seeliger, P. Mascagni, D. Mutter, J. Marescaux, N. Padoy. 
    Rendezvous: Attention Mechanisms for the Recognition of Surgical Action Triplets in Endoscopic Videos. 
    Medical Image Analysis, 78 (2022) 102433.
***  
Created on Thu Oct 21 15:38:36 2021
#==============================================================================  
Copyright 2021 The Research Group CAMMA Authors All Rights Reserved.
(c) Research Group CAMMA, University of Strasbourg, France
@ Laboratory: CAMMA - ICube
@ Author: Chinedu Innocent Nwoye
@ Website: http://camma.u-strasbg.fr
#==============================================================================
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
#==============================================================================
"""

import os
import random
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, ConcatDataset, DataLoader, Subset


class CholecT50():
    def __init__(self, 
                dataset_dir, 
                dataset_variant="cholect45-crossval",
                test_fold=1,
                augmentation_list=['original', 'vflip', 'hflip', 'contrast', 'rot90'],
                # ADDED
                img_height = 256,
                img_width = 448,
                normalization_mean = [0.485, 0.456, 0.406],
                normalization_std = [0.229, 0.224, 0.225],
                overfit=False,
                overfit_n_samples=100,
                data_efficient_training=[]
                # END_ADDED
                ):
        """ Args
                dataset_dir : common path to the dataset (excluding videos, output)
                list_video  : list video IDs, e.g:  ['VID01', 'VID02']
                aug         : data augumentation style
                split       : data split ['train', 'val', 'test']
            Call
                batch_size: int, 
                shuffle: True or False
            Return
                tuple ((image), (tool_label, verb_label, target_label, triplet_label))
        """
        # ADDED: -> Ability to set custom img_width, img_height as well as mean and std for normalization.
        #           In the original code these values were fixed.
        #        -> Overfitting for hyperparameter testing purposes
        # img size
        self.img_height = img_height
        self.img_width = img_width
        # normalization
        self.normalization_mean = normalization_mean
        self.normalization_std = normalization_std
        # overfitting
        self.overfit = overfit
        self.overfit_n_samples = overfit_n_samples
        # END_ADDED
        self.dataset_dir = dataset_dir
        self.list_dataset_variant = {
            "cholect45-crossval": "for CholecT45 dataset variant with the official cross-validation splits.",
            "cholect50-crossval": "for CholecT50 dataset variant with the official cross-validation splits",
            "cholect50-challenge": "for CholecT50 dataset variant as used in CholecTriplet challenge",
            "cholect50": "for the CholecT50 dataset with original splits used in rendezvous paper",
            "cholect45": "a pointer to cholect45-crossval",
        }
        assert dataset_variant in self.list_dataset_variant.keys(), print(dataset_variant, "is not a valid dataset variant")
        video_split  = self.split_selector(case=dataset_variant)
        train_videos = sum([v for k,v in video_split.items() if k!=test_fold], []) if 'crossval' in dataset_variant else video_split['train']
        test_videos  = sum([v for k,v in video_split.items() if k==test_fold], []) if 'crossval' in dataset_variant else video_split['test']
        if 'crossval' in dataset_variant:
            val_videos   = train_videos[-5:]
            train_videos = train_videos[:-5]
        else:
            val_videos   = video_split['val']

        # ADDED: data efficient training
        if (data_efficient_training):
            for video in data_efficient_training:
                if (video not in train_videos):
                    raise ValueError(f"Video{video:2d} isn't in the train dataset.")
            train_videos = data_efficient_training

        print("-" * 50)
        print(f"Train videos: {train_videos}")
        print(f"  Val videos: {val_videos}")
        print(f" Test videos: {test_videos}")
        print("-" * 50)
        # END_ADDED

        self.train_records = ['VID{}'.format(str(v).zfill(2)) for v in train_videos]
        self.val_records   = ['VID{}'.format(str(v).zfill(2)) for v in val_videos]
        self.test_records  = ['VID{}'.format(str(v).zfill(2)) for v in test_videos]
        self.augmentations = {
            'original': self.no_augumentation,
            'vflip': transforms.RandomVerticalFlip(0.4),
            'hflip': transforms.RandomHorizontalFlip(0.4),
            'contrast': transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0, hue=0),
            'rot90': transforms.RandomRotation(90,expand=True),
            'brightness': transforms.RandomAdjustSharpness(sharpness_factor=1.6, p=0.5),
            'contrast': transforms.RandomAutocontrast(p=0.5),
        }
        self.augmentation_list = []
        for aug in augmentation_list:
            self.augmentation_list.append(self.augmentations[aug])
        trainform, testform = self.transform()

        self.build_train_dataset(trainform)
        self.build_val_dataset(trainform)
        self.build_test_dataset(testform)
    
    def list_dataset_variants(self):
        print(self.list_dataset_variant)

    def list_augmentations(self):
        print(self.augmentations.keys())

    def split_selector(self, case='cholect50'):
        switcher = {
            'cholect50': {
                'train': [1, 15, 26, 40, 52, 65, 79, 2, 18, 27, 43, 56, 66, 92, 4, 22, 31, 47, 57, 68, 96, 5, 23, 35, 48, 60, 70, 103, 13, 25, 36, 49, 62, 75, 110],
                'val'  : [8, 12, 29, 50, 78],
                'test' : [6, 51, 10, 73, 14, 74, 32, 80, 42, 111]
            },
            'cholect50-challenge': {
                'train': [1, 15, 26, 40, 52, 79, 2, 27, 43, 56, 66, 4, 22, 31, 47, 57, 68, 23, 35, 48, 60, 70, 13, 25, 49, 62, 75, 8, 12, 29, 50, 78, 6, 51, 10, 73, 14, 32, 80, 42],
                'val':   [5, 18, 36, 65, 74],
                'test':  [92, 96, 103, 110, 111]
            },
            'cholect45-crossval': {
                1: [79,  2, 51,  6, 25, 14, 66, 23, 50,],
                2: [80, 32,  5, 15, 40, 47, 26, 48, 70,],
                3: [31, 57, 36, 18, 52, 68, 10,  8, 73,],
                4: [42, 29, 60, 27, 65, 75, 22, 49, 12,],
                5: [78, 43, 62, 35, 74,  1, 56,  4, 13,],
            },
            'cholect50-crossval': {
                1: [79,  2, 51,  6, 25, 14, 66, 23, 50, 111],
                2: [80, 32,  5, 15, 40, 47, 26, 48, 70,  96],
                3: [31, 57, 36, 18, 52, 68, 10,  8, 73, 103],
                4: [42, 29, 60, 27, 65, 75, 22, 49, 12, 110],
                5: [78, 43, 62, 35, 74,  1, 56,  4, 13,  92],
            },
        }
        return switcher.get(case)

    def no_augumentation(self, x):
        return x

    def transform(self):
        # RDV uses ImageNet mean and std for normalization, instead we apply Cholec80 mean and std
        # CHANGED: [0.485, 0.456, 0.406] -> self.normalization_mean / [0.229, 0.224, 0.225] -> self.normalization_std
        normalize = transforms.Normalize(mean=self.normalization_mean, std=self.normalization_std)
        # We will use (224, 224) img size.
        # CHANGED: (256, 448) -> (self.self.img_height, self.img_width)
        op_test   = [transforms.Resize((self.img_height, self.img_width)), transforms.ToTensor(), normalize,]
        op_train  = [transforms.Resize((self.img_height, self.img_width))] + self.augmentation_list + [transforms.Resize((self.img_height, self.img_width)), transforms.ToTensor(), normalize,]
        testform  = transforms.Compose(op_test)
        trainform = transforms.Compose(op_train)
        return trainform, testform

    def build_train_dataset(self, transform):
        iterable_dataset = []
        for video in self.train_records:
            dataset = T50(img_dir = os.path.join(self.dataset_dir, 'data', video), 
                        triplet_file = os.path.join(self.dataset_dir, 'triplet', '{}.txt'.format(video)), 
                        tool_file = os.path.join(self.dataset_dir, 'instrument', '{}.txt'.format(video)),  
                        verb_file = os.path.join(self.dataset_dir, 'verb', '{}.txt'.format(video)),  
                        target_file = os.path.join(self.dataset_dir, 'target', '{}.txt'.format(video)), 
                        transform=transform)
            iterable_dataset.append(dataset)
        self.train_dataset = ConcatDataset(iterable_dataset)

    def build_val_dataset(self, transform):
        iterable_dataset = []
        for video in self.val_records:
            dataset = T50(img_dir = os.path.join(self.dataset_dir, 'data', video), 
                        triplet_file = os.path.join(self.dataset_dir, 'triplet', '{}.txt'.format(video)), 
                        tool_file = os.path.join(self.dataset_dir, 'instrument', '{}.txt'.format(video)),  
                        verb_file = os.path.join(self.dataset_dir, 'verb', '{}.txt'.format(video)),  
                        target_file = os.path.join(self.dataset_dir, 'target', '{}.txt'.format(video)), 
                        transform=transform)
            iterable_dataset.append(dataset)
        # Unlike in the original RDV code we will mean validation metrics over videos 
        # the same way as it is done for test metrics.
        # CHANGED: ConcatDataset(iterable_dataset) -> iterable_dataset
        self.val_dataset = iterable_dataset

    def build_test_dataset(self, transform):
        iterable_dataset = []
        for video in self.test_records:
            dataset = T50(img_dir = os.path.join(self.dataset_dir, 'data', video), 
                triplet_file = os.path.join(self.dataset_dir, 'triplet', '{}.txt'.format(video)), 
                tool_file = os.path.join(self.dataset_dir, 'instrument', '{}.txt'.format(video)),  
                verb_file = os.path.join(self.dataset_dir, 'verb', '{}.txt'.format(video)),  
                target_file = os.path.join(self.dataset_dir, 'target', '{}.txt'.format(video)), 
                transform=transform)
            iterable_dataset.append(dataset)
        self.test_dataset = iterable_dataset
        
    def build(self):
        # ADDED: overfitting for hyperparameter testing purposes
        if (self.overfit):
            print("".join(["#"] * 30) + " Overfit enabled! " + "".join(["#"] * 30))
            print(f"Overfitting to {self.overfit_n_samples} samples from the train dataset.")
            print("Train dataset = Validation dataset = Test dataset")

            # equally distribute samples
            size = len(self.train_dataset)
            step = int(size / self.overfit_n_samples)
            
            self.train_dataset = Subset(self.train_dataset, list(range(0, self.overfit_n_samples * step, step)))
            self.val_dataset = [self.train_dataset]
            self.test_dataset = [self.train_dataset] # original test dataset is a list of per-video datasets

            return (self.train_dataset, self.val_dataset, self.test_dataset)
        # END_ADDED
        return (self.train_dataset, self.val_dataset, self.test_dataset)

    
class T50(Dataset):
    def __init__(self, img_dir, triplet_file, tool_file, verb_file, target_file, transform=None, target_transform=None):
        self.triplet_labels = np.loadtxt(triplet_file, dtype=np.int, delimiter=',',)
        self.tool_labels = np.loadtxt(tool_file, dtype=np.int, delimiter=',',)
        self.verb_labels = np.loadtxt(verb_file, dtype=np.int, delimiter=',',)
        self.target_labels = np.loadtxt(target_file, dtype=np.int, delimiter=',',)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.triplet_labels)
    
    def __getitem__(self, index):
        triplet_label = self.triplet_labels[index, 1:]
        tool_label = self.tool_labels[index, 1:]
        verb_label = self.verb_labels[index, 1:]
        target_label = self.target_labels[index, 1:]
        basename = "{}.png".format(str(self.triplet_labels[index, 0]).zfill(6))
        img_path = os.path.join(self.img_dir, basename)
        image    = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            triplet_label = self.target_transform(triplet_label)
        return image, (tool_label, verb_label, target_label, triplet_label)


if __name__ == "__main__":
    print("Refers to https://github.com/CAMMA-public/cholect45 for the usage guide.")
