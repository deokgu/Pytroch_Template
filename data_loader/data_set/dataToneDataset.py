import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset

class DataToneDataset(Dataset):
    def __init__(self, data_dir, mode = "train", transform=None):
        self.mode = mode 
        self.data_dir = data_dir
        self.is_tarin = True if mode == "train" else False
        self.data_dic = {}
        self.__ratio = 1

        self.set_data()
    
    def set_data(self,):
        # fold_list = ["cyto_negative", "cyto_negativ_test", "cyto_positive", "cyto_positive_test"]
        test_list = ["cyto_positive", "cyto_negative",]
        val_list = ["cyto_positive_test", "cyto_negativ_test"]

        fold_lists = test_list if self.if_train else val_list
        _count = 0
        for fold in fold_lists:
            for (path, dir, files) in os.walk(os.path.join(self.data_dir, fold)):
                for filename in files:
                    if filename[0] == ".": # not in: pass
                        continue
                    folder = os.path.split(path)[-1]
                    if folder[0] == ".":
                        continue
                    
                    image_path = os.path.join(path, filename)
                    if fold == "cyto_positive":
                        label = 1
                    else:
                        label = 0

                    self.data_dic[_count] = {"path": image_path, "image":None, "label":label}
                    _count += 1

    def __getitem__(self, index):
        if self.data_dic[index]["image"]== None :
            _image = Image.open(self.data_dic[index]["path"])
            self.data_dic[index]["image"]= _image
        else:
            _image = self.data_dic[index]["image"]  # Image.open

        if self.mode == "train":
            image_transform = self.transform["train"](_image)
        elif self.mode == "val":
            image_transform = self.transform["val"](_image)
        return image_transform, torch.tensor(self.data_dic[index]["label"])

    @property
    def ratio(self,):
        return self.__ratio
    @ratio.setter
    def ratio(self, ratio):
        self.__ratio =  ratio

    def __len__(self):
        return int(len(self.data_dic)* self.__ratio)      