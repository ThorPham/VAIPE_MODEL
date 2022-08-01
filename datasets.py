import json
import pandas as pd
import os
import cv2
import glob
import numpy as np
import json
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from torch.utils.data import Dataset,DataLoader

import torch
from transformers import AutoModel, AutoTokenizer,BertTokenizer
BERT_MODEL = "bert-base-uncased"
PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
bert_model = AutoModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

def loadJsonFile(file):
    with open(file,"r") as file:
        data = json.load(file)
    return data

class VAIPEDataset(Dataset):
    def __init__(self,file_json_map="dataset.json"):
        self.list_data = loadJsonFile(file_json_map)
    
    def __len__(self):
        return len(self.list_data)
    
    def __getitem__(self, idx):
        data = self.list_data[idx]
        feature_images,text_features,gt_pre, gt_pres_map = self.prepare_data(data)
        return feature_images,text_features,gt_pre, gt_pres_map
    def prepare_data(self,data):
        name_pill, name_pres, gt_pre, gt_pres_map = data["name_pill"], data["name_pres"], data["gt_pre"], data["gt_pres_map"]
        feature_images = torch.load(os.path.join("features",name_pill.replace("json","pth")))
        text_features = []
        for text in name_pres:
            input_ids = torch.tensor([tokenizer.encode(text.upper())])
            with torch.no_grad():
                features = bert_model(input_ids) 
                text_features.append(features["pooler_output"].reshape(-1))
        text_features = torch.stack(text_features)
        # matrix = self.convert2matrix(gt_pre,gt_pres_map)
        return feature_images,text_features, gt_pre, gt_pres_map
    def convert2matrix(self,x,y):
        x = torch.tensor(x)
        y = torch.tensor(y)
        x_ = torch.broadcast_to(x, (y.size(0), x.size(0)))
        y_ = torch.broadcast_to(y, (x.size(0), y.size(0)))
        z = (x_.T == y_).long()
        return z

if __name__ == "__main__":
    data = VAIPEDataset()
    train = DataLoader(data,batch_size=4,shuffle=True)
    for i in train:
        print(i)

           
         
         
