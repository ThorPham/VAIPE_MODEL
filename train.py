import torch
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
from torch.utils.data import Dataset
# from features import get_feature
from model import VAIPENet ,MultiSimilarityLoss1,MultiSimilarityLoss2
from datasets import VAIPEDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
import torch
# import some common libraries
import numpy as np
import os, json, cv2, random
from detectron2.modeling import build_backbone
from detectron2.modeling.poolers import ROIPooler
import matplotlib.pyplot as plt
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.modeling import build_model
from detectron2.engine import DefaultTrainer
from detectron2.structures import ImageList,Boxes
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T
setup_logger()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



net = VAIPENet()
net.to(device)
optimizer = optim.Adam(net.parameters(), lr=0.01)
EPOCHS = 10
criterion1 = MultiSimilarityLoss1()
criterion2 = MultiSimilarityLoss2()


data_train = VAIPEDataset()

for epoch in range(EPOCHS):
   for feature_images,text_features, gt_pre, gt_pres_map in tqdm(data_train):
      feature_images =  feature_images.to(device)
      text_features = text_features.to(device)
      gt_pre = torch.tensor(gt_pre).to(device)
      gt_pres_map = torch.tensor(gt_pres_map).to(device)

      optimizer.zero_grad()
      emb_img,emb_text = net(feature_images,text_features)
      loss1 = criterion1(emb_img, emb_text,gt_pre,gt_pres_map)
      loss2 = criterion2(emb_img, gt_pre)
      total_loss =  loss1 + loss2
      total_loss.backward()
      optimizer.step()
      print("loss1",loss1.item())
      print("loss2",loss2.item())