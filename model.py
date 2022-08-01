# Some basic setup:
# Setup detectron2 logger
import torch
# import some common libraries
import numpy as np
import os, json, cv2, random

import matplotlib.pyplot as plt
# import some common detectron2 utilities
from detectron2 import model_zoo

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


class VAIPENet(nn.Module):
    def __init__(self):
        super(VAIPENet,self).__init__()
        # for text emb
        self.lstm = nn.LSTM(768, 256)
        self.fc1 =  weight_norm(nn.Linear(256,128))
        
        # for image emb
        self.fc2 =  weight_norm(nn.Linear(2048,256))
        self.fc3 = weight_norm(nn.Linear(256,128))
    
        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        for l in [self.fc1,self.fc2,self.fc3]:
            r = np.sqrt(6.) / np.sqrt(l.in_features +
                                    l.out_features)
            l.weight.data.uniform_(-r, r)
            l.bias.data.fill_(0)    
    def forward_text_emb(self,x):
        x,_ = self.lstm(x)
        x = F.relu(self.fc1(x))
        # self attention
        attention = torch.matmul(x,x.T)
        score = F.softmax(attention,dim=1)
        x = torch.matmul(score,x)
        return l2norm(x,dim=-1)
    
    def forward_img_emb(self,x):

        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        attention = torch.matmul(x,x.T)
        score = F.softmax(attention,dim=1)
        x = torch.matmul(score,x)
        return l2norm(x,dim=-1)
    
    def forward(self,x,y):
        '''
        x : img
        y : text
        '''
        emb_img = self.forward_img_emb(x)
        emb_text = self.forward_text_emb(y)
       
        return emb_img,emb_text

class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0.2, ):
      super(ContrastiveLoss, self).__init__()
      self.margin = margin
        
    def forward(self, matrix, x, y):

      x_ = torch.broadcast_to(x, (y.size(0), x.size(0)))
      y_ = torch.broadcast_to(y, (x.size(0), y.size(0)))
      z = (x_.T == y_).long()
    #   positive = torch.sum(z - matrix*z)
    #   negative = torch.sum((((1-z)*matrix - self.margin).clamp(min=0)))
      positive = matrix*z
      negative = (1-z)*matrix
      return (torch.sum(positive) - torch.sum(negative) + self.margin)/(z.nelement())

class InstanceLoss(nn.Module):
    def __init__(self, margin=0.2, ):
        super(InstanceLoss, self).__init__()
        self.margin = margin 
    def forward(self,matrix,x):
        x_ = torch.broadcast_to(x, (x.size(0), x.size(0)))
        y_ = torch.broadcast_to(x, (x.size(0), x.size(0)))
        z = (x_.T == y_).long()
        positive = (1 - torch.eye(len(x)).to(device))*z - (1 - torch.eye(len(x)).to(device))*z*matrix
        negative = ((1-z)*matrix- self.margin).clamp(min=0)
        return torch.sum(positive + negative)

class MultiSimilarityLoss1(nn.Module):
    def __init__(self):
        super(MultiSimilarityLoss1, self).__init__()
        self.thresh = 0.5
        self.margin = 0.1

        self.scale_pos = 2.0
        self.scale_neg = 40.
    def forward(self,emb_img,emb_text,gt_pre,gt_pres_map):
        matrix = torch.matmul(emb_img, emb_text.T)
        print(matrix)
        print(emb_img)
        print(emb_text)
        z = torch.zeros(gt_pre.shape[0],gt_pres_map.shape[0]).to(device)
        for i in range(gt_pre.shape[0]):
            for j in range(gt_pres_map.shape[0]):
                if gt_pre[i]==gt_pres_map[j] :
                    z[i,j] =1
        matrix_p = z*matrix
        matrix_p = matrix_p[matrix_p > 0]
        if len(matrix_p) == 0 :
            pos_loss = 0.
        else:
            pos_loss  = 1.0 / self.scale_pos * torch.log(
                    1 + torch.sum(torch.exp(-self.scale_pos * (matrix_p - self.thresh))))
            pos_loss /= len(matrix_p)
        matrix_n = (1-z)*matrix
        matrix_n = matrix_n[matrix_n>0]
        if len(matrix_n) == 0:
            neg_loss = 0.
        else:
            neg_loss = 1.0 / self.scale_neg * torch.log(
                1 + torch.sum(torch.exp(self.scale_neg * (matrix_n - self.thresh))))
            neg_loss /= len(matrix_n)
        loss = pos_loss + neg_loss
        if loss == 0:
            return torch.zeros([], requires_grad=True)
        return pos_loss + neg_loss

class MultiSimilarityLoss2(nn.Module):
    def __init__(self):
        super(MultiSimilarityLoss2, self).__init__()
        self.thresh = 0.5
        self.margin = 0.1

        self.scale_pos = 2.0
        self.scale_neg = 40.
    def forward(self,emb_img,gt_pre):
        matrix = torch.matmul(emb_img, emb_img.T)
        # x_ = torch.broadcast_to(x, (x.size(0), x.size(0)))
        # y_ = torch.broadcast_to(x, (x.size(0), x.size(0)))
        # z = (x_.T == y_).long()
        z = torch.zeros(gt_pre.shape[0],gt_pre.shape[0]).to(device)
        for i in range(gt_pre.shape[0]):
            for j in range(gt_pre.shape[0]):
                if gt_pre[i]==gt_pre[j] and gt_pre[i]!=141:
                    z[i,j] =1
        diagonals = torch.eye(len(emb_img)).to(device)
        matrix_p = (1 - diagonals)*z*matrix
        matrix_p = matrix_p[matrix_p > 0]
        if len(matrix_p) == 0 :
            pos_loss = 0.
        else:
            pos_loss  = 1.0 / self.scale_pos * torch.log(
                    1 + torch.sum(torch.exp(-self.scale_pos * (matrix_p - self.thresh))))
            pos_loss /= len(matrix_p)
        matrix_n = (1-z)*matrix
        matrix_n = matrix_n[matrix_n>0]
        if len(matrix_n) == 0:
            neg_loss = 0.
        else:
            neg_loss = 1.0 / self.scale_neg * torch.log(
                1 + torch.sum(torch.exp(self.scale_neg * (matrix_n - self.thresh))))
            neg_loss /= len(matrix_n)
        loss = pos_loss + neg_loss
        if loss == 0:
            return torch.zeros([], requires_grad=True)
        return pos_loss + neg_loss
if __name__ == "__main__":
   input1 = torch.randn((3,7,7,256))
   input2 = torch.tensor([4,6])
   net = VAIPENet(256,108)
   print(net)
   out, matrixDistance = net(input1,input2)