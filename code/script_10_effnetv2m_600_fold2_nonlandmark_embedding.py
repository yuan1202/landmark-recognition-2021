#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os

import random
from datetime import datetime
from typing import Dict, Tuple, Any
import pickle
from tqdm import tqdm

import math

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2
import albumentations
from torch.utils.data import Dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset

import timm


# In[ ]:


DATA_DIR = '../input/'
LOAD_MODEL = 'effnetv2m_in21k_fold2_epoch8'

IMAGE_SIZE = 600
BATCH_SIZE = 48
NUM_WORKERS = 4
USE_AMP = True


# In[ ]:


class LandmarkDataset(Dataset):
    def __init__(self, csv, transform=None):

        self.csv = csv.reset_index()
        self.transform = transform

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        
        row = self.csv.iloc[index]

        image = cv2.imread(row.filepath)[:,:,::-1]

        if self.transform is not None:
            res = self.transform(image=image)
            image = res['image'].astype(np.float32)
        else:
            image = image.astype(np.float32)

        image = image.transpose(2, 0, 1)
        
        return torch.tensor(image)


transforms = albumentations.Compose([
    albumentations.Resize(IMAGE_SIZE, IMAGE_SIZE),
    albumentations.Normalize()
])


# In[ ]:


class Swish(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish_module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)


class DenseCrossEntropy(nn.Module):
    def forward(self, x, target):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        loss = -logprobs * target
        loss = loss.sum(-1)
        return loss.mean()


class ArcMarginProduct_subcenter(nn.Module):
    def __init__(self, in_features, out_features, k=3):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features*k, in_features))
        self.reset_parameters()
        self.k = k
        self.out_features = out_features
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
    
    def forward(self, features):
        cosine_all = F.linear(F.normalize(features), F.normalize(self.weight))
        cosine_all = cosine_all.view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine_all, dim=2)
        return cosine   


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=True):
        super(GeM,self).__init__()
        if p_trainable:
            self.p = Parameter(torch.ones(1)*p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)
    
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class EffnetV2m_Landmark(nn.Module):

    def __init__(self, out_dim, load_pretrained=True):
        super().__init__()

        self.backbone = timm.create_model('tf_efficientnetv2_m_in21k', pretrained=False)
        self.feat = nn.Sequential(
            nn.Linear(self.backbone.num_features, 512, bias=True),
            nn.BatchNorm1d(512),
            Swish_module()
        )
        self.backbone.global_pool = GeM()
        self.backbone.classifier = nn.Identity()
        
        # self.swish = Swish_module()
        self.metric_classify = ArcMarginProduct_subcenter(512, out_dim)


    def extract(self, x):
        return self.backbone(x)[:, :, 0, 0]

    @autocast()
    def forward(self, x):
        x = self.extract(x)
        return self.feat(x)


# In[ ]:


out_dim = 81313

load = torch.load('./model_checkpoints/{}.pth'.format(LOAD_MODEL))
model_only_weight = {k[7:] if k.startswith('module.') else k: v for k, v in load['model_state_dict'].items()}

model = EffnetV2m_Landmark(out_dim=out_dim).cuda()
model.load_state_dict(model_only_weight)
model = nn.DataParallel(model)

model = model.eval()


# In[ ]:


# get dataframe
df = pd.read_csv('../input/recognition_solution_v2.1.csv')
df = df.loc[df['landmarks'].isna()]
df['filepath'] = df['id'].apply(lambda x: os.path.join(DATA_DIR, 'test_2019', x[0], x[1], x[2], f'{x}.jpg'))

dataset = LandmarkDataset(df, transform=transforms)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False, pin_memory=True)


# In[ ]:


with torch.no_grad():
    
    embeddings = np.zeros((len(df) , 512), dtype=np.float16)
    
    for idx, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        
        data = data.cuda()

        with autocast():
            embedding = F.normalize(model(data))
        
        embeddings[idx*BATCH_SIZE:idx*BATCH_SIZE+embedding.size(0), :] = embedding.detach().cpu().numpy()



# In[ ]:



np.save("./embeddings/{}_test2019nonlandmark_embeddings".format(LOAD_MODEL), embeddings)

