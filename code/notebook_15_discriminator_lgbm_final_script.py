#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import joblib
from datetime import datetime
from typing import Dict, Tuple, Any
from tqdm import tqdm
import pickle
from collections import defaultdict
import gc

import math
import numpy as np
import pandas as pd

from scipy.special import softmax
from sklearn.model_selection import train_test_split, StratifiedKFold

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
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau

import lightgbm as lgb

import timm

import psutil
def get_used_memory():
    return psutil.Process(os.getpid()).memory_info().vms/1024**3
def get_used_memory_txt():
    return 'Used memory: {:.2f}'.format(get_used_memory())
initial_used_memory=get_used_memory()
print(get_used_memory_txt())


# In[2]:


df = pd.read_csv('../input/train.csv')
df_full = pd.read_csv('../input/train_full.csv')
df_lm = df_full[df_full['landmark_id'].isin(df['landmark_id'].unique())].reset_index(drop=True)
df_lm.drop(columns='url', inplace=True)
lm_img2embd_map = defaultdict(lambda: -1)
lm_img2embd_map.update({img_id: i for i, img_id in enumerate(df_lm['id'])})
lm_id2class_map = {id_: i for i, id_ in enumerate(sorted(df_lm['landmark_id'].unique()))}
df_lm['class'] = df_lm['landmark_id'].map(lambda x: lm_id2class_map[x])
lm_img2cls_map = defaultdict(lambda: -1)
lm_img2cls_map.update({img: c for img, c in zip(df_lm['id'], df_lm['class'])})


# In[3]:


df_nlm = pd.read_csv('../input/recognition_solution_v2.1.csv')
df_nlm = df_nlm[df_nlm['landmarks'].isna()].reset_index(drop=True)
nlm_img2embd_map = defaultdict(lambda: -1)
nlm_img2embd_map.update({img_id: i for i, img_id in enumerate(df_nlm['id'])})
nlm_img2cls_map = defaultdict(lambda: -1)
nlm_img2cls_map.update({img: 81313 for img in df_nlm['id']})


# In[4]:


del df, df_full, df_lm, df_nlm


# In[5]:


df_train = pd.read_csv('./final/VD_dataframe_withFolds.csv')


# In[6]:


embedding_lm = np.load('../input/model_v0.6_landmark_embeddings_f16_all.npy')
embedding_nlm = np.load('../input/model_v0.6_non_landmark_embeddings_f16.npy')


# In[7]:


FOLD = 0

trn_idx = df_train.loc[df_train['fold'] != FOLD].index.values
vld_idx = df_train.loc[df_train['fold'] == FOLD].index.values

trn_array = np.zeros((len(trn_idx), 512*4), dtype=np.float32)
trn_label = df_train['target'].iloc[trn_idx].values

vld_array = np.zeros((len(vld_idx), 512*4), dtype=np.float32)
vld_label = df_train['target'].iloc[vld_idx].values


# In[10]:


for i, idx in tqdm(enumerate(trn_idx), total=len(trn_idx)):
    
    embd0_idx = lm_img2embd_map[df_train['img_id'].iloc[idx]]
    if embd0_idx == -1:
        embd0_idx = nlm_img2embd_map[df_train['img_id'].iloc[idx]]
        embd0 = embedding_nlm[embd0_idx]
    else:
        embd0 = embedding_lm[embd0_idx]
        
    embd1_idx = lm_img2embd_map[df_train['img_id_crossed'].iloc[idx]]
    if embd1_idx == -1:
        embd1_idx = nlm_img2embd_map[df_train['img_id_crossed'].iloc[idx]]
        embd1 = embedding_nlm[embd1_idx]
    else:
        embd1 = embedding_lm[embd1_idx]
        
    trn_array[i, :512] = embd0
    trn_array[i, 512:1024] = embd1
    trn_array[i, 1024:1536] = np.nan_to_num(embd1 - embd0)
    embd0[embd0 == 0] = .001
    embd1[embd1 == 0] = .001
    trn_array[i, 1536:] = np.nan_to_num(embd1/embd0)
    
for i, idx in tqdm(enumerate(vld_idx), total=len(vld_idx)):
    
    embd0_idx = lm_img2embd_map[df_train['img_id'].iloc[idx]]
    if embd0_idx == -1:
        embd0_idx = nlm_img2embd_map[df_train['img_id'].iloc[idx]]
        embd0 = embedding_nlm[embd0_idx]
    else:
        embd0 = embedding_lm[embd0_idx]
        
    embd1_idx = lm_img2embd_map[df_train['img_id_crossed'].iloc[idx]]
    if embd1_idx == -1:
        embd1_idx = nlm_img2embd_map[df_train['img_id_crossed'].iloc[idx]]
        embd1 = embedding_nlm[embd1_idx]
    else:
        embd1 = embedding_lm[embd1_idx]
        
    vld_array[i, :512] = embd0
    vld_array[i, 512:1024] = embd1
    vld_array[i, 1024:1536] = np.nan_to_num(embd1 - embd0)
    embd0[embd0 == 0] = .001
    embd1[embd1 == 0] = .001
    vld_array[i, 1536:] = np.nan_to_num(embd1/embd0)


# In[12]:


del embedding_lm, embedding_nlm
gc.collect()


# In[13]:


model = lgb.LGBMClassifier(objective='binary', learning_rate=0.03, n_estimators=6000, n_jobs=30)


# In[ ]:


model.fit(
    X=trn_array,
    y=trn_label,
    eval_set=(vld_array, vld_label),
    eval_metric=['auc', 'acc', 'binary_logloss'],
    early_stopping_rounds=100,
    verbose=10
)


# In[15]:


joblib.dump(model, './model_checkpoints/lgbm_discriminator_final.lgb')


# In[ ]:




