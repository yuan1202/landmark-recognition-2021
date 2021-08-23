#!/usr/bin/env python
# coding: utf-8

# In[1]:
# =============================================================================


import os
from datetime import datetime
from typing import Dict, Tuple, Any
from tqdm import tqdm

import math
import numpy as np
import pandas as pd

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

import timm


# In[2]:
# =============================================================================


# ---------------------------------------
# parameters

MODEL_DIR = './model_checkpoints/'
LOAD_MODEL = 'rexnet_200_fold0_final'
DATA_DIR = '../input/'
LOG_DIR = './logs/'
DEVICE = 'cuda:0'
MODEL_NAME = 'rexnet_200_step1'

TRAIN_STEP = 1
FOLD = 0

IMAGE_SIZE = 512
BATCH_SIZE = 48
NUM_EPOCHS = 40
NUM_WORKERS = 4
LR = 1e-3
SCHEDULER_PEAK = 0.02
USE_AMP = True


# In[3]:
# =============================================================================


# ---------------------------------------
# utils

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# In[4]:
# =============================================================================


# ---------------------------------------
# data pipeline definitions

class LandmarkDataset(Dataset):
    def __init__(self, csv, mode, transform=None):

        self.csv = csv.reset_index()
        self.mode = mode
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
        if self.mode == 'test':
            return torch.tensor(image)
        else:
            return torch.tensor(image), torch.tensor(row.landmark_id)


def get_transforms():

    transforms_train = albumentations.Compose([
        albumentations.Resize(IMAGE_SIZE, IMAGE_SIZE),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.ImageCompression(quality_lower=99, quality_upper=100),
        albumentations.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=10, border_mode=0, p=0.7),
        albumentations.Cutout(max_h_size=int(IMAGE_SIZE * 0.4), max_w_size=int(IMAGE_SIZE * 0.4), num_holes=1, p=0.5),
        albumentations.Normalize()
    ])

    transforms_val = albumentations.Compose([
        albumentations.Resize(IMAGE_SIZE, IMAGE_SIZE),
        albumentations.Normalize()
    ])

    return transforms_train, transforms_val


def get_df():

    df = pd.read_csv(os.path.join(DATA_DIR, 'train_0.csv'))

    if TRAIN_STEP == 0:
        # df_train = pd.read_csv(os.path.join(DATA_DIR, 'train_url.csv')).drop(columns=['url'])
        df_train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    else:
        cls_81313 = df.landmark_id.unique()
        # df_train = pd.read_csv(os.path.join(DATA_DIR, 'train_url.csv')).drop(columns=['url']).set_index('landmark_id').loc[cls_81313].reset_index()
        df_train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv')).set_index('landmark_id').loc[cls_81313].reset_index()
        
    df_train['filepath'] = df_train['id'].apply(lambda x: os.path.join(DATA_DIR, 'train', x[0], x[1], x[2], f'{x}.jpg'))
    df = df_train.merge(df, on=['id','landmark_id'], how='left')

    landmark_id2idx = {landmark_id: idx for idx, landmark_id in enumerate(sorted(df['landmark_id'].unique()))}
    idx2landmark_id = {idx: landmark_id for idx, landmark_id in enumerate(sorted(df['landmark_id'].unique()))}
    df['landmark_id'] = df['landmark_id'].map(landmark_id2idx)

    out_dim = df.landmark_id.nunique()

    return df, out_dim


# In[5]:
# =============================================================================


# ---------------------------------------
# model definitions

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
    @autocast()
    def forward(self, x):
        return Swish.apply(x)


class DenseCrossEntropy(nn.Module):
    @autocast()
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
    
    @autocast()
    def forward(self, features):
        cosine_all = F.linear(F.normalize(features), F.normalize(self.weight))
        cosine_all = cosine_all.view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine_all, dim=2)
        return cosine   


class ArcFaceLossAdaptiveMargin(nn.modules.Module):
    def __init__(self, margins, s=30.0):
        super().__init__()
        self.crit = DenseCrossEntropy()
        self.s = s
        self.margins = margins
            
    @autocast()
    def forward(self, logits, labels, out_dim):
        ms = []
        ms = self.margins[labels.cpu().numpy()]
        cos_m = torch.from_numpy(np.cos(ms)).float().cuda()
        sin_m = torch.from_numpy(np.sin(ms)).float().cuda()
        th = torch.from_numpy(np.cos(math.pi - ms)).float().cuda()
        mm = torch.from_numpy(np.sin(math.pi - ms) * ms).float().cuda()
        labels = F.one_hot(labels, out_dim).float()
        logits = logits.float()
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * cos_m.view(-1, 1) - sine * sin_m.view(-1, 1)
        phi = torch.where(cosine > th.view(-1, 1), phi, cosine - mm.view(-1, 1))
        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s
        loss = self.crit(output, labels)
        return loss



class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        
        super().__init__()
        
        self.p=p
        self.eps=eps
        
    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)


class RexNet20_Landmark(nn.Module):

    def __init__(self, out_dim, load_pretrained=True):
        super(RexNet20_Landmark, self).__init__()

        self.net = timm.create_model('rexnet_200', pretrained=load_pretrained)
        self.feat = nn.Linear(self.net.features[-1].out_channels, 512)
        self.swish = Swish_module()
        self.metric_classify = ArcMarginProduct_subcenter(512, out_dim)
        self.net.head.global_pool = GeM()
        self.net.head.fc = nn.Identity()

    def extract(self, x):
        return self.net(x).squeeze()

    @autocast()
    def forward(self, x):
        x = self.extract(x)
        logits_m = self.metric_classify(self.swish(self.feat(x)))
        return logits_m


# In[6]:
# =============================================================================


# ---------------------------------------
# training utils

def global_average_precision_score(
        y_true: Dict[Any, Any],
        y_pred: Dict[Any, Tuple[Any, float]]
) -> float:
    """
    Compute Global Average Precision score (GAP)
    Parameters
    ----------
    y_true : Dict[Any, Any]
        Dictionary with query ids and true ids for query samples
    y_pred : Dict[Any, Tuple[Any, float]]
        Dictionary with query ids and predictions (predicted id, confidence
        level)
    Returns
    -------
    float
        GAP score
    """
    indexes = list(y_pred.keys())
    indexes.sort(
        key=lambda x: -y_pred[x][1],
    )
    queries_with_target = len([i for i in y_true.values() if i is not None])
    correct_predictions = 0
    total_score = 0.
    for i, k in enumerate(indexes, 1):
        relevance_of_prediction_i = 0
        if y_true[k] == y_pred[k][0]:
            correct_predictions += 1
            relevance_of_prediction_i = 1
        precision_at_rank_i = correct_predictions / i
        total_score += precision_at_rank_i * relevance_of_prediction_i

    return 1 / queries_with_target * total_score
    

def train_epoch(model, loader, optimizer, criterion, scaler, scheduler):

    model.train()
    train_loss = []
    bar = tqdm(loader)
    for (data, target) in bar:

        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()

        if not USE_AMP:
            logits_m = model(data)
            loss = criterion(logits_m, target)
            loss.backward()
            optimizer.step()
        else:
            with autocast():
                logits_m = model(data)
                loss = criterion(logits_m, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        torch.cuda.synchronize()
        
        # OneCycleLR stepping
        scheduler.step()
            
        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description('loss: %.5f, smth: %.5f' % (loss_np, smooth_loss))

    return train_loss


def val_epoch(model, valid_loader, criterion, get_output=False):

    model.eval()
    val_loss = []
    PRODS_M = []
    PREDS_M = []
    TARGETS = []

    with torch.no_grad():
        for (data, target) in tqdm(valid_loader):
            data, target = data.cuda(), target.cuda()

            logits_m = model(data)

            lmax_m = logits_m.max(1)
            probs_m = lmax_m.values
            preds_m = lmax_m.indices

            PRODS_M.append(probs_m.detach().cpu())
            PREDS_M.append(preds_m.detach().cpu())
            TARGETS.append(target.detach().cpu())

            loss = criterion(logits_m, target)
            val_loss.append(loss.detach().cpu().numpy())

        val_loss = np.mean(val_loss)
        PRODS_M = torch.cat(PRODS_M).numpy()
        PREDS_M = torch.cat(PREDS_M).numpy()
        TARGETS = torch.cat(TARGETS)

    if get_output:
        return LOGITS_M
    else:
        acc_m = (PREDS_M == TARGETS.numpy()).mean() * 100.
        y_true = {idx: target if target >=0 else None for idx, target in enumerate(TARGETS)}
        y_pred_m = {idx: (pred_cls, conf) for idx, (pred_cls, conf) in enumerate(zip(PREDS_M, PRODS_M))}
        gap_m = global_average_precision_score(y_true, y_pred_m)
        return val_loss, acc_m, gap_m


# In[7]:
# =============================================================================


# get dataframe
df, out_dim = get_df()
print(f"out_dim = {out_dim}")

# get adaptive margin
tmp = np.sqrt(1 / np.sqrt(df['landmark_id'].value_counts().sort_index().values))
margins = (tmp - tmp.min()) / (tmp.max() - tmp.min()) * 0.45 + 0.05


# In[8]:
# =============================================================================


# get augmentations
transforms_train, transforms_val = get_transforms()

# get train and valid dataset
df_train = df[df['fold'] != FOLD]
df_valid = df[df['fold'] == FOLD].reset_index(drop=True).query("index % 15==0")

dataset_train = LandmarkDataset(df_train, 'train', transform=transforms_train)
dataset_valid = LandmarkDataset(df_valid, 'val', transform=transforms_val)
train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True, drop_last=True)
valid_loader = DataLoader(dataset_valid, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
 


# In[9]:
# =============================================================================

# load weight
load = torch.load(os.path.join(MODEL_DIR, f'{LOAD_MODEL}.pth'))
# remove metric module weight
model_only_weight = {
    k[7:] if k.startswith('module.') else k: v for k, v in load['model_state_dict'].items() if 'metric_classify' not in k
}

# model
model = RexNet20_Landmark(out_dim=out_dim).cuda()
model.load_state_dict(model_only_weight, strict=False)

model = nn.DataParallel(model)

# loss func
def criterion(logits_m, target):
    arc = ArcFaceLossAdaptiveMargin(margins=margins, s=80)
    loss_m = arc(logits_m, target, out_dim)
    return loss_m

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scaler = GradScaler(enabled=True)

# scheduler
scheduler = OneCycleLR(optimizer, max_lr=LR, steps_per_epoch=len(train_loader), pct_start=SCHEDULER_PEAK, epochs=NUM_EPOCHS)


# In[10]:
# =============================================================================


# train & valid loop
gap_m_max = 0.
model_file = os.path.join(MODEL_DIR, f'{MODEL_NAME}_fold{FOLD}.pth')

for epoch in range(NUM_EPOCHS):
    
    curr_time = datetime.strftime(datetime.now(), '%Y%b%d_%HH%MM%SS')
    print(curr_time, 'Epoch:', epoch)
    
    train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, scheduler)
    val_loss, acc_m, gap_m = val_epoch(model, valid_loader, criterion)

    content = curr_time + ' ' + f'Fold {FOLD}, Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {np.mean(train_loss):.5f}, valid loss: {(val_loss):.5f}, acc_m: {(acc_m):.6f}, gap_m: {(gap_m):.6f}.'
    print(content)
    
    with open(os.path.join(MODEL_DIR, f'{MODEL_NAME}.txt'), 'a') as appender:
        appender.write(content + '\n')

    print('gap_m_max ({:.6f} --> {:.6f}). Saving model ...'.format(gap_m_max, gap_m))
    
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        },
        model_file
    )
    gap_m_max = gap_m

print(datetime.strftime(datetime.now(), '%Y%b%d_%HH%MM%SS'), 'Training Finished!')

torch.save(
    {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 
    os.path.join(MODEL_DIR, f'{MODEL_NAME}_fold{FOLD}_final.pth')
)

