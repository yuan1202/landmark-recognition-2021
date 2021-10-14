#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 21:50:18 2021

@author: virilo


"""

top_k_retrieve=300
top_k=20

version=f"v3_top_{top_k}"



import torch 

FP16,float_precision=True,torch.float32
device='cuda:0'
batchsize = 128

import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--batch_id', default=-1, type=int, help='batch number in output filename', required=False)
parser.add_argument('--remove_public_nl', default=0, type=int, help="1 remove non embeddings in public testset - 0 don't remove", required=False)
parser.add_argument('--landmarks_filter', default=0, type=int, help="1 use only landmarks in train - 0 don't filter", required=False)
parser.add_argument('--first_landmark', default=-1, type=int, help='first landmark_idx to filter (inclusive)', required=False)
parser.add_argument('--last_landmark', default=-1, type=int, help='first landmark_idx to filter (inclusive)', required=False)
parser.add_argument('--model_id', default='first-try-g-512-quantile-transformer', type=str, help='first landmark_idx to filter (inclusive)', required=False)





args, unknown = parser.parse_known_args()
landmarks_filter=(args.landmarks_filter==1)
remove_public_nl=(args.remove_public_nl==1)

print("adversarials: ", args.batch_id, args.first_landmark, args.last_landmark, landmarks_filter, remove_public_nl)


CLEAN_DATASET_SAMPLES=1580470
NUM_CLASSES=81313

import os, socket, sys

EMBEDDINGS_ROOT_PATH='../input/' if not ON_LOCAL_VIRILO else '/media/wd-2021/results_landmark/embeddings/'
DATA_DIR       = '../input/' if not ON_LOCAL_VIRILO else '/fast-drive/google-landmark/'


import pandas as pd

def get_used_GPU():
    print("no gpu object.  ??? free gpu ram ", flush=True)

import psutil, os
def get_used_memory():
    return psutil.Process(os.getpid()).memory_info().vms/1024**3
def get_used_memory_txt():
    return 'Used memory: {:.2f}'.format(get_used_memory())
initial_used_memory=get_used_memory()
print(get_used_memory_txt())

model_id= args.model_id
print("float_precision:", float_precision, "\model_id:", model_id)
LOAD_Q_TRANSFORMED=True


from tqdm.notebook import tqdm

import warnings
warnings.filterwarnings("ignore")

import gc


import numpy as np


import pickle
PICKLE_PROTOCOL=4  # there are issues with pickle.HIGHEST_PROTOCOL unsing local recent python versions and then kaggle kernels

def load_pickle(filename):
    ### LOAD PICKLE
    with open(filename, "rb") as pfile:
        x = pickle.load(pfile)
    return x
def save_pickle(x, filename=None, makedirs=True):
    if makedirs and os.path.dirname(filename)!='': os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as pfile:
        pickle.dump(x, pfile, protocol=PICKLE_PROTOCOL)
read_pickle=load_pickle


def to_hex_id(i):
    #return i.to_bytes(((i.bit_length() + 7) // 8),"big").hex(), 
    #return hex(i)[2:]
    return format(i, '#018x')[2:]

def to_int_id(id_):
    return int(id_, 16)


def cos_similarity_matrix(a, b, eps=1e-8):
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt
def get_topk_cossim(test_emb, tr_emb, batchsize = 64, k=10, device='cuda:0',verbose=True):
    tr_emb = torch.tensor(tr_emb, dtype = float_precision, device=torch.device(device))
    test_emb = torch.tensor(test_emb, dtype = float_precision, device=torch.device(device))
    vals = []
    inds = []
    for test_batch in tqdm(test_emb.split(batchsize),disable=1-verbose):
        sim_mat = cos_similarity_matrix(test_batch, tr_emb)
        vals_batch, inds_batch = torch.topk(sim_mat, k=k, dim=1)
        vals += [vals_batch.detach().cpu()]
        inds += [inds_batch.detach().cpu()]
    vals = torch.cat(vals)
    inds = torch.cat(inds)
    
    print("before delete and empty_cache")
    get_used_GPU()
    
    del tr_emb, test_emb
    torch.cuda.empty_cache()
    
    return vals, inds

def get_topk_cossim_sub(test_emb, tr_emb, vals_x, batchsize = 64, k=10, device='cuda:0',verbose=True):
    tr_emb = torch.tensor(tr_emb, dtype = float_precision, device=torch.device(device))
    test_emb = torch.tensor(test_emb, dtype = float_precision, device=torch.device(device))
    vals_x = torch.tensor(vals_x, dtype = float_precision, device=torch.device(device))
    vals = []
    inds = []
    for test_batch in tqdm(test_emb.split(batchsize),disable=1-verbose):
        sim_mat = cos_similarity_matrix(test_batch, tr_emb)
        sim_mat = torch.clamp(sim_mat,0,1) - vals_x.repeat(sim_mat.shape[0], 1)
        
        vals_batch, inds_batch = torch.topk(sim_mat, k=k, dim=1)
        vals += [vals_batch.detach().cpu()]
        inds += [inds_batch.detach().cpu()]
    vals = torch.cat(vals)
    inds = torch.cat(inds)
    return vals, inds

def global_average_precision_score(y_true, y_pred, ignore_non_landmarks=False):
    indexes = np.argsort(y_pred[1])[::-1]
    queries_with_target = (y_true < NUM_CLASSES).sum()
    correct_predictions = 0
    total_score = 0.
    i = 1
    for k in indexes:
        if ignore_non_landmarks and y_true[k] == NUM_CLASSES:
            continue
        if y_pred[0][k] == NUM_CLASSES:
            continue
        relevance_of_prediction_i = 0
        if y_true[k] == y_pred[0][k]:
            correct_predictions += 1
            relevance_of_prediction_i = 1
        precision_at_rank_i = correct_predictions / i
        total_score += precision_at_rank_i * relevance_of_prediction_i
        i += 1
    return 1 / queries_with_target * total_score

def comp_metric(y_true, logits, ignore_non_landmarks=False):
    
    score = global_average_precision_score(y_true, logits, ignore_non_landmarks=ignore_non_landmarks)
    return score
# In[28]:


'''
Split GLDv2_extended in two batches, 

    batch 1: landmark_idx: 0 to 40916 (101709 landmark_id)
    batch 2: 40917 to 81312
    
    
model_id='first-try-g-512-quantile-transformer'


current_train_landmark_id=set(list(set(train_df['landmark_id'].values))[:-1000])

'''


(landmark_mapping_idx_to_id, landmark_mapping_id_to_idx)=load_pickle(f'{DATA_DIR}/gld-v2-1/idx_mapping.pkl')

NON_LANDMARK =  91
NON_LANDMARK_2019 =  92
DIRTY_SCRAPPED_NON_LANDMARK =  93
CLEAN__SCRAPPED_NON_LANDMARK =  94
CLEAN__SCRAPPED_NON_LANDMARK_V2 =  95
CLEAN__SCRAPPED_NON_LANDMARK_V3 =  96
CLEAN__SCRAPPED_NON_LANDMARK_V4 =  97
EXTRA_LABELS=81314

landmark_mapping_idx_to_id=np.hstack([landmark_mapping_idx_to_id, [NON_LANDMARK, NON_LANDMARK_2019, DIRTY_SCRAPPED_NON_LANDMARK, CLEAN__SCRAPPED_NON_LANDMARK, CLEAN__SCRAPPED_NON_LANDMARK_V2, CLEAN__SCRAPPED_NON_LANDMARK_V3, CLEAN__SCRAPPED_NON_LANDMARK_V4]])
landmark_mapping_id_to_idx={landmark_mapping_idx_to_id[i]:i for i in range(len(landmark_mapping_idx_to_id))}
landmark_mapping_idx_to_id[EXTRA_LABELS]==NON_LANDMARK
landmark_mapping_id_to_idx[NON_LANDMARK]==EXTRA_LABELS

IDX_NON_LANDMARK = landmark_mapping_id_to_idx[NON_LANDMARK]
IDX_NON_LANDMARK_2019 = landmark_mapping_id_to_idx[NON_LANDMARK_2019]
IDX_DIRTY_SCRAPPED_NON_LANDMARK = landmark_mapping_id_to_idx[DIRTY_SCRAPPED_NON_LANDMARK]
IDX_CLEAN__SCRAPPED_NON_LANDMARK = landmark_mapping_id_to_idx[CLEAN__SCRAPPED_NON_LANDMARK]
IDX_CLEAN__SCRAPPED_NON_LANDMARK_V2 = landmark_mapping_id_to_idx[CLEAN__SCRAPPED_NON_LANDMARK_V2]
IDX_CLEAN__SCRAPPED_NON_LANDMARK_V3 = landmark_mapping_id_to_idx[CLEAN__SCRAPPED_NON_LANDMARK_V3]
IDX_CLEAN__SCRAPPED_NON_LANDMARK_V4 = landmark_mapping_id_to_idx[CLEAN__SCRAPPED_NON_LANDMARK_V4]




print(f"Load pickles. {get_used_memory_txt()}", flush=True)

print(f"# step e. {get_used_memory_txt()}", flush=True)





#VIRILO B5 
# targets_train_old=targets_train
# targets_train = load_pickle(f'{EMBEDDINGS_ROOT_PATH}/{model_id}/targets_train.pkl')
#VIRILO B5 

pd.read_csv("/media/wd-2021/landmark-more/zip-landmarks-scrapper/row-order/3.2M_landmarks_row_order.csv")['id']
"/media/wd-2021/landmark-more/zip-landmarks-scrapper/row-order/2019_non_landmarks_row_order.csv"


np_extended=np.load("/fast-drive/google-landmark/model-landmark-recognition/model_v0.7_landmark_embeddings_f16_all.npy")
np_scrapped=np.load("/media/wd-2021/landmark-more/zip-landmarks-scrapper/embeddings/model_v0.7_scrapped_landmarks_embeddings_f16.npy")
np_nlm2019=np.load("/fast-drive/google-landmark/model-landmark-recognition/model_v0.7_non_landmark_embeddings_f16.npy")
np_dirty=np.load("/media/wd-2021/landmark-more/zip-landmarks-scrapper/embeddings/model_v0.7_dirty-nlm_embeddings_f16.npy")

len_extended=len(np_extended)
len_scrapped=len(np_scrapped)
len_nlm2019=len(np_nlm2019)
len_dirty=len(np_dirty)
tr_embeddings = np.vstack([np_extended, np_scrapped, np_nlm2019, np_dirty])
np_extended, np_scrapped, np_nlm2019, np_dirty=None, None, None, None
gc.collect()
del np_extended, np_scrapped, np_nlm2019, np_dirty
gc.collect()



'''
    - extended
    - scrapped L
    - NL 2019
    - Dirty
    '''
    
row_order_extended=pd.read_csv("/media/wd-2021/landmark-more/zip-landmarks-scrapper/row-order/3.2M_landmarks_row_order.csv")
row_order_scrapped=pd.read_csv("/media/wd-2021/landmark-more/zip-landmarks-scrapper/row-order/scrapped-landmarks_v2.csv")
row_order_nlm2019=pd.read_csv("/media/wd-2021/landmark-more/zip-landmarks-scrapper/row-order/2019_non_landmarks_row_order.csv")
row_order_dirty=pd.read_csv("/media/wd-2021/landmark-more/zip-landmarks-scrapper/row-order/dirty_scrapped_non_landmarks_row_order_and_similarity.csv")
 
targets_train=np.array([landmark_mapping_id_to_idx[x] for x in row_order_extended['landmark_id'].values ] + \
    [landmark_mapping_id_to_idx[x] for x in row_order_scrapped['landmarks'].values ]+ \
    [NON_LANDMARK]*len_nlm2019 + \
    [NON_LANDMARK]*len_dirty
        
)
len(targets_train)==len(tr_embeddings)


# tr_embeddings_old=tr_embeddings


if FP16: tr_embeddings=tr_embeddings.astype(np.float16)
gc.collect()

# if os.path.isfile(f"{EMBEDDINGS_ROOT_PATH}/{model_id}/train_ids.pkl"):
#     train_ids= load_pickle(f"{EMBEDDINGS_ROOT_PATH}/{model_id}/train_ids.pkl")
# else:
#     print("ERROR, not found: train_ids.pkl")
#     train_ids= None
# train_ids_copy=train_ids.copy()
train_ids=np.array([to_int_id(x) for x in row_order_extended['id'].values] + \
    [to_int_id(x) for x in row_order_scrapped['id'].values] + \
    [to_int_id(x) for x in row_order_nlm2019['id'].values] + \
    [to_int_id(x) for x in row_order_dirty['id'].values] 
, dtype=np.uint64)
len(train_ids)==len(tr_embeddings)

row_order_extended, row_order_scrapped, row_order_nlm2019, row_order_dirty= None,None,None,None
gc.collect()
''' IGNORE IT
if landmarks_filter:
    train_df=pd.read_csv(f'{COMPETITION_PATH}/train.csv')
    # current_train_landmark_idx=set([landmark_mapping_id_to_idx[x] for x in train_df['landmark_id'].values])
    current_train_landmark_id=set(train_df['landmark_id'].values)
    # np.max(targets_train)
    # current_landmark_filter=[x in current_train_landmark_idx for x in targets_train]
    current_landmark_filter=[landmark_mapping_idx_to_id[x] in current_train_landmark_id for x in targets_train]
    print(f"landmarks_filter: {np.sum(current_landmark_filter)}/{len(targets_train)}")
    
    tr_embeddings=tr_embeddings[current_landmark_filter];print("tr_embeddings", len(tr_embeddings))
    targets_train=targets_train[current_landmark_filter];print("targets_train", len(targets_train))
    if train_ids is not None:
        train_ids=train_ids[current_landmark_filter];print("train_ids", len(train_ids))

if args.first_landmark!=-1 and args.last_landmark!=-1:
    1/0
    print(f"Selecting train rows for landmarks {args.first_landmark} to {args.last_landmark}. {get_used_memory_txt()}", flush=True)
    batch_selection=[( x<=args.last_landmark and x>=args.first_landmark) for x in targets_train]
    if train_ids is not None:
        print("saving train_ids.pkl")
        train_ids=train_ids[batch_selection]
        save_pickle(train_ids, f"batch_{args.batch_id}.train_ids.pkl")
        train_ids=None
    tr_embeddings=tr_embeddings[batch_selection]
    targets_train=targets_train[batch_selection]    
    batch_selection=None
    print(len(tr_embeddings),len(targets_train))

END OF IGNORE IT'''


# nonlandmark_embeddings = load_pickle(f'{EMBEDDINGS_ROOT_PATH}/{model_id}/nonlandmark_embeddings.pkl')
# (test_embeddings, test_hex_ids)=load_pickle("tmp_embeddings.pkl")

'''
if remove_public_nl and os.path.isfile(f"{EMBEDDINGS_ROOT_PATH}/{model_id}/nonlandmark_ids.pkl"):
    nonlandmark_ids=load_pickle(f"{EMBEDDINGS_ROOT_PATH}/{model_id}/nonlandmark_ids.pkl")
    nonlandmark_ids=[to_hex_id(x) for x in nonlandmark_ids]
    sample_submission_df=pd.read_csv(f'{COMPETITION_PATH}/sample_submission.csv')
    sample_submission_ids=set(sample_submission_df['id'].values)
    nl_selection=[(x not in sample_submission_ids) for x in nonlandmark_ids]
    print(f"remove_public_nl: ({np.sum(nl_selection)}/{len(nonlandmark_ids)}) | {len(nonlandmark_ids)-np.sum(nl_selection)} non-landmark removed")
    nonlandmark_embeddings=nonlandmark_embeddings[nl_selection]
    nonlandmark_ids=None
    # nonlandmark_ids=load_pickle(args.nonlandmark_ids)
    # nonlandmark_ids=nonlandmark_ids[nl_selection]
    print("nonlandmark_embeddings:", len(nonlandmark_embeddings))
elif remove_public_nl:
    print("remove_public_nl ERROR, not found: nonlandmark_ids.pkl")   
elif os.path.isfile(f"{EMBEDDINGS_ROOT_PATH}/{model_id}/nonlandmark_ids.pkl"):
    print("loading nonlandmark_ids.pkl")
    nonlandmark_ids=load_pickle(args.nonlandmark_ids)

if  os.path.isfile(args.nonlandmark_ids):
    print("saving nonlandmark_ids.pkl")
#    train_ids=train_ids[batch_selection]
    save_pickle(nonlandmark_ids, f"nonlandmark_ids.pkl")
    nonlandmark_ids=None
'''
print("tr_embeddings:", len(tr_embeddings), flush=True)

# calculate A-B
print(f"# calculate A-B. {get_used_memory_txt()}", flush=True)
EMB_SIZE = 512

i=0

print(f"# step 1. {get_used_memory_txt()}", flush=True)
get_used_GPU()
# vals_nl, inds_nl = get_topk_cossim(tr_embeddings[:,i*EMB_SIZE:(i+1)*EMB_SIZE], nonlandmark_embeddings[:,i*EMB_SIZE:(i+1)*EMB_SIZE], k=5)

tr_emb = torch.tensor(tr_embeddings, dtype = float_precision, device=torch.device(device))
adversarial_emb=tr_emb

'''
train_ids= load_pickle('/fast-drive/google-landmark/first-try-g-512-quantile-transformer/train_ids.pkl')
targets_train= load_pickle('/fast-drive/google-landmark/first-try-g-512-quantile-transformer/targets_train.pkl')
'''

vals = []
inds = []
targets=targets_train.copy()
ids=train_ids.copy()

adversarials={}
best_friends={}



for iteration, batch in tqdm(enumerate(tr_emb.split(batchsize))):
    batch_ids=ids[:len(batch)]
    ids=ids[len(batch):]
    batch_targets=targets[:len(batch)]
    targets=targets[len(batch):]
    
    sim_mat = cos_similarity_matrix(batch, adversarial_emb)
    advers_vals, advers_inds = torch.topk(sim_mat, k=top_k_retrieve, dim=1)
    advers_vals=advers_vals.detach().cpu().numpy()
    advers_inds=advers_inds.detach().cpu().numpy()
    advers_ids=train_ids[advers_inds]    
    advers_targets=targets_train[advers_inds]    
    # save_pickle((advers_targets, advers_vals,advers_inds, batch_targets, batch_ids, advers_ids),"tmp.pkl")
    # (advers_targets, advers_vals,advers_inds, batch_targets, batch_ids, advers_ids)= load_pickle("tmp.pkl")
    
    for i in range(len(batch)):
        vals_i=advers_vals[i,:]
        inds_i=advers_inds[i,:]
        ids_i=advers_ids[i,:]
        labels_i=advers_targets[i,:]
        label=batch_targets[i]
        img_id=batch_ids[i]
        
        
        sel=[l!=label for l in labels_i]
        adversarials[img_id]=(ids_i[sel][:top_k],vals_i[sel][:top_k])
        
        sel=[l==label for l in labels_i]
        best_friends[img_id]=(ids_i[sel][:top_k],vals_i[sel][:top_k])
        
    
    # if iteration%(100000//batchsize)==0 or iteration==100:
    #     print(f"iteration {iteration} saving...", end='')
    #     save_pickle(adversarials, f"all_data_adversarials_{version}.pkl")
    #     save_pickle(best_friends, f"all_data_best_friends_{version}.pkl")
    #     print("done")


targets_train=None
train_ids=None
gc.collect()

save_pickle(adversarials, f"all_data_adversarials_{version}.pkl")
save_pickle(best_friends, f"all_data_best_friends_{version}.pkl")
    
    

'''

(33e6*(12*100))/(1024**3)

print("before delete and empty_cache")
get_used_GPU()

del tr_emb, adversarial_emb
torch.cuda.empty_cache()

'''