# Training with augmentation for 180 epochs after training with no augmentation for 180 epochs.
# coding: utf-8

# In[1]:

# Adapted from Tiramisu_56_BC_IN_4ch-Dice.ipynb of xujq
#get_ipython().magic('load_ext autoreload')
#get_ipython().magic('autoreload 2')

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import io_utils_lidc3d, train_utils_aug as train_utils
import joint_transforms
import loss
import Tiramisu

import matplotlib.pyplot as plt
import time
#% matplotlib inlinea


# In[2]:

print(torch.__version__)


# In[3]:

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'


# ## Load data

# In[5]:

#DATA_PATH = '/home/xujq/data/ILD_Geneva/ILD_seg_slice/'
#DATA_PATH = '/home/shenxk/data/ILD/ILD_geneva/ILD_seg_slice/'
DATA_PATH = '/home/shenxk/data/LungCancerData/LIDC_prep'

batch_size = 3
crop_size = 64
num_z = 10

# In[14]:

MEAN, STD = io_utils_lidc3d.MEAN, io_utils_lidc3d.STD
transform = ['ToTensor', 'normalize']

joint_transformer, dataset, dataloader = {}, {}, {}

# ==== Train ====
# Fine tune with AnnoRandomCrop and RandomHorizontalFlip in training set to avoid overfitting
joint_transformer['train'] = transforms.Compose([    
    #joint_transforms_3d.JointRandomCrop_3d(96) # commented for fine-tuning
    #joint_transforms_3d.JointAnnoRandomCrop_3d(128),
    joint_transforms.JointAnnoCenterCrop(crop_size),
    #joint_transforms.JointRandomHorizontalFlip()
    ])

joint_transformer['val'] = transforms.Compose([    
    #joint_transforms_3d.JointRandomCrop_3d(96) # commented for fine-tuning
    joint_transforms.JointAnnoCenterCrop(crop_size)
    #joint_transforms.JointRandomRotate(),
    #joint_transforms.JointRandomHorizontalFlip()
    ])

dataset['train'] = io_utils_lidc3d.lidc2d(DATA_PATH, 'train', [num_z, crop_size, crop_size], split_files = 'train.npy',
                                    slice_split_files = 'train_slices.npy', joint_transform=joint_transformer['train'])
dataloader['train'] = torch.utils.data.DataLoader(dataset['train'], batch_size=batch_size, 
                                                  shuffle=True, num_workers=2)
# ==== Val ====
dataset['val'] = io_utils_lidc3d.lidc2d(DATA_PATH, 'val', [num_z, crop_size, crop_size], split_files = 'val.npy',
                                  slice_split_files = 'val_slices.npy', joint_transform=joint_transformer['val'])
dataloader['val'] = torch.utils.data.DataLoader(dataset['val'], batch_size=1, 
                                                shuffle=False, num_workers=2)
# ==== Test ====
dataset['test'] = io_utils_lidc3d.lidc2d(DATA_PATH, 'test', [num_z, crop_size, crop_size], split_files = 'test.npy',
                                   slice_split_files = 'test_slices.npy', joint_transform=None)
dataloader['test'] = torch.utils.data.DataLoader(dataset['test'], batch_size=1, 
                                                 shuffle=False, num_workers=2)


# In[17]:

print("Train: %d" % len(dataloader['train'].dataset.slice_names))
print("Val: %d"   % len(dataloader['val'].dataset.slice_names))
print("Test: %d"  % len(dataloader['test'].dataset.slice_names))
#print("Classes: %d" % len(dataloader['train'].dataset.classes))

imgs, labels = next(iter(dataloader['train']))
print("Train images: ", imgs.size())
print("Train labels: ", labels.size())

imgs_val, labels_val = next(iter(dataloader['val']))
print("Val images: ", imgs_val.size())
print("Val labels: ", labels_val.size())

'''
num_withAnnos = 0
for ind, data in enumerate(dataloader['train']):
    for i in range(data[0].shape[0]):
        if np.any(data[1].numpy()[i] < 7):
            #print(np.where(imgs[1].numpy()[i] == 1))
            num_withAnnos += 1
            
print('Number of images with annotation: %d' % num_withAnnos)
'''
# ## Experimental settings

# In[8]:

LR = 1e-3
LR_DECAY = 0.95
DECAY_EVERY_N_EPOCHS = 20


# In[9]:

model = Tiramisu.FCDensenet46(num_classes=2, dense_bn_size=None, dense_compression=1.0).cuda()
# criterion = nn.CrossEntropyLoss(weight=io_utils.class_weight.cuda()).cuda()
# criterion = loss.LossMulti(jaccard_weight=0.2, class_weights=io_utils_lidc3d.class_weight)
# criterion = nn.BCELoss(weight = io_utils_lidc3d.class_weight).cuda()
# criterion = nn.BCELoss().cuda()
criterion = nn.NLLLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

para_num = sum([p.data.nelement() for p in model.parameters()])
print('Number of params: {}'.format(para_num))


# In[10]:

print("Let's use", torch.cuda.device_count(), "GPUs!")

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)


# In[11]:

CHECKPOINT_PATH = '/home/shenxk/Documents/nodule_seg3d/results/checkpoint/'
experiment_name = 'DenseU_2d_CenterAnnoCrop'
experiment = train_utils.Experiment(model, criterion=criterion, optimizer=optimizer,
                                    checkpoint_path=CHECKPOINT_PATH, experiment_name=experiment_name)


# In[11]:

#print(model)

# Load existing weights
#experiment.load_weights(path = os.path.join(CHECKPOINT_PATH, 'FCDenseNet_56_BC_IN_4ch_dice_3d_checkpoint.pth.tar'), load_opt = True)

# In[12]:

# patience = 50
# i, epoch = 0, 0

# while i < patience:
#     i, epoch = i+1, epoch+1
for i in range(50):
    
    start_time = time.time()
    loss_trn, iou_trn, dice_trn, sen_trn, ppv_trn = experiment.train(dataloader['train'], output=True)
    loss_val, iou_val, dice_val, sen_val, ppv_val = experiment.val(dataloader['val'], output=True)
    end_time = time.time()
    print('Time consuming: %d min %d s' % (int((end_time - start_time)//60), int((end_time - start_time)%60)))
    
    experiment.save_weights(loss_val, iou_val, dice_val, sen_val, ppv_val)
    experiment.adjust_learning_rate(LR, LR_DECAY, DECAY_EVERY_N_EPOCHS)
    
#     if experiment.current_is_best:
#         i = 0


# In[15]:

experiment.save_csv()