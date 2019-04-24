
# coding: utf-8

# In[1]:

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
import os

print(torch.__version__)

os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

# Load data
DATA_PATH = '/home/shenxk/data/LungCancerData/LIDC_prep'

experiment_name = 'DenseU_2d_assignCenter_noWeight_crop128'

batch_size = 16
crop_size = 128
num_z = 10
nEpochs = 120


# In[2]:

LR = 1e-3
LR_DECAY = 0.95
DECAY_EVERY_N_EPOCHS = 20

model = Tiramisu.FCDensenet46(num_classes=2, dense_bn_size=None, dense_compression=1.0).cuda()
criterion = nn.NLLLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

para_num = sum([p.data.nelement() for p in model.parameters()])
print('Number of params: {}'.format(para_num))
print("Let's use", torch.cuda.device_count(), "GPUs!")

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)


# In[3]:

CHECKPOINT_PATH = '/home/shenxk/Documents/nodule_seg3d/results/checkpoint'
#experiment_name = 'DenseU_2d_assignCenter_noWeight_fromScratch'
experiment = train_utils.Experiment(model, criterion=criterion, optimizer=optimizer,
                                    checkpoint_path=CHECKPOINT_PATH, experiment_name=experiment_name)


# In[6]:

joint_transformer, dataset, dataloader = {}, {}, {}

# ==== Train ====
# Fine tune with AnnoRandomCrop and RandomHorizontalFlip in training set to avoid overfitting
joint_transformer['train'] = transforms.Compose([    
    joint_transforms.JointAnnoCenterCrop(crop_size),
    joint_transforms.JointRandomHorizontalFlip()
    ])

joint_transformer['val'] = transforms.Compose([    
     joint_transforms.JointAnnoCenterCrop(crop_size)
    ])

dataset['train'] = io_utils_lidc3d.lidc2d_assignCenter(
                      DATA_PATH, 'train', [num_z, crop_size, crop_size], split_files = 'train.npy',
                      slice_split_files = 'train_slices.npy',
                      slice_nodule_split_files = 'train_slices_nodule.npy',
                      transform = ['ToTensor'], joint_transform=joint_transformer['train'])
dataloader['train'] = torch.utils.data.DataLoader(dataset['train'], batch_size=batch_size, 
                                                  shuffle=True, num_workers=1)
# ==== Val ====
dataset['val'] = io_utils_lidc3d.lidc2d_assignCenter(
                    DATA_PATH, 'val', [num_z, crop_size, crop_size], split_files = 'val.npy',
                    slice_split_files = 'val_slices.npy',
                    slice_nodule_split_files = 'val_slices_nodule.npy',
                    transform = ['ToTensor'], joint_transform=None)
dataloader['val'] = torch.utils.data.DataLoader(dataset['val'], batch_size=1, 
                                                shuffle=False, num_workers=2)

print("Train: %d" % len(dataloader['train'].dataset.slice_names))
print("Val: %d"   % len(dataloader['val'].dataset.slice_names))

imgs, labels = next(iter(dataloader['train']))
print("Train images: ", imgs.size())
print("Train labels: ", labels.size())

imgs_val, labels_val = next(iter(dataloader['val']))
print("Val images: ", imgs_val.size())
print("Val labels: ", labels_val.size())


# In[7]:

for i in range(nEpochs):
    start_time = time.time()
    loss_trn, iou_trn, dice_trn, sen_trn, ppv_trn = experiment.train(dataloader['train'], output=True)
    loss_val, iou_val, dice_val, sen_val, ppv_val = experiment.val(dataloader['val'], output=True)
    end_time = time.time()
    print('Time consuming: %d min %d s' % (int((end_time - start_time)//60), int((end_time - start_time)%60)))
    
    experiment.save_weights(loss_val, iou_val, dice_val, sen_val, ppv_val)
    experiment.adjust_learning_rate(LR, LR_DECAY, DECAY_EVERY_N_EPOCHS)
    
experiment.save_csv()


# In[9]:

# ==== Test ====
dataset['test'] = io_utils_lidc3d.lidc2d_assignCenter(
                    DATA_PATH, 'test', [num_z, crop_size, crop_size], split_files = 'test.npy',
                    slice_split_files = 'test_slices.npy',
                    slice_nodule_split_files = 'test_slices_nodule.npy',
                    transform = ['ToTensor'], joint_transform=None)
dataloader['test'] = torch.utils.data.DataLoader(dataset['test'], batch_size=1, 
                                                shuffle=False, num_workers=2)

#experiment.load_weights(os.path.join(CHECKPOINT_PATH, 'DenseU_2d_assignCenter_noWeight_fromScratch_checkpoint.pth.tar'))

loss_test, iou_test, dice_test, sen_test, ppv_test = experiment.val(dataloader['test'], output=True)

