import cv2
import numpy as np
import os
import torch
import torch.utils.data as data
from torchvision.datasets.folder import default_loader

import matplotlib.pyplot as plt
import pylidc as pl

import glob

#MEAN = (0.5, 0.5, 0.5)
#STD  = (1.0, 1.0, 1.0)

MEAN = 0.5
STD = 1.0

blacklist = ['0146198_9.npy', '0222074_3.npy', '0612395_32.npy', '0367118_6.npy']

classes = ['Non-nodule', 'Nodule']
#class_weight = torch.FloatTensor([0.3, 0.7])

def load_label(label_name, label_path, slice_num, slice0):
    # Load label and slice given label name and the slice number.
    bbox = np.load(os.path.join(label_path, 'bbox', label_name))
    mask_pos = np.load(os.path.join(label_path, 'mask', label_name))
    
    mask_pos_rec = np.copy(mask_pos)

    for i in range(len(mask_pos)):
        mask_pos_rec[i] = mask_pos[i] + bbox[i,0]
        mask_pos_now = mask_pos_rec[:, mask_pos_rec[2,:]==slice_num]

    mask_pos_now = (mask_pos_now[0,:], mask_pos_now[1,:])
    
    try:
        mask_rec = np.zeros((slice0.shape[0], slice0.shape[1]))
        mask_rec[mask_pos_now] = 1
        
    except AttributeError:
        print(os.path.join(slice_path, slice_name))
        print(label_name)
        #return np.zeros((slice0.shape[0], slice0.shape[1])), np.zeros((slice0.shape[0], slice0.shape[1]))
    return mask_rec


def load_data(label_name, label_path, slice_path, slice_num):
    # Load label and slice given label name and the slice number.
    slice0 = cv2.imread(os.path.join(slice_path, slice_name))
    mask_rec = load_label(label_name, label_path, slice_num, slice0)
    
    start_slice = int(label_name.split('_')[0][-3:])
    num_slices = int(label_name.split('_')[1].split('.')[0])
    slice_name = label_name[:4]+'0'*(3-len(str(slice_num)))+str(slice_num)+'.png'
    #slice_name = label_name[:4]+'0'*(3-len(str(start_slice)))+str(start_slice+i)+'.png'
    
    return slice0[:,:,2], mask_rec


def transform_ToTensor(data):
    # transform numpy array to float tensor
    
    data = torch.from_numpy(np.array(data, dtype='uint8'))
                        
    if isinstance(data, torch.ByteTensor):
        return data.float().div(255)
    else:
        return data
    
def transform_normalize(tensor, mean, std):
    if not torch.is_tensor(tensor):
        raise TypeError('input is not a tensor.')

    # This is faster than using broadcasting, don't change without benchmarking
    #for t, m, s in zip(tensor, mean, std):   
        
    tensor.sub_(mean).div_(std)
    #print(tensor.size())
    return tensor

def JointAssignCenterCrop(imgs, crop_center, crop_size):
    """Crops the given numpy array with the assigned point in the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size).
    
    The method is revised by Xinke to center crop according to connected regions.
    """
    img, label = imgs[0], imgs[1]
    ih, iw = img.shape[0], img.shape[1]

    th, tw = crop_size, crop_size
    x1 = max(int(round(crop_center[0] - th / 2.)), 0) # If the lower bound is negative, set 0 as the lower bound
    y1 = max(int(round(crop_center[1] - tw / 2.)), 0)

    for i in range(len(imgs)):
        imgs[i] = imgs[i][min(x1, ih - th): min(x1 + th, ih), min(y1, iw - tw): min(y1 + tw, iw)]
    return imgs

    
class lidc3d(data.Dataset):
    def __init__(self, root, split='train', crop_size = [10, 64, 64], transform = ['ToTensor', 'normalize'],
                joint_transform = None, input_2dResult = False, input_2dModelName = None, 
                 split_files = 'train.npy', blacklist = blacklist):
        self.root = root
        assert split in ('train', 'val', 'test')
        self.split = split
        
        label_path = os.path.join(root, 'labels')
        img_path = os.path.join(root, 'slices')
        #label_names = os.listdir(os.path.join(label_path, 'mask'))
        label_names = np.load(split_files)
        for i in range(len(blacklist)):
            label_names = np.delete(label_names, np.where(label_names == blacklist[i]))
        
        self.label_names = label_names
        self.label_path = label_path
        self.img_path = img_path
        self.crop_size = crop_size
        self.joint_transform = joint_transform
        self.transform = transform
        
        self.input_2dResult = input_2dResult
        self.input_2dModelName = input_2dModelName
        
        self.mean = MEAN
        self.std = STD
        
    def __getitem__(self, index):
        label_name  = self.label_names[index]
        start_slice = int(label_name.split('_')[0][-3:])
        num_slices = int(label_name.split('_')[1].split('.')[0])
        scan_name = label_name.split('_')[0][:4]
        
        num_crop = self.crop_size[0]
        start_crop = max(int(start_slice + num_slices / 2 - num_crop / 2), 0)
        
        #print(label_name)
        img0, label0 = load_data(label_name, self.label_path, self.img_path, start_crop)
        img_shape = label0.shape
        img = np.zeros((num_crop, img_shape[0], img_shape[1])) # Use one channel for now
        label = np.zeros((num_crop, img_shape[0], img_shape[1]))
        
        try:
            for slice_num in range(start_crop, start_crop + num_crop):
                img[slice_num-start_crop], label[slice_num-start_crop] = load_data(
                                          label_name, self.label_path, self.img_path, slice_num)
                
        except AttributeError:
            # The start_crop + num_crop may exceed the max slice number.
            list_slices = glob.glob(os.path.join(root, 'slices', scan_name+'*'))
            list_slices.sort()
            
            if start_crop + num_crop > max_slice + 1:
                start_crop = max_slice + 1 - num_crop
            
                for slice_num in range(start_crop, start_crop + num_crop):
                    img[slice_num-start_crop], label[slice_num-start_crop] = load_data(
                                          label_name, self.label_path, self.img_path, slice_num)
            else:
                print('Other error')
                
                
        if self.input_2dResult:
            assert self.input_2dModelName is not None
            pred = np.load(os.path.join(path.replace(self.split, self.split + '_preds/' + self.input_2dModelName)))
            
        if self.joint_transform is not None:
            if self.input_2dResult:
                img, label, pred = self.joint_transform([img, label, pred])
                pred = torch.from_numpy(np.array(pred, dtype='uint8')).long()   
            else:
                img, label = self.joint_transform([img, label])
        
        #img = img.transpose((3,0,1,2))
        #print(img.shape)
        
        if 'ToTensor' in self.transform:
            img = transform_ToTensor(img)
                        
        if 'normalize' in self.transform:
            img = transform_normalize(img, self.mean, self.std)

        label = torch.from_numpy(np.array(label, dtype='uint8')).long()
        
        try:
            assert img.size() == label.size()
        except:
            print(label_name)
            

        if self.input_2dResult:
            return img, label, pred
        
        else:
            return img, label
    
    def __len__(self):
        return len(self.label_names)
    

class lidc2d(data.Dataset):
    def __init__(self, root, split='train', crop_size = [10, 64, 64], transform = ['ToTensor', 'normalize'],
                joint_transform = None, split_files = 'train.npy', slice_split_files = 'train_slices.npy',
                 blacklist = blacklist):
        self.root = root
        assert split in ('train', 'val', 'test')
        self.split = split
        
        label_path = os.path.join(root, 'labels')
        img_path = os.path.join(root, 'slices')
        #label_names = os.listdir(os.path.join(label_path, 'mask'))
        label_names = np.load(split_files)
        slice_names = np.load(slice_split_files)
        for i in range(len(blacklist)):
            #posit = np.where(label_names == blacklist[i])[0]
            slice_names = np.delete(slice_names, np.where(label_names == blacklist[i]), axis = 0)
            label_names = np.delete(label_names, np.where(label_names == blacklist[i]))

        slice_names_flattened = []
        for sublist in slice_names:
            for val in sublist:
                slice_names_flattened.append(val)
        
        self.label_names = label_names
        self.slice_names = slice_names_flattened
        self.label_path = label_path
        self.img_path = img_path
        self.crop_size = crop_size
        self.joint_transform = joint_transform
        self.transform = transform
        self.crop_size = crop_size
        
        self.mean = MEAN
        self.std = STD
        
    def __getitem__(self, index):
        #print(len(self.label_names))
        label_name  = self.label_names[int(index//self.crop_size[0])] # !!label_name和slice_name需要严格排序
        slice_name = self.slice_names[index]
        
        slice_num = int(slice_name.split('.')[0][-3:])
        
        img = cv2.imread(os.path.join(self.img_path, slice_name))
        img = img[:,:,2]
        label = load_label(label_name, self.label_path, slice_num, img)
        
        if self.joint_transform is not None:
            img, label = self.joint_transform([img, label])
        
        if 'ToTensor' in self.transform:
            img = transform_ToTensor(img)
                        
        if 'normalize' in self.transform:
            img = transform_normalize(img, self.mean, self.std)

        label = torch.from_numpy(np.array(label, dtype=np.int32)).long()
        
        assert img.size() == label.size()        
        return img, label
    
    def __len__(self):
        return len(self.slice_names)
    
    
class lidc2d_curriculum(data.Dataset):
    def __init__(self, root, split='train', crop_size = [10, 64, 64], transform = ['ToTensor', 'normalize'],
                joint_transform = None, split_files = 'train.npy', 
                 slice_split_files = 'train_slices.npy', slice_nodule_split_files = 'train_slices_nodule.npy',
                 augratio = 1, blacklist = blacklist):
        self.root = root
        assert split in ('train', 'val', 'test')
        self.split = split
        
        label_path = os.path.join(root, 'labels')
        img_path = os.path.join(root, 'slices')
        #label_names = os.listdir(os.path.join(label_path, 'mask'))
        label_names = np.load(split_files)
        slice_names = np.load(slice_split_files)
        slice_nodule_names = np.load(slice_nodule_split_files)
        for i in range(len(blacklist)):
            #posit = np.where(label_names == blacklist[i])[0]
            slice_names = np.delete(slice_names, np.where(label_names == blacklist[i]), axis = 0)
            slice_nodule_names = np.delete(slice_nodule_names, np.where(label_names == blacklist[i]), axis = 0)
            label_names = np.delete(label_names, np.where(label_names == blacklist[i]))
                   
        slice_names_flattened = []
        slice_nums = [0]
        for i in range(len(slice_names)):
            sublist = slice_names[i]
            numslices_sublist = 0
            for value in sublist:
                if value in slice_nodule_names[i]:
                    for p in range(augratio):
                        slice_names_flattened.append(value)
                    numslices_sublist += augratio
                else:
                    slice_names_flattened.append(value)
                    numslices_sublist += 1

            slice_nums.append(slice_nums[-1] + numslices_sublist)

        self.label_names = label_names
        self.slice_names = slice_names_flattened
        self.slice_nums = slice_nums
        self.label_path = label_path
        self.img_path = img_path
        self.crop_size = crop_size
        self.joint_transform = joint_transform
        self.transform = transform
        
        self.mean = MEAN
        self.std = STD
        
    def __getitem__(self, index):
        #print(len(self.label_names))
        slice_nums = self.slice_nums
        for i in range(len(slice_nums)):
            if (index >= slice_nums[i]) & (index < slice_nums[i+1]):
                index_label = i
                break
        
        label_name  = self.label_names[index_label] 
        slice_name = self.slice_names[index]
        
        slice_num = int(slice_name.split('.')[0][-3:])
        img = cv2.imread(os.path.join(self.img_path, slice_name))
        img = img[:,:,2]

        label = load_label(label_name, self.label_path, slice_num, img)
        
        if self.joint_transform is not None:
            img, label = self.joint_transform([img, label])

        if 'ToTensor' in self.transform:
            img = transform_ToTensor(img)
                        
        if 'normalize' in self.transform:
            img = transform_normalize(img, self.mean, self.std)

        label = torch.from_numpy(np.array(label, dtype=np.int32)).long()
        
        assert img.size() == label.size()        
        return img, label
    
    def __len__(self):
        return len(self.slice_names)
    

class lidc2d_assignCenter(data.Dataset):
    def __init__(self, root, split='train', crop_size = [10, 64, 64], transform = ['ToTensor', 'normalize'],
                joint_transform = None, split_files = 'train.npy', 
                 slice_split_files = 'train_slices.npy', slice_nodule_split_files = 'train_slices_nodule.npy',
                 augratio = 1, blacklist = blacklist):
        self.root = root
        assert split in ('train', 'val', 'test')
        self.split = split
        
        label_path = os.path.join(root, 'labels')
        img_path = os.path.join(root, 'slices')
        #label_names = os.listdir(os.path.join(label_path, 'mask'))
        label_names = np.load(split_files)
        slice_names = np.load(slice_split_files)
        slice_nodule_names = np.load(slice_nodule_split_files)
        for i in range(len(blacklist)):
            #posit = np.where(label_names == blacklist[i])[0]
            slice_names = np.delete(slice_names, np.where(label_names == blacklist[i]), axis = 0)
            slice_nodule_names = np.delete(slice_nodule_names, np.where(label_names == blacklist[i]), axis = 0)
            label_names = np.delete(label_names, np.where(label_names == blacklist[i]))
                   
        slice_names_flattened = []
        slice_nums = [0]
        for i in range(len(slice_names)):
            sublist = slice_names[i]
            numslices_sublist = 0
            for value in sublist:
                if value in slice_nodule_names[i]:
                    for p in range(augratio):
                        slice_names_flattened.append(value)
                    numslices_sublist += augratio
                else:
                    slice_names_flattened.append(value)
                    numslices_sublist += 1

            slice_nums.append(slice_nums[-1] + numslices_sublist)

        self.label_names = label_names
        self.slice_names = slice_names_flattened
        self.slice_nums = slice_nums
        self.label_path = label_path
        self.img_path = img_path
        self.crop_size = crop_size
        self.joint_transform = joint_transform
        self.transform = transform
        
        self.mean = MEAN
        self.std = STD
        
    def __getitem__(self, index):
        #print(len(self.label_names))
        slice_nums = self.slice_nums
        for i in range(len(slice_nums)):
            if (index >= slice_nums[i]) & (index < slice_nums[i+1]):
                index_label = i
                break
        
        label_name  = self.label_names[index_label] 
        slice_name = self.slice_names[index]
        
        slice_num = int(slice_name.split('.')[0][-3:])
        img = cv2.imread(os.path.join(self.img_path, slice_name))
        img = img[:,:,2]

        bbox = np.load(os.path.join(self.label_path, 'bbox', label_name))
        mask_pos = np.load(os.path.join(self.label_path, 'mask', label_name))

        mask_pos_rec = np.copy(mask_pos)

        for i in range(len(mask_pos)):
            mask_pos_rec[i] = mask_pos[i] + bbox[i,0]
            mask_pos_now = mask_pos_rec[:, mask_pos_rec[2,:]==slice_num]

        mask_pos_now = (mask_pos_now[0,:], mask_pos_now[1,:])
    
        label = np.zeros((img.shape[0], img.shape[1]))
        label[mask_pos_now] = 1
        
        #print(mask_pos_rec)
        crop_center = [np.mean(mask_pos_rec[0]), np.mean(mask_pos_rec[1])]
        #print(crop_center)
        
        img, label = JointAssignCenterCrop([img, label], crop_center, self.crop_size[1])
        
        if self.joint_transform is not None:
            img, label = self.joint_transform([img, label])

        if 'ToTensor' in self.transform:
            img = transform_ToTensor(img)
                        
        if 'normalize' in self.transform:
            img = transform_normalize(img, self.mean, self.std)

        label = torch.from_numpy(np.array(label, dtype=np.int32)).long()
        
        assert img.size() == label.size()        
        return img, label
    
    def __len__(self):
        return len(self.slice_names)