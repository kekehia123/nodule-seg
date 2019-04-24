# This version load the files in order (Line 137).
# Line 9 chenged.
# Line 145 changed.
# Line 129 changed. (Decide if the file ends with .png)
import cv2
import numpy as np
import os
import torch
import torch.utils.data as data
#from torchvision.datasets.folder import is_image_file, default_loader
from torchvision.datasets.folder import default_loader

import matplotlib.pyplot as plt

# ========== set the constants ==========
#classes = ['healthy', 'emphysema', 'ground_glass', 'fibrosis', 'micronodules', 'consolidation', 'reticulation', 'null', 'non_lung']
classes = ['healthy', 'emphysema', 'ground_glass', 'fibrosis', 'micronodules', 'consolidation', 'reticulation', 'null', 'non_lung']

classes_abbr = ['H', 'Em', 'GG', 'Fib', 'MN', 'Cons', 'Ret',]

class_weight = torch.FloatTensor(
    [0.04123975, 0.15628265, 0.07671421, 0.04550556, 0.02957947, 0.48473181, 0.16594656, 0, 1e-5]
)

colors = [
    [93, 172, 129], # healthy, r
    [181, 68, 52], # emphysema, r
    [165, 222, 228], # ground_glass, r
    [0, 137, 167], # fibrosis, g, 粉
    [180, 129, 187], # micronodules, g, 绿
    [255, 177, 27], # consolidation, b, 黄
    [0, 92, 175], # reticulation, b, 蓝
    [128, 128, 128], # null
    [0, 0, 0], # non_lung
]

MEAN = (0.5, 0.5, 0.5)
STD  = (1.0, 1.0, 1.0)
# ========== ========== ========== ==========

def modify_out(tensor, size=8):
    ret = np.zeros(tensor.shape)
    out = tensor.clone()
    for row in range(tensor.shape[0]):
        for col in range(tensor.shape[1]):
            row_min, row_max = max(0, row-size), min(tensor.shape[0], row+size)
            col_min, col_max = max(0, col-size), min(tensor.shape[1], col+size)
            num, index = 0, 8
            for i in range(0, 7):
                current = np.sum(out[row_min: row_max, col_min: col_max] == i)
                if current > num:
                    index = i
                    num = current
            ret[row, col] = index
    return ret

def mask_overlay(image, mask, color=(255, 255, 0)):
    """
    Helper function to visualize mask on the top of the lung
    """
    mask = np.dstack((mask, mask, mask)) * np.array(color)
    mask = mask.astype(np.uint8)
    weighted_sum = cv2.addWeighted(mask, 0.5, image, 0.5, 0.)
    img = image.copy()
    ind = mask[:, :, 1] > 0    
    img[ind] = weighted_sum[ind]    
    return img

def view_tensor(tensor, image=None, mask=None):
    # imshow the tensor
    img = tensor.clone().numpy()
    assert img.ndim in (2, 3)
    if img.ndim == 3:
        # input is rgb image
        img = img.transpose((1, 2, 0))
        img = img * STD + MEAN
        img = img[:, :, 0]
#         img[:, :, 0], img[:, :, 2] = img[:, :, 1].copy(), img[:, :, 1].copy()
        plt.imshow(img, cmap=plt.cm.bone)
    else:
        # input is label/pred        
        if mask is not None:
            img = modify_out(img)
            mask = mask.clone().numpy()
            img[mask == len(classes)- 1] = len(classes) - 1
        
        label = [classes[i] for i in range(0, len(classes)-2) if np.sum(img==i) > 0]
        print('Contained label:', label)
        
        if image is None:
            r, g, b = img.copy(), img.copy(), img.copy()
            for i in range(len(colors)):
                r[img == i] = colors[i][0]
                g[img == i] = colors[i][1]
                b[img == i] = colors[i][2]
            img = np.zeros((img.shape[0], img.shape[1], 3))
            img[:, :, 0], img[:, :, 1], img[:, :, 2] = r, g, b
            show_img = img
        else:
            image = image.clone().numpy().transpose((1, 2, 0))
            image = (image * STD + MEAN) * 255
            show_img = np.dstack((image[:, :, 0], image[:, :, 0], image[:, :, 0])).astype(np.uint8)
            for i in range(0, len(classes)-2):
                show_img = mask_overlay(show_img, (img==i).astype(np.uint8), color=colors[i])
#             for i in range(0, len(classes)-2):
#                 if i <= 2:
#                     r[img == i] = colors[i][0]
#                 elif i <= 4:
#                     g[img == i] = colors[i][1]
#                 else:
#                     b[img == i] = colors[i][2]
#             img = np.zeros((img.shape[0], img.shape[1], 3))
#             img[:, :, 0], img[:, :, 1], img[:, :, 2] = r, g, b
        
        plt.imshow(show_img, cmap=plt.cm.bone)
    
    #return img, label


class Dataset(data.Dataset):

    def __init__(self, root, split='train', joint_transform=None,
                 transform=None, loader=default_loader, no_ret = True):
        self.root = root
        assert split in ('train', 'val', 'test')
        self.split = split
        self.transform = transform
        self.joint_transform = joint_transform
        self.loader = loader
        self.classes = classes
        self.class_weight = class_weight        
        self.mean = MEAN
        self.std  = STD
        
        path = os.path.join(root, split)
        #self.imgs = [os.path.join(path, file) for file in os.listdir(path) if is_image_file(file)]
        self.imgs = [os.path.join(path, file) for file in os.listdir(path) if file[-4:] == '.png']
        self.imgs.sort()  # Make the files in order (to make 6 consecutive slices together)
        self.no_ret = no_ret

    def __getitem__(self, index):
        path = self.imgs[index]
        try:
            img = self.loader(path)
        except:
            print(path)
        label = self.loader(path.replace(self.split, self.split + '_mask')).convert('L')

        if self.joint_transform is not None:
            img, label = self.joint_transform([img, label])

        if self.transform is not None:
            img = self.transform(img)

        #label = torch.Tensor(np.array(label, dtype='uint8')).long()
        label = torch.from_numpy(np.array(label, dtype='uint8')).long()
        
        # Remove the class reticulation
        if self.no_ret:
            label[label == 6] = 7
        
        return img, label

    def __len__(self):
        return len(self.imgs)