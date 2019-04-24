import itertools
import numpy as np
import os
import pandas as pd
import shutil
from sklearn.metrics import confusion_matrix
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import io_utils_lidc3d
import matplotlib.pyplot as plt


class Metrics(object):
    
    def __init__(self):
        self.num_classes = len(io_utils_lidc3d.classes)
        
    def iou(self, pred, targets):
        try:
            assert pred.size() == targets.size()
        except:
            print(pred.size(), targets.size())
            assert pred.size() == targets.size()
        
        pred, targets = pred.numpy(), targets.numpy()
        #print('uniques of pred:', np.unique(pred), 'uniques of targets:', np.unique(targets))
        I = np.sum((pred == 1) & (targets == 1))
        U = np.sum((pred == 1) | (targets == 1))
        return np.array([I, U])

    def dice(self, pred, targets):
        assert pred.size() == targets.size()
        pred, targets = pred.numpy(), targets.numpy()
        nomi = 2 * np.sum((pred == 1) & (targets == 1))
        denomi = np.sum(pred == 1) + np.sum(targets == 1)
        return np.array([nomi, denomi])

    #def ASD(prd, targets):

    def sen(self, pred, targets):
        assert pred.size() == targets.size()
        pred, targets = pred.numpy(), targets.numpy()
        nomi = np.sum((pred == 1) & (targets == 1))
        denomi = np.sum(targets == 1)
        return np.array([nomi, denomi])

    def ppv(self, pred, targets):
        assert pred.size() == targets.size()
        pred, targets = pred.numpy(), targets.numpy()
        nomi = np.sum((pred == 1) & (targets == 1))
        denomi = np.sum(pred == 1)
        return np.array([nomi, denomi])


class Experiment(object):
    
    def __init__(self, model, criterion, optimizer, checkpoint_path,
                 epoch=0, experiment_name=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.epoch = epoch
        self.checkpoint_path = checkpoint_path
        self.experiment_name = experiment_name
        
        self.num_classes = len(io_utils_lidc3d.classes)
        self.metrics = Metrics()
        self.train_from_scratch = True
        self.record={
            'train': {'loss': [], 'iou': [], 'dice': [], 'sen': [], 'ppv': []},
            'val': {'loss': [], 'iou': [], 'dice': [], 'sen': [], 'ppv': []},
            'test': {'loss': [], 'iou': [], 'dice': [], 'sen': [], 'ppv': []},
        }
        self.iou_best = 0
        self.idx = 0
    
    def save_weights(self, loss, iou, dice, sen, ppv):
        # save current model state and best model
        if dice == None:
            dice = self.dice
        checkpoint_name = self.experiment_name+'_checkpoint.pth.tar' if self.experiment_name is not None else 'checkpoint.pth.tar'
        checkpoint_name = os.path.join(self.checkpoint_path, checkpoint_name)
        torch.save({
            'Epoch': self.epoch,
            'loss': loss, 'iou': iou, 'dice':dice, 'sen': sen, 'ppv': ppv,
            'iou_best': self.iou_best,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, checkpoint_name)
        if self.current_is_best:
            best_model_name = self.experiment_name+'_best_model.pth.tar' if self.experiment_name is not None else 'best_model.pth.tar'
            best_model_name = os.path.join(self.checkpoint_path, best_model_name)
            shutil.copyfile(checkpoint_name, best_model_name)
            self.best_model_path = best_model_name
        
    def load_weights(self, path=None, load_opt=False):
        # load the existed model
        
        if path == None:
            path = self.best_model_path
        if not os.path.isfile(path):
            print('=> no checkpoint found at "{}"'.format(path))
            return
        
        checkpoint = torch.load(path)        
        self.model.load_state_dict(checkpoint['state_dict'])
        
        if load_opt:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epoch = checkpoint['Epoch']
            self.loss, self.iou, self.dice = checkpoint['loss'], checkpoint['iou'], checkpoint['dice']
            self.sen, self.ppv = checkpoint['sen'], checkpoint['ppv']
            self.iou_best = checkpoint['iou_best']
        print("=> loaded checkpoint '{}' (Epoch {})".format(path, checkpoint['Epoch']), 'IoU:', checkpoint['iou'])
        
        self.train_from_scratch = False
        
    def output_record(self, loss, iou, dice, sen, ppv, mode):
        # output current train/val loss, bacc, iou, etc. to the txt file
        assert mode in ('train', 'val', 'test')
        file = os.path.join(self.checkpoint_path, self.experiment_name+'_checkpoint.txt')
        if os.path.isfile(file):
            file = open(file, 'a')
        else:
            file = open(file, 'w')
        
        if mode == 'train':
            print('Epoch {:d}, idx {:d}'.format(self.epoch, self.idx), '==> time:', 
                  time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), file=file)
            print('Train - Loss: {:.4f}, IoU: {:.4f}, Dice: {:.4f}, Sen: {:.4f}, PPV: {:.4f}'.format(loss, iou, dice, sen, ppv), file=file)
        elif mode == 'val':
            print('Val -Loss: {:.4f}, IoU: {:.4f}, Dice: {:.4f}, Sen: {:.4f}, PPV: {:.4f}'.format(loss, iou, dice, sen, ppv), file=file)
        
        '''
        time_elapsed = time.time() - self.since
        if mode == 'train':
            print('Train Time {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60), file=file)
        elif mode =='val':
            print('Total Time {:.0f}m {:.0f}s\n'.format(
                time_elapsed // 60, time_elapsed % 60), file=file)
        '''
        file.close()
    
    def train(self, loader, output=False):
        self.epoch, self.idx = self.epoch+1, self.idx+1
        self.model.train()
        loss_train, count = 0, 0
        iou_train, dice_train, sen_train, ppv_train = np.zeros((self.num_classes-1, 2)), np.zeros((self.num_classes-1, 2)), \
                                       np.zeros((self.num_classes-1, 2)), np.zeros((self.num_classes-1, 2))
        #print(iou_train)
        self.since = time.time()
        
        for i, data in enumerate(loader):
            # load the data
            imgs = data[0].to(torch.device('cuda'))
            labels = data[1].to(torch.device('cuda'))
            
            # calculate the loss and optimize
            self.optimizer.zero_grad()
            imgs = imgs.float().unsqueeze(1).to(torch.device('cuda'))

            out = self.model(imgs)
            #print('train:', out.size())
            loss = self.criterion(out, labels)
            loss.backward()
            self.optimizer.step()
            
            # update the evaluation
            count += data[0].size(0)
            loss_train += loss.item() * data[0].size(0)
            
            #labels = torch.squeeze(labels)
            
            pred = self.get_predictions(out)
            
            temp = self.metrics.iou(pred, labels.detach().cpu())
            iou_train += self.metrics.iou(pred, labels.detach().cpu())
            dice_train += self.metrics.dice(pred, labels.detach().cpu())
            sen_train += self.metrics.sen(pred, labels.detach().cpu())
            ppv_train += self.metrics.ppv(pred, labels.detach().cpu())
            #print(temp)
            #print(iou_train)
        
        loss_train /= count
        self.record['train']['loss'].append(loss_train)
        
        iou_train = iou_train[:,0] / iou_train[:,1]
        iou_avg = np.sum(iou_train) / len(iou_train)        
        self.record['train']['iou'].append(iou_avg)
        dice_train = dice_train[:,0] / dice_train[:,1]
        dice_avg = np.sum(dice_train) / len(dice_train)
        self.record['train']['dice'].append(dice_avg)
        sen_train = sen_train[:,0] / sen_train[:,1]
        sen_avg = np.sum(sen_train) / len(sen_train)
        self.record['train']['sen'].append(sen_avg)
        ppv_train = ppv_train[:,0] / ppv_train[:,1]
        ppv_avg = np.sum(ppv_train) / len(ppv_train)
        self.record['train']['ppv'].append(ppv_avg)
        
        self.output_record(loss_train, iou_avg, dice_avg, sen_avg, ppv_avg, mode='train')
        
        if output:
            print('Epoch: 【{}】 idx: 【{}】 | Train - Loss: {:.4f}'.format(self.epoch, self.idx, loss_train))
            print('Train - IoU:', np.round(iou_avg, 4), ';',np.round(iou_train, 4))
            print('Train - Dice:', np.round(dice_avg, 4), ';',np.round(dice_train, 4))
            print('Train - Sen:', np.round(sen_avg, 4), ';', np.round(sen_train, 4))
            print('Train - PPV:', np.round(ppv_avg, 4), ';', np.round(ppv_train, 4))
        
        return loss_train, iou_avg, dice_avg, sen_avg, ppv_avg 
    
    def val(self, loader, update=True, output = False):
        self.model.eval()
        loss_val, count = 0, 0
        iou_val, dice_val, sen_val, ppv_val = np.zeros((self.num_classes-1, 2)), np.zeros((self.num_classes-1, 2)), \
                                 np.zeros((self.num_classes-1, 2)), np.zeros((self.num_classes-1, 2))
        self.current_is_best = False
        
        with torch.no_grad():
            for i, data in enumerate(loader):
                # load the data
                imgs = data[0].to(torch.device('cuda'))
                labels = data[1].to(torch.device('cuda'))

                # calculate the loss
                imgs = imgs.float().unsqueeze(1).to(torch.device('cuda'))
                
                out = self.model(imgs)
                #print('val:', out.size())
#                 out = self.model(imgs)
                loss = self.criterion(out, labels)

                # update the evaluation
                count += data[0].size(0)
                loss_val += loss.item() * data[0].size(0)
                #labels = torch.squeeze(labels, dim=1)
                
                pred = self.get_predictions(out)
                #print(pred.size(), labels.detach().cpu().size())
                
                iou_val  += self.metrics.iou(pred, labels.detach().cpu())
                dice_val += self.metrics.dice(pred, labels.detach().cpu())
                sen_val += self.metrics.sen(pred, labels.detach().cpu())
                ppv_val += self.metrics.ppv(pred, labels.detach().cpu())
        
        # adjust the record
        loss_val /= count
        iou_val = iou_val[:, 0] / iou_val[:, 1]
        iou_avg = np.sum(iou_val) / len(iou_val)
        dice_val = dice_val[:, 0] / dice_val[:, 1]
        dice_avg = np.sum(dice_val) / len(dice_val)
        sen_val = sen_val[:, 0] / sen_val[:, 1]
        sen_avg = np.sum(sen_val) / len(sen_val)
        ppv_val = ppv_val[:, 0] / ppv_val[:, 1]
        ppv_avg = np.sum(ppv_val) / len(ppv_val)
        
        if update:
            self.record['val']['loss'].append(loss_val)            
            self.record['val']['iou'].append(iou_avg)
            self.record['val']['dice'].append(dice_avg)
            self.record['val']['sen'].append(sen_avg)    
            self.record['val']['ppv'].append(ppv_avg) 
            
            self.output_record(loss_val, iou_avg, dice_avg, sen_avg, ppv_avg, mode='val')
        
        if output:
            print('Val- Loss: {:.4f}'.format(loss_val))
            print('Val - IoU:', np.round(iou_avg, 4), ';', np.round(iou_val, 4))
            print('Val - Dice:', np.round(dice_avg, 4), ';', np.round(dice_val, 4))
            print('Val - Sen:', np.round(sen_avg, 4), ';', np.round(sen_val, 4))
            print('Val - PPV:', np.round(ppv_avg, 4), ';', np.round(ppv_val, 4), '\n')
            
        if iou_avg > self.iou_best*1.005:
            # assist to save the model
            self.current_is_best = True
            self.iou_best = iou_avg
            self.idx = 0
        
        self.loss = loss_val
        self.iou  = iou_avg
        self.dice = dice_avg
        self.sen = sen_avg
        self.ppv = ppv_avg
        
        return loss_val, iou_avg, dice_avg, sen_avg, ppv_avg
    
    
    def inference_time(self, loader):
        self.model.eval()
        
        num = 0
        with torch.no_grad():
            since = time.time()
            for i, data in enumerate(loader):
                # load the data
                imgs = data[0].to(torch.device('cuda'))
                labels = data[1].to(torch.device('cuda'))

                # calculate the loss
                out = self.model(imgs)
#                 out = self.model(imgs)

                # update the evaluation
                pred = self.get_predictions(out)
                num += data[0].size(0)
        print('Number of data:', num)
        print(np.round(1000*(time.time() - since) / num, 4))
    
    def get_predictions(self, tensor):
        # input is the output of the model corresponding to the input image.
        # use softmax to compute the biggest expection
#         tensor = F.softmax(tensor, dim=1)
        tensor = tensor.detach().cpu()
        _, indices = tensor.max(dim=1)
        
        return indices
    
    def adjust_learning_rate(self, lr, decay, interval):
        # adjust the learning rate according to the current epoch and the interval to decay
        new_lr = lr * (decay ** (self.epoch // interval))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
    
    def save_csv(self):
        # save the train and val loss, bacc, iou, etc. to the csv file
        file_name = self.experiment_name+'_checkpoint.csv' if self.experiment_name is not None else 'checkpoint.csv'
        file_path = os.path.join(self.checkpoint_path, file_name)
        
        file = pd.DataFrame()
        file['Epoch'] = list(range(self.epoch))
        file['Loss_train'] = self.record['train']['loss']
        file['Loss_val'] = self.record['val']['loss']
        file['IoU_train']  = self.record['train']['iou']
        file['IoU_val']  = self.record['val']['iou']
        file['Dice_train']  = self.record['train']['dice']
        file['Dice_val']  = self.record['val']['dice']
        file['Sen_train'] = self.record['train']['sen']
        file['Sen_val'] = self.record['val']['sen']
        file['PPV_train'] = self.record['train']['ppv']
        file['PPV_val'] = self.record['val']['ppv']
        
        file.to_csv(file_path, index=False)
    
    def modify_out(self, tensor, size=4):
        ret = torch.zeros(tensor.size())
        out = tensor.clone().numpy()
        for row in range(tensor.size(0)):
            for col in range(tensor.size(1)):
                row_min, row_max = max(0, row-size), min(tensor.size(0), row+size)
                col_min, col_max = max(0, col-size), min(tensor.size(1), col+size)
                values = np.unique(out[row_min: row_max, col_min: col_max])
                num = 0
                for i in values:
                    current = np.sum(out[row_min: row_max, col_min: col_max] == i)
                    if current > num:
                        index = int(i)
                        num = current
                ret[row, col] = index
        return ret
    
    def view_sample(self, loader, num=1):
        # load the data        
        with torch.no_grad():
            for idx, data in enumerate(loader):
                
                if idx not in (34, 69, 92):
                    continue
                print(idx)
                
                imgs = data[0].to(torch.device('cuda'))
                labels = data[1].to(torch.device('cpu'))

#                 num = min(num, data[0].size(0))
                self.model.eval()
                since = time.time()
                out = self.model(imgs)
                pred = self.get_predictions(out)
                print('Use time:', np.round((time.time() - since) * 1000, 3), 'ms')

                plt.figure(figsize=(16,16))
                plt.subplot(1, 3, 1)
                plt.title('Input image')
                plt.axis('off')
                io_utils.view_tensor(data[0][0])
                plt.subplot(1, 3, 2)
                plt.title('Ground truth')
                plt.axis('off')
                io_utils.view_tensor(labels[0], image=data[0][0])
                plt.subplot(1, 3, 3)
                plt.title('Prediction')
                plt.axis('off')
                io_utils.view_tensor(pred[0], image=data[0][0], mask=labels[0])
                plt.show()
                
                if idx == num:
     
                    break
        
        #return pred[0], data[0][0], labels[0]