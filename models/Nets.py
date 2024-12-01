#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

def conv3x3(in_channels, out_channels, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        nn.BatchNorm2d(out_channels, track_running_stats=True),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        in_channels = 3
        num_classes = args.num_classes
        
        hidden_size = 64

        self.conv0 = conv3x3(in_channels, hidden_size)
        self.conv1 = conv3x3(hidden_size, hidden_size)
        self.conv2 = conv3x3(hidden_size, hidden_size)
        self.conv3 = conv3x3(hidden_size, hidden_size)

        self.linear_0  = nn.Linear( hidden_size, num_classes+1, bias=False)
        self.linear_1  = nn.Linear( hidden_size, num_classes+1, bias=False)
        self.linear_2  = nn.Linear( hidden_size, num_classes+1, bias=False)
 
        self.linear_final = nn.Linear(hidden_size*2*2, num_classes)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x, out_layer=None, stop_scale=True):
        # features = self.features(x)

        # define network list ------------------------------------------------
        outs = []    
        features = self.conv0(x)
        outs.append(features)
        features = self.conv1(features)
        outs.append(features)
        features = self.conv2(features)
        outs.append(features)
        features = self.conv3(features)
        outs.append(features)

        # define FINAL logits of complete network
        outs_this = outs[-1]
        outs_this = outs_this.view(outs_this.size(0), -1)                       
        logits_final = self.linear_final(outs_this)

        # define AUXILIARY logits ------------------------------------------
        logit_list = []
        outs_0 = outs[0]
        outs_0 = F.avg_pool2d(outs_0, outs_0.shape[2])
        outs_0 = outs_0.view(outs_0.size(0), -1)
        logits_0 = self.linear_0(outs_0) 
        if stop_scale:
            logits_0 =  logits_0[:,:10]
        else:
            logits_0 =  logits_0[:,:10] / self.sigmoid(logits_0[:,-1])[:,None]
        logit_list.append(logits_0)

        outs_1 = outs[1]
        outs_1 = F.avg_pool2d(outs_1, outs_1.shape[2])
        outs_1 = outs_1.view(outs_1.size(0), -1)       
        logits_1 = self.linear_1(outs_1)
        if stop_scale:
            logits_1 =  logits_1[:,:10]
        else:
            logits_1 =  logits_1[:,:10] / self.sigmoid(logits_1[:,-1])[:,None]
        logit_list.append(logits_1)

        outs_2 = outs[2]
        outs_2 = F.avg_pool2d(outs_2, outs_2.shape[2])
        outs_2 = outs_2.view(outs_2.size(0), -1)   
        logits_2 = self.linear_2(outs_2) 
        if stop_scale:
            logits_2 =  logits_2[:,:10]
        else:
            logits_2 =  logits_2[:,:10] / self.sigmoid(logits_2[:,-1])[:,None]
        logit_list.append(logits_2)


        if out_layer == -1:
            return logits_final
        else:
            return logits_final, logit_list
    
    def extract_features(self, x):
        features = self.features(x)
        features = features.view((features.size(0), -1))
        return features
        
'''MobileNet in PyTorch.
See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes, track_running_stats=True) 
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, track_running_stats=True)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out

class MobileNetCifar(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]  # paras:  3,309,476

    def __init__(self, num_classes=10):
        super(MobileNetCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32, track_running_stats=True) 
        self.layers = self._make_layers(in_planes=32)
        self.linear_3  = nn.Linear( 256, num_classes+1, bias=False) 
        self.linear_6  = nn.Linear( 512, num_classes+1, bias=False) 
        self.linear_9  = nn.Linear( 512, num_classes+1, bias=False)
        self.linear_final = nn.Linear(1024, num_classes, bias=False) 
        self.sigmoid = nn.Sigmoid()

       
    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)
    
    def forward(self, x, out_layer=None, stop_scale=True):
        out = F.relu(self.bn1(self.conv1(x)))
        if out_layer==None:
            out_layer = -1

        # define network list ------------------------------------------------
        outs = []                              
        for l in range(len(self.layers)):
            out = self.layers[l](out)
            outs.append(out)    

        # define FINAL logits of complete network
        outs_this = outs[-1]
        outs_this = F.avg_pool2d(outs_this, outs_this.shape[2])
        outs_this = outs_this.view(outs_this.size(0), -1)                       
        logits_final = self.linear_final(outs_this)

        # define AUXILIARY logits ------------------------------------------
        logit_list = []
        outs_3 = outs[3]
        outs_3 = F.avg_pool2d(outs_3, outs_3.shape[2])
        outs_3 = outs_3.view(outs_3.size(0), -1)      
        logits_3 = self.linear_3(outs_3) 
        if stop_scale:
            logits_3 =  logits_3[:,:100]
        else:
            logits_3 =  logits_3[:,:100] / self.sigmoid(logits_3[:,-1])[:,None]
        logit_list.append(logits_3) 

        outs_6 = outs[6]
        outs_6 = F.avg_pool2d(outs_6, outs_6.shape[2])
        outs_6 = outs_6.view(outs_6.size(0), -1)        
        logits_6 = self.linear_6(outs_6)
        if stop_scale:
            logits_6 =  logits_6[:,:100]
        else:
            logits_6 =  logits_6[:,:100] / self.sigmoid(logits_6[:,-1])[:,None]
        logit_list.append(logits_6) 
 
        outs_9 = outs[9]
        outs_9 = F.avg_pool2d(outs_9, outs_9.shape[2])
        outs_9 = outs_9.view(outs_9.size(0), -1)            
        logits_9 = self.linear_9(outs_9)
        if stop_scale:
            logits_9 =  logits_9[:,:100]
        else:
            logits_9 =  logits_9[:,:100] / self.sigmoid(logits_9[:,-1])[:,None]    
        logit_list.append(logits_9)  

        if out_layer == -1:
            return logits_final
        else:
            return logits_final, logit_list
      
    def extract_features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        return out
    



class MobileNetCifarTiny(nn.Module):      
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2)]                                  # paras:  91,044
 
    def __init__(self, num_classes=10):
        super(MobileNetCifarTiny, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(32, track_running_stats=False)
        ''' 
        track_running_stats – a boolean value that when set to True , 
         this module tracks the running mean and variance, and when set to False , 
         this module does not track such statistics and always uses batch statistics 
         in both training and eval modes. Default: True
        '''
        self.bn1 = nn.BatchNorm2d(32, track_running_stats=True) # DJ
        self.layers = self._make_layers(in_planes=32)
        self.linear = nn.Linear(256, num_classes) 
    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        logits = self.linear(out)
        return logits
    
    def extract_features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        return out    