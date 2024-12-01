#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import math
import pdb
import copy
from torch.optim import Optimizer
from datetime import datetime


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

    
class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, pretrain=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.loss_func_per = nn.CrossEntropyLoss(reduction='none')
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True, num_workers=4, pin_memory=True)
        self.pretrain = pretrain

    def train(self, net, body_lr, head_lr, out_layer=-1, local_eps=None):
        net.train()
        # For ablation study
        """
        body_params = []
        head_params = []
        for name, p in net.named_parameters():
            if 'features.0' in name or 'features.1' in name: # active
                body_params.append(p)
            else: # deactive
                head_params.append(p)
        """
        body_params = [p for name, p in net.named_parameters() if 'linear' not in name]
        head_params = [p for name, p in net.named_parameters() if 'linear' in name]
        optimizer = torch.optim.SGD([{'params': body_params, 'lr': body_lr},
                                     {'params': head_params, 'lr': head_lr}],  
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)
        epoch_loss = []
        
        if local_eps is None:
            if self.pretrain:
                local_eps = self.args.local_ep_pretrain
            else:
                local_eps = self.args.local_ep
        for iter in range(local_eps):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                logits = net(images)
                loss = self.loss_func(logits, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
    # Multi-Heads Knowledge Distillation
    def train_MHKD(self, net, body_lr, head_lr, out_layer=-1, local_eps=None, Temperature=1, KD_weight = 1005):

        net.train()

        body_params = [p for name, p in net.named_parameters() if 'linear' not in name]
        head_params = [p for name, p in net.named_parameters() if 'linear' in name]
        optimizer = torch.optim.SGD([{'params': body_params, 'lr': body_lr},
                                     {'params': head_params, 'lr': head_lr}],
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)
        epoch_loss = []
        if local_eps is None:
            if self.pretrain:
                local_eps = self.args.local_ep_pretrain
            else:
                local_eps = self.args.local_ep
        for iter in range(local_eps):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                
                if KD_weight==1005:
                    logits_final, logits_aux = net(images, out_layer, False) 
                    y_pred = logits_final.data.max(1, keepdim=True)[1].view_as(labels) 
                    KD_Idx = y_pred.eq(labels.data)
                    loss_aux, loss_kd = 0., 0.
                    for l in range(len(logits_aux)):
                        loss_aux += self.loss_func(logits_aux[l], labels)
                    # dense distiilation
                    for l in range(len(logits_aux)):
                        for i in range(l+1,3):
                            loss_kd  += nn.KLDivLoss(log_target=True)(nn.functional.log_softmax(logits_aux[l][KD_Idx] / Temperature, dim=-1), nn.functional.log_softmax(logits_aux[i][KD_Idx] / Temperature, dim=-1)) * (Temperature**2)
                        loss_kd  += nn.KLDivLoss(log_target=True)(nn.functional.log_softmax(logits_aux[l][KD_Idx] / Temperature, dim=-1), nn.functional.log_softmax(logits_final[KD_Idx] / Temperature, dim=-1)) * (Temperature**2)
                                   
                    loss = self.loss_func(logits_final, labels) * 1 +  loss_aux*10 + loss_kd*1
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


    # knowledge_distillation
    def train_KD(self, trainedTutor, tutee, body_lr, head_lr, out_layer=-1, local_eps=None, Temperature=5.0, KD_weight = 10):
        trainedTutor.eval()  # tutor set to evaluation mode
        tutee.train()        # tutee set to train mode
        body_params = [p for name, p in tutee.named_parameters() if 'linear' not in name]
        head_params = [p for name, p in tutee.named_parameters() if 'linear' in name]
        paras_dict = [{'params': body_params, 'lr': body_lr},{'params': head_params, 'lr': head_lr}]
        optimizer = torch.optim.SGD(paras_dict, momentum=self.args.momentum, weight_decay=self.args.wd)
        epoch_loss = []
        if local_eps is None:
            if self.pretrain:
                local_eps = self.args.local_ep_pretrain
            else:
                local_eps = self.args.local_ep
        for iter in range(local_eps):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                # Forward pass with the tutor model - do not save gradients here as we do not change the tutor's weights
                with torch.no_grad():
                    tutor_logits = trainedTutor(images, -1)

                tutee_logits = tutee(images, out_layer)
                tutee_loss = self.ce_loss(tutee_logits, labels)

                # Weighted sum of the two losses
                Temperature = 2
                soft_targets = nn.functional.softmax(tutor_logits / Temperature, dim=-1)
                soft_prob = nn.functional.log_softmax(tutee_logits / Temperature, dim=-1)
                soft_targets_loss = -torch.sum(soft_targets * soft_prob) / soft_prob.size()[0] * (Temperature**2)
                loss = 0.25*soft_targets_loss + 0.75*tutee_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return tutee.state_dict(), sum(epoch_loss) / len(epoch_loss)


class LocalUpdatePerFedAvg(object):    
    def __init__(self, args, dataset=None, idxs=None, pretrain=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.pretrain = pretrain

    def train(self, net, lr, beta=0.001, momentum=0.9):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
        epoch_loss = []
        
        for local_ep in range(self.args.local_ep):
            batch_loss = []
            
            if len(self.ldr_train) / self.args.local_ep == 0:
                num_iter = int(len(self.ldr_train) / self.args.local_ep)
            else:
                num_iter = int(len(self.ldr_train) / self.args.local_ep) + 1
                
            train_loader_iter = iter(self.ldr_train)
            
            for batch_idx in range(num_iter):
                temp_net = copy.deepcopy(list(net.parameters()))
                    
                # Step 1
                for g in optimizer.param_groups:
                    g['lr'] = lr
                    
                try:
                    images, labels = next(train_loader_iter)
                except:
                    train_loader_iter = iter(self.ldr_train)
                    images, labels = next(train_loader_iter)
                    
                    
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                
                net.zero_grad()
                
                logits = net(images)

                loss = self.loss_func(logits, labels)
                loss.backward()
                optimizer.step()
                
                
                # Step 2
                for g in optimizer.param_groups:
                    g['lr'] = beta
                    
                try:
                    images, labels = next(train_loader_iter)
                except:
                    train_loader_iter = iter(self.ldr_train)
                    images, labels = next(train_loader_iter)
                    
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                    
                net.zero_grad()
                
                logits = net(images)

                loss = self.loss_func(logits, labels)
                loss.backward()
                
                # restore the model parameters to the one before first update
                for old_p, new_p in zip(net.parameters(), temp_net):
                    old_p.data = new_p.data.clone()
                    
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss) 
    
    def one_sgd_step(self, net, lr, beta=0.001, momentum=0.9):
        net.train()
        # train and update

        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
        
        test_loader_iter = iter(self.ldr_train)

        # Step 1
        for g in optimizer.param_groups:
            g['lr'] = lr

        try:
            images, labels = next(train_loader_iter)
        except:
            train_loader_iter = iter(self.ldr_train)
            images, labels = next(train_loader_iter)


        images, labels = images.to(self.args.device), labels.to(self.args.device)

        net.zero_grad()

        logits = net(images)

        loss = self.loss_func(logits, labels)
        loss.backward()
        optimizer.step()

        # Step 2
        for g in optimizer.param_groups:
            g['lr'] = beta

        try:
            images, labels = next(train_loader_iter)
        except:
            train_loader_iter = iter(self.ldr_train)
            images, labels = next(train_loader_iter)

        images, labels = images.to(self.args.device), labels.to(self.args.device)

        net.zero_grad()

        logits = net(images)

        loss = self.loss_func(logits, labels)
        loss.backward()

        optimizer.step()


        return net.state_dict()

class LocalUpdateFedRep(object):
    def __init__(self, args, dataset=None, idxs=None, pretrain=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.pretrain = pretrain

    def train(self, net, lr):
        net.train()

        # train and update
        body_params = [p for name, p in net.named_parameters() if 'linear' not in name]
        head_params = [p for name, p in net.named_parameters() if 'linear' in name]
        
        optimizer = torch.optim.SGD([{'params': body_params, 'lr': 0.0, 'name': "body"},
                                     {'params': head_params, 'lr': lr, "name": "head"}],
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)

        local_eps = self.args.local_ep
        
        for iter in range(local_eps):
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                logits = net(images)

                loss = self.loss_func(logits, labels)
                loss.backward()
                optimizer.step()
        
        for g in optimizer.param_groups:
            if g['name'] == "body":
                g['lr'] = lr
            elif g['name'] == 'head':
                g['lr'] = 0.0
        
        for batch_idx, (images, labels) in enumerate(self.ldr_train):
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            net.zero_grad()
            logits = net(images)

            loss = self.loss_func(logits, labels)
            loss.backward()
            optimizer.step()

        return net.state_dict()

class LocalUpdateFedProx(object):
    def __init__(self, args, dataset=None, idxs=None, pretrain=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.pretrain = pretrain

    def train(self, net, body_lr, head_lr):
        net.train()
        g_net = copy.deepcopy(net)
        
        body_params = [p for name, p in net.named_parameters() if 'linear' not in name]
        head_params = [p for name, p in net.named_parameters() if 'linear' in name]
        
        optimizer = torch.optim.SGD([{'params': body_params, 'lr': body_lr},
                                     {'params': head_params, 'lr': head_lr}],
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)

        epoch_loss = []
        
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                logits = net(images)

                loss = self.loss_func(logits, labels)
                
                # for fedprox
                fed_prox_reg = 0.0
                for l_param, g_param in zip(net.parameters(), g_net.parameters()):
                    fed_prox_reg += (self.args.mu / 2 * torch.norm((l_param - g_param)) ** 2)
                loss += fed_prox_reg
                
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss) 
    
class LocalUpdateDitto(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
            
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net, w_ditto=None, lam=0, idx=-1, lr=0.1, last=False, momentum=0.9):
        net.train()
        # train and update
        bias_p=[]
        weight_p=[]
        for name, p in net.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
                
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)

        local_eps = self.args.local_ep
        args = self.args 
        epoch_loss=[]
        num_updates = 0
        
        for iter in range(local_eps):
            done=False
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                w_0 = copy.deepcopy(net.state_dict())
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if w_ditto is not None:
                    w_net = copy.deepcopy(net.state_dict())
                    for key in w_net.keys():
                        w_net[key] = w_net[key] - args.lr*lam*(w_0[key] - w_ditto[key])
                    net.load_state_dict(w_net)
                    optimizer.zero_grad()
                
                num_updates += 1
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

'''
if any(KD_Idx):
    for l in range(len(logits_aux)):
        loss_kd  += nn.KLDivLoss(log_target=True)(nn.functional.log_softmax(logits_aux[l][KD_Idx] / Temporature, dim=-1), nn.functional.log_softmax(logits_final[KD_Idx] / Temporature, dim=-1)) * (Temporature**2)
        y_pred_reverse = logits_aux[l].data.max(1, keepdim=True)[1].view_as(labels) 
        KD_Idx_reverse = y_pred_reverse.eq(labels.data) 
        if any(KD_Idx_reverse):
            loss_kd  += nn.KLDivLoss(log_target=True)(nn.functional.log_softmax(logits_final[KD_Idx_reverse] / Temporature, dim=-1), nn.functional.log_softmax(logits_aux[l][KD_Idx_reverse] / Temporature, dim=-1)) * (Temporature**2) 
else:
    for l in range(len(logits_aux)):
        y_pred_reverse = logits_aux[l].data.max(1, keepdim=True)[1].view_as(labels)
        KD_Idx_reverse = y_pred_reverse.eq(labels.data)
        if any(KD_Idx_reverse):
            loss_kd  += nn.KLDivLoss(log_target=True)(nn.functional.log_softmax(logits_final[KD_Idx_reverse] / Temporature, dim=-1), nn.functional.log_softmax(logits_aux[l][KD_Idx_reverse] / Temporature, dim=-1)) * (Temporature**2) 
'''

#loss_kd  += nn.KLDivLoss(log_target=True)(nn.functional.log_softmax(logits_aux[2][KD_Idx] / Temporature, dim=-1), nn.functional.log_softmax(logits_final[KD_Idx] / Temporature, dim=-1)) * (Temporature**2)
#loss_kd  += nn.KLDivLoss(log_target=True)(nn.functional.log_softmax(logits_aux[1][KD_Idx] / Temporature, dim=-1), nn.functional.log_softmax(logits_aux[2][KD_Idx] / Temporature, dim=-1)) * (Temporature**2)
#loss_kd  += nn.KLDivLoss(log_target=True)(nn.functional.log_softmax(logits_aux[0][KD_Idx] / Temporature, dim=-1), nn.functional.log_softmax(logits_aux[1][KD_Idx] / Temporature, dim=-1)) * (Temporature**2)
                    
