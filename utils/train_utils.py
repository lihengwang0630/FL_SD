from torchvision import datasets, transforms
from models.Nets import CNNCifar, MobileNetCifar, MobileNetCifarTiny
from utils.sampling import iid, noniid, iid_unbalanced, noniid_unbalanced
from prettytable import PrettyTable 
import random                       
import math

trans_mnist = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])
trans_cifar10_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
trans_cifar10_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
trans_cifar100_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                               std=[0.267, 0.256, 0.276])])
trans_cifar100_val = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                              std=[0.267, 0.256, 0.276])])

def get_data(args, env='fed'):
    if env == 'single':
        if args.dataset == 'cifar10':
            dataset_train = datasets.CIFAR10('../data/cifar10', train=True, download=True, transform=trans_cifar10_train)
            dataset_test = datasets.CIFAR10('../data/cifar10', train=False, download=True, transform=trans_cifar10_val)
            
        elif args.dataset == 'cifar100':
            dataset_train = datasets.CIFAR100('../data/cifar100', train=True, download=True, transform=trans_cifar100_train)
            dataset_test = datasets.CIFAR100('../data/cifar100', train=False, download=True, transform=trans_cifar100_val)
        return dataset_train, dataset_test
    
    elif env == 'fed':
        if args.unbalanced:
            if args.dataset == 'cifar10':
                dataset_train = datasets.CIFAR10('../data/cifar10', train=True, download=True, transform=trans_cifar10_train)
                dataset_test = datasets.CIFAR10('../data/cifar10', train=False, download=True, transform=trans_cifar10_val)
                if args.iid:
                    dict_users_train = iid_unbalanced(dataset_train, args.num_users, args.num_batch_users, args.moved_data_size)
                    dict_users_test = iid_unbalanced(dataset_test, args.num_users, args.num_batch_users, args.moved_data_size)
                else:
                    dict_users_train, rand_set_all = noniid_unbalanced(dataset_train, args.num_users, args.num_batch_users, args.moved_data_size, args.shard_per_user)
                    dict_users_test, rand_set_all = noniid_unbalanced(dataset_test, args.num_users, args.num_batch_users, args.moved_data_size, args.shard_per_user, rand_set_all=rand_set_all)
            elif args.dataset == 'cifar100':
                dataset_train = datasets.CIFAR100('../data/cifar100', train=True, download=True, transform=trans_cifar100_train)
                dataset_test = datasets.CIFAR100('../data/cifar100', train=False, download=True, transform=trans_cifar100_val)
                if args.iid:
                    dict_users_train = iid_unbalanced(dataset_train, args.num_users, args.num_batch_users, args.moved_data_size)
                    dict_users_test = iid_unbalanced(dataset_test, args.num_users, args.num_batch_users, args.moved_data_size)
                else:
                    dict_users_train, rand_set_all = noniid_unbalanced(dataset_train, args.num_users, args.num_batch_users, args.moved_data_size, args.shard_per_user)
                    dict_users_test, rand_set_all = noniid_unbalanced(dataset_test, args.num_users, args.num_batch_users, args.moved_data_size, args.shard_per_user, rand_set_all=rand_set_all)
            else:
                exit('Error: unrecognized dataset')

        else:
            if args.dataset == 'mnist':
                dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
                dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
                # sample users
                if args.iid:
                    dict_users_train = iid(dataset_train, args.num_users, args.server_data_ratio)
                    dict_users_test = iid(dataset_test, args.num_users, args.server_data_ratio)
                else:
                    dict_users_train, rand_set_all = noniid(dataset_train, args.num_users, args.shard_per_user, args.server_data_ratio)
                    dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user, args.server_data_ratio, rand_set_all=rand_set_all)
            elif args.dataset == 'cifar10':
                dataset_train = datasets.CIFAR10('../data/cifar10', train=True, download=True, transform=trans_cifar10_train)
                dataset_test = datasets.CIFAR10('../data/cifar10', train=False, download=True, transform=trans_cifar10_val)
                if args.iid:
                    dict_users_train = iid(dataset_train, args.num_users, args.server_data_ratio)
                    dict_users_test = iid(dataset_test, args.num_users, args.server_data_ratio)
                else:
                    dict_users_train, rand_set_all = noniid(dataset_train, args.num_users, args.shard_per_user, args.server_data_ratio)
                    dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user, args.server_data_ratio, rand_set_all=rand_set_all)
            elif args.dataset == 'cifar100':
                dataset_train = datasets.CIFAR100('../data/cifar100', train=True, download=True, transform=trans_cifar100_train)
                dataset_test = datasets.CIFAR100('../data/cifar100', train=False, download=True, transform=trans_cifar100_val)
                if args.iid:
                    dict_users_train = iid(dataset_train, args.num_users, args.server_data_ratio)
                    dict_users_test = iid(dataset_test, args.num_users, args.server_data_ratio)
                else:
                    # breakpoint()
                    dict_users_train, rand_set_all = noniid(dataset_train, args.num_users, args.shard_per_user, args.server_data_ratio)
                    dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user, args.server_data_ratio, rand_set_all=rand_set_all)
            else:
                exit('Error: unrecognized dataset')

        return dataset_train, dataset_test, dict_users_train, dict_users_test

def get_model(args):
    # dj: add net_tutor, net_glob > net_tutee
    if args.model == 'cnn' and args.dataset in ['cifar10', 'cifar100']:
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'mobile' and args.dataset in ['cifar10', 'cifar100']:
        net_glob = MobileNetCifar(num_classes=args.num_classes).to(args.device)
    elif args.model == 'mobiletiny' and args.dataset in ['cifar10', 'cifar100']:
        net_glob = MobileNetCifarTiny(num_classes=args.num_classes).to(args.device)                      
    elif args.model == 'mobileKD' and args.dataset in ['cifar10', 'cifar100']:
        net_tutor = MobileNetCifar(num_classes=args.num_classes).to(args.device)
        net_tutee = MobileNetCifarTiny(num_classes=args.num_classes).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp' and args.dataset == 'mnist':
        net_glob = MLP(dim_in=784, dim_hidden=256, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    if args.model == 'mobileKD' or args.model == 'resnet18KD':
        return net_tutor, net_tutee
    else:
        return net_glob

def get_layer_list(model):
    layer_list = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        layer_list.append(name)
    return layer_list


def count_model_parameters(model): 
    table = PrettyTable(["Modules", "Parameters", "AccumParas"])
    total_params = 0
    accum_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        accum_params+=params
        table.add_row([name, params, accum_params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def count_layer_parameters(model, layers):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        if name in layers:
            params = parameter.numel()
            total_params+=params
    return total_params