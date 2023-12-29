import os
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import warnings
import argparse
warnings.filterwarnings("ignore")

from data_utils import get_data_loaders, num_classes
from model_utils import get_networks

from methods import pstarc          
                        
tta_methods = {'pstarc': pstarc.tta}


def eval(loader, netFE, netC):
    netFE.eval()
    netC.eval()
    preds_dict ={'labels': torch.zeros(len(loader.dataset)), 'preds':torch.zeros(len(loader.dataset))}
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs, labels = data[0], data[1]
            inputs = inputs.cuda()
            indx = data[-1]

            output = netFE(inputs)
            outputs = netC(output)
            outputs = nn.Softmax(-1)(outputs)
            _ , preds = torch.max(outputs, dim=1)
            preds_dict['preds'][indx] = preds.detach().clone().cpu().float()
            preds_dict['labels'][indx] = labels.float()
    return preds_dict


def get_acc(preds_dict):
    acc_dict = {}
    acc_dict['total_acc'] = ((preds_dict['preds'] == preds_dict['labels']).float().sum()/preds_dict['preds'].shape[0]*100).item()

    matrix = confusion_matrix(preds_dict['labels'].float(), preds_dict['preds'].float())
    acc = matrix.diagonal() / matrix.sum(axis=1) * 100
    acc_dict['avg_acc'] = acc.mean()
    aa = [str(np.round(i, 2)) for i in acc]
    acc_dict['cls_acc'] = " ".join(aa)
    acc_dict['cls_acc'] = [np.round(i, 2) for i in acc]
    return acc_dict               


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LPA")
    parser.add_argument("--seed", type=int, default=2024, help="random seed")
    parser.add_argument("--worker", type=int, default=4, help="number of workers")
    parser.add_argument("--gpu_id", type=str, default="0", help="device id to run")
    parser.add_argument("--dataset", type=str, default="visda")
    parser.add_argument("--dshift", type=str, default="t2v")
    parser.add_argument("--tta_method", type=str, default="pstarc", help="which subset")
    parser.add_argument("--tta_bs", type=int, default=128, help="which subset")
    parser.add_argument("--tta_lr", type=float, default=5e-4, help="learning rate")

    parser.add_argument("--eval_mode", action='store_true')
    parser.add_argument('--opt_bn', action='store_true')  
    parser.add_argument('--opt_fe', action='store_true') 
    parser.add_argument('--opt_cls', action='store_true')  

    parser.add_argument("--lamda", type=float, default=0.5)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--thresh", type=float, default=0.5, help="which subset")

    parser.add_argument("--bottleneck", type=int, default=256)
    parser.add_argument("--layer", type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument("--classifier", type=str, default="bn", choices=["ori", "bn"])

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    args.class_num = num_classes[args.dataset]

    data_loader = get_data_loaders(args)
    netFE, netC = get_networks(args)

    print(f'Using pSTarC for TTA on dataset: {args.dataset} and domain shift: {args.dshift}')
    netFE, netC, preds_dict = tta_methods[args.tta_method](args, data_loader, netFE, netC)

    acc_dict = get_acc(preds_dict)
    print(f'\nFinal metrics:\n{acc_dict}')





