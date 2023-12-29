import torch 
import torch.nn as nn
from collections import OrderedDict

from models import network, network_domainnet, network_res50


def get_visda_networks(args):
    netF = network.ResBase(res_name=args.net).cuda()

    netB = network.feat_bootleneck(
        type=args.classifier,
        feature_dim=netF.in_features,
        bottleneck_dim=args.bottleneck,).cuda()

    netC = network.feat_classifier(
        type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    modelpath = f'weights/{args.dataset}/source/source_F.pt'
    netF.load_state_dict(torch.load(modelpath))
    modelpath = f'weights/{args.dataset}/source/source_B.pt'
    netB.load_state_dict(torch.load(modelpath))
    modelpath = f'weights/{args.dataset}/source/source_C.pt'
    netC.load_state_dict(torch.load(modelpath))

    netFE = nn.Sequential(netF, netB)

    return netFE, netC


def get_domainnet_networks(args):
    netFE = network_domainnet.Classifier(args).cuda()

    netC = network_res50.feat_classifier(
        type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    source = args.dshift.split('2')[0]
    modelpath = f'weights/{args.dataset}/{source}/source.pth.tar'
    model_dict = torch.load(modelpath)['state_dict']

    new_state_dict = OrderedDict()
    for k, v in model_dict.items():
        name = k[7:] # remove 'module.' of DataParallel/DistributedDataParallel
        new_state_dict[name] = v

    netFE.load_state_dict(new_state_dict)

    netC_dict = OrderedDict()
    for k, v in netC.state_dict().items():
        netC_dict[k] = new_state_dict[k]
    netC.load_state_dict(netC_dict)

    return netFE, netC


def get_officehome_networks(args):
    netFE = network_res50.ResNet_FE().cuda()

    netC = network_res50.feat_classifier(
        type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    source = args.dshift.split('2')[0]
    modelpath = f'weights/{args.dataset}/{source}/source_F.pt'
    netFE.load_state_dict(torch.load(modelpath))
    modelpath = f'weights/{args.dataset}/{source}/source_C.pt'
    netC.load_state_dict(torch.load(modelpath))

    return netFE, netC


net_dict = {'visda': get_visda_networks, 'domainnet126': get_domainnet_networks, 'officehome': get_officehome_networks}
def get_networks(args):
    return net_dict[args.dataset](args)