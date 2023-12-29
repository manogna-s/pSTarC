import torch
import torch.optim as optim
import torch.nn as nn


def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def generate_features(args, netC, num_features=100, num_epochs=50, feature_dim=256):
    """Generate pseudo source features
    Given classifier, optimizes a randomly initialized feature bank
    """
    netC.train()
    pseudo_features = torch.randn((args.class_num * num_features, feature_dim)).cuda()
    pseudo_features.requires_grad = True
    pseudo_features.requires_grad_(True)
    
    optim_feats  = optim.Adam ([pseudo_features], lr=0.01)

    for t in range(num_epochs):
        optim_feats.zero_grad()
        scores = nn.Softmax(dim=1)(netC(pseudo_features))
        loss_ent = torch.mean(Entropy(scores))
        
        loss_div = torch.sum(torch.mean(scores, 0) * torch.log(torch.mean(scores, 0) + 1e-6))
        loss = 0.5 * loss_ent + loss_div * 10
        
        loss.backward()
        optim_feats.step()

    return pseudo_features


def collect_all_params(model):
    """Collect all trainable parameters.
    Walk the model's modules and collect all parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        for np, p in m.named_parameters():
            if np in ['weight', 'bias'] and p.requires_grad:
                params.append(p)
                names.append(f"{nm}.{np}")
    return params, names


def collect_bn_params(model):
    """Collect the affine scale + shift parameters from batch norms.
    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm) or isinstance(m, nn.GroupNorm) or isinstance(m, nn.BatchNorm1d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names
    

def get_tta_optimizer(args, netFE, netC):
    print('Optimizing ')
    param_group = []
    if args.opt_bn:
        print('BN params')
        params, names = collect_bn_params(netFE)
        param_group += [params]
    elif args.opt_fe:
        print('All backbone params')
        params, names = collect_all_params(netFE)
        param_group += [params]
    if args.opt_cls:
        print('Classifier params\n')
        params, names = collect_all_params(netC)
        param_group += [params]
    
    param_group = [elem for sublist in param_group for elem in sublist]
    optimizer = optim.SGD(param_group, lr =args.tta_lr, momentum = 0.9, weight_decay = 0, nesterov = True)
    return optimizer


def tta(args, loader, netFE, netC):

    netC.train()

    pseudo_feats = generate_features(args, netC, num_features=20, num_epochs=50, feature_dim= 256)

    pseudo_scores = nn.Softmax(dim=1)(netC(pseudo_feats))
    pseudo_maxprobs, pseudo_label_bank = torch.max(pseudo_scores, dim=1)

    fea_bank = pseudo_feats.cpu()
    fea_bank = torch.nn.functional.normalize(fea_bank)
    score_bank = pseudo_scores
    label_bank = pseudo_label_bank.cpu()


    preds_dict ={'labels': torch.zeros(len(loader.dataset)), 'preds':torch.zeros(len(loader.dataset)), 'scores': torch.zeros(len(loader.dataset), args.class_num)}

    optimizer = get_tta_optimizer(args, netFE, netC)

    iter_test = iter(loader)
    
    for i in range(len(loader)):
        inputs, labels, indx = next(iter_test)
        image, weak, strong = inputs
        image, weak, strong = image.cuda(), weak.cuda(), strong.cuda()

        if weak.shape[0]>1:

            netFE.train()
            netC.train()

            features_w = netFE(weak)
            outputs = netC(features_w)
            softmax_out = nn.Softmax(dim=1)(outputs)
            max_prob, pseudo_label = torch.max(softmax_out, dim=1)            
            ent_batch = Entropy(softmax_out)
            ent_thresh = torch.mean(ent_batch)

            p_w = softmax_out
            
            features_s = netFE(strong)
            p_s = nn.Softmax(dim=1)(netC(features_s))
            loss_aug_attr = -torch.sum(p_w * p_s, dim=1)
            

            with torch.no_grad():
                output_f_norm = torch.nn.functional.normalize(features_w)
                output_f_ = output_f_norm.cpu().detach().clone()

                distance = output_f_ @ fea_bank.T

                distance_near, idx_near = torch.topk(distance, dim=-1, largest=True, k=args.K + 1)
                distance_near, idx_near = distance_near[:, 1:], idx_near[:, 1:] # batch x K
                score_near = score_bank[idx_near]  # batch x K x C

                score_near_cls = torch.zeros(weak.shape[0], args.K, args.class_num).cuda() 
                for cls_idx in range(args.class_num):
                    src_cls_feats = torch.nn.functional.normalize(fea_bank[label_bank==cls_idx])
                    src_cls_scores = score_bank[label_bank==cls_idx]
                    curr_cls_feats = output_f_[pseudo_label.cpu()==cls_idx]
                    n_curr_cls = curr_cls_feats.shape[0]
                    if n_curr_cls==0: continue

                    cls_dist = curr_cls_feats @ src_cls_feats.T
                    
                    
                    cls_dist_near, cls_idx_near = torch.topk(cls_dist, dim=-1, largest=True, k=args.K + 1)
                    cls_dist_near, cls_idx_near = cls_dist_near[:, 1:], cls_idx_near[:, 1:] 
                    cls_score_near = src_cls_scores[cls_idx_near]
                    score_near_cls[pseudo_label.cpu()==cls_idx] = cls_score_near
                score_near[ent_batch<ent_thresh] = score_near_cls[ent_batch<ent_thresh]
                score_near[ent_batch>ent_thresh] = (p_w[ent_batch>ent_thresh]).detach().clone().unsqueeze(1).expand(-1, args.K, -1)

            # nn
            softmax_out_un = softmax_out.unsqueeze(1).expand(-1, args.K, -1)  # batch x K x C

            loss_attr= -(softmax_out_un * score_near).sum(-1).sum(1)

            mask = torch.ones((features_w.shape[0], features_w.shape[0]))
            diag_num = torch.diag(mask)
            mask_diag = torch.diag_embed(diag_num)
            mask = mask - mask_diag
            copy = softmax_out.T  

            dot_neg = softmax_out @ copy  # batch x batch

            loss_disp = (dot_neg * mask.cuda()).sum(-1)  # batch

            loss_aug = torch.mean(loss_aug_attr)
            loss_attr = torch.mean(loss_attr)
            loss_disp = torch.mean(loss_disp)
            loss = loss_aug + loss_attr + loss_disp * args.lamda 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

            netFE.eval()
            netC.eval()
            outputs = netC(netFE(image))
            _ , preds = torch.max(outputs, dim=1)
            preds_dict['preds'][indx] = preds.detach().clone().cpu().float()
            preds_dict['labels'][indx] = labels.float()
            preds_dict['scores'][indx] = outputs.detach().clone().cpu().float()
            batch_acc = torch.sum(preds_dict['preds'][indx]==preds_dict['labels'][indx])/preds.shape[0]
    
    return netFE, netC, preds_dict