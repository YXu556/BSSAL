import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from collections import defaultdict, Counter
from joblib import dump
from sklearn.ensemble import RandomForestClassifier

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import WeightedRandomSampler

from models import *
from dataset import *

SITES = ['Garfield', 'Randolph', 'Adams', 'Coahoma', 'Harvey', 'Haskell']
SITES_NO = dict(zip(SITES, np.arange(6)))
priors = {
    'prior_mu': 0,
    'prior_sigma': 0.1,
    'posterior_mu_initial': (0, 0.1),  # (mean, std) normal_
    'posterior_rho_initial': (-5, 0.1),  # (mean, std) normal_
}


def train_sup_epoch(model, optimizer, criterion, dataloader, device, args):
    losses = AverageMeter('Loss', ':.4e')

    model.train()
    for idx, (X, y) in enumerate(dataloader):

        optimizer.zero_grad()

        X = X.float().to(device)
        y = y.long().to(device)

        if 'BBB' in model._get_name():
            outputs = torch.zeros(X.shape[0], args.nclasses, args.num_ens).to(device)
            kl = 0.0
            for i in range(args.num_ens):
                net_out, _kl = model(X)
                kl += _kl
                outputs[:, :, i] = F.log_softmax(net_out, dim=1)
            kl = kl / args.num_ens
            Lx, Lw = criterion(outputs.squeeze(), y, kl)
            loss = Lx + Lw  # todo
        else:
            outputs = model(X)
            loss = criterion(outputs, y)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), X.size(0))  # todo

    return losses.avg


def train_semi_epoch(model, epoch, steps_per_epoch, optimizer, ema_optimizer, criterion, labeled_iter, unlabeled_iter, device, args):
    losses = AverageMeter('Loss', ':.4e')
    losses_w = AverageMeter('Prior loss', ':.4e')
    losses_x = AverageMeter('Sup loss', ':.4e')
    losses_u = AverageMeter('Unsup loss', ':.4e')
    pseudo_labels = list()
    real_labels = list()

    # progress_bar = tqdm(range(args.steps_per_epoch), desc=f'{args.method.upper()} Epoch {epoch + 1}/{args.epochs}')
    model.train()
    # for batch_idx in progress_bar:
    for batch_idx in range(args.steps_per_epoch):
        sample_x, sample_u = next(labeled_iter), next(unlabeled_iter)
        inputs_x, targets_x = sample_x
        inputs_x = inputs_x.float().to(device)
        targets_x = targets_x.long().to(device)
        inputs_u, _targets_u = sample_u
        inputs_u = inputs_u.float().to(device)

        size_x = inputs_x.size(0)
        size_u = inputs_u.size(0)

        if size_u != size_x:
            idx_u = torch.randperm(size_u)[:size_x]
            inputs_u = inputs_u[idx_u]
            _targets_u = _targets_u[idx_u]

        # Transform label to one-hot
        targets_x = torch.zeros(size_x, args.nclasses, device=device).scatter_(1, targets_x.view(-1, 1).long(), 1)

        with torch.no_grad():
            # compute guessed labels of Funlabel samples
            outputs_u, _ = model(inputs_u)
            outputs_u2, _ = model(inputs_u)
            p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
            pt = p ** (1 / args.T)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

        if args.pseudo_threshold is not None:
            pseudo_conf, _ = torch.max(targets_u, dim=1)
            pseudo_mask = pseudo_conf > args.pseudo_threshold
            inputs_u = inputs_u[pseudo_mask]
            targets_u = targets_u[pseudo_mask]
            _targets_u = _targets_u[pseudo_mask]
            size_u = inputs_u.size(0)

        pseudo_labels.append(targets_u.argmax(-1).cpu())
        real_labels.append(_targets_u)

        # aug u for training
        transform_train = torchvision.transforms.Compose([
            RandomTempShift(),
            ToTensor(args.in_ch),
        ])
        aug_u = []
        if size_u > 0:
            for input_u in inputs_u:
                aug_u.append(transform_train(input_u.cpu().numpy()))
            inputs_u = torch.vstack(aug_u).to(device)

        # mixup
        all_inputs = torch.cat([inputs_x, inputs_u], dim=0)
        all_targets = torch.cat([targets_x, targets_u], dim=0)

        l = np.random.beta(args.alpha, args.alpha)
        l = max(l, 1 - l)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        if args.pseudo_threshold is not None:
            weight_x = torch.ones(size_x).to(args.device)
            weight_u = pseudo_conf[pseudo_mask]#torch.ones(size_u).to(args.device)#
            all_weights = torch.cat([weight_x, weight_u], dim=0)
            weight_a, weight_b = all_weights, all_weights[idx]
            mixed_weight = l * weight_a + (1 - l) * weight_b
            weight_x = mixed_weight[:size_x]
            weight_u = mixed_weight[size_x:]

        # interleave labeled and unlabed samples between batches to get correct batchnorm calculation
        mixed_input = list(torch.split(mixed_input, size_x))
        mixed_input = interleave(mixed_input, size_x)

        net_out, _kl = model(mixed_input[0])
        logits = [net_out]
        kl = _kl
        for input in mixed_input[1:]:
            net_out, _kl = model(input)
            logits.append(net_out)
            kl += _kl
        kl = kl / len(mixed_input)

        # put interleaved samples back
        logits = interleave(logits, size_x)
        logits_x = logits[0]
        if len(logits) > 1:
            logits_u = torch.cat(logits[1:], dim=0)
        else:
            logits_u = None

        if args.pseudo_threshold is not None:
            Lx, Lu, Lw, w = criterion(logits_x, mixed_target[:size_x], logits_u, mixed_target[size_x:], kl,
                                      epoch + batch_idx / steps_per_epoch, weight_x, weight_u)
        else:
            Lx, Lu, Lw, w = criterion(logits_x, mixed_target[:size_x], logits_u, mixed_target[size_x:], kl,
                                      epoch + batch_idx / steps_per_epoch)
        # print(Lx.item(), Lu.item(), (real_labels[-1]==pseudo_labels[-1]).sum()/real_labels[-1].shape[0])
        loss = Lw + Lx + w * Lu

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

        # record loss
        losses.update(loss.item(), inputs_x.size(0))
        losses_w.update(Lw.item(), inputs_x.size(0))
        losses_x.update(Lx.item(), inputs_x.size(0))
        losses_u.update(Lu.item(), inputs_x.size(0))

    pseudo_labels = torch.concat(pseudo_labels)
    real_labels = torch.concat(real_labels)

    return losses.avg, pseudo_labels, real_labels


def interleave(xy, batch):
    if xy[-1].shape[0] != batch:
        res = xy[-1]
        xy = xy[:-1]
    else:
        res = None
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    if res is not None:
        return [torch.cat(v, dim=0) for v in xy] + [res]
    else:
        return [torch.cat(v, dim=0) for v in xy]


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


class WeightEMA(object):
    def __init__(self, model, ema_model, lr=0.001, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype==torch.float32:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)
                # customized weight decay
                # param.mul_(1 - self.wd)


def get_pool_index(select_type, model, dataset, step, args, D2=None):
    X = torch.tensor(dataset.x[:, :args.in_ch]).float().to(args.device)
    if select_type == 2:  # Bayesian BvSB
        with torch.no_grad():
            classes_All = list()
            for d in range(args.dropout_iterations):
                logits, _ = model(X)
                classes_All.append(logits.argmax(-1).cpu().numpy())
        classes_All = np.array(classes_All).T
        class_prob = np.apply_along_axis(lambda x: np.bincount(x, minlength=args.nclasses), 1,
                                         classes_All) / args.dropout_iterations
        class_prob.sort()
        uncertainty = 1 - (class_prob[:, -1] - class_prob[:, -2])
        thre = np.sort(uncertainty)[-step]
        # 1
        pooled_index = np.random.choice(np.where(uncertainty >= thre)[0], step, replace=False)
        # 2
        # pi1 = np.where(uncertainty > thre)[0]
        # pi2 = np.random.choice(np.where(uncertainty==thre)[0], step-pi1.size, replace=False)
        # pooled_index = np.concatenate([pi1, pi2])
        # 3
        # pooled_index = uncertainty.argsort()[-step:][::-1]
    elif select_type == 3:  # BvSB
        with torch.no_grad():
            if 'BBB' in model._get_name():
                score_All = np.zeros((X.shape[0], args.nclasses))
                for d in range(args.dropout_iterations):
                    logits, _ = model(X)
                    score_All += F.softmax(logits, dim=1).cpu().numpy()
                y_score = score_All / args.dropout_iterations
            else:
                logits = model(X)
                y_score = F.softmax(logits, dim=1).cpu().numpy()
        y_score.sort()
        uncertainty = 1 - (y_score[:, -1] - y_score[:, -2])
        pooled_index = uncertainty.argsort()[-step:][::-1]
    elif select_type == 4:  # entropy
        # sub_idx = np.random.choice(np.arange(X.shape[0]), 1000, replace=False)
        # X_sub = X[sub_idx]
        with torch.no_grad():
            if 'BBB' in model._get_name():
                logits, _ = model(X, sample=False)
            else:
                logits = model(X)
        y_score = F.softmax(logits, dim=1).cpu().numpy()
        uncertainty = -np.sum(y_score * np.log(y_score), axis=1)
        pooled_index = uncertainty.argsort()[-step:][::-1]
    elif select_type == 5:

        with torch.no_grad():
            classes_All = list()
            for d in range(args.dropout_iterations):
                logits, _ = model(X)
                classes_All.append(logits.argmax(-1).cpu().numpy())
        classes_All = np.array(classes_All).T
        class_prob = np.apply_along_axis(lambda x: np.bincount(x, minlength=args.nclasses), 1,
                                         classes_All) / args.dropout_iterations
        uncertainty = -np.sum(class_prob * np.log(class_prob+1e-6), axis=1)

        # similarity
        pseudo_class = class_prob.argmax(-1)
        similarity = np.zeros(X.shape[0])
        ndvi = dataset.ndvi
        D2_unlabeled = (ndvi > (ndvi.max(-1)/2).reshape(-1, 1)).argmax(1)
        for c in range(6):
            d2 = D2[c]
            if d2 is None:
                continue
            sim = 1 - abs(D2_unlabeled[pseudo_class == c] - d2) / 19
            similarity[pseudo_class == c] = sim
            pooled_index = (uncertainty-similarity).argsort()[-step:][::-1]
    elif select_type == 6:
        with torch.no_grad():
            classes_All = list()
            for d in range(args.dropout_iterations):
                logits, _ = model(X)
                classes_All.append(logits.argmax(-1).cpu().numpy())
        classes_All = np.array(classes_All).T
        class_prob = np.apply_along_axis(lambda x: np.bincount(x, minlength=args.nclasses), 1,
                                         classes_All) / args.dropout_iterations
        uncertainty = -np.sum(class_prob * np.log(class_prob+1e-6), axis=1)
        pooled_index = uncertainty.argsort()[-step:][::-1]
    elif select_type == 7:

        with torch.no_grad():
            classes_All = list()
            for d in range(args.dropout_iterations):
                logits, _ = model(X)
                classes_All.append(logits.argmax(-1).cpu().numpy())
        classes_All = np.array(classes_All).T
        class_prob = np.apply_along_axis(lambda x: np.bincount(x, minlength=args.nclasses), 1,
                                         classes_All) / args.dropout_iterations
        pseudo_class = class_prob.argmax(-1)
        class_prob.sort()
        uncertainty = 1 - (class_prob[:, -1] - class_prob[:, -2])

        # similarity
        similarity = np.zeros(X.shape[0])
        ndvi = dataset.ndvi
        D2_unlabeled = (ndvi > (ndvi.max(-1)/2).reshape(-1, 1)).argmax(1)
        for c in range(6):
            d2 = D2[c]
            if d2 is None:
                continue
            sim = 1 - abs(D2_unlabeled[pseudo_class == c] - d2) / 19
            similarity[pseudo_class == c] = sim
        pooled_index = (uncertainty-similarity).argsort()[-step:][::-1]
    return pooled_index


def evaluation(model, criterion, dataloader, device, num_class, args):
    losses = AverageMeter('Loss', ':.4e')
    y_true_list = list()
    y_pred_list = list()
    model.eval()
    with torch.no_grad():
        for idx, (X, y) in enumerate(dataloader):
            X = X.float().to(device)
            y = y.long().to(device)

            if 'BBB' in model._get_name():
                logits, _kl = model(X, sample=False)
                output = F.log_softmax(logits, dim=1)
                Lx, Lw = criterion(output, y, _kl)
                loss = Lx + Lw
            else:
                logits = model(X)
                loss = criterion(logits, y)
            losses.update(loss.item(), X.size(0))

            y_true_list.append(y)
            y_pred_list.append(logits.argmax(-1))

    y_true = torch.cat(y_true_list).cpu().numpy()
    y_pred = torch.cat(y_pred_list).cpu().numpy()

    scores = accuracy(y_pred, y_true, num_class)

    return losses.avg, scores


def fast_eval(model, dataset, device, num_class, args, criterion=None):
    X = torch.tensor(dataset.x[:, :args.in_ch]).float().to(device)
    y = torch.tensor(dataset.y).long().to(device)

    with torch.no_grad():
        if 'BBB' in model._get_name():
            outputs, _kl = model(X, sample=False)
            if criterion is not None:
                Lx, Lw = criterion(F.log_softmax(outputs, dim=1), y, _kl)
                loss = Lx + Lw
        else:
            outputs = model(X)
            if criterion is not None:
                loss = criterion(outputs, y)

    scores = accuracy(outputs.argmax(-1).cpu().numpy(), y.cpu().numpy(), num_class)
    if criterion is not None:
        return loss.item(), scores
    else:
        return scores


class ELBO(nn.Module):
    def __init__(self, beta):
        super(ELBO, self).__init__()
        self.beta = beta

    def forward(self, input, target, kl):
        assert not target.requires_grad
        Lx = F.nll_loss(input, target)
        Lw = self.beta * kl
        return  Lx, Lw


class SemiELBOLoss(nn.Module):
    def __init__(self, lambda_u, epochs, beta, train_size, pseudo_threshold=None):
        super(SemiELBOLoss, self).__init__()
        self.lambda_u = lambda_u
        self.epochs = epochs
        self.beta = beta
        self.pseudo_threshold = pseudo_threshold
        self.train_size = train_size

    def linear_rampup(self, current):
        rampup_length = self.epochs
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current / rampup_length, 0.0, 1.0)
            return float(current)

    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, kl, epoch, weight_x=None, weight_u=None):  # todo
        Lx = -torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1)  # sum
        if outputs_u is not None:
            probs_u = torch.softmax(outputs_u, dim=1)
            Lu = ((probs_u - targets_u) ** 2).sum(1)
        if weight_x is not None:
            Lx = (Lx * weight_x).sum() / weight_x.sum()
            if weight_u.sum() > 0:
                Lu = (Lu*weight_u).sum() / weight_u.sum()
            else:
                Lu = torch.tensor(0)
        else:
            Lx = Lx.mean()
            Lu = Lu.mean()
        Lw = self.beta * kl
        return Lx, Lu, Lw, self.lambda_u * self.linear_rampup(epoch)



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, num_classes=21):
    num = target.shape[0]

    confusion_matrix = get_confusion_matrix(output, target, num_classes)
    TP = confusion_matrix.diagonal()
    FP = confusion_matrix.sum(1) - TP
    FN = confusion_matrix.sum(0) - TP

    po = TP.sum() / num
    pe = (confusion_matrix.sum(0) * confusion_matrix.sum(1)).sum() / num ** 2
    if pe == 1:
        kappa = 1
    else:
        kappa = (po - pe) / (1 - pe)

    p = TP / (TP + FP + 1e-12)
    r = TP / (TP + FN + 1e-12)
    f1 = 2 * p * r / (p + r + 1e-12)

    oa = po
    kappa = kappa
    macro_f1 = f1.mean()
    weight = confusion_matrix.sum(0) / confusion_matrix.sum()
    weighted_f1 = (weight * f1).sum()
    class_f1 = f1

    return dict(
        oa=oa,
        kappa=kappa,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        class_f1=class_f1,
        confusion_matrix=confusion_matrix
    )


def get_confusion_matrix(y_pred, y_true, num_classes=6):
    idx = y_pred * num_classes + y_true
    return np.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


# -------------------------------------- #
#              data utils                #
# -------------------------------------- #
def cycle(iterable):  # Don't use itertools.cycle, as it repeats the same shuffle
    while True:
        for x in iterable:
            yield x


def get_data(train_fn, test_fn):
    test_name = test_fn.stem

    train_data = pd.read_pickle(train_fn)
    test_data = pd.read_pickle(test_fn)

    if 'test' not in test_name and 'site' in train_fn.stem:
        train_data = train_data.loc[train_data['site'] != SITES_NO.get(test_name.split('_')[0])]

    return train_data.values, test_data.values


def get_trainval_dataloader(trainval_data, args):
    transform_train = torchvision.transforms.Compose([
        RandomTempShift(),
        ToTensor(args.in_ch),
    ])
    transform_val = torchvision.transforms.Compose([
        ToTensor(args.in_ch),
    ])

    indices = np.arange(trainval_data.shape[0])
    np.random.shuffle(indices)

    num_val = int(args.num * args.val_ratio)
    num_train = args.num# - num_val

    train_idxs = indices[:num_train]
    val_idxs = indices[-num_val:]

    train_data = trainval_data[train_idxs]
    valid_data = trainval_data[val_idxs]

    X_train, y_train = train_data[:, :190], train_data[:, -1]
    X_valid, y_valid = valid_data[:, :190], valid_data[:, -1]

    trainset = WrapDataset(X_train, y_train, transform_train)
    validset = WrapDataset(X_valid, y_valid, transform_val)

    if args.balance:
        freq = Counter(y_train)
        class_weight = {x:  1.0 / freq[x] for x in freq}
        train_weights = [class_weight[x] for x in y_train]
        sampler = WeightedRandomSampler(train_weights, (len(train_weights) // args.batchsize + 1) * args.batchsize)
        # sampler = WeightedRandomSampler(train_weights, len(train_weights))
        train_loader = torch.utils.data.DataLoader(trainset, sampler=sampler, batch_size=args.batchsize)
    else:
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True)
    # train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=args.batchsize, shuffle=False)

    return train_loader, valid_loader


def get_test_dataloader(test_data, args):
    transform_val = torchvision.transforms.Compose([
        ToTensor(args.in_ch),
    ])
    X_test, y_test = test_data[:, :190], test_data[:, -1]
    if test_data.shape[1] > 191:
        site = test_data[:, -2]
    else:
        site = None
    testset = WrapDataset(X_test, y_test, transform_val, site=site)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.testbatchsize, shuffle=False)

    return testset, test_loader


def get_ssl_dataloader(trainval_data, args):
    transform_train = torchvision.transforms.Compose([
        RandomTempShift(),
        ToTensor(args.in_ch),
    ])
    transform_val = torchvision.transforms.Compose([
        ToTensor(args.in_ch),
    ])

    indices = np.arange(trainval_data.shape[0])
    np.random.shuffle(indices)

    num_val = int(args.num * args.val_ratio)
    num_train = args.num# - num_val

    train_idxs = indices[:num_train]
    val_idxs = indices[-num_val:]
    unlabeled_idxs = indices[num_train:-num_val]

    train_data = trainval_data[train_idxs]
    valid_data = trainval_data[val_idxs]
    unlabeled_data = trainval_data[unlabeled_idxs]

    X_train, y_train = train_data[:, :190], train_data[:, -1]
    X_valid, y_valid = valid_data[:, :190], valid_data[:, -1]
    X_unlabeled, y_unlabeled = unlabeled_data[:, :190], unlabeled_data[:, -1]

    trainset = WrapDataset(X_train, y_train, transform_train)
    validset = WrapDataset(X_valid, y_valid, transform_val)
    unlabeledset = WrapDataset(X_unlabeled, y_unlabeled, transform_val)

    if args.balance:
        freq = Counter(y_train)
        class_weight = {x:  1.0 / freq[x] for x in freq}
        train_weights = [class_weight[x] for x in y_train]
        sampler = WeightedRandomSampler(train_weights, (len(train_weights) // args.batchsize + 1) * args.batchsize)
        # sampler = WeightedRandomSampler(train_weights, len(train_weights))
        train_loader = torch.utils.data.DataLoader(trainset, sampler=sampler, batch_size=args.batchsize)
    else:
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=args.batchsize, shuffle=False)
    unlabeled_loader = torch.utils.data.DataLoader(unlabeledset, batch_size=args.batchsize, shuffle=True)

    return train_loader, valid_loader, unlabeled_loader


def get_pool_dataset(trainval_data, num_val, select_type):
    X_pool, y_pool = trainval_data[:, :190], trainval_data[:, -1]
    if trainval_data.shape[1] > 191:
        site = trainval_data[:, -2]
    else:
        site = None
    poolset = ALDataset(X_pool, y_pool, num_val, select_type, site)
    return poolset


def get_al_dataloader(pool_dataset, args):
    transform_train = torchvision.transforms.Compose([
        RandomTempShift(),
        ToTensor(args.in_ch),
    ])
    transform_val = torchvision.transforms.Compose([
        ToTensor(args.in_ch),
    ])
    samples = pool_dataset.get_samples()
    X_labeled = samples['x_labeled']
    y_labeled = samples['y_labeled']
    X_valid = samples['x_valid']
    y_valid = samples['y_valid']
    X_unlabeled = samples['x_unlabeled']
    y_unlabeled = samples['y_unlabeled']

    train_data = np.hstack([X_labeled, y_labeled.reshape(-1, 1)])
    valid_data = np.hstack([X_valid, y_valid.reshape(-1, 1)])

    X_train, y_train = train_data[:, :190], train_data[:, -1]
    X_valid, y_valid = valid_data[:, :190], valid_data[:, -1]

    trainset = WrapDataset(X_train, y_train, transform_train, ndvi=True)
    validset = WrapDataset(X_valid, y_valid, transform_val)
    unlabeledset = WrapDataset(X_unlabeled, y_unlabeled, transform_val, ndvi=True)

    if args.balance:
        freq = Counter(y_train)
        class_weight = {x:  1.0 / freq[x] for x in freq}
        train_weights = [class_weight[x] for x in y_train]
        sampler = WeightedRandomSampler(train_weights, (len(train_weights) // args.batchsize + 1) * args.batchsize)
        # sampler = WeightedRandomSampler(train_weights, len(train_weights))
        train_loader = torch.utils.data.DataLoader(trainset, sampler=sampler, batch_size=args.batchsize)
    else:
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True)

    valid_loader = torch.utils.data.DataLoader(validset, batch_size=args.batchsize, shuffle=False)
    unlabeled_loader = torch.utils.data.DataLoader(unlabeledset, batch_size=args.batchsize, shuffle=True)

    return train_loader, valid_loader, unlabeled_loader, unlabeledset


def get_al_unlabeledset(pool_dataset, args):
    transform_val = torchvision.transforms.Compose([
        ToTensor(args.in_ch),
    ])
    samples = pool_dataset.get_samples()
    X_unlabeled = samples['x_unlabeled']
    y_unlabeled = samples['y_unlabeled']
    unlabeledset = WrapDataset(X_unlabeled, y_unlabeled, transform_val, samples.get('site_unlabeled'))

    return unlabeledset


# -------------------------------------- #
#              model utils                #
# -------------------------------------- #
def get_model(modelname, num_classes, args):
    modelname = modelname.lower()  # make case invariant
    if modelname == 'bnn':
        model = BBBMLP(num_classes, args.in_ch, priors, device=args.device)
    elif modelname == 'mlp':
        model = MLP(num_classes, args.in_ch).to(args.device)
    return model


def save(model, path="model.pth", **kwargs):
    model_state = model.state_dict()
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    torch.save(dict(model_state=model_state, **kwargs), path)


# -------------------------------------- #
#              model utils                #
# -------------------------------------- #
def overall_performance(args):
    overall_metrics = defaultdict(list)

    cms = []
    for seed in range(1, 6):
        fold_dir = args.output_dir / f'Seed_{seed}'
        if fold_dir.exists():
            test_metrics = pd.read_csv(fold_dir / f'testlog_{args.unlabeled}.csv').iloc[0].to_dict()
            for metric, value in test_metrics.items():
                overall_metrics[metric].append(value)
            cm = np.load(fold_dir / f'test_conf_mat_{args.unlabeled}.npy')
            cms.append(cm)

    print(f'Overall result across 5 trials:')
    for metric, values in overall_metrics.items():
        values = np.array(values)
        if isinstance(values[0], (str)) or np.any(np.isnan(values)):
            continue
        if 'loss' in metric:# or 'f1' in metric:
            print(f"{metric}: {np.mean(values):.4}")  # ±{np.std(value}s):.4")
        else:
            values *= 100
            print(f"{metric}: {np.mean(values):.2f}±{np.std(values):.2f}")

    for oa in overall_metrics['oa']:
        print(oa, end='\t')
