# This is the main python script for sup-learn for now
import copy
import json
import random
import argparse
import sklearn.metrics
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from joblib import dump
from sklearn.ensemble import RandomForestClassifier

import torch
import torch.nn as nn
from torch.nn import functional as F

from utils import *
from methods.ssl import train_semisupervise
from methods.al import train_active
from methods.ssal import train_semiactive

# path to dataset
DATAPATH = Path(r'data')


def parse_args(doy=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--labeled', default='train_2019_site', help='labeled dataset')
    parser.add_argument('--unlabeled', default='test_2019_site', help='unlabeled dataset')
    parser.add_argument('-c', '--nclasses', type=int, default=6,
                        help='num of classes (default: 6)')
    parser.add_argument('-d', '--doy', default=280, type=int,
                        help='The end doy for end-of-the-season/in-season experiments (default 280)')
    parser.add_argument('-n', '--num', default=300, type=int,
                        help='number of labeled samples (training and validation) (default 300)')
    parser.add_argument('--useall', action='store_true',
                        help='wheather to use all training data')
    parser.add_argument("--balance", action='store_true', #default=True,
                        help='class balanced batches for train')
    parser.add_argument("--val_ratio", default=0.1, type=float,
                        help='Ratio of training data to use for validation. Default 10%.')
    parser.add_argument('-b', '--batchsize', type=int, default=64,
                        help='batch size')
    parser.add_argument('--testbatchsize', type=int, default=4096,
                        help='batch size')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10)')
    parser.add_argument('--model', type=str, default="BNN",
                        help='select model architecture from [BNN|...].')
    parser.add_argument('--pretrained', default=None, type=str,
                        help='path to pretrained checkpoint')
    parser.add_argument('--num_ens', type=int, default=1,
                        help='sample times during training')
    parser.add_argument('--beta', type=float, default=1e-7,
                        help='trade-off coefficient for bayesian loss')
    parser.add_argument('--output_dir', default='results/checkpoints',
                        help='logdir to store progress and models (defaults to ./results)')
    parser.add_argument('-s', '--suffix', default=None,
                        help='suffix to output_dir')
    parser.add_argument('--eval', action='store_true',# default=True,
                        help='evaluate model on validation set')
    parser.add_argument('--overall', action='store_true',
                        help='print overall results, if exists')
    parser.add_argument('--device', type=str, default=None,
                        help='torch.Device. either "cpu" or "cuda". default will check by torch.cuda.is_available() ')

    # Specific parameters for each training method
    subparsers = parser.add_subparsers(dest='method')

    # SSL
    ssl = subparsers.add_parser('ssl')
    ssl.add_argument("--steps_per_epoch", type=int, default=10, help='n steps per epoch')
    ssl.add_argument('--lambda_u', type=float, default=1,
                     help='trade off for unlabeled loss')
    ssl.add_argument('--ema_decay', type=float, default=0.999,
                     help='ema decay for mixmatch setting')
    ssl.add_argument('-T', type=float, default=0.5,
                     help='temperature for mixmatch setting')
    ssl.add_argument('--alpha', type=float, default=0.25,
                     help='alpha for mixmatch setting')
    ssl.add_argument('--report_ema', action='store_true',
                    help='evaluate on ema model')
    ssl.add_argument("--pseudo_threshold", default=0.9, type=float,
                     help='confidence threshold for assigning pseudo labels')

    # AL
    al = subparsers.add_parser('al')
    al.add_argument('-sel', '--select_type', type=int, default=1,
                    help='[1: Random | 2: BBvSB | 3: BvSB | 4: Entropy | 5: MI]')
    al.add_argument('--dropout_iterations', type=int, default=100)
    al.add_argument('--start_num', type=int, default=30)
    al.add_argument('--step', type=int, default=30)

    # SSAL
    ssal = subparsers.add_parser('ssal')
    ssal.add_argument("--steps_per_epoch", type=int, default=10, help='n steps per epoch')
    ssal.add_argument('--lambda_u', type=float, default=1,
                     help='trade off for unlabeled loss')
    ssal.add_argument('--ema_decay', type=float, default=0.999,
                     help='ema decay for mixmatch setting')
    ssal.add_argument('-T', type=float, default=0.5,
                     help='temperature for mixmatch setting')
    ssal.add_argument('--alpha', type=float, default=0.25,
                     help='alpha for mixmatch setting')
    ssal.add_argument('--report_ema', action='store_true',
                    help='evaluate on ema model')
    ssal.add_argument("--pseudo_threshold", default=0.9, type=float,
                     help='confidence threshold for assigning pseudo labels')
    ssal.add_argument('-sel', '--select_type', type=int, default=2,
                    help='[1: Random | 2: BBvSB | 3: BvSB | 4: Entropy | 5: MI]')
    ssal.add_argument('--dropout_iterations', type=int, default=100)
    ssal.add_argument('--start_num', type=int, default=30)
    ssal.add_argument('--step', type=int, default=30)

    args = parser.parse_args()

    if doy is not None:
        args.doy = doy
    args.in_ch = args.doy - 100 + 10

    args.train_fn = DATAPATH / f'{args.labeled}.pkl'
    args.test_fn = DATAPATH / f'{args.unlabeled}.pkl'
    args.output_dir = Path(args.output_dir) / f"{args.model}_{args.labeled}_R{args.num}_DoY{args.doy}"
    if args.method is not None:
        args.output_dir = args.output_dir.parent / f"{args.method}_{args.output_dir.stem}"
        if 'al' in args.method:
            args.output_dir = args.output_dir.parent / f"{args.output_dir.name}_{args.select_type}"
    if args.suffix:
        args.output_dir = args.output_dir.parent / f"{args.output_dir.name}_{args.suffix}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for fold in range(1, 6):  # 5 Random Seeds
        (args.output_dir / f'Seed_{fold}').mkdir(parents=True, exist_ok=True)

    # write training config to file
    if not args.eval:
        with open(str(args.output_dir / 'train_config.json'), 'w') as f:
            args.train_fn = str(args.train_fn)
            args.test_fn = str(args.test_fn)
            args.output_dir = str(args.output_dir)
            f.write(json.dumps(vars(args), indent=4))
            args.train_fn = Path(args.train_fn)
            args.test_fn = Path(args.test_fn)
            args.output_dir = Path(args.output_dir)

    args.datapath = DATAPATH

    # Setup device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(args)

    return args


def main(args):
    # load data to ram
    print('Pre-load all data')
    trainval_data, test_data = get_data(args.train_fn, args.test_fn)
    # test dataloader kept the same among five trails
    testset, testdataloader = get_test_dataloader(test_data, args)
    if args.useall:
        args.num = trainval_data.shape[0]

    # iter among 5 seeds
    for seed in range(1, 6):
        print('-------------- SEED-{} ----------------'.format(seed))
        args.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        print(f'Starting seed {seed}...')
        args.seed_dir = args.output_dir / f'Seed_{seed}'
        print(f"Logging results to {args.seed_dir}")

        print("=> creating model '{}'".format(args.model))
        device = torch.device(args.device)
        model = get_model(args.model, args.nclasses, args)
        best_model_path = args.seed_dir / 'model_best.pth'

        if args.pretrained is not None:
            pretrained_path = f"{args.pretrained}/Seed_{seed}/model_best.pth"
            print("=> loaded checkpoint '{}'".format(str(pretrained_path)))
            model_dict = model.state_dict()
            checkpoint = torch.load(pretrained_path)
            state_dict = checkpoint["model_state"]
            model_dict.update(state_dict)
            model.load_state_dict(model_dict)

        if not args.eval:
            if args.method == 'ssl':
               train_semisupervise(model, args, trainval_data, device, best_model_path)
            elif args.method == 'al':
                train_active(model, args, trainval_data, testset, device, best_model_path)
            elif args.method == 'ssal':
                train_semiactive(model, args, trainval_data, testset, device, best_model_path)
            else:
               train_supervised(model, args, trainval_data, device, best_model_path)

        print('Restoring best model weights for testing...')
        checkpoint = torch.load(best_model_path)
        state_dict = checkpoint['model_state']
        criterion = checkpoint['criterion']
        model.load_state_dict(state_dict)

        # test_loss, scores = evaluation(model, criterion, testdataloader, device, args.nclasses, args)
        test_loss, scores = fast_eval(model, testset, device, args.nclasses, args, criterion)

        scores_msg = ", ".join([f"{k}={v:.4f}" for (k, v) in scores.items() if k not in ['class_f1', 'confusion_matrix']])
        print(f"Test results : \n\n {scores_msg} \n\n")

        scores['epoch'] = 'test'
        scores['testloss'] = test_loss
        conf_mat = scores.pop('confusion_matrix')
        class_f1 = scores.pop('class_f1')

        log_df = pd.DataFrame([scores]).set_index("epoch")
        log_df.to_csv(args.seed_dir / f"testlog_{args.unlabeled}.csv")
        np.save(args.seed_dir / f"test_conf_mat_{args.unlabeled}.npy", conf_mat)
        np.save(args.seed_dir / f"test_class_f1_{args.unlabeled}.npy", class_f1)


def train_supervised(model, args, trainval_data, device, best_model_path):
    # load source dataloader
    print("=> creating train/val dataloader")
    traindataloader, valdataloader = get_trainval_dataloader(trainval_data, args=args)

    # criterion & optimizer
    if 'BBB' in model._get_name():
        criterion = ELBO(args.beta).to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, amsgrad=True, weight_decay=5e-4)

    val_loss_min = np.Inf
    print(f"Training {args.model} in {args.labeled}")
    for epoch in range(args.epochs):
        train_loss = train_sup_epoch(model, optimizer, criterion, traindataloader, device, args)
        val_loss, scores = evaluation(model, criterion, valdataloader, device, args.nclasses, args)
        scores_msg = ", ".join([f"{k}={v:.4f}" for (k, v) in scores.items() if k not in ['class_f1', 'confusion_matrix']])

        if (epoch + 1) % args.print_freq == 0:
            print(f"epoch {epoch + 1}: trainloss={train_loss:.4f}, valloss={val_loss:.4f} " + scores_msg)

        if val_loss < val_loss_min:
            val_loss_min = val_loss
            save(model, path=best_model_path, criterion=criterion)
    print(f"saving model to {str(best_model_path)}\n")


if __name__ == '__main__':
    args = parse_args()
    if not args.overall:
        main(args)
    overall_performance(args)