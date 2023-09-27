import copy
import argparse
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

import torch
import torchvision
import torch.nn as nn
from torch.nn import functional as F


from utils import *

# path to dataset
DATAPATH = Path(r'D:\DadaX\PhD\Research\1. CropMapping\data')


def parse_args(year=None):
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
    parser.add_argument("--balance", action='store_true', default=True,
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

    if year is not None:
        args.labeled = args.labeled.replace('2019', str(year))
        args.unlabeled = args.unlabeled.replace('2019', str(year))
    args.in_ch = args.doy - 100 + 10

    args.train_fn = DATAPATH / f'csv/train_new/{args.labeled}.pkl'  # todo
    args.test_fn = DATAPATH / f'csv/test_new/{args.unlabeled}.pkl'  # todo
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

    args.datapath = DATAPATH

    # Setup device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(args)

    return args


def model_testing(model, dataset, device, args):
    X = torch.tensor(dataset.x[:, :args.in_ch]).float().to(device)
    model.eval()
    with torch.no_grad():
        if 'BBB' in model._get_name():
            outputs, _kl = model(X, sample=False)
        else:
            outputs = model(X)
    y_pred = outputs.argmax(1).cpu().numpy()
    return y_pred


def main(args):
    # data
    print('Pre-load all data')
    _, test_data = get_data(args.train_fn, args.test_fn)
    # test dataloader kept the same among five trails
    testset, testdataloader = get_test_dataloader(test_data, args)

    # get model
    device = torch.device(args.device)
    model = get_model(args.model, args.nclasses, args)

    # load model to eval
    print('testing...')
    y_preds = []
    for fold in range(1, 6):  # 5 Random Seeds
        ckpt_pth = args.output_dir / f'Seed_{fold}'/ 'model_best.pth'
        checkpoint = torch.load(ckpt_pth)
        model.load_state_dict(checkpoint['model_state'])
        y_pred = model_testing(model, testset, device, args)
        y_preds.append(y_pred)

    y_count = np.apply_along_axis(lambda x: np.bincount(x, minlength=args.nclasses), 0, np.array(y_preds))
    y_pred = y_count.argmax(0)
    y = testset.y
    sites = testset.site
    site_accs = []
    for i in range(len(SITES)):
        site = SITES[i]
        scores = accuracy(y_pred[sites==i], y[sites==i], args.nclasses)
        scores['site'] = site
        site_accs.append(scores)
    log_df = pd.DataFrame(site_accs).set_index("site")
    log_df.to_csv(args.output_dir / "site_acc.csv")


if __name__ == '__main__':
    for year in [2019, 2020, 2021]:
        print('==============', year, '==============')
        args = parse_args(year)
        main(args)
