from utils import *


def train_semisupervise(model, args, trainval_data, device, best_model_path):
    # load labeled/val/unlabeled dataloader
    print("=> creating ssl dataloader")
    labeleddataloader, valdataloader, unlabeleddataloader = get_ssl_dataloader(trainval_data, args=args)
    labeled_iter, unlabeled_iter = iter(cycle(labeleddataloader)), iter(cycle(unlabeleddataloader))
    print([Counter(valdataloader.dataset.y)[i] for i in range(5)])
    # create ema model
    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.detach_()

    # criterion & optimizer
    train_criterion = SemiELBOLoss(args.lambda_u, args.epochs, args.beta, args.pseudo_threshold).to(device)
    criterion = ELBO(args.beta).to(device)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, amsgrad=True, weight_decay=5e-4)
    ema_optimizer = WeightEMA(model, ema_model, lr=args.lr, alpha=args.ema_decay)

    log = []
    best_epoch = 0
    val_loss_min = np.Inf
    print(f"Training {args.model} in {args.labeled}")
    for epoch in range(args.epochs):
        train_loss, pseudo_labels, real_labels = train_semi_epoch(model, epoch, args.steps_per_epoch, optimizer, ema_optimizer, train_criterion,
                                 labeled_iter, unlabeled_iter, device, args)
        if args.report_ema:
            val_loss, scores = evaluation(ema_model, criterion, valdataloader, device, args.nclasses, args)
        else:
            val_loss, scores = evaluation(model, criterion, valdataloader, device, args.nclasses, args)

        pseudo_num = real_labels.shape[0]
        pseudo_acc = (pseudo_labels == real_labels).sum() / pseudo_num

        scores_msg = ", ".join([f"{k}={v:.4f}" for (k, v) in scores.items() if k not in ['class_f1', 'confusion_matrix']])

        if (epoch + 1) % args.print_freq == 0:
            print(f"epoch {epoch + 1}: trainloss={train_loss:.4f}, valloss={val_loss:.4f} " + scores_msg)

        scores['epoch'] = epoch+1
        scores['pseudo_num'] = pseudo_num
        scores['pseudo_acc'] = pseudo_acc
        log.append(scores)
        log_df = pd.DataFrame(log).set_index("epoch")
        log_df.to_csv(best_model_path.parent / "trainlog.csv")

        if val_loss < val_loss_min:
            best_epoch = epoch
            val_loss_min = val_loss
            if args.report_ema:
                save(ema_model, path=best_model_path, criterion=criterion)
            else:
                save(model, path=best_model_path, criterion=criterion)
    print(f"saving model to {str(best_model_path)} from epoch {best_epoch+1}\n")
