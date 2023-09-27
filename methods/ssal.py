import copy
from utils import *


def train_semiactive(model, args, trainval_data, testset, device, best_model_path):
    # create Pool dataset | it is not a real dataset, still need to wrap
    pool_dataset = get_pool_dataset(trainval_data, int(args.num*args.val_ratio), args.select_type)

    ns_str = np.arange(args.start_num, args.num+1, args.step)
    if ns_str[-1] != args.num:
        ns_str = np.append(ns_str, args.num)

    log = []
    for i, ns in enumerate(ns_str):
        print("STEP {}".format(i + 1))
        if i == 0:
            pool_dataset.get_init(ns)
        else:
            if args.select_type == 1:
                pool_dataset.update(ns-ns_str[i-1])
            else:
                pool_dataset.update(ns-ns_str[i-1], pooled_index)

        labeleddataloader, valdataloader, unlabeleddataloader, unlabeledset = get_al_dataloader(pool_dataset, args)
        labeled_iter, unlabeled_iter = iter(cycle(labeleddataloader)), iter(cycle(unlabeleddataloader))

        y_labeled = pool_dataset.get_labeled_y()
        # site_labeled = pool_dataset.get_labeled_site()  # todo save labeled_y_site
        # (args.seed_dir / 'samples').mkdir(parents=True, exist_ok=True)
        # np.save(args.seed_dir / 'samples' / f'{i+1}.npy', np.vstack([y_labeled, site_labeled]))
        print(f"\nLabeled number: {y_labeled.shape[0]} - {np.bincount(y_labeled)[1:]}")

        # if args.pretrained is not None:
        #     pretrained_path = f"{args.pretrained}/Seed_{args.seed}/model_best.pth"
        #     model_dict = model.state_dict()
        #     checkpoint = torch.load(pretrained_path)
        #     state_dict = checkpoint["model_state"]
        #     model_dict.update(state_dict)
        #     model.load_state_dict(model_dict)
        # else:
        #     for module in model.children():
        #         if 'BBB' in module._get_name():
        #             module.reset_parameters()

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

        val_loss_min = np.Inf
        model_best = None
        best_epoch = 0
        semi_epoch_log = []
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

            scores_msg = ", ".join([f"{k}={v:.4f}" for (k, v) in scores.items() if k in ['oa', 'kappa', 'macro_f1']])

            if (epoch + 1) % args.print_freq == 0:
                print(f"{args.seed_dir.name} - Step {i + 1} - epoch {epoch + 1}: trainloss={train_loss:.4f}, "
                      f"valloss={val_loss:.4f} " + scores_msg)
            if i == len(ns_str) - 1:
                scores['epoch'] = epoch + 1
                scores['pseudo_num'] = pseudo_num
                scores['pseudo_acc'] = pseudo_acc
                semi_epoch_log.append(scores)
                semi_epoch_log_df = pd.DataFrame(semi_epoch_log).set_index("epoch")
                semi_epoch_log_df.to_csv(best_model_path.parent / "semi_epoch_trainlog.csv")

            if val_loss < val_loss_min:
                # save
                nums = []
                accs = []
                for c in range(6):
                    num = Counter(pseudo_labels.numpy()).get(c)
                    if num is None:
                        num = 0
                        acc = 0
                    else:
                        acc = (pseudo_labels[pseudo_labels == c] == real_labels[pseudo_labels == c]).sum() / num
                    nums.append(num)
                    accs.append(acc)

                best_epoch = epoch
                val_loss_min = val_loss
                if args.report_ema:
                    model_best = copy.deepcopy(ema_model)
                    if i == len(ns_str) - 1:
                        save(ema_model, path=best_model_path, criterion=criterion)
                else:
                    model_best = copy.deepcopy(model)
                    if i == len(ns_str) - 1:
                        save(model, path=best_model_path, criterion=criterion)

        print(f'Testing...on model from epoch {best_epoch+1}')
        scores = fast_eval(model_best, testset, device, args.nclasses, args)
        # test_loss, scores = evaluation(model_best, criterion, testdataloader, device, args.nclasses, args)
        scores_msg = ", ".join([f"{k}={v:.4f}" for (k, v) in scores.items() if k not in ['class_f1', 'confusion_matrix']])
        print(f"{scores_msg}\n")

        scores['step'] = i+1
        scores['pseudo_nums'] = np.array(nums)
        scores['pseudo_accs'] = np.array(accs)

        # scores['testloss'] = test_loss
        log.append(scores)
        log_df = pd.DataFrame(log).set_index("step")
        log_df.to_csv(best_model_path.parent / "trainlog.csv")

        # calculate uncertainty
        if i < len(ns_str) - 1 and args.select_type != 1:
            # print('Calculate uncertainty...\n')
            pooled_index = get_pool_index(args.select_type, model_best, unlabeledset, ns_str[i+1]-ns, args)

