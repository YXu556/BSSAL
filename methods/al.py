import copy
from utils import *


def train_active(model, args, trainval_data, testset, device, best_model_path):
    # create Pool dataset | it is not a real dataset, still need to wrap

    pool_dataset = get_pool_dataset(trainval_data, int(args.num*args.val_ratio), args.select_type)

    # todo save the site info of the final training set
    # todo save y label at each step
    ns_str = np.arange(args.start_num, args.num+1, args.step)
    if ns_str[-1] != args.num:
        ns_str = np.append(ns_str, args.num)

    log = []
    for i, ns in enumerate(ns_str):
        print("STEP {}".format(i + 1))
        if i == 0 and (args.pretrained is None or args.select_type == 1):
            pool_dataset.get_init(ns)
        elif i == 0 and args.pretrained is not None:
            unlabeledset = get_al_unlabeledset(pool_dataset, args)
            pooled_index = get_pool_index(args.select_type, model, unlabeledset, ns, args)
            pool_dataset.update(ns, pooled_index)
        else:
            if args.select_type == 1:
                pool_dataset.update(ns-ns_str[i-1])
            else:
                pool_dataset.update(ns-ns_str[i-1], pooled_index)

        # get AL dataloader [labeled|val|unlabeled]
        labeleddataloader, valdataloader, _, unlabeledset = get_al_dataloader(pool_dataset, args)

        y_labeled = pool_dataset.get_labeled_y()
        # site_labeled = pool_dataset.get_labeled_site()  # todo save labeled_y_site
        # (args.seed_dir / 'samples').mkdir(parents=True, exist_ok=True)
        # np.save(args.seed_dir / 'samples' / f'{i+1}.npy', np.vstack([y_labeled, site_labeled]))
        print(f"\nLabeled number: {y_labeled.shape[0]} - {np.bincount(y_labeled)[1:]}")

        # criterion & optimizer
        if 'BBB' in model._get_name():
            criterion = ELBO(args.beta).to(device)
        else:
            criterion = nn.CrossEntropyLoss().to(device)
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, amsgrad=True, weight_decay=5e-4)

        # if args.pretrained is not None:
        #     pretrained_path = f"{args.pretrained}/Seed_{args.seed}/model_best.pth"
        #     model_dict = model.state_dict()
        #     checkpoint = torch.load(pretrained_path)
        #     state_dict = checkpoint["model_state"]
        #     model_dict.update(state_dict)
        #     model.load_state_dict(model_dict)
        # else:
        #     for module in model.modules():
        #         if module._get_name() == 'BBBLinear':
        #             module.reset_parameters()
        #         elif module._get_name() == 'Linear':
        #             model.init_fc(module)

        val_loss_min = np.Inf
        model_best = None
        best_epoch = 0
        print(f"Training {args.model} in {args.labeled}")
        for epoch in range(args.epochs):
            train_loss = train_sup_epoch(model, optimizer, criterion, labeleddataloader, device, args)
            val_loss, scores = evaluation(model, criterion, valdataloader, device, args.nclasses, args)
            scores_msg = ", ".join(
                [f"{k}={v:.4f}" for (k, v) in scores.items() if k in ['oa', 'kappa', 'macro_f1']])

            if (epoch + 1) % args.print_freq == 0:
                print(f"{args.seed_dir.name} - Step {i+1} - epoch {epoch + 1}: trainloss={train_loss:.4f}, valloss={val_loss:.4f} " + scores_msg)

            if val_loss < val_loss_min:
                best_epoch = epoch
                val_loss_min = val_loss
                model_best = copy.deepcopy(model)
                if i == len(ns_str) - 1:
                    save(model_best, path=best_model_path, criterion=criterion)

        print(f'Testing...on model from epoch {best_epoch+1}')
        scores = fast_eval(model_best, testset, device, args.nclasses, args)
        # test_loss, scores = evaluation(model_best, criterion, testdataloader, device, args.nclasses, args)
        scores_msg = ", ".join([f"{k}={v:.4f}" for (k, v) in scores.items() if k not in ['class_f1', 'confusion_matrix']])
        print(f"{scores_msg}\n")

        scores['step'] = i+1
        # scores['testloss'] = test_loss
        log.append(scores)
        log_df = pd.DataFrame(log).set_index("step")
        log_df.to_csv(best_model_path.parent / "trainlog.csv")

        # calculate uncertainty
        if i < len(ns_str) - 1 and args.select_type != 1:
            if args.select_type in [5, 7]:
                ndvi = labeleddataloader.dataset.ndvi
                y = labeleddataloader.dataset.y
                D2 = []
                for c in range(6):
                    if (y == c).sum() == 0:
                        D2.append(None)
                    else:
                        ndvi_c = ndvi[y == c].mean(0)
                        d2 = np.where(ndvi_c > ndvi_c.max() / 2)[0][0]
                        D2.append(d2)
            else:
                D2 = None
            pooled_index = get_pool_index(args.select_type, model_best, unlabeledset, ns_str[i+1]-ns, args, D2)
