import argparse
import os
import time

import torch
import torch.nn.functional as F
from torch import optim

from common.builder import *
from common.loss import loss_Jo_AR, get_js_avg, get_js_list
from dataset import data_loader
from model.cnn import CNN


def parser_args():
    parser = argparse.ArgumentParser()

    # ------------------- Init-related --------------------
    parser.add_argument('--random_seed', type=int, default=0, help='set the random seed')
    parser.add_argument('--device', type=str, default='cuda:0', help='Set the GPU')
    parser.add_argument('--prefetch', type=int, default=1, help='Set the workers num')
    parser.add_argument('--log_freq', type=int, default=1)

    # -------------------   Dataset   ---------------------
    parser.add_argument('--config', type=str, default='config/cifar100')
    parser.add_argument('--batch_size', type=int, default=128)
    
    # ------------------- Train-related -------------------
    # Model
    parser.add_argument('--model', type=str, default='cnn', help='model name used for logging')
    parser.add_argument('--init_method', type=str, default='He', help='Xavier or He')

    # Noise
    parser.add_argument('--noise_type', type=str, default='asymmetric',
                        help='symmetric or asymmetric, openset, instance')
    parser.add_argument('--noise_ratio', type=float, default=0.4, help='Set the noisy ratio')

    # Optim
    parser.add_argument('--opt', type=str, default='Adam', help='optimizer name')
    parser.add_argument('--weight-decay', type=float, default=1e-5)

    # Loss method
    parser.add_argument('--loss_type', type=str, default='soft', help="ce, hard, soft, adaptive")

    # Learning rate
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr-decay', type=str, default='Step')
    parser.add_argument('--warmup-lr-scale', type=float, default=10.0)

    # Epochs
    parser.add_argument('--warmup_epochs', type=int, default=10, help='warmup epochs')
    parser.add_argument('--epochs', type=int, default=200, help='train epochs')
    parser.add_argument('--resume', type=str, default=None, help='path to a training checkpoint')

    args = parser.parse_args()
    config = load_from_cfg(args.config)
    override_config_items = [k for k, v in args.__dict__.items() if k != 'config' and v is not None]
    for item in override_config_items:
        config.set_item(item, args.__dict__[item])

    print(config)
    return config


def adjust_learning_rate(optimizer, epoch, al_plan, be_plan):
    for param_group in optimizer.param_groups:
        param_group['lr'] = al_plan[epoch]
        param_group['betas'] = (be_plan[epoch], 0.999)  # Only change beta1


def save_training_checkpoint(path, epoch, net, net2, optimizer, optimizer2,
                             best_accuracy, best_epoch, best_accuracy2, best_epoch2, js_avg):
    checkpoint = {
        'epoch': epoch,
        'net': net.state_dict(),
        'net2': net2.state_dict(),
        'optimizer': optimizer.state_dict(),
        'optimizer2': optimizer2.state_dict(),
        'best_accuracy': best_accuracy,
        'best_epoch': best_epoch,
        'best_accuracy2': best_accuracy2,
        'best_epoch2': best_epoch2,
        'js_avg': js_avg.detach().cpu(),
    }
    temporary_path = path + '.tmp'
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, path)


def load_training_checkpoint(path, device, net, net2, optimizer, optimizer2):
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Checkpoint not found: {path}')

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    net.load_state_dict(checkpoint['net'])
    net2.load_state_dict(checkpoint['net2'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    optimizer2.load_state_dict(checkpoint['optimizer2'])

    return {
        'start_epoch': int(checkpoint['epoch']),
        'best_accuracy': float(checkpoint.get('best_accuracy', 0.0)),
        'best_epoch': checkpoint.get('best_epoch'),
        'best_accuracy2': float(checkpoint.get('best_accuracy2', 0.0)),
        'best_epoch2': checkpoint.get('best_epoch2'),
        'js_avg': torch.as_tensor(checkpoint.get('js_avg', 1.0), device=device),
    }


def main(cfg):
    # Set the random seed
    init_seeds(cfg.random_seed)

    # Get device
    device = cfg.device
    print(f'Train on: {device}')

    # Build Logger
    logger, result_dir = build_logger(cfg)

    # Get net optim
    net = CNN(n_outputs=cfg.num_classes).to(device)
    net2 = CNN(n_outputs=cfg.num_classes).to(device)
    init_weights(net, init_method=cfg.init_method)
    init_weights(net2, init_method=cfg.init_method)
    optimizer = optim.Adam(net.parameters(), cfg.lr, weight_decay=cfg.weight_decay)
    optimizer2 = optim.Adam(net2.parameters(), cfg.lr, weight_decay=cfg.weight_decay)

    # Set learing rate
    mom1 = 0.9
    mom2 = 0.1
    alpha_plan = [cfg.lr] * cfg.epochs
    beta1_plan = [mom1] * cfg.epochs

    decay_start = 80
    for i in range(decay_start, cfg.epochs):
        alpha_plan[i] = float(cfg.epochs - i) / (cfg.epochs - decay_start) * cfg.lr
        beta1_plan[i] = mom2

    # Set the dataloader
    train_loader, valid_loader, n_train_samples, n_validating_samples = data_loader.build_dataloader(cfg.dataset, cfg)

    logger.msg(f"Categories: {cfg.num_classes}, Training Samples: {n_train_samples},"
               f" Valid Samples: {n_validating_samples}, Model: {cfg.model}")
    logger.msg(f"Noise Type: {cfg.noise_type}, Noise Ratio: {cfg.noise_ratio}")
    logger.msg(f'Optimizer: {cfg.opt}')

    # ----------------loss----------------
    loss_function = loss_Jo_AR

    # ----------------- meter ---------------
    train_loss = AverageMeter()
    train_accuracy = AverageMeter()
    epoch_train_time = AverageMeter()
    best_accuracy, last_accuracy, best_epoch = 0.0, 0.0, None
    best_accuracy2, last_accuracy2, best_epoch2 = 0.0, 0.0, None
    js_avg = torch.tensor(1.).to(device)
    start_epoch = 0

    resume_path = getattr(cfg, 'resume', None)
    if resume_path:
        resume_state = load_training_checkpoint(
            resume_path, device, net, net2, optimizer, optimizer2)
        start_epoch = resume_state['start_epoch']
        best_accuracy = resume_state['best_accuracy']
        best_epoch = resume_state['best_epoch']
        best_accuracy2 = resume_state['best_accuracy2']
        best_epoch2 = resume_state['best_epoch2']
        js_avg = resume_state['js_avg']
        logger.msg(f'Resumed checkpoint: {resume_path} (completed epochs: {start_epoch})')

    if start_epoch >= cfg.epochs:
        logger.msg(f'Checkpoint already completed {start_epoch} epochs; target is {cfg.epochs}.')
        return

    # ------------------ training -------------------
    for epoch in range(start_epoch, cfg.epochs):
        start_time = time.time()

        net.train()
        net2.train()
        adjust_learning_rate(optimizer, epoch, alpha_plan, beta1_plan)
        adjust_learning_rate(optimizer2, epoch, alpha_plan, beta1_plan)

        optimizer.zero_grad()
        optimizer2.zero_grad()

        train_loss.reset()
        train_accuracy.reset()

        JSD = torch.zeros(n_train_samples, device=device)
        clean_num = 0
        selection_num = 0
        jsd_offset = 0

        # Train this epoch
        pbar = tqdm(train_loader, ncols=150, ascii=' >', leave=False, desc='training', total=len(train_loader))
        for it, samples in enumerate(pbar):
            iter_start = time.time()
            curr_lr = [group['lr'] for group in optimizer.param_groups][0]

            # Divided data
            in_X_weak, in_X_strong = samples[0].to(device), samples[1].to(device)
            target, target_gt = samples[2].long().to(device), samples[3].long().to(device)

            # predictions
            prob_weak = net(in_X_weak)
            prob_strong = net(in_X_strong)
            net2_prob = net2(in_X_weak).detach()

            js_list = get_js_list(prob_weak.softmax(dim=1), target).detach()

            # get the accuracy of this step
            train_acc = accuracy(prob_weak.softmax(dim=1), target_gt)
            pbar.set_postfix_str(f'TrainAcc: {train_accuracy.avg:3.2f}%; TrainLoss: {train_loss.avg:3.2f}')

            # update the model
            if epoch < cfg.warmup_epochs:
                pbar.set_description(f'WARMUP TRAINING (lr={curr_lr:.3e})')
                loss_all = F.cross_entropy(prob_weak, target)
                JSD[jsd_offset:(jsd_offset + in_X_weak.size(0))] = js_list
                jsd_offset += in_X_weak.size(0)
                clean_list = js_list < js_avg
                if sum(clean_list) != 0:
                    loss2 = F.cross_entropy(net2(in_X_weak[clean_list]), target[clean_list]) + F.cross_entropy(net2(in_X_strong[clean_list]), target[clean_list])
                    optimizer2.zero_grad()
                    loss2.backward()
                    optimizer2.step()

            else:
                pbar.set_description(f'ROBUST TRAINING (lr={curr_lr:.3e})')
                loss_all, js_list_cor, clean_list = loss_function(prob_weak, prob_strong, target, js_avg, js_list, net2_prob)
                JSD[jsd_offset:(jsd_offset + in_X_weak.size(0))] = js_list_cor
                jsd_offset += in_X_weak.size(0)
                if sum(clean_list) != 0:
                    clean_samples_weak, clean_sample_strong, clean_target = in_X_weak[clean_list], in_X_strong[clean_list], target[clean_list]
                    prob_weak_2 = net2(clean_samples_weak)
                    prob_strong_2 = net2(clean_sample_strong)
                    loss2 = F.cross_entropy(prob_weak_2, clean_target) + F.cross_entropy(prob_strong_2, clean_target)
                    optimizer2.zero_grad()
                    loss2.backward()
                    optimizer2.step()

            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

            selection_num += clean_list.sum().item()
            clean_num += (target[clean_list] == target_gt[clean_list]).sum().item()

            train_accuracy.update(train_acc[0], in_X_weak.size(0))
            train_loss.update(loss_all.item(), in_X_weak.size(0))
            epoch_train_time.update(time.time() - iter_start, 1)

            if (cfg.log_freq is not None and (it + 1) % cfg.log_freq == 0) or (it + 1 == len(train_loader)):
                total_mem = torch.cuda.get_device_properties(0).total_memory / 2 ** 30 if torch.cuda.is_available() else 0.0
                mem = torch.cuda.memory_reserved() / 2 ** 30 if torch.cuda.is_available() else 0.0
                console_content = f"Epoch:[{epoch + 1:>3d}/{cfg.epochs:>3d}]  " \
                                  f"Iter:[{it + 1:>4d}/{len(train_loader):>4d}]  " \
                                  f"Train Accuracy:[{train_accuracy.avg:6.2f}]  " \
                                  f"Loss:[{train_loss.avg:4.4f}]  " \
                                  f"GPU-MEM:[{mem:6.3f}/{total_mem:6.3f} Gb]  " \
                                  f"{epoch_train_time.avg:6.2f} sec/iter"
                logger.debug(console_content)

        eval_result = evaluate(valid_loader, net, device)
        test_accuracy = eval_result['accuracy']
        test_loss = eval_result['loss']
        is_best_net1 = best_epoch is None or test_accuracy > best_accuracy
        if is_best_net1:
            best_accuracy = test_accuracy
            best_epoch = epoch + 1

        eval_result2 = evaluate(valid_loader, net2, device)
        test_accuracy2 = eval_result2['accuracy']
        test_loss2 = eval_result2['loss']
        is_best_net2 = best_epoch2 is None or test_accuracy2 > best_accuracy2
        if is_best_net2:
            best_accuracy2 = test_accuracy2
            best_epoch2 = epoch + 1

        runtime = time.time() - start_time
        logger.info(f'epoch: {epoch + 1:>3d} | '
                    f'train loss: {train_loss.avg:>6.4f} | '
                    f'train accuracy: {train_accuracy.avg:>6.3f} | '
                    f'test loss: {test_loss:>6.4f} | '
                    f'test accuracy: {test_accuracy:>6.3f} | '
                    f'best accuracy: {best_accuracy:6.3f} @ epoch: {best_epoch:03d} |'
                    f'net2 loss: {test_loss2:>6.4f} | '
                    f'net2 acc: {test_accuracy2:>6.3f} | '
                    f'best accuracy: {best_accuracy2:6.3f} @ epoch: {best_epoch2:03d} |' 
                    f'epoch runtime: {runtime:6.2f} sec | '
                    f'selected clean rate: {clean_num/(selection_num+1e-8):6.3f} | '
                    f'clean num: {clean_num} | '
                    f'selected num: {selection_num},'
                    )

        js_avg = get_js_avg(JSD).detach().to(device)
        latest_checkpoint = os.path.join(result_dir, 'checkpoint_latest.pt')
        save_training_checkpoint(
            latest_checkpoint, epoch + 1, net, net2, optimizer, optimizer2,
            best_accuracy, best_epoch, best_accuracy2, best_epoch2, js_avg)
        if is_best_net1:
            best_checkpoint = os.path.join(result_dir, 'checkpoint_best_net1.pt')
            save_training_checkpoint(
                best_checkpoint, epoch + 1, net, net2, optimizer, optimizer2,
                best_accuracy, best_epoch, best_accuracy2, best_epoch2, js_avg)
            logger.msg(f'Best net1 checkpoint saved: {best_checkpoint}')
        if is_best_net2:
            best_checkpoint2 = os.path.join(result_dir, 'checkpoint_best_net2.pt')
            save_training_checkpoint(
                best_checkpoint2, epoch + 1, net, net2, optimizer, optimizer2,
                best_accuracy, best_epoch, best_accuracy2, best_epoch2, js_avg)
            logger.msg(f'Best net2 checkpoint saved: {best_checkpoint2}')
        print(js_avg, min(JSD))

    logger.msg(f'Latest checkpoint saved: {latest_checkpoint}')


if __name__ == '__main__':
    cfg = parser_args()
    main(cfg)
