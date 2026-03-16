import os
import time
import numpy as np
import torch.nn
from torch.utils.tensorboard import SummaryWriter
from utils.compute_std import compute_std
from alg.opt import *
from alg import alg, modelopera
from utils.util import set_random_seed, save_checkpoint, print_args, train_valid_target_eval_names, \
    alg_loss_dict
from datautil.getdataloader import get_img_dataloader
import clip
import argparse
import os
import sys
from utils.util import Tee, img_param_init, print_environ, update_excel


def get_args():
    parser = argparse.ArgumentParser(description='DG')
    parser.add_argument('--algorithm', type=str, default="MILES")
    parser.add_argument('--alpha', type=float,
                        default=0.5, help='DANN dis alpha')
    parser.add_argument('--anneal_iters', type=int,
                        default=500, help='Penalty anneal iters used in VREx')
    parser.add_argument('--batch_size', type=int,
                        default=32, help='batch_size')
    parser.add_argument('--beta', type=float,
                        default=0.01, help='DIFEX beta')
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--checkpoint_freq', type=int,
                        default=1, help='Checkpoint every N epoch')
    parser.add_argument('--classifier', type=str,
                        default="linear", choices=["linear", "wn"])
    parser.add_argument('--data_file', type=str, default='',
                        help='root_dir')
    parser.add_argument('--dataset', type=str, default='terra_incognita')
    parser.add_argument('--data_dir', type=str, default='/mlspace/datasets/terra_incognita/', help='data dir')
    parser.add_argument('--dis_hidden', type=int,
                        default=256, help='dis hidden dimension')
    parser.add_argument('--disttype', type=str, default='2-norm',
                        choices=['1-norm', '2-norm', 'cos', 'norm-2-norm', 'norm-1-norm'])
    parser.add_argument('--gpu_id', type=str, nargs='?',
                        default='1', help="device id to run")
    parser.add_argument('--groupdro_eta', type=float,
                        default=1, help="groupdro eta")
    parser.add_argument('--inner_lr', type=float,
                        default=1e-2, help="learning rate used in MLDG")
    parser.add_argument('--lam', type=float,
                        default=0.1, help="tradeoff hyperparameter used in VREx")
    parser.add_argument('--layer', type=str, default="bn",
                        choices=["ori", "bn"])
    parser.add_argument('--lr', type=float, default=0.002, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.75, help='for sgd cosine scheduler')
    parser.add_argument('--lr_decay1', type=float,
                        default=1.0, help='for pretrained featurizer')
    parser.add_argument('--lr_decay2', type=float, default=1.0,
                        help='inital learning rate decay of network')
    parser.add_argument('--lr_gamma', type=float,
                        default=0.0003, help='for optimizer')
    parser.add_argument('--max_epoch', type=int,
                        default=50, help="max iterations")
    parser.add_argument('--mixupalpha', type=float,
                        default=0.1, help='mixup hyper-param')
    parser.add_argument('--mldg_beta', type=float,
                        default=1, help="mldg hyper-param")
    parser.add_argument('--mmd_gamma', type=float,
                        default=0.5, help='MMD, CORAL hyper-param')
    parser.add_argument('--momentum', type=float,
                        default=0.9, help='for optimizer')
    parser.add_argument('--net', type=str, default='ViT-B/16',
                        help="featurizer: vgg16, resnet50, resnet101,DTNBase")
    parser.add_argument('--N_WORKERS', type=int, default=0)
    parser.add_argument('--rsc_f_drop_factor', type=float,
                        default=1 / 3, help='rsc hyper-param')
    parser.add_argument('--rsc_b_drop_factor', type=float,
                        default=1 / 3, help='rsc hyper-param')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--schuse', action='store_true')
    parser.add_argument('--schusech', type=str, default='cos')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--split_style', type=str, default='strat',
                        help="the style to split the train and eval datasets")
    parser.add_argument('--task', type=str, default="img_dg", \
                        choices=["img_dg", 'img_dg_single'], help='now only support image tasks')
    parser.add_argument('--tau', type=float, default=0.82, help="AndMask tau")
    parser.add_argument('--test_envs', type=int, nargs='+',
                        default=[0], help='target domains')
    parser.add_argument('--output', type=str,
                        default="/mlspace/DeepDG/scripts/ImageNet_ViT-B-16/env1/CLIP_E2E",
                        help='result output path')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--compute_std', type=bool, default=True)
    parser.add_argument('--weight', type=float, default=2.0, help="FACT weight")
    parser.add_argument('--rampup_length', type=int, default=5, help="FACT rampup_length")
    parser.add_argument('--mtl_ema', type=float, default=0.99)
    parser.add_argument('--ema_decay', type=float, default=0.9995)
    parser.add_argument('--optimizer', type=str, default='SGD', help='type of optimizer')
    parser.add_argument('--num_heads', type=int, default=4, help='number of attention head')
    parser.add_argument('--T', type=float, default=1, help='temperature of prediction')
    parser.add_argument('--temperature', type=float, default=0.1, help='temperature of contrastive')
    parser.add_argument('--metric', type=str, default="euclidean", help='temperature of contrastive')
    parser.add_argument('--uniform', type=float, default=1.0, help='range of uniform')
    parser.add_argument('--pk', type=int, default=20, help='number of prototypes')
    parser.add_argument('--qratio', type=int, default=2, help='number of prototypes')
    parser.add_argument('--ratio', type=float, default=0.25, help='furrier ratio')
    parser.add_argument('--amp', '-a', action='store_true', help='if specified, turn amp on')
    parser.add_argument('--rate', type=float, default=0.85, help='the learning rate scheduler of cls')
    parser.add_argument('--init_method', type=str, default='zeroshot_classifier', help='init method')
    parser.add_argument('--sch_prompt', type=str, default='lambda', help='the scheduler of prompt')
    parser.add_argument('--model_type', type=str, default='clip', help='the type of model (CLIP or ResNet)')
    parser.add_argument('--min_lr', type=float, default=0, help="learning rate")
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--augmented", action='store_true')
    parser.add_argument('--prec', type=str, default="fp16", help='precision type')
    parser.add_argument('--steps_per_epoch', type=int, default=100)
    parser.add_argument('--scale', type=float, default=12)
    parser.add_argument('--shots_per_class', type=int, default=None)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--nesterov', type=bool, default=True)
    parser.add_argument('--warmup_epoch', type=int, default=0)
    parser.add_argument("--hyper1", type=float, default=0.8)
    parser.add_argument("--hyper2", type=float, default=0.5)
    parser.add_argument("--hyper3", type=float, default=0.1)
    parser.add_argument('--ridg_reg', type=float, default=0.01, help='ridg_reg')

    args = parser.parse_args()
    args.data_dir = args.data_file + args.data_dir
    args.data_dir = args.data_file + args.data_dir
    if ',' in args.gpu_id:
        args.gpu_id = [int(gpu) for gpu in args.gpu_id.split(',')]
    else:
        args.gpu_id = [int(args.gpu_id)]

    torch.cuda.set_device(args.gpu_id[0])
    os.makedirs(args.output, exist_ok=True)
    sys.stdout = Tee(os.path.join(args.output, 'out.txt'))
    sys.stderr = Tee(os.path.join(args.output, 'err.txt'))
    args = img_param_init(args)
    print_environ()
    return args


def main():
    args = get_args()
    set_random_seed(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load(args.net, device=args.device)

    args.preprocess = preprocess
    loss_list = alg_loss_dict(args)
    train_loaders, eval_loaders = get_img_dataloader(args)
    eval_name_dict = train_valid_target_eval_names(args)

    args.in_features = clip_model.visual.output_dim
    algorithm_class = alg.get_algorithm_class(args.algorithm)

    class_names = eval_loaders[0].dataset.classes
    args.class_names = [name.lower().replace("_", " ") for name in class_names]

    clip_model.eval()
    algorithm = algorithm_class(args, clip_model).cuda()

    algorithm = torch.compile(algorithm)
    algorithm.train()
    if args.algorithm not in ('CLIP_ZS', 'COCOOP', 'COOP'):
        opt = get_optimizer(algorithm, args)
        sch = get_scheduler(opt, args)
    else:
        opt = sch = None
    s = print_args(args, [])
    writer = SummaryWriter(args.output)
    print('=======hyper-parameter used========')
    print(s)
    if 'DIFEX' in args.algorithm:
        ms = time.time()
        n_steps = args.max_epoch * args.steps_per_epoch
        print('start training fft teacher net')
        opt1 = get_optimizer(algorithm.teaNet, args, isteacher=True)
        sch1 = get_scheduler(opt1, args)
        algorithm.teanettrain(train_loaders, n_steps, opt1, sch1)
        print('complet time:%.4f' % (time.time() - ms))

    acc_record = {}
    acc_type_list = ['train', 'valid', 'target']
    train_minibatches_iterator = zip(*train_loaders)
    best_valid_acc, target_acc = 0, 0

    print('===========start training===========')

    sss = time.time()
    for epoch in range(args.max_epoch):
        for iter_num in range(args.steps_per_epoch):
            minibatches_device = [(data) for data in next(train_minibatches_iterator)]
            # Reset optim, because it doesn't like the sharp jump in gradient magnitudes that happens at this step.
            if (args.algorithm == 'VREx' or args.algorithm == 'IRM') and algorithm.update_count == args.anneal_iters:
                opt = get_optimizer(algorithm, args)
                sch = get_scheduler(opt, args)
            step_vals = algorithm.update(minibatches_device, opt, sch)

        if (epoch == (args.max_epoch - 1) or (epoch % args.checkpoint_freq == 0)):
            print('===========epoch %d===========' % (epoch))
            s = ''
            for item in loss_list:
                s += (item + '_loss:%.4f,' % step_vals[item])  # print all loss respectively
            print(s[:-1])
            s = ''
            for item in acc_type_list:
                if item != 'train':
                    acc_record[item] = np.mean(np.array([modelopera.accuracy(
                        algorithm, eval_loaders[i]) for i in eval_name_dict[item]]))
                    s += (item + '_acc:%.4f,' % acc_record[item])
            print(s[:-1])

            if acc_record['valid'] >= best_valid_acc:
                best_valid_acc = acc_record['valid']
                target_acc = acc_record['target']
                save_checkpoint(f'best_model.pkl', algorithm, args.output)

            # Visulization
            for item in loss_list:
                writer.add_scalar(f'loss/{item}_loss', step_vals[item], epoch)  # visualize loss
            for item in acc_type_list:
                if item != 'train':
                    writer.add_scalar(f'acc/{item}_acc', acc_record[item], epoch)  # visualize acc


            if args.save_model_every_checkpoint:
                save_checkpoint(f'model_epoch{epoch}.pkl', algorithm, args.output)
            print('total cost time: %.4f' % (time.time() - sss))

    print('valid acc: %.4f' % best_valid_acc)
    print('DG result: %.4f' % target_acc)

    writer.close()
    with open(os.path.join(args.output, 'done.txt'), mode="a") as f:
        f.write('total cost time:%s\n' % (str(time.time() - sss)))
        f.write('valid acc:%.4f\n' % (best_valid_acc))
        f.write('target acc seed%d:%.4f\n\n' % (args.seed, target_acc))
    f.close()
    if args.compute_std and args.seed == 3:
        compute_std(args.output)
    update_excel(target_acc, None, args.alpha, args.beta,
                 os.path.join(os.path.dirname(os.path.dirname(args.output)), 'results_officehome.xlsx'), args.dataset)

if __name__ == '__main__':
    main()
