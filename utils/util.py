# coding=utf-8
import random
import numpy as np
import sys
import torch.nn as nn
import torchvision
import PIL
import torch
from torch.nn.modules.batchnorm import _BatchNorm
import time
import zipfile

def set_random_seed(seed=0):
    # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(filename, alg, output):
    trainable_params = {name: param.detach().clone() for name, param in alg.named_parameters() if param.requires_grad}
    torch.save({
        'state_dict': trainable_params,
    }, os.path.join(output, filename))


def load_checkpoint(filename, alg, output):
    checkpoint = torch.load(os.path.join(output, filename), map_location='cuda')
    saved_state_dict = checkpoint['state_dict']
    model_dict = alg.state_dict()
    model_dict.update({k: v for k, v in saved_state_dict.items() if k in model_dict})
    alg.load_state_dict(model_dict)
    print("Model loaded from {}".format(os.path.join(output, filename)))

def load_checkpoint1(filename, alg, output):
    checkpoint = torch.load(os.path.join(output, filename), map_location='cuda')
    saved_state_dict = checkpoint['miles_learner_state_dict']
    model_dict = alg.state_dict()
    # unmatched_keys = [k for k in saved_state_dict.keys() if k not in model_dict]
    # if unmatched_keys:
    #     print(f"Unmatched keys in checkpoint: {unmatched_keys}")
    model_dict.update({k: v for k, v in saved_state_dict.items() if k in model_dict})
    alg.load_state_dict(model_dict)
    print("Model loaded from {}".format(os.path.join(output, filename)))

def load_checkpoint_with_mapping(filename, alg, output, train_classes, target_classes):
    """
    加载权重并将 1000 类别的权重映射到 200 类别的分类头，支持 bias 为空的情况
    """
    checkpoint = torch.load(os.path.join(output, filename), map_location='cuda')
    saved_state_dict = checkpoint['state_dict']
    train_class_to_idx = {cls_name: idx for idx, cls_name in enumerate(train_classes)}
    target_class_to_idx = {cls_name: idx for idx, cls_name in enumerate(target_classes)}
    # duplicates = [cls_name for cls_name in train_classes if train_classes.count(cls_name) > 1]
    # if duplicates:
    #     print(f"重复的类名: {set(duplicates)}")
    # else:
    #     print("没有重复的类名")

    if hasattr(alg, 'classifier') and hasattr(alg.classifier, 'fc') and isinstance(alg.classifier.fc, nn.Linear):
        fc_weight_key = [key for key in saved_state_dict.keys() if 'fc.weight' in key]
        fc_bias_key = [key for key in saved_state_dict.keys() if 'fc.bias' in key]

        if fc_weight_key:
            old_fc_weight = saved_state_dict[fc_weight_key[0]]
        else:
            raise KeyError("fc.weight not found in checkpoint.")

        old_fc_bias = None
        if fc_bias_key:
            old_fc_bias = saved_state_dict[fc_bias_key[0]]

        num_target_classes = len(target_classes)
        new_fc_weight = torch.zeros((num_target_classes, old_fc_weight.size(1)), device=old_fc_weight.device).to(alg.clip_model.dtype)
        new_fc_bias = torch.zeros(num_target_classes, device=old_fc_weight.device).to(alg.clip_model.dtype) if old_fc_bias is not None else None

        for target_class, target_idx in target_class_to_idx.items():
            if target_class in train_class_to_idx:
                train_idx = train_class_to_idx[target_class]
                new_fc_weight[target_idx] = old_fc_weight[train_idx]
                if old_fc_bias is not None:
                    new_fc_bias[target_idx] = old_fc_bias[train_idx]

        alg.classifier.fc.weight.data = new_fc_weight
        if new_fc_bias is not None:
            alg.classifier.fc.bias.data = new_fc_bias

    elif hasattr(alg, 'classifier') and isinstance(alg.classifier, nn.Parameter):
        old_classifier_key = [key for key in saved_state_dict.keys() if 'classifier' in key]
        if old_classifier_key:
            old_classifier = saved_state_dict[old_classifier_key[0]]
        else:
            raise KeyError("classifier not found in checkpoint.")

        num_target_classes = len(target_classes)
        new_classifier = torch.zeros((num_target_classes, old_classifier.size(1)), device=old_classifier.device).to(alg.clip_model.dtype)

        for target_class, target_idx in target_class_to_idx.items():
            if target_class in train_class_to_idx:
                train_idx = train_class_to_idx[target_class]
                new_classifier[target_idx] = old_classifier[train_idx]

        alg.classifier.data = new_classifier

    if hasattr(alg, 'adapter') and isinstance(alg.adapter, nn.Linear):
        adapter_weight_key = [key for key in saved_state_dict.keys() if 'adapter.weight' in key]
        if adapter_weight_key:
            old_adapter_weight = saved_state_dict[adapter_weight_key[0]]
        else:
            raise KeyError("adapter.weight not found in checkpoint.")
        alg.adapter.weight.data = old_adapter_weight

    model_dict = alg.state_dict()
    model_dict.update({k: v for k, v in saved_state_dict.items() if k in model_dict and 'classifier' not in k and 'adapter' not in k})
    alg.load_state_dict(model_dict)

    print(f"Checkpoint loaded and weights mapped to {len(target_classes)} classes.")

def save_all_checkpoint(filename, alg, args):
    save_dict = {
        "network": alg.network.state_dict()
    }
    torch.save(save_dict, os.path.join(args.output, filename))


def load_all_checkpoint(filename, alg, args):
    checkpoint = torch.load(os.path.join(args.output, filename),
                            map_location=torch.device('cuda', args.gpu_id))
    model = checkpoint['network']
    alg.network.load_state_dict(model)


def train_valid_target_eval_names(args):
    eval_name_dict = {'train': [], 'valid': [], 'target': []}
    t = 0
    for i in range(args.domain_num):
        if i not in args.test_envs:
            eval_name_dict['train'].append(t)
            t += 1
    for i in range(args.domain_num):
        if i not in args.test_envs:
            eval_name_dict['valid'].append(t)
        else:
            eval_name_dict['target'].append(t)
        t += 1
    return eval_name_dict


def alg_loss_dict(args):
    loss_dict = {'ANDMask': ['total'],
                 'CORAL': ['class', 'coral', 'total'],
                 'DANN': ['class', 'dis', 'total'],
                 'ERM': ['class'],
                 'Mixup': ['class'],
                 'Mixup1': ['class'],
                 'MLDG': ['total'],
                 'MMD': ['class', 'mmd', 'total'],
                 'GroupDRO': ['group'],
                 'RSC': ['class'],
                 'VREx': ['loss', 'nll', 'penalty'],
                 'IRM': ['loss', 'nll', 'penalty'],
                 'MTL': ['loss'],
                 'DIFEX': ['class', 'dist', 'exp', 'align', 'total'],
                 'FACT': ['class', 'loss_aug', 'loss_ori_tea', 'loss_aug_tea', 'total'],
                 'DNA': ['total', 'loss_c', 'loss_v'],
                 'DAPC': ['total', 'class', 'consistency', 'contrast_loss'],
                 'DAPC_EMA': ['total', 'class', 'consistency', 'contrast_loss'],
                 'PCL': ['total', 'loss_cls', 'loss_pcl'],
                 'SAGM_DG': ['loss'],
                 'CLIP_Linear': ['loss_cls'],
                 'CLIP_ZS': ['loss_cls'],
                 'CLIP_Aug': ['cls_loss', 'clip_aug_loss', 'domain_loss'],
                 'CLIP_E2E': ['cls_loss', 'clip_aug_loss', 'domain_loss'],
                 'COOP': ['cls_loss'],
                 'COCOOP': ['cls_loss'],
                 'TIP_Adapter': ['cls_loss'],
                 'DiffDG': ['cls_loss', 'clip_aug_loss', 'domain_loss'],
                 'RIDG': ['loss'],
                 'SCIPD': ['class', 'contra'],
                 'SCIPD_S': ['class', 'contra'],
                 'DiffDG_SD': ['ce_loss', 'kl_loss', 'loss'],
                 'CLIPFit': ['total'],
                 'MILES': ['total'],
                 }

    return loss_dict[args.algorithm]


def print_args(args, print_list):
    s = "==========================================\n"
    l = len(print_list)
    for arg, content in args.__dict__.items():
        if l == 0 or arg in print_list:
            s += "{}:{}\n".format(arg, content)
    return s


def print_environ():
    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))


class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def img_param_init(args):
    dataset = args.dataset
    if dataset == 'office':
        domains = ['amazon', 'dslr', 'webcam']
    elif dataset == 'office-caltech':
        domains = ['amazon', 'dslr', 'webcam', 'caltech']
    elif dataset == 'office-home':
        domains = ['Art', 'Clipart', 'Product', 'RealWorld']
    elif dataset == 'digits_dg':
        domains = ['mnist', 'mnist_m', 'svhn', 'syn']
    elif dataset == 'dg5':
        domains = ['mnist', 'mnist_m', 'svhn', 'syn', 'usps']
    elif dataset == 'PACS':
        domains = ['art_painting', 'cartoon', 'photo', 'sketch']
    elif dataset == 'VLCS':
        domains = ['Caltech101', 'LabelMe', 'SUN09', 'VOC2007']
    elif dataset == 'terra_incognita':
        domains = ['location_100', 'location_38', 'location_43', 'location_46']
    elif dataset == 'domainnet':
        domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
    elif dataset == 'ImageNet':
        domains = ['imagenet', 'imagenetv2', 'imagenet-sketch', 'imagenet-a', 'imagenet-r']
    elif dataset == 'cifar100':
        domains = ['CIFAR-100', 'CIFAR-100-C']
    elif dataset == 'domainnets':
        domains = ['clipart', 'infograph', 'painting', 'real', 'sketch']
    else:
        print('No such dataset exists!')
    args.domains = domains
    args.img_dataset = {
        'office': ['amazon', 'dslr', 'webcam'],
        'office-caltech': ['amazon', 'dslr', 'webcam', 'caltech'],
        'office-home': ['Art', 'Clipart', 'Product', 'RealWorld'],
        'PACS': ['art_painting', 'cartoon', 'photo', 'sketch'],
        'digits_dg': ['mnist', 'mnist_m', 'svhn', 'syn'],
        'dg5': ['mnist', 'mnist_m', 'svhn', 'syn', 'usps'],
        'VLCS': ['Caltech101', 'LabelMe', 'SUN09', 'VOC2007'],
        'terra_incognita': ['location_100', 'location_38', 'location_43', 'location_46'],
        'domainnet': ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch'],
        'ImageNet': ['imagenet-origin', 'imagenetv2', 'imagenet-sketch', 'imagenet-a', 'imagenet-r'],
        'cifar100': ['CIFAR-100', 'CIFAR-100-C'],
        'domainnets': ['clipart', 'infograph', 'painting', 'real', 'sketch'],
    }
    if dataset == 'digits_dg' or dataset == 'dg5':
        args.input_shape = (3, 32, 32)
        args.num_classes = 10
    else:
        args.input_shape = (3, 224, 224)
        if args.dataset == 'office-home':
            args.num_classes = 65
        elif args.dataset == 'office':
            args.num_classes = 31
        elif args.dataset == 'PACS':
            args.num_classes = 7
        elif args.dataset == 'VLCS':
            args.num_classes = 5
        elif args.dataset == 'terra_incognita':
            args.num_classes = 10
        elif args.dataset == 'domainnet':
            args.num_classes = 345
        elif args.dataset == 'domainnets':
            args.num_classes = 345
        elif args.dataset == 'ImageNet':
            args.num_classes = 1000
        elif args.dataset == 'cifar100':
            args.num_classes = 100
    return args


def get_current_consistency_weight(a, epoch, weight, rampup_length, rampup_type='step'):
    if rampup_type == 'step':
        rampup_func = step_rampup
    elif rampup_type == 'linear':
        rampup_func = linear_rampup
    elif rampup_type == 'sigmoid':
        rampup_func = sigmoid_rampup
    else:
        raise ValueError("Rampup schedule not implemented")

    return weight * rampup_func(a, epoch, rampup_length)


def sigmoid_rampup(a, current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-a * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def step_rampup(current, rampup_length):
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return 0.0


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def PJS_loss(prob, label):
    row_index = torch.arange(0, prob.size(0))
    prob_y = prob[row_index, label]
    loss = (torch.log(2 / (1 + prob_y)) + prob_y * torch.log(2 * prob_y / (1 + prob_y))).mean()
    return loss


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)


'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''

# _, term_width = os.popen('stty size', 'r').read().split()
term_width = 80

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


import re
import pandas as pd
import os


# Step 1: 读取 done.txt 文件内容并提取 mean_std
def parse_result(file_path):
    with open(file_path, 'r') as file:
        data = file.read()

    # 使用正则表达式提取 mean_std 的值
    matches = re.findall(r"mean_std: ([\d.]+)\+([\d.]+)", data)
    if matches:
        # 取最后一个匹配结果
        mean, std = map(float, matches[-1])
        return mean, std
    else:
        raise ValueError("未找到 mean_std 的值！")


import re

def parse_target_acc(file_path):
    with open(file_path, 'r') as file:
        data = file.read()

    # 只读 target acc: 后的第一个浮点数
    matches = re.findall(r"target acc(?: [^:]*)?:\s*([\d.]+)", data)
    if matches:
        return float(matches[-1])  # 返回最后一个匹配
    else:
        raise ValueError("未找到 target acc 的值！")


# Step 2: 更新 Excel 表格
def update_excel(mean, std, algorithm, colum, excel_path, dataset_name):
    if not os.path.exists(excel_path):
        df = pd.DataFrame(index=pd.MultiIndex(levels=[[], []], codes=[[], []], names=["algorithm", "metric"]))
        with pd.ExcelWriter(excel_path) as writer:
            df.to_excel(writer, sheet_name=dataset_name)
    else:
        # 读取指定 sheet，如果不存在则创建
        if (not os.path.exists(excel_path)) or (not zipfile.is_zipfile(excel_path)):
            df = pd.DataFrame(index=pd.MultiIndex(levels=[[], []], codes=[[], []], names=["algorithm", "metric"]))
        else:
            try:
                df = pd.read_excel(excel_path, sheet_name=dataset_name, index_col=[0, 1])
            except (ValueError, zipfile.BadZipFile):
                df = pd.DataFrame(index=pd.MultiIndex(levels=[[], []], codes=[[], []], names=["algorithm", "metric"]))

    # 如果 shot 不在列中，动态添加
    if colum not in df.columns:
        df[colum] = None

    # 如果 algorithm 不在索引中，动态添加
    if (algorithm, 'ACC') not in df.index:
        df.loc[(algorithm, 'ACC'), :] = None
    if (algorithm, 'Std') not in df.index:
        df.loc[(algorithm, 'Std'), :] = None

    # 填入均值和标准差
    df.loc[(algorithm, 'ACC'), colum] = mean
    df.loc[(algorithm, 'Std'), colum] = std

    # 保存更新后的表格到指定 sheet
    with pd.ExcelWriter(excel_path, mode='a', if_sheet_exists='replace') as writer:
        df.to_excel(writer, sheet_name=dataset_name)

    print(f"Successfully updated Excel file: {excel_path}, sheet: {dataset_name}")

# try:
#     # 解析结果并更新 Excel
#     mean, std = parse_result('/mlspace/DeepDG/scripts/ImageNet_ViT-B-16/MILES/imagenetv2/done.txt')
#     update_excel(mean, std, '0', 0.1, '/mlspace/DeepDG/scripts/ImageNet_ViT-B-16/results.xlsx','ImageNet')
# except Exception as e:
#     print(f"发生错误：{e}")
