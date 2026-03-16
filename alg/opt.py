# coding=utf-8
import torch


def get_params(alg, args, inner=False, alias=True, isteacher=False):
    if args.schuse:
        if args.schusech == 'cos':
            initlr = args.lr
        else:
            initlr = 1.0
    else:
        if inner:
            initlr = args.inner_lr
        else:
            initlr = args.lr

    if args.algorithm == 'PCL':
        params = [
            {'params': alg.fea_proj.parameters()},
            {'params': alg.fc_proj},
            {'params': alg.classifier}
        ]
        return params
    if args.algorithm == 'MILES':
        params = [param for name, param in alg.miles_learner.named_parameters() if param.requires_grad]
        return params

    if args.algorithm == 'CLIPFit':
        params = [param for name, param in alg.fit_learner.named_parameters() if param.requires_grad]
        return params

    if args.algorithm in ['CLIP_E2E', 'DiffDG', 'RIDG', 'CLIP_Linear', 'CLIP_Aug', 'SCIPD', 'SAGM_DG', 'TIP_Adapter']:
        params = [{'params': alg.classifier.parameters()}]
        return params

    if isteacher:
        params = [
            {'params': alg[0].parameters(), 'lr': args.lr_decay1 * initlr},
            {'params': alg[1].parameters(), 'lr': args.lr_decay2 * initlr},
            {'params': alg[2].parameters(), 'lr': args.lr_decay2 * initlr}
        ]
        return params
    if inner:
        params = [
            {'params': alg[0].parameters(), 'lr': args.lr_decay1 *
                                                  initlr},
            {'params': alg[1].parameters(), 'lr': args.lr_decay2 *
                                                  initlr}
        ]
    elif "SCIPD_S" == args.algorithm:
        params = [
            {"params": alg.featurizer.parameters(), "lr": args.lr_decay1 * initlr},
            {"params": alg.classifier.parameters(), "lr": args.lr_cls * initlr},
        ]
    elif alias:
        params = [
            {'params': alg.featurizer.parameters(), 'lr': args.lr_decay1 * initlr},
            {'params': alg.classifier.parameters(), 'lr': args.lr_decay2 * initlr}
        ]
    else:
        params = [
            {'params': alg[0].parameters(), 'lr': args.lr_decay1 * initlr},
            {'params': alg[1].parameters(), 'lr': args.lr_decay2 * initlr}
        ]
    if ('DANN' in args.algorithm) or ('CDANN' in args.algorithm):
        params.append({'params': alg.discriminator.parameters(),
                       'lr': args.lr_decay2 * initlr})
    if ('CDANN' in args.algorithm):
        params.append({'params': alg.class_embeddings.parameters(),
                       'lr': args.lr_decay2 * initlr})
    if ('DIFEX' in args.algorithm):
        params.append({'params': alg.bottleneck.parameters(),
                       'lr': args.lr_decay2 * initlr})
    if ('DAPC' in args.algorithm):
        params.append({'params': alg.atten_head.parameters(),
                       'lr': args.lr_decay2 * initlr})
    if ('SD' in args.algorithm):
        params.append({'params': alg.project.parameters(),
                       'lr': args.lr_decay2 * initlr})
    return params


def get_optimizer(alg, args, inner=False, alias=True, isteacher=False):
    params = get_params(alg, args, inner, alias, isteacher)
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay, eps=1e-3)
    else:
        optimizer = torch.optim.AdamW(
            params, lr=args.lr, weight_decay=args.weight_decay)
    return optimizer

def get_optimizer1(network, args):
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            network, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    else:
        optimizer = torch.optim.AdamW(
            network, lr=args.lr, weight_decay=args.weight_decay)
    return optimizer


def get_scheduler(optimizer, args):
    if not args.schuse:
        return None
    if args.schusech == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.max_epoch * args.steps_per_epoch)
    elif args.schusech == 'rule':
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=get_lambda_schedule(args.rate, args.steps_per_epoch))
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lambda step: 1 - step / (
                                                              args.max_epoch * args.steps_per_epoch))
    if args.warmup_epoch > 0:
        scheduler = ConstantWarmupScheduler(
            optimizer, scheduler, args.warmup_epoch * args.steps_per_epoch + 1,
            1e-5
        )

    return scheduler


def get_lambda_schedule(rate, steps_per_epoch):
    def _rule(epoch):
        lamda = math.pow(rate, (epoch / steps_per_epoch))
        return lamda
        # warmup_steps = 100
        # if epoch < warmup_steps:
        #     return float(epoch + 1) / float(max(1, warmup_steps))
        # return 1.0

    return _rule


import math
from torch.optim.lr_scheduler import _LRScheduler


class _BaseWarmupScheduler(_LRScheduler):

    def __init__(
            self,
            optimizer,
            successor,
            warmup_epoch,
            last_epoch=-1,
            verbose=False
    ):
        self.successor = successor
        self.warmup_epoch = warmup_epoch
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if self.last_epoch >= self.warmup_epoch:
            self.successor.step(epoch)
            self._last_lr = self.successor.get_last_lr()
        else:
            super().step(epoch)


class ConstantWarmupScheduler(_BaseWarmupScheduler):

    def __init__(
            self,
            optimizer,
            successor,
            warmup_epoch,
            cons_lr,
            last_epoch=-1,
            verbose=False
    ):
        self.cons_lr = cons_lr
        super().__init__(
            optimizer, successor, warmup_epoch, last_epoch, verbose
        )

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.successor.get_last_lr()
        return [self.cons_lr for _ in self.base_lrs]
