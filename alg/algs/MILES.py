# coding=utf-8
import torch
from alg.algs.base import Algorithm
from network.miles_learner import MilesLearner
import torch.nn.functional as F
from torch.amp import autocast as autocast
from torch.amp import GradScaler
from network.loss import ProxyPLoss
from utils.entropy_loss import EntropyMinimization
import torch.nn as nn


class MILES(Algorithm):
    """
    MILES
    """

    def __init__(self, args, clip_model):
        super(MILES, self).__init__(args)
        if args.prec == "fp32" or args.amp:
            clip_model.float()
        self.amp = args.amp
        self.alpha = args.alpha
        self.beta = args.beta
        self.scaler = GradScaler('cuda', enabled=args.amp)
        self.logit_scale = clip_model.logit_scale
        self.entropy = EntropyMinimization(args.T)

        self.miles_learner = MilesLearner(args, args.class_names, clip_model, args.dim, args.dropout)
        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.miles_learner.named_parameters():
            if 'imageLatentSampling' not in name and 'hard_' not in name:
                param.requires_grad_(False)

        for name, param in self.miles_learner.text_encoder.named_parameters():
            if any(key in name for key in ['c_proj.bias']):
                param.requires_grad_(True)

        for name, param in self.miles_learner.image_encoder.named_parameters():
            if any(key in name for key in ['ln', 'bn', 'block_poolers']):
                param.requires_grad_(True)

        self.miles_learner.text_proj.requires_grad_(True)
        self.miles_learner.img_proj.requires_grad_(True)
        total_params = sum(p.numel() for p in self.miles_learner.parameters() if p.requires_grad)
        print(f"Total Parameters: {total_params / 1e6:.2f}M")
        self.miles_learner.to(args.device)

        self.proxy_loss = ProxyPLoss(args.num_classes, args.scale)

        self.num_classes = args.num_classes
        self.class_names = args.class_names
        self.device = args.device
        self.domain_names = [domain for i, domain in enumerate(args.domains) if i not in args.test_envs]
        self.steps_per_epoch = args.steps_per_epoch
        self.current_step = 0

    def update(self, minibatches, opt, sch):
        self.miles_learner.train()
        self.current_step += 1
        inputs = torch.cat([data[0].cuda().float() for data in minibatches])
        labels = torch.cat([data[1].cuda().long() for data in minibatches])
        domain_labels = torch.cat([data[2].cuda().long() for data in minibatches])

        noise_num = 1
        if not noise_num == 1:
            labels = labels.expand(noise_num, *labels.size()).transpose(0, 1)
            labels = labels.reshape(-1, *labels.size()[2:])
        with (((autocast('cuda', enabled=self.amp)))):
            image_features, class_features = self.miles_learner(inputs, num=noise_num)
            logits = self.get_class_logits(image_features, class_features)

            style_features = self.miles_learner.refresh_style()
            y_domain = self.get_class_logits(image_features, style_features)
            entropy = self.entropy(y_domain, y_domain)
            K = entropy.size(-1)
            domain_loss = - torch.mean(torch.sum(entropy, dim=-1)) + torch.log(
                torch.tensor(float(K), device=entropy.device))  # Maximizing entropy
            max_entropy_indices = torch.argmax(entropy, dim=1)  # equal torch.argmax(y_domain, dim=1)
            hard_style_features = style_features[max_entropy_indices]
            hard_positives = self.miles_learner.hard_positives(image_features, hard_style_features)

            all_features = torch.cat([image_features, hard_positives], dim=0)
            all_lables = torch.cat([labels, labels], dim=0)
            proj_features, proj_class = self.miles_learner.get_proj_features(all_features, class_features)
            N = image_features.shape[0]
            proj_images, _ = torch.split(proj_features, [N, proj_features.shape[0] - N], dim=0)
            random_indices = torch.randint(0, self.num_classes - 1, (labels.size(0),)).to(self.device)
            random_indices = (random_indices + (random_indices >= labels).long()) % self.num_classes
            hard_negatives = proj_class[random_indices]

            hard_negatives = self.miles_learner.hard_negatives(proj_images.detach(), hard_negatives.detach())
            total_loss = F.cross_entropy(logits, labels) + domain_loss
            loss_similarity_text = self.alpha * self.miles_learner.forward_similarity(labels)

            loss_con = self.beta * self.proxy_loss(proj_features, all_lables, proj_class,
                                                   torch.cat([hard_negatives, hard_negatives], dim=0))

            total_loss += loss_similarity_text + loss_con
            opt.zero_grad()
            self.scaler.scale(total_loss).backward()
            self.scaler.step(opt)
            self.scaler.update()
            if sch:
                sch.step()
        return {'total': total_loss.item()}

    def predict(self, x):
        image_features, class_features = self.miles_learner(x)
        return self.get_class_logits(image_features, class_features)

    def get_class_logits(self, image_features, class_features):
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        class_features = class_features / class_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ class_features.t()
        return logits
