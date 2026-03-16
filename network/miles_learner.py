import torch
import torch.nn as nn
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from datautil.templates import *
from network.loss import DirectionLoss
import torch.nn.functional as F
import random
from datautil.util import style_template, format_prompt
from clip.model import AttentionPool2d
import json
import os
_tokenizer = _Tokenizer()

def save_attn_weights(module, input, output):
    module.attn_weights = output[1]


class Fusion(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Fusion, self).__init__()
        self.fusion = nn.Linear(input_dim, output_dim)

    def forward(self, a, b=None):
        if b is None:
            return self.fusion(a)
        return self.fusion(torch.cat([a, b], dim=-1))


class CrossAttention(nn.Module):
    def __init__(self, q_dim, kv_dim, output_dim, num_heads=4):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.scale = q_dim ** -0.5

        # Linear projections for Q, K, and V
        self.q_proj = nn.Linear(q_dim, output_dim)  # Q 的线性投影
        self.k_proj = nn.Linear(kv_dim, output_dim)  # K 的线性投影
        self.v_proj = nn.Linear(kv_dim, output_dim)  # V 的线性投影

        # Attention dropout
        self.attn_drop = nn.Dropout(0)

    def forward(self, query, key, value):
        N = query.shape[0]
        C = self.output_dim
        # Apply linear projections to Q, K, and V
        q = self.q_proj(query).reshape(N, self.num_heads, C // self.num_heads).permute(1, 0, 2)
        k = self.k_proj(key).reshape(-1, self.num_heads, C // self.num_heads).permute(1, 0, 2)
        v = self.v_proj(value).reshape(-1, self.num_heads, C // self.num_heads).permute(1, 0, 2)

        # Compute attention weights
        attn_weights = (q @ k.transpose(-2, -1)) * self.scale  # Dot product of q and k ==> [num_heads, Nq, Nk]
        attn_weights = attn_weights.softmax(dim=-1)  # Apply softmax to the dot product
        attn_weights = self.attn_drop(attn_weights)  # Apply dropout

        x = (attn_weights @ v).transpose(1, 2).reshape(N, C)  # [Nq, num_heads, C // num_heads] ==> [N, C]

        return x, attn_weights


class LatentSampling(nn.Module):
    def __init__(self, ouptut_dim, q_dim, kv_dim, reduction=1):
        super(LatentSampling, self).__init__()

        self.reduction = reduction
        self.ouptut_dim = ouptut_dim
        self.q_dim = q_dim
        self.kv_dim = kv_dim
        self.hidden_dim = ouptut_dim // reduction
        self._build_up()

    def _build_up(self):
        self.crossAttention = CrossAttention(self.q_dim, self.kv_dim, self.hidden_dim)
        self.fc_variance = nn.Linear(self.hidden_dim, self.ouptut_dim)
        self.fc_mean = nn.Linear(self.hidden_dim, self.ouptut_dim)

    def forward(self, query, key):
        # batch (grid*grid) dim
        attn_feat, attn_weights = self.crossAttention(
            query, key, key
        )
        # batch * patch_num * feat_dim
        variance = self.fc_variance(attn_feat)
        mu = self.fc_mean(attn_feat)
        return variance, mu

    def sample(self, mu, variance, num=1):
        var = variance.expand(num, *variance.size()).transpose(0, 1)
        m = mu.expand(num, *mu.size()).transpose(0, 1)
        noise = torch.randn_like(var).to(var.device)
        noise = var * noise + m
        return noise


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.text_projection.dtype

    def forward(self, prompts, tokenized_prompts, block=False):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # 转换形状: NLD -> LND

        block_features = []
        if block:
            token_indices = tokenized_prompts.argmax(dim=-1), torch.arange(x.shape[1])
            for resblock in self.transformer.resblocks:
                x = resblock(x)
                cls_token = x[token_indices]
                block_features.append(cls_token)  # 提取每个块的 CLS token
            block_features = torch.stack(block_features, dim=1)
        else:
            x = self.transformer(x)

        # 恢复形状: LND -> NLD
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)

        cls_token_indices = torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)
        x = x[cls_token_indices] @ self.text_projection

        if block:
            block_features = self.ln_final(block_features).type(self.dtype)
            block_features = block_features @ self.text_projection
            block_features = torch.cat([x.unsqueeze(1), block_features], dim=1)
            return x, block_features
        return x


class ImageEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.visual = clip_model.visual
        self.dtype = clip_model.visual.conv1.weight.dtype
        if hasattr(self.visual, 'layer1'):
            self.block_poolers = nn.ModuleList([
                AttentionPool2d(56, 256, num_heads=1, output_dim=2048).type(self.dtype),  # layer1
                AttentionPool2d(28, 512, num_heads=1, output_dim=2048).type(self.dtype),  # layer2
                AttentionPool2d(14, 1024, num_heads=1, output_dim=2048).type(self.dtype),  # layer3
                AttentionPool2d(7, 2048, num_heads=1, output_dim=2048).type(self.dtype),  # layer4
            ])

    def forward(self, x):
        if hasattr(self.visual, 'transformer'):
            return self.forward_vit(x)
        elif hasattr(self.visual, 'layer1'):
            return self.forward_resnet(x)
        else:
            raise TypeError("Unsupported visual encoder type")

    def forward_vit(self, x):
        x = self.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype,
                                                                             device=x.device), x],
                      dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.visual.positional_embedding.to(x.dtype)
        x = self.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND

        block_features = []
        for resblock in self.visual.transformer.resblocks:
            x = resblock(x)
            cls_token = x[0, :, :]
            block_features.append(cls_token)  # Extract features from each block
        block_features = torch.stack(block_features, dim=1)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.visual.ln_post(x[:, 0, :])

        if self.visual.proj is not None:
            x = x @ self.visual.proj

        return x, block_features  # Return final output and features from each block [B,12,768]

    def forward_resnet(self, x):
        def stem(x):
            x = self.visual.relu1(self.visual.bn1(self.visual.conv1(x)))
            x = self.visual.relu2(self.visual.bn2(self.visual.conv2(x)))
            x = self.visual.relu3(self.visual.bn3(self.visual.conv3(x)))
            x = self.visual.avgpool(x)
            return x

        x = x.type(self.dtype)
        x = stem(x)

        pooled_features = []
        for layer_id, layer in enumerate(
                [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]):
            x = layer(x)
            pooled = self.block_poolers[layer_id](x)
            pooled_features.append(pooled)

        x = self.visual.attnpool(x)  # [B, D]
        pooled_features = torch.stack(pooled_features, dim=1)
        return x, pooled_features


class MilesLearner(nn.Module):
    def __init__(self, args, classnames, clip_model, dim=256, dropout=0):
        super().__init__()
        self.device = args.device
        self.output_dim = args.in_features
        clip_model_, preprocess = clip.load(args.net, device=args.device)
        self.dtype = clip_model.dtype
        if args.prec == "fp32" or args.amp:
            clip_model_.float()
        self.n_style = 20
        self.ctx_dim = clip_model.ln_final.weight.shape[0]
        self.init_func = [nn.init.normal_, nn.init.xavier_uniform_, nn.init.xavier_normal_, nn.init.kaiming_normal_,
                          nn.init.kaiming_uniform_]
        clip_model_.eval()
        self.clip_model_ = clip_model_
        self.image_encoder = ImageEncoder(clip_model)
        self.text_encoder = TextEncoder(clip_model)
        if len(args.gpu_id) > 1:
            self.text_encoder = torch.nn.DataParallel(self.text_encoder, device_ids=args.gpu_id)
        # For a Vision Transformer (ViT) model
        if hasattr(clip_model.visual, 'transformer'):
            block_dim = clip_model.visual.transformer.width
            print(f"block_features last dimension (transformer width): {block_dim}")

        # For a ResNet model
        elif hasattr(clip_model.visual, 'layer1'):
            block_dim = 2048
            print(f"block_features last dimension (ResNet output channels): {block_dim}")

        self.imageLatentSampling = LatentSampling(block_dim, block_dim, args.in_features, 1).to(
            self.dtype)

        if args.dataset == "ImageNet":
            json_path = os.path.join(os.path.dirname(__file__), '../datautil/imagenet_prompt.json')
            with open(json_path, 'r') as f:
                gpt3_prompt = json.load(f)
            print('Using GPT-3 generated prompts for ImageNet')
            with torch.no_grad():
                clip_weights = []
                for classname in classnames:
                    # Tokenize the prompts
                    classname = classname.replace("_", " ")
                    texts = []
                    for t in gpt3_prompt[classname]:
                        texts.append(t)
                    texts = clip.tokenize(texts)
                    if torch.cuda.is_available():
                        texts = texts.cuda()
                    class_embeddings = clip_model_.encode_text(texts)
                    class_embeddings = class_embeddings.mean(dim=0, keepdim=True)
                    clip_weights.append(class_embeddings)
        else:
            self.template = IMAGENET_TEMPLATES
            with torch.no_grad():
                clip_weights = []
                # Using multiple text templates to ensure textual diversity during training
                for classname in classnames:
                    # Tokenize the prompts for each class
                    classname = classname.replace('_', ' ')  # Replace underscores with spaces
                    texts = [format_prompt(t, classname) for t in self.template]
                    x_tokenized = clip.tokenize(texts).to(args.device)  # Tokenize each class prompts
                    class_embeddings = clip_model_.encode_text(x_tokenized)
                    class_embedding = class_embeddings.mean(dim=0, keepdim=True)  # Average embeddings for each class
                    clip_weights.append(class_embedding)
        clip_weights = torch.cat(clip_weights, dim=0)
        class_embeddings_fixed = clip_weights / clip_weights.norm(dim=-1, keepdim=True)
        self.class_embeddings_fixed = class_embeddings_fixed

        temp = "a photo of a {}"
        prompts_ = [format_prompt(temp, c) for c in args.class_names]
        # print(f"Prompts: {prompts_}")
        tokenized_prompts_ = torch.cat([clip.tokenize(p) for p in prompts_])
        tokenized_prompts_ = tokenized_prompts_.to(args.device)

        base_style_list = style_template()
        tokenized_base_style = torch.cat([clip.tokenize(s) for s in base_style_list]).to(self.device)
        with torch.no_grad():
            self.clip_tokenized_prompt = tokenized_prompts_
            self.clip_prompt = clip_model.token_embedding(tokenized_prompts_).type(self.dtype)

            self.base_style_embedding = clip_model.token_embedding(tokenized_base_style)[:, 1:2, :].squeeze().type(
                self.dtype)  # style embedding

        self.img_proj = nn.Linear(args.in_features, dim).type(self.dtype)
        self.text_proj = nn.Linear(args.in_features, dim).type(self.dtype)

        self.hard_positives = Fusion(args.in_features * 2, args.in_features).to(self.dtype)
        self.hard_negatives = Fusion(dim * 2, dim).to(self.dtype)

        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-07)
        self.dropout = nn.Dropout(dropout)

    def forward(self, images, num=1):
        image_features, block_features = self.image_encoder(images.type(self.dtype))
        avg_features = block_features.mean(dim=1)  # [B, 12, 768] -> [B, 768]
        prompts = self.clip_prompt  # 修改不用初始化函数，关于跟域有关的提示，self.clip_prompt 变成形参
        tokenized_prompts = self.clip_tokenized_prompt
        text_features, block_text_features = self.text_encoder(prompts, tokenized_prompts, True)
        avg_text_features = block_text_features.mean(dim=1)

        mu, variance = self.imageLatentSampling(avg_features.detach(),
                                                avg_text_features.detach())  # [B, 12, 768], [B, 512]

        noises = self.imageLatentSampling.sample(mu, variance, num)

        if num == 1:
            noises = noises.squeeze(1)
        else:
            noises = noises.reshape(-1, *noises.size()[2:])
            image_features = image_features.expand(num, *image_features.size()
                                                   ).transpose(0, 1)
            image_features = image_features.reshape(-1, *image_features.size()[2:])
        if hasattr(self.image_encoder.visual, 'transformer'):
            image_features = image_features + noises @ self.image_encoder.visual.proj
        else:
            image_features = image_features + self.image_encoder.visual.attnpool.c_proj(noises)
        image_features = self.dropout(image_features)
        return image_features, text_features

    def forward_features(self, images):
        image_features, _ = self.image_encoder(images.type(self.dtype))
        prompts = self.clip_prompt  # 修改不用初始化函数，关于跟域有关的提示，self.clip_prompt 变成形参
        tokenized_prompts = self.clip_tokenized_prompt
        text_features = self.text_encoder(prompts, tokenized_prompts)
        return image_features, text_features

    def get_old_features(self, images):
        image_features = self.clip_model_.visual(images.type(self.dtype))
        class_features = self.class_embeddings_fixed
        return image_features, class_features

    def get_text_features(self, text_prompts):
        tokenized_prompts_ = torch.cat([clip.tokenize(p) for p in text_prompts])
        tokenized_prompts_ = tokenized_prompts_.to(self.device)
        with torch.no_grad():
            prompts = self.clip_model_.token_embedding(tokenized_prompts_).type(self.dtype)
            tokenized_prompts = tokenized_prompts_

        text_features = self.text_encoder(prompts, tokenized_prompts)
        return text_features

    def forward_similarity(self, labels=None):
        if labels is None:
            unique_labels = torch.arange(len(self.clip_prompt))  # 全选所有标签
        else:
            unique_labels = torch.unique(labels)

        prompts = self.clip_prompt[unique_labels]
        tokenized_prompts = self.clip_tokenized_prompt[unique_labels]
        text_features = self.text_encoder(prompts, tokenized_prompts)  # tokenized_prompts 编号

        text_features_old = self.class_embeddings_fixed[unique_labels]

        text_features_old = text_features_old / text_features_old.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        score = self.cos(text_features, text_features_old)
        score = 1.0 - torch.mean(score)
        return score

    def forward_direction(self, images, labels, image_features, text_features):
        directional_loss = DirectionLoss()
        text_features_old = self.class_embeddings_fixed[labels]
        image_features_old = self.clip_model_.visual(images.type(self.dtype))

        img_directional = image_features - image_features_old
        img_directional = F.normalize(img_directional, dim=-1)
        text_embeddings = text_features - text_features_old
        text_directional = F.normalize(text_embeddings, dim=-1)
        loss = directional_loss(img_directional, text_directional).mean()
        return loss

    def get_proj_features(self, image_features, text_features):
        proj_image_features = self.img_proj(image_features)
        proj_text_features = self.text_proj(text_features)
        return proj_image_features, proj_text_features

    def refresh_style(self, refresh_type="Select"):
        new_styles = []
        num_base = self.base_style_embedding.shape[0]

        if refresh_type == "Mix":
            for _ in range(self.n_style):
                _lambda = torch.distributions.Beta(0.1, 0.1).sample((self.base_style_embedding.shape[0],)).to(
                    self.device)
                normalized_lambda = _lambda / _lambda.sum()
                normalized_lambda = normalized_lambda.view(self.base_style_embedding.shape[0], 1)
                new_style = normalized_lambda * self.base_style_embedding
                new_style = torch.sum(new_style, dim=0)
                new_style = new_style.view(1, new_style.shape[0])
                new_styles.append(new_style)

        elif refresh_type == "Random":
            for _ in range(self.n_style):
                new_style = torch.empty(1, self.ctx_dim, dtype=torch.float, device=self.device)
                init_func_id = random.randint(0, len(self.init_func) - 1)
                self.init_func[init_func_id](new_style)
                new_styles.append(new_style)

        elif refresh_type == "Select":
            if num_base >= self.n_style:
                indices = torch.randperm(num_base)[:self.n_style].tolist()
            else:
                indices = [random.randint(0, num_base - 1) for _ in range(self.n_style)]
            for idx in indices:
                new_style = self.base_style_embedding[idx].view(1, -1).to(self.device)
                new_styles.append(new_style)

        else:
            raise ValueError(f"Unknown refresh_type: {refresh_type}")

        new_styles = torch.stack(new_styles)  # shape -> [n_style, 1, dim]
        return self.init_stylized_text(new_styles)

    def init_stylized_text(self, style_embedding):  # Used for domain loss.
        base_text = "x-like style"
        style_position = 1
        base_text_list = [base_text] * self.n_style
        tokenized_base_text = torch.cat([clip.tokenize(p) for p in base_text_list]).to(self.device)
        with torch.no_grad():
            stylized_base_text_embedding = self.clip_model_.token_embedding(
                tokenized_base_text).type(self.dtype)  # Convert basic-style tokens into embeddings.
        stylized_base_text_embedding[:, style_position:style_position + 1, :] = style_embedding
        return self.text_encoder(stylized_base_text_embedding, tokenized_base_text)
