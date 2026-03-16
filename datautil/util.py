# coding=utf-8
import numpy as np
import torch
from itertools import cycle
from math import sqrt
from collections import Counter
import random
from tqdm import tqdm
import clip
import os
from torchvision import transforms
# import faiss
import gc


# 找到给定列表test_envs中第一个大于给定值d的元素的索引
# d在test_env前面则减去0，前面没有test_env占着位置，否则减去前面占用test——envs大小
def Nmax(test_envs, d):
    for i in range(len(test_envs)):
        if d < test_envs[i]:
            return i
    return len(test_envs)


def split_meta_train_test(minibatches, num_meta_test=1):
    n_domains = len(minibatches)
    perm = torch.randperm(n_domains).tolist()
    pairs = []
    meta_train = perm[:(n_domains - num_meta_test)]
    meta_test = perm[-num_meta_test:]

    for i, j in zip(meta_train, cycle(meta_test)):
        xi, yi = minibatches[i][0], minibatches[i][1]
        xj, yj = minibatches[j][0], minibatches[j][1]

        min_n = min(len(xi), len(xj))
        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs


def random_pairs_of_minibatches_by_domainperm(minibatches):
    perm = torch.randperm(len(minibatches)).tolist()  # domain id

    pairs = []

    for i in range(len(minibatches)):
        j = i + 1 if i < (len(minibatches) - 1) else 0
        xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1]
        xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]

        min_n = min(len(xi), len(xj))

        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs


def random_pairs_of_minibatches_by_domainperm1(minibatches):
    pairs = []
    perm1 = torch.randperm(len(minibatches)).tolist()
    for i in range(0, len(minibatches), 2):
        j = random.randint(0, len(minibatches) - 1)
        idx = torch.randperm(len(minibatches[j][0]))

        xi, yi = minibatches[j][0][idx], minibatches[j][1][idx]
        xj, yj = minibatches[perm1[i]][0], minibatches[perm1[i]][1]

        min_n = min(len(xi), len(xj))

        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs


def random_pairs_of_minibatches(args, minibatches):
    ld = len(minibatches)
    pairs = []
    tdlist = np.arange(ld)  # minibatch 一个domain
    txlist = np.arange(args.batch_size)  # 一个domain中的数据
    for i in range(ld):
        for j in range(args.batch_size):
            (tdi, tdj), (txi, txj) = np.random.choice(tdlist, 2,
                                                      replace=False), np.random.choice(txlist, 2,
                                                                                       replace=True)  # 取不同域的数据
            if j == 0:
                xi, yi, di = torch.unsqueeze(
                    minibatches[tdi][0][txi], dim=0), minibatches[tdi][1][txi], minibatches[tdi][2][txi]
                xj, yj, dj = torch.unsqueeze(
                    minibatches[tdj][0][txj], dim=0), minibatches[tdj][1][txj], minibatches[tdj][2][txj]
            else:
                xi, yi, di = torch.vstack((xi, torch.unsqueeze(minibatches[tdi][0][txi], dim=0))), torch.hstack(
                    (yi, minibatches[tdi][1][txi])), torch.hstack((di, minibatches[tdi][2][txi]))
                xj, yj, dj = torch.vstack((xj, torch.unsqueeze(minibatches[tdj][0][txj], dim=0))), torch.hstack(
                    (yj, minibatches[tdj][1][txj])), torch.hstack((dj, minibatches[tdj][2][txj]))
        pairs.append(((xi, yi, di), (xj, yj, dj)))
        # x: data
    return pairs


def colorful_spectrum_mix(img1, img2, uniform=1, ratio=1.0, ):
    """Input image size: ndarray of [H, W, C]"""
    lam = np.random.uniform(0, uniform)

    assert img1.shape == img2.shape
    b, c, h, w = img1.shape
    h_crop = int(h * sqrt(ratio))
    w_crop = int(w * sqrt(ratio))
    h_start = h // 2 - h_crop // 2  # 0
    w_start = w // 2 - w_crop // 2  # 0

    img1_fft = torch.fft.fft2(img1, dim=(-2, -1))
    img2_fft = torch.fft.fft2(img2, dim=(-2, -1))
    img1_abs, img1_pha = torch.abs(img1_fft), torch.angle(img1_fft)
    img2_abs, img2_pha = torch.abs(img2_fft), torch.angle(img2_fft)

    # img1_abs = torch.fft.fftshift(img1_abs, dim=(-2, -1))
    # img2_abs = torch.fft.fftshift(img2_abs, dim=(-2, -1))

    img1_abs_ = torch.clone(img1_abs)
    img2_abs_ = torch.clone(img2_abs)
    img1_abs[:, :, h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img2_abs_[:, :, h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img1_abs_[:, :,
                                                                                                h_start:h_start + h_crop,
                                                                                                w_start:w_start + w_crop]
    img2_abs[:, :, h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img1_abs_[:, :, h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img2_abs_[:, :,
                                                                                                h_start:h_start + h_crop,
                                                                                                w_start:w_start + w_crop]

    # img1_abs = torch.fft.ifftshift(img1_abs, dim=(-2, -1))
    # img2_abs = torch.fft.ifftshift(img2_abs, dim=(-2, -1))

    img21 = img1_abs * (torch.exp(1j * img1_pha))
    img12 = img2_abs * (torch.exp(1j * img2_pha))
    img21 = torch.real(torch.fft.ifftn(img21, dim=(-2, -1)))
    img12 = torch.real(torch.fft.ifftn(img12, dim=(-2, -1)))
    # img21 = np.uint8(np.clip(img21, 0, 255))
    # img12 = np.uint8(np.clip(img12, 0, 255))

    return img21, img12


def make_weights_for_balanced_classes(dataset):
    counts = Counter()
    classes = []
    for y in dataset:
        y = int(y)
        counts[y] += 1
        classes.append(y)

    n_classes = len(counts)

    weight_per_class = {}
    for y in counts:
        weight_per_class[y] = 1 / (counts[y] * n_classes)

    weights = torch.zeros(len(dataset))
    for i, y in enumerate(classes):
        weights[i] = weight_per_class[int(y)]

    return weights


def get_features(loaders, model, device):
    model.eval()  # very importance
    all_features = []
    all_labels = []
    all_domains = []

    with torch.no_grad():
        for i, loader in enumerate(loaders):
            features_list = []
            labels_list = []
            domains_list = []

            for data in tqdm(loader):
                images, labels, domains = data[0].to(device), data[1].to(device), data[2].to(device)

                # Encode images and move features to CPU immediately to save GPU memory
                features = model.encode_image(images).cpu()
                features_list.append(features)
                labels_list.append(labels.cpu())
                domains_list.append(domains.cpu())

                # Clean up GPU memory
                del images, labels, domains, features
                torch.cuda.empty_cache()
                gc.collect()

            # Concatenate all features, labels, and domains
            all_features.append(torch.cat(features_list).numpy())
            all_labels.append(torch.cat(labels_list).numpy())
            all_domains.append(torch.cat(domains_list).numpy())

            # Clean up list variables to save memory
            del features_list, labels_list, domains_list
            torch.cuda.empty_cache()
            gc.collect()

    return all_features, all_labels, all_domains


# def generate_augmented_features(loaders, model, device, train_id, num_augmentations=5):
#     model.eval()
#     all_features = []
#     all_labels = []
#     all_domains = []
#
#     # Use no_grad() for inference
#     with torch.no_grad():
#         for i, loader in enumerate(loaders):
#             features_list = []
#             labels_list = []
#             domains_list = []
#
#             for data in tqdm(loader):
#                 images, labels, domains = data[0], data[1], data[2]
#
#                 if i in train_id:
#                     # Perform multiple augmentations and compute features
#                     augmented_features = []
#                     for _ in range(num_augmentations):
#                         augmented_images = torch.stack([train_transform(img) for img in images])
#                         features = model.encode_image(augmented_images.to(device))
#                         augmented_features.append(features.cpu())
#                         # Clean up augmented images to save memory
#                         del augmented_images
#
#                     # Average the augmented features
#                     mean_features = torch.mean(torch.stack(augmented_features), dim=0)
#                     features_list.append(mean_features.cpu())
#
#                     # Clean up augmented features to save memory
#                     # del augmented_features, mean_features
#                 else:
#                     # Directly compute features without augmentation
#                     features = model.encode_image(images.to(device))
#                     features_list.append(features.cpu())
#
#                 labels_list.append(labels.cpu())
#                 domains_list.append(domains.cpu())
#
#                 # Clean up images, labels, domains to save memory
#                 del images, labels, domains, features
#                 gc.collect()
#
#             # Concatenate and store features, labels, and domains
#             all_features.append(torch.cat(features_list).numpy())
#             all_labels.append(torch.cat(labels_list).numpy())
#             all_domains.append(torch.cat(domains_list).numpy())
#
#             # Clean up lists to save memory
#             del features_list, labels_list, domains_list
#             gc.collect()
#
#     return all_features, all_labels, all_domains

import datautil.imgdata.util as imgutil


def generate_augmented_features(args, loaders, model, device, train_id, num_augmentations=1):
    model.eval()
    all_features = []
    all_labels = []
    all_domains = []
    train_transform = imgutil.image_augment(args=args)
    # Use no_grad() for inference
    with torch.no_grad():
        for i, loader in enumerate(loaders):
            features_list = []
            labels_list = []
            domains_list = []

            for data in tqdm(loader):
                images, labels, domains = data[0], data[1], data[2]

                if i in train_id:
                    # Perform multiple augmentations and compute features
                    augmented_features = []
                    for _ in range(num_augmentations):
                        augmented_images = torch.stack([train_transform(img) for img in images])
                        features = model.encode_image(augmented_images.to(device))
                        features_list.append(features.cpu())
                        # = augmented_features
                        labels_list.append(labels.cpu())
                        domains_list.append(domains.cpu())
                        # Clean up augmented images to save memory
                        del augmented_images

                    # Average the augmented features
                    # mean_features = torch.mean(torch.stack(augmented_features), dim=0)
                    # features_list.append(mean_features.cpu())

                    # Clean up augmented features to save memory
                    # del augmented_features, mean_features
                else:
                    # Directly compute features without augmentation
                    features = model.encode_image(images.to(device))
                    features_list.append(features.cpu())

                    labels_list.append(labels.cpu())
                    domains_list.append(domains.cpu())

                # Clean up images, labels, domains to save memory

                del images, labels, domains, features
                gc.collect()

            # Concatenate and store features, labels, and domains
            all_features.append(torch.cat(features_list).numpy())
            all_labels.append(torch.cat(labels_list).numpy())
            all_domains.append(torch.cat(domains_list).numpy())

            # Clean up lists to save memory
            del features_list, labels_list, domains_list
            gc.collect()

    return all_features, all_labels, all_domains


def load_embeddings(cache_file):
    """
    Loads the embeddings from a file
    """
    save_dict = torch.load(cache_file)

    all_features, all_labels, all_domains = save_dict['all_features'], \
        save_dict['all_labels'], save_dict['all_domains']
    return all_features, all_labels, all_domains


def get_domain_text_embs(model, args, text_prompts):
    """
    Gets the text embeddings of the prompts describing the source and target domains.
    """
    # print(text_prompts)
    all_texts = []
    for i, t in enumerate(text_prompts):
        # if i not in args.test_envs:
        texts = [[t.format(c)] for c in args.class_names]
        # 提取每个类别的文本embedding 特征
        text_emb = zeroshot_classifier(model, texts).permute(2, 1, 0)  # 可以用于当作分类器的权重
        # print("text_emb:", text_emb.shape)  # [emb_dim, num_classes, 1 domain]
        all_texts.append(text_emb)
    text_pairs = torch.cat(all_texts, dim=-1)  # 在 domain 维度上拼接
    text_pairs = text_pairs.permute(2, 1, 0)
    # print("text pairs", text_pairs.shape)

    mask = torch.ones(len(text_pairs), dtype=torch.bool)
    indices = args.test_envs
    mask[indices] = 0

    source_embeddings = text_pairs[mask]
    target_embeddings = text_pairs[indices]
    # print("source embeddings", source_embeddings.shape)  # [num_domain, num_classes, emb_size]
    # print("target embeddings", target_embeddings.shape)
    # source_embeddings = text_pairs
    return source_embeddings, target_embeddings


def zeroshot_classifier(model, prompts):
    """ Computes CLIP text embeddings for a list of prompts."""
    #  return (1 domain,num_classes,emb_size)
    model.eval()
    assert type(prompts[0]) == list, "prompts must be a list of lists"
    with torch.no_grad():
        zeroshot_weights = []
        for texts in prompts:
            texts = clip.tokenize(texts).cuda()  # tokenize
            class_embedding = model.encode_text(texts)  # embed with text encoder
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
    return zeroshot_weights


def zeroshot_classifier1(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embedding = class_embeddings.mean(dim=0)
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights


def clip_classifier(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights


def save_checkpoint(args, model):
    checkpoint_dir = os.path.join(args.output, "checkpoint")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    path = f'{args.output}/checkpoint/model_best_{args.seed}.pth'
    print(f'Saving checkpoint...')
    state = {
        "net": model.state_dict()
    }
    torch.save(state, path)
    return path


def load_checkpoint(args, model):
    path = f'{args.output}/checkpoint/model_best_{args.seed}.pth'
    print(f"==> loading checkpoint {path}")
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['net'])
    return model


def questions_template_pool():
    questions = ['Question: What category and style in this image?',
                 'Question: What category and style is depicted in this image?',
                 'Question: Describe the category and style of this image, please.',
                 'Question: What category and style are represented in this image?',
                 'Question: How would you describe the composition of the image?',
                 'Question: What category does this image belong to?',
                 'Question: Is there any specific category or genre that this image belongs to?',
                 'Question: Can you identify the class and style of this image?',
                 'Question: How would you characterize the category and style of this image?',
                 ]

    question = random.choice(questions)
    return question


# def answers_rtemplate_pool():
#     answers = ['Answer: A {} style of a {}.',
#                'Answer: A {} style of an image {}.',
#                'Answer: A {} style of a subject of {}.',
#                'Answer: A stock photo depicts a {} style of {}.',
#                'Answer: This image depicts a {} style of a subject of {}.',
#                'Answer: Yes, the image contains a {} style of a {}.',
#                'Answer: Certainly, it\'s a {} design showcased in a {} style.',
#                'Answer: This image is in the {} category, displaying a {} style.',
#                'Answer: This illustration captures a {} setting with a {} style.',
#                'Answer: Of course, it\'s a {} depicted in a {} style.',
#                'Answer: Certainly, it\'s a {} design displayed in a {} style.',
#                ]
#     answer = random.choice(answers)
#     return answer


def answers_template_pool(style):
    answer = fill_rstyle(answers_rtemplate_pool(), style)
    return answer


# def answers_template_pool_terra(style):
#     answer = fill_fstyle(answers_ftemplate_pool(), style)
#     return answer

def answers_rtemplate_pool():
    answers = ['A {} style of a {}.',
               'A {} style of a subject of {}.',
               'This image depicts a {} style of {}.',
               'This image depicts a {} style of a subject of {}.',
               'The image contains a {} style of a {}.',
               'It\'s a {} design showcased in a {} style.',
               'This image is in the {} category, displaying a {} style.',
               'This illustration captures a {} image with a {} style.',
               'It\'s a {} rendered in a {} style.',
               'It\'s a {} design displayed in a {} style.',
               # 'A {} style capturing {}.',
               # 'This image shows a {} style of a {}.',
               # 'An illustration in a {} style of a {}.',
               # 'The image is rendered in a {} style of a {}.',
               ]
    answer = random.choice(answers)
    return answer


# def answers_ftemplate_pool():
#     answers = ['A {} angle of a {}.',
#                'A {} angle  of a subject of {}.',
#                'This image depicts a {} angle  of {}.',
#                'This image depicts a {} angle  of a subject of {}.',
#                'The image contains a {} angle  of a {}.',
#                'It\'s a {} design showcased in a {} angle.',
#                'This image is in the {} category, displaying a {} angle.',
#                'This illustration captures a {} image with a {} angle.',
#                'It\'s a {} rendered in a {} angle.',
#                'It\'s a {} design displayed in a {} angle.',
#                # 'A {} style capturing {}.',
#                # 'This image shows a {} style of a {}.',
#                # 'An illustration in a {} style of a {}.',
#                # 'The image is rendered in a {} style of a {}.',
#                ]
#     answer = random.choice(answers)
#     return answer


def fill_rstyle(sentence, style):
    style_index = sentence.find("style")
    if style_index != -1:
        closest_bracket_index = sentence.rfind("{}", 0, style_index)
        if closest_bracket_index != -1:
            sentence = sentence[:closest_bracket_index] + style.lower().replace("_", " ") + sentence[
                                                                                            closest_bracket_index + 2:]
    return sentence


def fill_fstyle(sentence, style):
    style_index = sentence.find("angle")
    if style_index != -1:
        closest_bracket_index = sentence.rfind("{}", 0, style_index)
        if closest_bracket_index != -1:
            sentence = sentence[:closest_bracket_index] + style.lower().replace("_", " ") + sentence[
                                                                                            closest_bracket_index + 2:]
    return sentence


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def extract_fp_per_class(fx, n, gpu_id, record_mean=True):  # record_mean=True means taking the class mean
    if n == 1:
        fp = torch.mean(fx, dim=0, keepdim=True)
    elif record_mean:
        n -= 1
        fm = torch.mean(fx, dim=0, keepdim=True)
        if n >= len(fx):
            fp = fx
        else:
            fp = kmeans(fx, n, gpu_id)
        fp = torch.cat([fm, fp], dim=0)
    else:
        if n >= len(fx):
            fp = fx
        else:
            fp = kmeans(fx, n, gpu_id)
    return fp


def kmeans(fx, n, gpu_id, metric='euclidean'):
    device = fx.device

    if metric == 'cosine':
        fn = fx / torch.clamp(torch.norm(fx, dim=1, keepdim=True), min=1e-20)
    elif metric == 'euclidean':
        fn = fx
    else:
        raise KeyError(f"Unsupported metric '{metric}'")

    fn = fn.detach().cpu().numpy()

    # # Use FAISS for KMeans clustering
    # d = fn.shape[1]
    # kmeans = faiss.Kmeans(d, n, niter=20, verbose=False)
    #
    # # Get GPU resources for faiss and transfer KMeans to GPU
    # res = faiss.StandardGpuResources()  # Create GPU resource
    # gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, kmeans.index)  # Move to GPU (GPU ID = 0)
    #
    # kmeans.index = gpu_index  # Update kmeans to use GPU

    # Train the model using GPU
    kmeans.train(fn)

    # Directly use the centroids computed by faiss
    centroids = kmeans.centroids
    fp = torch.FloatTensor(centroids).to(device)

    return fp


def style_template():
    templates = [
        "surrealism",
        "minimalist",
        "retro",
        "pixel-art",
        "collage",
        "pointillism",
        "stained-glass",
        "comic-book",
        "illustration",
        "fantasy",
        "landscape",
        "portrait",
        "chiaroscuro",
        # "cubism",
        # "abstract",
        # "art-nouveau",
        # "pop-art",
        # "gothic",
        # "impressionism",
        # "expressionism",
        # "baroque",
        # "futurism",
        # "realism",
        # "surrealist-collage",
        # "neon",
        # "graffiti",
        # "watercolor",
        # "oil-painting",
        # "digital-art",
        # "retro-futurism",
        # "manga",
        # "art-deco",
        # "psychedelic",
        # "steampunk",
        # "victorian",
        # "neo-classical",
        # "fauvism",
        # "sculpture"
    ]
    return templates

import re


def format_prompt(temp: str, cls_name: str):
    if cls_name.lower().startswith("the "):
        temp = re.sub(r"\b(a|the) (?=\{\})", "", temp)
    return temp.format(cls_name)


if __name__ == '__main__':
    sentences = answers_rtemplate_pool()
    for sentence in sentences:
        sentence = fill_fstyle(sentence, "realistic")
        print(sentence)
