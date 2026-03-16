# coding=utf-8
import os
from torch.utils.data import DataLoader
from datautil.imgdata.imgdataload import ImageDataset
import datautil.imgdata.util as imgutil


def get_dataloader_imagenet(args):
    names = args.img_dataset[args.dataset]
    image_train = imgutil.image_train
    image_test = imgutil.image_test

    class_index_path = args.data_dir + 'imagenet_class_index.json'
    # class_index_path = args.data_dir + 'imagenet1k_classes.txt'

    data_dir = args.data_dir
    train_dataset = ImageDataset(args.dataset, args.task, data_dir, names[0] + "/train",
                                 transform=image_train(args),
                                 shots_per_class=args.shots_per_class,
                                 class_index_path=class_index_path)  # few-shot for training
    val_dataset = ImageDataset(args.dataset, args.task, data_dir, names[0] + "/val",
                               transform=image_test(args), class_index_path=class_index_path)

    train_loader = [DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.N_WORKERS,
        drop_last=True,
        shuffle=True
    )]

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=100,
        num_workers=2 * args.N_WORKERS,
        pin_memory=True,
        drop_last=False,
        shuffle=False
    )

    target_loaders = {}
    for target_domain in names[1:]:
        target_dir = os.path.join(args.data_dir, target_domain)
        if os.path.isdir(target_dir):
            target_dataset = ImageDataset(args.dataset, args.task, target_dir, domain_name="",
                                          transform=image_test(args), class_index_path=class_index_path)
            target_loaders[target_domain] = DataLoader(
                dataset=target_dataset,
                batch_size=100,
                num_workers=2 * args.N_WORKERS,
                pin_memory=True,
                drop_last=False,
                shuffle=False
            )

    return train_loader, val_loader, target_loaders
