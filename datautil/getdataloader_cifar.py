# coding=utf-8
import os
from torch.utils.data import DataLoader
from datautil.imgdata.imgdataload import ImageDataset
import datautil.imgdata.util as imgutil


def get_dataloader_cifar(args):
    names = args.img_dataset[args.dataset]
    image_train = imgutil.image_train
    image_test = imgutil.image_test

    train_dataset = ImageDataset(args.dataset, args.task, args.data_dir, names[0] + "/train",
                                 transform=image_train(args))
    val_dataset = ImageDataset(args.dataset, args.task, args.data_dir, names[0] + "/test",
                               transform=image_test(args))

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

    target_dir = args.data_dir + names[1]
    target_loaders = {}
    for corruption in os.listdir(target_dir):
        corruption_path = os.path.join(target_dir, corruption)
        if os.path.isdir(corruption_path):
            target_dataset = ImageDataset(args.dataset, args.task, corruption_path, domain_name="",
                                          transform=image_test(args))
            target_loaders[corruption] = DataLoader(
                dataset=target_dataset,
                batch_size=100,
                num_workers=2 * args.N_WORKERS,
                pin_memory=True,
                drop_last=False,
                shuffle=False
            )

    return train_loader, val_loader, target_loaders
