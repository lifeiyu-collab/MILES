# coding=utf-8
import torch
import time

def accuracy(network, loader):
    correct = 0
    total = 0

    network.eval()
    with torch.no_grad():
        for data in loader:
            x = data[0].cuda().float()
            y = data[1].cuda().long()

            # start_time = time.time()
            p = network.predict(x)
            # end_time = time.time()
            # print(f"Infer time: {end_time - start_time:.3f} seconds")
            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float()).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float()).sum().item()
            total += len(x)
    network.train()
    return correct / total


