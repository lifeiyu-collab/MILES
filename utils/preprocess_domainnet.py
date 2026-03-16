import os
file = open("./domain_net_duplicates.txt", 'r')
root = r'/mlspace/datasets/DomainNet/'

for line in file.readlines():
        os.remove(os.path.join(root, line.strip()))
