# import numpy as np
# import os
# from PIL import Image
#
# # CIFAR-10 类标签对应的名称（根据 CIFAR-10 的标准标签）
# class_names = [
#     'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
# ]
#
# # Path to the directory where the .npy files are located
# npy_directory = '/mnt/d/datasets/CIFAR-10-C'
#
# # Output to the same directory as input
# output_directory = npy_directory  #
#
# # Load the labels from labels.npy
# labels = np.load(os.path.join(npy_directory, 'labels.npy'))
#
# # List of .npy files to process
# npy_files = [
#     'frost.npy', 'gaussian_blur.npy', 'gaussian_noise.npy', 'glass_blur.npy',
#     'impulse_noise.npy', 'jpeg_compression.npy', 'motion_blur.npy', 'pixelate.npy',
#     'saturate.npy', 'shot_noise.npy', 'snow.npy', 'spatter.npy',
#     'speckle_noise.npy', 'zoom_blur.npy'
# ]
#
# # Process each .npy file
# for npy_file in npy_files:
#     # Load the numpy array of images
#     npy_path = os.path.join(npy_directory, npy_file)
#     images = np.load(npy_path)
#
#     # Create a folder for each type of distortion (e.g., 'fog', 'gaussian_noise', etc.)
#     distortion_dir = os.path.join(output_directory, npy_file.split('.')[0])
#     os.makedirs(distortion_dir, exist_ok=True)
#
#     # Save each image into the appropriate label folder
#     for idx, (img, label) in enumerate(zip(images, labels)):
#         # Convert the numpy array to a PIL image
#         img = Image.fromarray(img.astype('uint8'))
#
#         # Create a directory for the class label if it doesn't exist
#         class_dir = os.path.join(distortion_dir, class_names[label])
#         os.makedirs(class_dir, exist_ok=True)
#
#         # Save the image to the appropriate class directory
#         img.save(os.path.join(class_dir, f"{idx:04d}.png"))  # Save image with zero-padded index
#
#     # Delete the original .npy file after processing
#     os.remove(npy_path)
#
#     print(f"Processed and removed: {npy_file}")


# import numpy as np
# import os
# from PIL import Image
#
# # # CIFAR-100 类标签对应的名称（CIFAR-100 的标准标签）
# classnames = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl',
#              'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair',
#              'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin',
#              'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp',
#              'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
#              'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck',
#              'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road',
#              'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
#              'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger',
#              'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
#              'worm']
#
# # Path to the directory where the .npy files are located
# npy_directory = '/mnt/d/datasets/CIFAR-100-C'  # 修改为你数据集的实际路径
#
# # Output to the same directory as input
# output_directory = npy_directory  # 可以修改为其他路径，如果需要
#
# # Load the labels from labels.npy
# labels = np.load(os.path.join(npy_directory, 'labels.npy'))
#
# # List of .npy files to process
# npy_files = [
#     'brightness.npy', 'contrast.npy', 'defocus_blur.npy', 'elastic_transform.npy',
#     'fog.npy', 'frost.npy', 'gaussian_blur.npy', 'gaussian_noise.npy', 'glass_blur.npy',
#     'impulse_noise.npy', 'jpeg_compression.npy', 'motion_blur.npy', 'pixelate.npy',
#     'saturate.npy', 'shot_noise.npy', 'snow.npy', 'spatter.npy', 'speckle_noise.npy',
#     'zoom_blur.npy'
# ]
#
# # Process each .npy file
# for npy_file in npy_files:
#     # Load the numpy array of images
#     npy_path = os.path.join(npy_directory, npy_file)
#     try:
#         images = np.load(npy_path)
#     except Exception as e:
#         print(f"Error loading {npy_file}: {e}")
#         continue  # Skip the current file and move to the next one
#
#     # Create a folder for each type of distortion (e.g., 'fog', 'gaussian_noise', etc.)
#     distortion_dir = os.path.join(output_directory, npy_file.split('.')[0])
#     os.makedirs(distortion_dir, exist_ok=True)
#
#     # Save each image into the appropriate label folder
#     for idx, (img, label) in enumerate(zip(images, labels)):
#         try:
#             # Convert the numpy array to a PIL image
#             img = Image.fromarray(img.astype('uint8'))
#
#             # Create a directory for the class label if it doesn't exist
#             class_dir = os.path.join(distortion_dir, classnames[label])
#             os.makedirs(class_dir, exist_ok=True)
#
#             # Save the image to the appropriate class directory
#             img.save(os.path.join(class_dir, f"{idx:04d}.png"))  # Save image with zero-padded index
#
#         except Exception as e:
#             print(f"Error processing image {idx} in {npy_file}: {e}")
#             continue  # Skip the current image and continue with the next one
#
#     # Optionally, delete the original .npy file after processing
#     # os.remove(npy_path)
#
#     print(f"Processed: {npy_file}")

# import torchvision.datasets as datasets
#
# cifar100 = datasets.CIFAR100(root='/mnt/d/datasets/', train=True, download=True)
# # 获取超类名称（粗标签）
# coarse_classes = cifar100.coarse_classes
# # 获取子类名称（细标签）
# fine_classes = cifar100.fine_classes
#
# print("超类（20个）:", coarse_classes)
# print("子类（100个）:", fine_classes)


import os
from PIL import Image
import pickle

# 解码 CIFAR-100 数据集
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# 创建文件夹
def create_class_folders(base_dir, class_names):
    for class_name in class_names:
        # 解码字节字符串为普通字符串
        class_name = class_name.decode('utf-8')
        class_dir = os.path.join(base_dir, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

# 保存图片到文件夹
def save_image(data, label, class_names, folder_path, idx):
    img = data.reshape(3, 32, 32).transpose(1, 2, 0)  # 转换为 32x32 RGB 图片
    img = Image.fromarray(img)
    label_name = class_names[label].decode('utf-8')  # 解码为普通字符串
    class_folder = os.path.join(folder_path, label_name)
    if not os.path.exists(class_folder):
        os.makedirs(class_folder)
    img.save(os.path.join(class_folder, f'{idx:04d}.png'))

# 将数据集转换为 /train/class_name/图片 和 /test/class_name/图片 格式
def convert_cifar100(input_path, output_dir):
    # 加载 CIFAR-100 数据集
    meta = unpickle(os.path.join(input_path, 'meta'))
    class_names = meta[b'fine_label_names']  # 获取类别名称

    # 加载训练和测试数据
    train_data = unpickle(os.path.join(input_path, 'train'))
    test_data = unpickle(os.path.join(input_path, 'test'))

    # 创建文件夹
    create_class_folders(os.path.join(output_dir, 'train'), class_names)
    create_class_folders(os.path.join(output_dir, 'test'), class_names)

    # 保存训练集图片
    for idx, (data, label) in enumerate(zip(train_data[b'data'], train_data[b'fine_labels'])):
        save_image(data, label, class_names, os.path.join(output_dir, 'train'), idx)

    # 保存测试集图片
    for idx, (data, label) in enumerate(zip(test_data[b'data'], test_data[b'fine_labels'])):
        save_image(data, label, class_names, os.path.join(output_dir, 'test'), idx)

# 使用示例
input_path = '/mnt/d/datasets/cifar-100-python'  # CIFAR-100 数据集路径
output_dir = '/mnt/d/datasets/CIFAR-100'   # 输出路径
convert_cifar100(input_path, output_dir)

print("complete!")


