import numpy as np
import re
import os


def round_up(value):
    # 替换内置round函数,实现保留2位小数的精确四舍五入
    return round(value * 100) / 100.0


def compute_std(path):
    with open(os.path.join(path, 'done.txt'), mode="a+") as f:  # a+ 追加读，和追加写，a只追加写
        f.seek(0)  # 指针默认在末尾，归为到文件头
        b = []
        lines = f.readlines()
        for line in lines:
            if line.isspace():
                continue
            else:
                if line.startswith('target acc'):
                    line = line.strip()  # 去掉换行符
                    a = float(line.split(":")[-1]) * 100
                    b.append(a)
        # 求均值
        arr_mean = round_up(np.mean(b))
        # 求标准差
        arr_std = round_up(np.std(b, ddof=1))
        f.write("mean_std: %.2f" % arr_mean)
        f.write('+')
        f.write("%.2f\n" % arr_std)
    f.close()

def compute_single_std(path):
    with open(os.path.join(path, 'done.txt'), 'r') as file:  # a+ 追加读，和追加写，a只追加写
        text_data = file.read()

    # 解析文本数据
    env_data = {}
    pattern = r"target (env\d+) seed\d+:(\d\.\d+)"
    matches = re.findall(pattern, text_data)

    # 将数据按环境进行分类
    for env, value in matches:
        if env not in env_data:
            env_data[env] = []
        env_data[env].append(float(value)* 100)

    # 计算均值和方差
    results = {}
    for env, values in env_data.items():
        mean_val = round_up(np.mean(values))  # 转换为百分比
        std_val = round_up(np.std(values, ddof=1))  # 转换为百分比
        results[env] = (mean_val, std_val)

    # 将结果写回文本
    output_text = text_data.strip() + "\n\n"
    for env, (mean_val, std_val) in results.items():
        output_text += f"{env} mean_std: {mean_val:.2f}+{std_val:.2f}\n"

    # 将结果保存回文件
    with open(os.path.join(path, 'done.txt'), 'w') as file:
        file.write(output_text)




if __name__ == '__main__':
        compute_std('/mlspace/MILES/scripts/ImageNet_ViT-B-16/MILES_16shot_new/imagenet-a/')
























