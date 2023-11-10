############################################################################################################
# 相较于简单版本的训练脚本 train_sample 增添了以下功能：
# 1. 使用argparse类实现可以在训练的启动命令中指定超参数
# 2. 可以通过在启动命令中指定 --seed 来固定网络的初始化方式，以达到结果可复现的效果
# 3. 使用了更高级的学习策略 cosine warm up：在训练的第一轮使用一个较小的lr（warm_up），从第二个epoch开始，随训练轮数逐渐减小lr。
# 4. 可以通过在启动命令中指定 --model 来选择使用的模型
# 5. 使用amp包实现半精度训练，在保证准确率的同时尽可能的减小训练成本(比如将小数点精确到后8位)
# 6. 实现了数据加载类的自定义实现
# 7. 可以通过在启动命令中指定 --tensorboard 来进行tensorboard可视化, 默认不启用。
#    注意，使用tensorboad之前需要使用命令 "tensorboard --logdir= log_path"来启动，结果通过网页 http://localhost:6006/'查看可视化结果
############################################################################################################
# --model 可选的超参如下：
# alexnet   zfnet   vgg   vgg_tiny   vgg_small   vgg_big   googlenet   xception   resnet_small   resnet   resnet_big   resnext   resnext_big
# densenet_tiny   densenet_small   densenet   densenet_big   mobilenet_v3   mobilenet_v3_large   shufflenet_small   shufflenet
# efficient_v2_small   efficient_v2   efficient_v2_large   convnext_tiny   convnext_small   convnext   convnext_big   convnext_huge
# vision_transformer_small   vision_transformer   vision_transformer_big   swin_transformer_tiny   swin_transformer_small   swin_transformer


############################################################################################################
# 训练命令示例： # python train.py --model alexnet --num_classes 5


# cd Based_CNN_Model
# conda activate pytorch_gpu
# python train.py --model xception --batch_size 8 --use_amp --tensorboard
# python train.py --model alexnet --batch_size 16  --use_amp
# python train.py --model resnet --batch_size 32  --use_amp --tensorboard


############################################################################################################


import os
import argparse
import math
import shutil
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
from torchsummary import summary

import models
from utils.lr_methods import warmup
from dataload.dataload_brain_mri import Brain_MRI_Load
from utils.train_engin import train_one_epoch, evaluate, evaluate_val, tensor_to_list
from utils.mri_preprocess import random_shift_3d, random_lr_flip_3d, silces_2d_extraction, numpy_to_tensor, silces_2_5d_extraction
from utils.compute_statistic import compute_mean_std
from utils.params_analysis import parse_value
import scipy.stats as stats


def tensor_to_list(tensor):
    if tensor.numel() == 1:  # 检查是否是标量
        return [tensor.item()]  # 返回一个包含单个元素的列表
    return tensor.tolist()  # 将其他 Tensor 转换为列表


parser = argparse.ArgumentParser()


parser.add_argument('--batch_size', type=int, default=8,
                    help='batch_size for training')
parser.add_argument('--lr', type=float, default=0.0002,
                    help='star learning rate')
parser.add_argument('--lrf', type=float, default=0.0001,
                    help='end learning rate')  # 预热学习率
parser.add_argument('--seed', default=False, action='store_true',
                    help='fix the initialization of parameters')
parser.add_argument('--tensorboard', default=True,
                    action='store_true', help=' use tensorboard for visualization')
parser.add_argument('--use_amp', default=False, action='store_true',
                    help=' training with mixed precision')  # 半精度
parser.add_argument('--data_path', type=str,
                    default=r"H:\Preprocess_MRI")  # 数据路径
parser.add_argument('--model', type=str, default="resnet",
                    help=' select a model for training')
parser.add_argument('--model_wegihts_path', type=str, default=r"E:\BaiduSyncdisk\paper\software\pytorch\brainage_project\results\weights\resnext2d\resnext2d.pth",
                    help='The storage address of the model weight')
parser.add_argument('--device', default='cuda',
                    help='device id (i.e. 0 or 0,1 or cpu)')
parser.add_argument('--dataset_used', type=list,
                    default=['OASIS-3', 'SALD', 'SLIM'], help='val dataset used')
parser.add_argument('--mix_train_val', type=bool, default=True,
                    help='training set and the validation set use the same data set but different data')
parser.add_argument('--val_scale', type=float, default=0.2,
                    help='The ratio of the verification set to the total data')
parser.add_argument('--channels', type=int, default=7,
                    help='Parameters of a 2d network')
parser.add_argument('--uncertainty', type=bool, default=False,
                    help='Whether to predict uncertainty')
parser.add_argument('--gp_use', type=bool, default=False,
                    help='Must work with gp network')
parser.add_argument('--kflod', type=int, default=10,
                    help='The number of folds of cross-validation')
parser.add_argument('--mean', type=parse_value, default=None,
                    help='Computed  mean of brain image in advance(Will automatically calculate if inputs is None)')
parser.add_argument('--std', type=parse_value, default=None,
                    help='Computed standard deviation of brain image in advance(Will automatically calculate if inputs is None)')
parser.add_argument('--mc_dropout', type=bool, default=None,
                    help='Use the Monte Carlo Dropout method')
parser.add_argument('--epc_ale_ratio', type=float,
                    default=False, help='Use the Monte Carlo Dropout method')
parser.add_argument('--gp_cov_momentum', type=float, default=0.99,
                    help='This parameter exclusively affects GP networks.')
parser.add_argument('--gp_cov_ridge_penalty', type=float, default=1e-3,
                    help='This parameter exclusively affects GP networks.')


opt = parser.parse_args()  # 定义好的超参数进行实例化


if opt.seed:
    def seed_torch(seed=7):
        random.seed(seed)  # Python random module.
        os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
        np.random.seed(seed)  # Numpy module.
        torch.manual_seed(seed)  # 为CPU设置随机种子
        torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        # 设置cuDNN：cudnn中对卷积操作进行了优化，牺牲了精度来换取计算效率。如果需要保证可重复性，可以使用如下设置:
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True
        # 实际上这个设置对精度影响不大，仅仅是小数点后几位的差别。所以如果不是对精度要求极高，其实不太建议修改，因为会使计算效率降低。
        print('random seed has been fixed')
    seed_torch()


def kflod_spilt(remaining_numbers, num_folds):
    # 随机打乱 remaining_numbers 列表
    random.shuffle(remaining_numbers)
    fold_size = len(remaining_numbers) // num_folds
    train_num = []
    test_num = []

    # 创建交叉验证的训练集和测试集
    for fold in range(num_folds):
        start = fold * fold_size
        end = (fold + 1) * fold_size

        # 提取测试数据
        test_data = remaining_numbers[start:end]

        # 提取训练数据，将测试数据从 remaining_numbers 中移除
        train_data = [x for x in remaining_numbers if x not in test_data]

        train_num.append(train_data)
        test_num.append(test_data)
    return train_num, test_num


def main(args):  # 传入参数
    # create model

    if args.uncertainty:  # When uncertainty is set manually, the actual output is 2
        output = 2
    else:
        output = 1

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(args)
    if args.gp_use:
        network_type, _ = network_type, model = models.find_model_using_name(
            args.model, args.channels, output, dropout=args.mc_dropout, gp_cov_momentum=args.gp_cov_momentum, gp_cov_ridge_penalty=args.gp_cov_ridge_penalty)
    else:
        network_type, _ = models.find_model_using_name(
            opt.model, opt.channels, output, dropout=args.mc_dropout)
    #

    if network_type == '3d':
        if args.mean == None or args.std == None:
            mean, std = compute_mean_std(args.data_path, network_type='3d')
        elif isinstance(args.mean, float) and isinstance(args.std, float):
            mean = args.mean
            std = args.std
        else:
            raise ValueError("3D Network input mean or std must be a float")

        data_transform = {  # 定义数据的预处理
            "test": transforms.Compose([
                transforms.Lambda(lambda x: numpy_to_tensor(x)),
                transforms.Lambda(lambda x: (x - mean) / std)])}

    elif network_type == '2d-slices':

        if args.mean == None or args.std == None:
            mean, std = compute_mean_std(
                args.data_path, network_type='2d-slices', channel=args.channels)
        elif isinstance(args.mean, list) and isinstance(args.std, list):
            mean = args.mean
            std = args.std
        else:
            raise ValueError("2D Network input mean or std must be a list")

        data_transform = {  # 定义数据的预处理
            "test": transforms.Compose([
                transforms.Lambda(lambda x: silces_2d_extraction(
                    x, channel=args.channels)),
                transforms.Lambda(lambda x: numpy_to_tensor(x)),
                transforms.Normalize(mean, std)])}

    elif network_type == '2.5d':
        if args.mean == None or args.std == None:
            mean, std = compute_mean_std(
                args.data_path, network_type='2.5d')  # 2.5d只有9个channel
        elif isinstance(args.mean, list) and isinstance(args.std, list):
            mean = args.mean
            std = args.std
        else:
            raise ValueError("2.5D Network input mean or std must be a list")
        data_transform = {  # 定义数据的预处理
            "test": transforms.Compose([
                transforms.Lambda(lambda x: silces_2_5d_extraction(x)),
                transforms.Lambda(lambda x: numpy_to_tensor(x)),
                transforms.Normalize(mean, std)])}
    nw = 0
    # nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers

    print('Using {} dataloader workers every process'.format(nw))

    if args.mix_train_val:
        data_len = len(Brain_MRI_Load(
            args.data_path, args.train_dataset_used + args.val_dataset_used, print_result=False))
        # 先读取一次数据取得数据集大小
        all_numbers = list(range(data_len))
        step = max(1, int(1 / args.val_scale))
        selected_numbers = all_numbers[::step]  # Independent verification set

        val_dataset = Brain_MRI_Load(
            args.data_path, args.train_dataset_used + args.val_dataset_used,
            transform=data_transform["test"], data_filter=selected_numbers, print_result=False)  # 自定义加载类

    else:
        data_len = len(Brain_MRI_Load(
            args.data_path, args.train_dataset_used, print_result=False))
        all_numbers = list(range(data_len))
        train_numbers, test_numbers = kflod_spilt(all_numbers, args.kflod)
        val_dataset = Brain_MRI_Load(
            args.data_path, args.val_dataset_used, transform=data_transform["test"], print_result=False)  # 自定义加载类

    print('The test data size is:', len(test_dataset))
    test_dataset = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=nw, collate_fn=test_dataset.collate_fn)

    all_mean = []
    all_var = []

    for k in range(args.kflod):
        print('training', str(k+1), 'models')

        if args.gp_use:
            _, model = models.find_model_using_name(args.model, args.channels, output, dropout=args.mc_dropout,
                                                    gp_cov_momentum=args.gp_cov_momentum, gp_cov_ridge_penalty=args.gp_cov_ridge_penalty)
        else:
            _, model = models.find_model_using_name(
                opt.model, opt.channels, output, dropout=args.mc_dropout)
        model.to(device)
        mdoel_path = os.path.join(args.model_wegihts_path, str(k+1) + 'flod')
        model.load_state_dict(torch.load(mdoel_path))
        model.eval()
        if k == 0:
            if network_type == '3d':
                # input_size是数据的维度,要增加一个灰度维度
                summary(model, input_size=(1,) + train_dataset[0][0].shape)
            elif network_type == '2d-slices' and network_type == '2.5d':
                # print(train_dataset[0][0],train_dataset[0][0].shape)
                summary(model, train_dataset[0][0].shape)

        if output == 2 or args.gp_use or args.mc_dropout:
            mean_list, var_list = evaluate_val(model=model, data_loader=val_dataset, device=device,
                                               network_type=network_type, output=output,
                                               gp_use=args.gp_use, mc_dropout=args.mc_dropout)
            all_mean.append(mean_list)
            all_var.append(var_list)
        else:
            mean_list = evaluate_val(model=model, data_loader=val_dataset, device=device,
                                     network_type=network_type, output=output,
                                     gp_use=args.gp_use, mc_dropout=args.mc_dropout)
            all_mean.append(mean_list)

    labels_list = []
    for step, data in enumerate(val_dataset):
        _, labels = data
        labels_list = labels_list + tensor_to_list(labels)

    if output == 2 or args.gp_use or args.mc_dropout:
        mean_pred = [sum(col) / len(col) for col in zip(*all_mean)]
        var_pred = [sum(col) / len(col) for col in zip(*all_var)]

        mae_result = [abs(a - b) for a, b in zip(mean_pred, labels_list)]
        cvmae = sum(mae_result) / len(mae_result)
        correlation_coefficient, p_value = stats.pearsonr(mean_pred, var_pred)

        print('cv mae: ', cvmae)
        print('coefficient: ', correlation_coefficient)
        print('p_value ', p_value)
    else:
        mean_pred = [sum(col) / len(col) for col in zip(*all_mean)]
        mae_result = [abs(a - b) for a, b in zip(mean_pred, labels_list)]

        cvmae = sum(mae_result) / len(mae_result)
        print('cv mae: ', cvmae)


if __name__ == '__main__':
    main(opt)
