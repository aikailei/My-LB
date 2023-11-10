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
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
from torchsummary import summary

import models
from utils.lr_methods import warmup
from dataload.dataload_brain_mri import Brain_MRI_Load
from utils.train_engin import train_one_epoch, evaluate
from utils.mri_preprocess import random_shift_3d, random_lr_flip_3d, silces_2d_extraction, numpy_to_tensor, silces_2_5d_extraction
from utils.compute_statistic import compute_mean_std
from utils.params_analysis import parse_value


parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=8,
                    help='batch_size for training')
parser.add_argument('--lr', type=float, default=0.0002,
                    help='star learning rate')
parser.add_argument('--lrf', type=float, default=0.0001,
                    help='end learning rate')  # 预热学习率
parser.add_argument('--seed', default=True, action='store_true',
                    help='fix the initialization of parameters')
parser.add_argument('--tensorboard', default=True,
                    action='store_true', help=' use tensorboard for visualization')
parser.add_argument('--use_amp', default=False, action='store_true',
                    help=' training with mixed precision')  # 半精度
parser.add_argument('--data_path', type=str,
                    default=r"H:\Preprocess_MRI")  # 数据路径
parser.add_argument('--model', type=str, default="resnet",
                    help=' select a model for testing')
parser.add_argument('--model_wegihts_path', type=str, default=r"E:\BaiduSyncdisk\paper\software\pytorch\brainage_project\results\weights\resnext2d\resnext2d.pth",
                    help='The storage address of the model weight')
parser.add_argument('--device', default='cuda',
                    help='device id (i.e. 0 or 0,1 or cpu)')
parser.add_argument('--dataset_used', type=list, default=['Beijing_Enhanced', 'DLBS', 'CORR', 'OASIS-3', 'SLIM',
                    'CCNP_CKG', 'CCNP_PEK', 'IXI', 'narratives', 'ABIDE-1', 'ABIDE-2', 'SALD'], help='test dataset used')
parser.add_argument('--mix_train_val', type=bool, default=False,
                    help='training set and the validation set use the same data set but different data')
parser.add_argument('--val_scale', type=float, default=0.2,
                    help='The ratio of the verification set to the total data')
parser.add_argument('--channels', type=int, default=9,
                    help='Parameters of a 2d network')
parser.add_argument('--uncertainty', type=bool, default=False,
                    help='Whether to predict uncertainty')
parser.add_argument('--gp_use', type=bool, default=False,
                    help='Must work with gp network')
parser.add_argument('--mean', type=parse_value, default=None,
                    help='Computed  mean of brain image in advance(Will automatically calculate if inputs is None)')
parser.add_argument('--std', type=parse_value, default=None,
                    help='Computed standard deviation of brain image in advance(Will automatically calculate if inputs is None)')
parser.add_argument('--mc_dropout', type=bool, default=None,
                    help='Use the Monte Carlo Dropout method')
parser.add_argument('--epc_ale_ratio', type=float,
                    default=False, help='Use the Monte Carlo Dropout method')


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


def main(args):  # 传入参数
    # create model
    if args.uncertainty == True:  # The network outputs two results directly to estimate the epistemic uncertainty
        output = 2
    else:
        output = 1

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(args)

    if args.gp_use:
        network_type, model = models.find_model_using_name(args.model, args.channels, output,
                                                           dropout=args.mc_dropout, gp_cov_momentum=args.gp_cov_momentum, gp_cov_ridge_penalty=args.gp_cov_ridge_penalty)
    else:
        network_type, model = models.find_model_using_name(
            args.model, args.channels, output, dropout=args.mc_dropout)
    model.to(device)
    model.load_state_dict(torch.load(args.model_wegihts_path))
    model.eval()
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

    # 对标pytorch封装好的ImageFlolder，我们自己实现了一个数据加载类 Brain_MRI_Load，并使用指定的预处理操作来处理图像，结果会同时返回图像和对应的标签。
    if args.mix_train_val:
        data_len = len(Brain_MRI_Load(
            args.data_path, args.dataset_used, print_result=False))
        # 先读取一次数据取得数据集大小
        all_numbers = list(range(data_len))
        step = max(1, int(1 / args.val_scale))
        selected_numbers = all_numbers[::step]  # 验证集
        test_dataset = Brain_MRI_Load(args.data_path, args.dataset_used,
                                      transform=data_transform["test"], data_filter=selected_numbers, print_result=False)
        print(test_dataset[102][1], test_dataset[204][1], test_dataset[304][1])

    else:
        test_dataset = Brain_MRI_Load(
            args.data_path, args.dataset_used, transform=data_transform["test"])  # 自定义加载类
    print('The test data size is:', len(test_dataset))
    nw = 0
    # nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers

    print('Using {} dataloader workers every process'.format(nw))

    # 使用 DataLoader 将加载的数据集处理成批量（batch）加载模式
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                              shuffle=False, pin_memory=True,  num_workers=nw, collate_fn=test_dataset.collate_fn)

    if network_type == '3d':
        # input_size是数据的维度,要增加一个灰度维度
        summary(model, input_size=(1,) + test_dataset[0][0].shape)

    elif network_type == '2d-slices' or network_type == '2.5d':
        summary(model, test_dataset[0][0].shape)

    if output == 1 and not args.gp_use and not args.mc_dropout:  # 没有估计不确定性的情况
        # train

        test_mae = evaluate(model=model, data_loader=test_loader, device=device,
                            network_type=network_type, output=output, gp_use=args.gp_use)
        print('test_mae:', test_mae)

    else:  # 估计了不确定性的状况

        epc_ale_ratio = args.epc_ale_ratio
        test_mae, cof, p_val = evaluate(model=model, data_loader=test_loader, device=device,
                                        network_type=network_type, output=output, gp_use=args.gp_use,
                                        mc_dropout=args.mc_dropout, epc_ale_ratio=epc_ale_ratio)
        print('test_mae:', test_mae, 'cof', cof, 'p_val', p_val)


if __name__ == '__main__':
    main(opt)
