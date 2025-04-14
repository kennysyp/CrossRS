import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys
sys.path.append(r"/home/gdut-627/huang/RRS-LPBA")
import numpy as np
import argparse
import random

import monai
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.backends.cudnn as cudnn

import SimpleITK as sitk
sitk.ProcessObject.SetGlobalWarningDisplay(False)

from data.data_utils import read_data_csv, process_label
from data.metric import dice_val, dice_val_substruct
from utils.util import csv_writter, AverageMeter, Dataset_epoch_crop_seg

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/home/gdut-627/huang/RRS-LPBA/data/', help='data')
parser.add_argument('--exp', type=str,
                    default='cross/joint2', help='model_name')
parser.add_argument('--batch_size', type=int, default=1, help='batch size per gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--label_percent', type=int, default=100, help='25 and 100')
parser.add_argument('--seed', type=int, default=19260817, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')

parser.add_argument('--sp', type=float, default=2.0,
                    help="seg supervised loss: suggested range 1 to 3")
parser.add_argument('--consistency', type=float, default=0.5,
                    help="cross consistency loss: suggested range 0.1 to 1")

args = parser.parse_args()

"""
test seg model
"""

train_data_path = args.root_path
snapshot_path = "/home/gdut-627/huang/RRS-LPBA/model/" + args.exp +\
    "_{}_{}/".format(args.consistency, args.sp)

# snapshot_path = "/home/gdut-627/huang/RRS-LPBA/model/" + args.exp + "/"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = args.batch_size
num_segmentation_classes = 55

model_name = "seg_net"


if not args.deterministic:
    cudnn.benchmark = True
    cudnn.deterministic = False
else:
    cudnn.benchmark = False
    cudnn.deterministic = True
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


if __name__ == '__main__':
    dict = process_label("/home/gdut-627/huang/RRS-LPBA/data/labels.txt")

    csv_writter('mono_seg_net_{}labels test'.format(args.label_percent), snapshot_path +  model_name)
    line = 'idx'
    for i in range(num_segmentation_classes):
        line = line + ',' + dict[i]
    line = line + ',' + "Avg Dice" + ',' + "Avg Dice Structure"
    csv_writter(line, snapshot_path +  model_name)

    test_path = read_data_csv(train_data_path + "test.csv")

    test_generator = Data.DataLoader(Dataset_epoch_crop_seg(test_path, norm=True),
                                     shuffle=False, num_workers=2)

    """ Create Model"""
    seg_net = monai.networks.nets.UNet(
        3,  # spatial dims
        1,  # input channels
        num_segmentation_classes,  # output channels
        (8, 8, 16, 16, 32, 32, 64, 64, 128),  # channel sequence
        (1, 2, 1, 2, 1, 2, 1, 2),   # convolutional strides
        dropout=0.2,
        norm='batch'
    ).to(device)
    seg_net.load_state_dict(torch.load(snapshot_path + model_name + '.pth'))
    print(f"seg_net load model from [{snapshot_path}{model_name}.pth]")


    print("===========================Start Testing================================")

    eval_dsc = AverageMeter()
    eval_dsc_substruct = AverageMeter()
    with torch.no_grad():
        stdy_idx = 1
        for img, true_seg in test_generator:
            seg_net.eval()
            img, true_seg = img.to(device), true_seg.to(device)

            predict_seg = seg_net(img)
            predict_seg = torch.softmax(predict_seg, dim=1)

            dsc_seg = dice_val(predict_seg, true_seg.long(), num_segmentation_classes)
            dsc_seg_substruct = dice_val(predict_seg, true_seg.long(), num_segmentation_classes, False)

            print('idx: {} Trans dsc: {:.4f} substruct: {:.4f}'.format(stdy_idx, dsc_seg.item(), dsc_seg_substruct.item()))

            eval_dsc.update(dsc_seg.item(), img.size(0))
            eval_dsc_substruct.update(dsc_seg_substruct.item(), img.size(0))
            line = dice_val_substruct(predict_seg, true_seg.long(), num_segmentation_classes, stdy_idx)
            line = line + ',' + str(dsc_seg.item()) + ',' + str(dsc_seg_substruct.item())
            csv_writter(line, snapshot_path +  model_name)
            stdy_idx += 1

    print('Seg Dice %: {:.3f} +- {:.3f}'.format(eval_dsc.avg, eval_dsc.std))
    print('Seg Dice Structures %: {:.3f} +- {:.3f}'.format(eval_dsc_substruct.avg, eval_dsc_substruct.std))
    csv_writter(f"{eval_dsc.avg:.3f} +- {eval_dsc.std:.3f}", snapshot_path +  model_name)
    csv_writter(f"{eval_dsc_substruct.avg:.3f} +- {eval_dsc_substruct.std:.3f}", snapshot_path +  model_name)
