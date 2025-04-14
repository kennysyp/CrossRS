import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys
sys.path.append(r"/home/gdut-627/huang/RRS-OASIS")
import numpy as np
import argparse
import random

import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.backends.cudnn as cudnn

import SimpleITK as sitk
sitk.ProcessObject.SetGlobalWarningDisplay(False)

from data.data_utils import read_data_csv, process_label, take_data_pairs
from data.metric import dice_val, dice_val_substruct, jacobian_determinant
from utils.util import csv_writter, AverageMeter, generate_grid, Predict_dataset_crop
from networks.symnet import SYMNet, SpatialTransform, SpatialTransformNearest, DiffeomorphicTransform, CompositionTransform

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/home/gdut-627/huang/RRS-OASIS/data/', help='data')
parser.add_argument('--exp', type=str,
                    default='cross/joint', help='model_name')
parser.add_argument('--batch_size', type=int, default=1, help='batch size per gpu')
parser.add_argument('--start_channel', type=int, default=7, help="number of start channels")
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--label_percent', type=int, default=40, help='10, 40 or 100')
parser.add_argument('--seed', type=int, default=19260817, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')

parser.add_argument('--local_ori', type=float, default=100.0,
                    help="Local Orientation Consistency loss: suggested range 1 to 1000")
parser.add_argument('--magnitude', type=float, default=0.1,
                    help="magnitude loss: suggested range 0.001 to 1.0")

args = parser.parse_args()

"""
test reg model
"""

train_data_path = args.root_path
# snapshot_path = "/home/gdut-627/huang/RRS-OASIS/model/" + args.exp +\
#       "_{}ori_{}mag/".format(args.local_ori, args.magnitude)

snapshot_path = "/home/gdut-627/huang/RRS-OASIS/model/" + args.exp + \
    "_{}label/".format(args.label_percent)

print("***********************************************")
print(snapshot_path)

# snapshot_path = "/home/gdut-627/huang/RRS-Main/model/" + args.exp + "/"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = args.batch_size
start_channel = args.start_channel

img_size = [160, 144, 192]
num_segmentation_classes = 36
range_flow = 100

model_name = "reg_net_best"

# model_name = "symnet_lc_30000"


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
    dict = process_label("/home/gdut-627/huang/RRS-OASIS/data/seg35_labels.txt")

    csv_writter('reg_net test', snapshot_path +  model_name)
    line = 'idx'
    for i in range(num_segmentation_classes):
        line = line + ',' + dict[i]
    line = line + ',' + "Avg Dice" + ',' + "Avg Dice Structure" + ',' + "njac_count" + ',' + "stdjac"
    csv_writter(line, snapshot_path +  model_name)

    test_path = read_data_csv(train_data_path + "test.csv")
    fixed_names = test_path[:5]
    moving_names = test_path[5:]

    """ Create Model"""
    reg_net = SYMNet(2, 3, start_channel).to(device)
    reg_net.load_state_dict(torch.load(snapshot_path + model_name + '.pth'))
    print(f"reg_net load model from [{snapshot_path}{model_name}.pth]")

    transform = SpatialTransform().cuda()
    transform_nearest = SpatialTransformNearest().cuda()
    diff_transform = DiffeomorphicTransform(time_step=7).cuda()
    com_transform = CompositionTransform().cuda()

    grid = generate_grid(img_size)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).cuda().float()

    test_generator = Data.DataLoader(
        Predict_dataset_crop(fixed_names, moving_names, norm=True),
        batch_size = batch_size,
        shuffle=False,
        num_workers=4,
    )

    print("===========================Start Testing================================")

    eval_dsc = AverageMeter()
    eval_dsc_substruct = AverageMeter()
    eval_det = AverageMeter()
    eval_stddet = AverageMeter()
    with torch.no_grad():
        stdy_idx = 1
        for data in test_generator:
            reg_net.eval()
            X, Y = data['move'].to(device), data['fixed'].to(device) 
            X_label, Y_label = data['move_label'].to(device), data['fixed_label'].to(device)

            F_xy, F_yx = reg_net(X, Y)
            F_X_Y_half = diff_transform(F_xy, grid, range_flow)
            F_Y_X_half_inv = diff_transform(-F_yx, grid, range_flow)
            F_X_Y = com_transform(F_X_Y_half, F_Y_X_half_inv, grid, range_flow)

            X_Y_label = transform_nearest(X_label, F_X_Y.permute(0, 2, 3, 4, 1) * range_flow, grid)

            F_XY = F_X_Y.permute(0, 2, 3, 4, 1).data.cpu().numpy()[0, :, :, :, :]
            F_XY = F_XY.astype(np.float32) * range_flow

            jac_det = jacobian_determinant(F_XY)
            njac = np.sum(jac_det <= 0) / np.prod(img_size) * 100
            std_det = np.std(jac_det)

            dsc_seg = dice_val(X_Y_label, Y_label, num_segmentation_classes)
            dsc_seg_substruct = dice_val(X_Y_label, Y_label, num_segmentation_classes, False)

            print(f"idx: {stdy_idx} Trans dsc: {dsc_seg.item():.4f} substruct: {dsc_seg_substruct.item():.4f}")
            print(f"det < 0: {njac}, std(det): {std_det}")

            eval_dsc.update(dsc_seg.item(), X.size(0))
            eval_dsc_substruct.update(dsc_seg_substruct.item(), X.size(0))
            eval_det.update(njac, X.size(0))
            eval_stddet.update(std_det, X.size(0))

            line = dice_val_substruct(X_Y_label, Y_label, num_segmentation_classes, stdy_idx)
            line = line + ',' + str(dsc_seg.item()) + ',' + str(dsc_seg_substruct.item()) + ',' + str(njac) + ',' + str(std_det)
            csv_writter(line, snapshot_path +  model_name)
            stdy_idx += 1

    print('reg Dice %: {:.3f} +- {:.3f}'.format(eval_dsc.avg, eval_dsc.std))
    print('reg Dice Structures %: {:.3f} +- {:.3f}'.format(eval_dsc_substruct.avg, eval_dsc_substruct.std))
    print(f"reg njac <=0: {eval_det.avg:.3f} +- {eval_det.std:.3f}")
    print(f"reg std(jac): {eval_stddet.avg:.3f} +- {eval_stddet.std:.3f}")

    csv_writter(f"Dice %: {eval_dsc.avg:.3f} +- {eval_dsc.std:.3f}", snapshot_path +  model_name)
    csv_writter(f"Dice Structures %: {eval_dsc_substruct.avg:.3f} +- {eval_dsc_substruct.std:.3f}", snapshot_path +  model_name)
    csv_writter(f"njac <=0: {eval_det.avg:.3f} +- {eval_det.std:.3f}", snapshot_path +  model_name)
    csv_writter(f"std(jac): {eval_stddet.avg:.3f} +- {eval_stddet.std:.3f}", snapshot_path +  model_name)
