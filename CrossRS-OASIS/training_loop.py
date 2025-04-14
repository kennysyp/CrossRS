import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys
sys.path.append(r"/home/gdut-627/huang/RRS-OASIS")
import datetime
import numpy as np
import logging
import argparse
import random
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use Agg backend (non-GUI)

import monai
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.backends.cudnn as cudnn

import SimpleITK as sitk
sitk.ProcessObject.SetGlobalWarningDisplay(False)

from data.data_utils import (
    read_data_csv, 
    take_data_pairs, 
    subdivide_list_of_data_pairs,
    create_batch_generator
)
from utils.util import generate_grid, plot_against_epoch_numbers, Dataset_pairs_crop, Dataset_epoch_crop_seg, \
    sigmoid_rampup
from networks.symnet import SYMNet, SpatialTransform, SpatialTransformNearest, smoothloss, DiffeomorphicTransform, \
    CompositionTransform, magnitude_loss, neg_Jdet_loss, NCC


import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/home/gdut-627/huang/RRS-OASIS/data/', help='data')
parser.add_argument('--exp', type=str,
                    default='cross/joint', help='model_name')
parser.add_argument('--max_epochs', type=int,
                    default=120, help='maximum epoch number to train')
parser.add_argument('--start_channel', type=int, default=7, help="number of start channels")
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--label_percent', type=int, default=40, help='10 or 40')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=19260817, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')

# reg loss
parser.add_argument('--local_ori', type=float, default=100.0,
                    help="Local Orientation Consistency loss: suggested range 1 to 1000")
parser.add_argument('--magnitude', type=float, default=0.1,
                    help="magnitude loss: suggested range 0.001 to 1.0")
parser.add_argument('--smooth', type=float, default=3.0,
                    help="Gradient smooth loss: suggested range 0.1 to 10")
parser.add_argument('--anatomy', type=float, default=2.0,
                    help="anatomy loss: suggested range 0.1 to 10")

# seg loss
parser.add_argument('--sp', type=float, default=3.0,
                    help="seg supervised loss: suggested range 1 to 3")
parser.add_argument('--consistency', type=float, default=0.1,
                    help="Local Orientation Consistency loss: suggested range 0.1 to 1")
parser.add_argument('--consistency_rampup', type=float, default=200.0,
                    help="consistency_rampup")

args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "/home/gdut-627/huang/RRS-OASIS/model/" + args.exp + \
    "_{}label/".format(args.label_percent)

reg_net_name = "reg_net"
seg_net_name = "seg_net"
seg_net_pretrained = "/home/gdut-627/huang/RRS-OASIS/model/mono_seg_{}labels/mono_seg_net_{}labels_best.pth".format(args.label_percent, args.label_percent)
reg_net_pretrained = "/home/gdut-627/huang/RRS-OASIS/model/SYMNet_lc_{}labels/symnet_lc_30000.pth".format(args.label_percent)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = args.batch_size
max_epochs = args.max_epochs

start_channel = args.start_channel

# reg lambda
lambda_ori = args.local_ori
lambda_mag = args.magnitude
lambda_smooth = args.smooth
lambda_ana = args.anatomy

# seg lambda
lambda_con = args.consistency # consistency loss weight
lambda_sp = args.sp # supervised segmentation loss weight

num_segmentation_classes = 36
img_size = [160, 144, 192]
range_flow = 100

seg_availabilities = ['00', '01', '10', '11']

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

def get_current_consistency_weight(epoch):
    # Consistency ramp-up
    return args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)


def dice(im1, atlas):
    unique_class = np.unique(atlas)
    dice = 0
    num_count = 0
    for i in unique_class:
        if (i == 0) or ((im1==i).sum()==0) or ((atlas==i).sum()==0):
            continue

        sub_dice = np.sum(atlas[im1 == i] == i) * 2.0 / (np.sum(im1 == i) + np.sum(atlas == i))
        dice += sub_dice
        num_count += 1
    return dice/num_count


def swap_training(network_to_train, network_to_not_train):
    """
    Switch out of training one network and into training another
    """
    for param in network_to_not_train.parameters():
        param.requires_grad = False
    for param in network_to_train.parameters():
        param.requires_grad = True
    
    network_to_train.train()
    network_to_not_train.eval()


if __name__ == "__main__":

    # make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    
    root_logger = logging.getLogger()
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
    logging.basicConfig(filename=snapshot_path+"log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(f"data: {datetime.datetime.now().strftime('%Y-%m-%d')}")
    logging.info(str(args))

    # Load data
    train_path = read_data_csv(train_data_path + "train_{}per.csv".format(args.label_percent))
    val_path = read_data_csv(train_data_path + "val.csv")

    data_pairs_train = take_data_pairs(train_path)
    data_pairs_valid = take_data_pairs(val_path)

    data_pairs_train_subdivided = subdivide_list_of_data_pairs(data_pairs_train)
    data_pairs_valid_subdivided = subdivide_list_of_data_pairs(data_pairs_valid)

    num_train_reg_net = len(data_pairs_train)
    num_valid_reg_net = len(data_pairs_valid)
    num_train_both = len(data_pairs_train_subdivided['01']) +\
        len(data_pairs_train_subdivided['10']) +\
        len(data_pairs_train_subdivided['11']) +\
        len(data_pairs_train_subdivided['00'])

    logging.info(f"""We have {num_train_both} pairs to train reg_net and seg_net together.""")
    logging.info(f"We have {num_valid_reg_net} pairs for reg_net validation.")

    dataset_pairs_train_subdivided = {
        seg_availability: Dataset_pairs_crop(
            data_pairs=data_list,
            norm=True
        )
        for seg_availability, data_list in data_pairs_train_subdivided.items()
    }

    dataset_pairs_valid_subdivided = {
        seg_availability: Dataset_pairs_crop(
            data_pairs=data_list,
            norm=True
        )
        for seg_availability, data_list in data_pairs_valid_subdivided.items()
    }

    dataset_valid = Dataset_epoch_crop_seg(
        names=val_path,
        norm=True,
    )

    """ Create Networks"""
    # Load Model
    seg_net = monai.networks.nets.UNet(
        3,  # spatial dims
        1,  # input channels
        num_segmentation_classes,  # output channels
        (8, 8, 16, 16, 32, 32, 64, 64, 128),  # channel sequence
        (1, 2, 1, 2, 1, 2, 1, 2),   # convolutional strides
        dropout=0.2,
        norm='batch'
    )
    seg_net.load_state_dict(torch.load(seg_net_pretrained))
    logging.info(f"seg_net load pretrained state dict from [{seg_net_pretrained}]")

    reg_net = SYMNet(2, 3, start_channel).to(device)
    reg_net.load_state_dict(torch.load(reg_net_pretrained))
    logging.info(f"reg_net load pretrained state dict from [{reg_net_pretrained}]")

    transform = SpatialTransform().cuda()
    transform_nearest = SpatialTransformNearest().cuda()
    diff_transform = DiffeomorphicTransform(time_step=7).cuda()
    com_transform = CompositionTransform().cuda()

    grid = generate_grid(img_size)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).cuda().float()

    loss_similarity = NCC(win=5)
    loss_smooth = smoothloss
    loss_magnitude = magnitude_loss
    loss_Jdet = neg_Jdet_loss

    dice_loss = monai.losses.DiceLoss(
        include_background=True,
        to_onehot_y=True,  # Our seg labels are single channel images indicating class index, rather than one-hot
        softmax=True,  # Note that our segmentation network is missing the softmax at the end
        reduction="mean"
    )

    # 给anatomy loss用的
    dice_loss2 = monai.losses.DiceLoss(
        include_background=True,
        to_onehot_y=False,
        softmax=False,
        reduction="mean"
    )

    def anatomy_loss(F_X_Y, X, Y, seg_net, gt_seg1=None, gt_seg2=None):
        if gt_seg1 is not None:
            # ground truth seg of target image
            seg1 = monai.networks.one_hot(gt_seg1, num_segmentation_classes)
        else:
            seg1 = seg_net(X).softmax(dim=1)

        if gt_seg2 is not None:
            # ground truth seg of target image
            seg2 = monai.networks.one_hot(gt_seg2, num_segmentation_classes)
        else:
            seg2 = seg_net(Y).softmax(dim=1)

        seg1_warp = transform(seg1, F_X_Y.permute(0, 2, 3, 4, 1) * range_flow, grid)

        return dice_loss2(seg1_warp, seg2)

    def reg_losses(batch):
        X, Y = batch['img1'].to(device), batch['img2'].to(device)

        F_xy, F_yx = reg_net(X, Y)

        F_X_Y_half = diff_transform(F_xy, grid, range_flow)
        F_Y_X_half = diff_transform(F_yx, grid, range_flow)

        F_X_Y_half_inv = diff_transform(-F_xy, grid, range_flow)
        F_Y_X_half_inv = diff_transform(-F_yx, grid, range_flow)

        X_Y_half = transform(X, F_X_Y_half.permute(0, 2, 3, 4, 1) * range_flow, grid)
        Y_X_half = transform(Y, F_Y_X_half.permute(0, 2, 3, 4, 1) * range_flow, grid)

        F_X_Y = com_transform(F_X_Y_half, F_Y_X_half_inv, grid, range_flow)
        F_Y_X = com_transform(F_Y_X_half, F_X_Y_half_inv, grid, range_flow)

        X_Y = transform(X, F_X_Y.permute(0, 2, 3, 4, 1) * range_flow, grid)
        Y_X = transform(Y, F_Y_X.permute(0, 2, 3, 4, 1) * range_flow, grid)

        loss_sim_mid = loss_similarity(X_Y_half, Y_X_half)
        loss_sim_full = loss_similarity(Y, X_Y) + loss_similarity(X, Y_X)
        loss_mag = loss_magnitude(F_X_Y_half * range_flow, F_Y_X_half * range_flow)
        loss_jdet = loss_Jdet(F_X_Y.permute(0, 2, 3, 4, 1) * range_flow, grid) + loss_Jdet(F_Y_X.permute(0, 2, 3, 4, 1) * range_flow, grid)
        loss_smo = loss_smooth(F_xy * range_flow) + loss_smooth(F_yx * range_flow)

        gt_seg1 = batch['seg1'].to(device) if 'seg1' in batch.keys() else None
        gt_seg2 = batch['seg2'].to(device) if 'seg2' in batch.keys() else None
        loss_ana = anatomy_loss(F_X_Y, X, Y, seg_net, gt_seg1, gt_seg2)

        return loss_sim_mid, loss_sim_full, loss_mag, loss_jdet, loss_smo, loss_ana
    
    """ DataLoader """
    dataloader_pairs_train_subdivided = {
        seg_availability: Data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=4,
            shuffle=True
        )
        if len(dataset) > 0 else []  # empty dataloaders are not a thing-- put an empty list if needed
        for seg_availability, dataset in dataset_pairs_train_subdivided.items()
    }

    dataloader_pairs_valid_subdivided = {
        seg_availability: Data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=4,
            shuffle=True  # Shuffle validation data because we will only take a sample for validation each time
        )
        if len(dataset) > 0 else []  # empty dataloaders are not a thing-- put an empty list if needed
        for seg_availability, dataset in dataset_pairs_valid_subdivided.items()
    }

    dataloader_seg_available_valid = Data.DataLoader(
        dataset_valid,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
    )

    batch_generator_train_reg = create_batch_generator(dataloader_pairs_train_subdivided)
    batch_generator_valid_reg = create_batch_generator(dataloader_pairs_valid_subdivided)

    # When training seg_net alone, we only consider data pairs for which at least one ground truth seg is available
    seg_train_sampling_weights = [len(dataloader_pairs_train_subdivided[s]) for s in seg_availabilities]
    logging.info(f"""When training seg_net alone, segmentation availabilities {seg_availabilities}
        will be sampled with respective weights {seg_train_sampling_weights}""")
    batch_generator_train_seg = create_batch_generator(dataloader_pairs_train_subdivided, seg_train_sampling_weights)

    logging.info(
        "===========================Start Training================================")
    seg_net.to(device)
    reg_net.to(device)

    learing_rate_reg = 1e-4
    optimizer_reg = torch.optim.Adam(reg_net.parameters(), learing_rate_reg)

    learing_rate_seg = 1e-3
    optimizer_seg = torch.optim.Adam(seg_net.parameters(), learing_rate_seg)

    reg_phase_training_batches_per_epoch = 400
    seg_phase_training_batches_per_epoch = 200  # Fewer batches needed, because seg_net converges more quickly
    reg_phase_num_validation_batches_to_use = 80
    val_interval = 2

    training_losses_reg = []
    validation_losses_reg = []
    training_losses_seg = []
    validation_losses_seg = []

    best_seg_validation_loss = float('inf')
    best_reg_validation_loss = float('inf')

    step = 1
    seg_step = 1
    for epoch_number in range(max_epochs):
        logging.info(f"Epoch {epoch_number+1}/{max_epochs}")
        # ------------------------------------------------
        #         reg_net training, with seg_net frozen
        # ------------------------------------------------
        swap_training(reg_net, seg_net)

        losses = []
        for batch in batch_generator_train_reg(reg_phase_training_batches_per_epoch):
            optimizer_reg.zero_grad()
            loss_sim_mid, loss_sim_full, loss_mag, loss_jdet, loss_smo, loss_ana = reg_losses(batch)
            loss = loss_sim_mid + loss_sim_full + lambda_mag * loss_mag + lambda_ori * loss_jdet \
                + lambda_smooth * loss_smo + lambda_ana * loss_ana
            loss.backward()
            optimizer_reg.step()
            losses.append(loss.item())
            logging.info(f"\tstep-{step} training loss: {loss.item():.4f} sim_mid: {loss_sim_mid.item():.4f} sim_full: {loss_sim_full.item():.4f} mag: {loss_mag.item():.4f} Jdet: {loss_jdet.item():.10f} smo: {loss_smo.item():.4f} ana: {loss_ana.item():.6f}")
            step = step + 1
        
        training_loss = np.mean(losses)
        logging.info(f"\tepoch-{epoch_number+1} reg training loss: {training_loss}")
        training_losses_reg.append([epoch_number, training_loss])

        if (epoch_number+1) % val_interval == 0:
            reg_net.eval()
            losses = []
            with torch.no_grad():
                for batch in batch_generator_valid_reg(reg_phase_num_validation_batches_to_use):
                    loss_sim_mid, loss_sim_full, loss_mag, loss_jdet, loss_smo, loss_ana = reg_losses(batch)
                    loss = loss_sim_mid + loss_sim_full + lambda_mag * loss_mag + lambda_ori * loss_jdet \
                        + lambda_smooth * loss_smo + lambda_ana * loss_ana
                    losses.append(loss.item())

            validation_loss = np.mean(losses)
            logging.info(f"\tepoch-{epoch_number+1} reg validation loss: {validation_loss}")
            validation_losses_reg.append([epoch_number, validation_loss])

            if validation_loss < best_reg_validation_loss:
                best_reg_validation_loss = validation_loss
                save_path = os.path.join(snapshot_path, 'reg_net_best.pth')
                torch.save(reg_net.state_dict(), save_path)

        # Free up memory
        del loss, loss_sim_mid, loss_sim_full, loss_mag, loss_jdet, loss_smo
        torch.cuda.empty_cache()

        # ------------------------------------------------
        #         seg_net training, with reg_net frozen
        # ------------------------------------------------
        swap_training(seg_net, reg_net)

        losses = []
        for batch in batch_generator_train_seg(seg_phase_training_batches_per_epoch):
            optimizer_seg.zero_grad()

            X, Y = batch['img1'].to(device), batch['img2'].to(device)

            F_xy, F_yx = reg_net(X, Y)

            F_X_Y_half = diff_transform(F_xy, grid, range_flow)
            F_Y_X_half = diff_transform(F_yx, grid, range_flow)

            F_X_Y_half_inv = diff_transform(-F_xy, grid, range_flow)
            F_Y_X_half_inv = diff_transform(-F_yx, grid, range_flow)

            F_X_Y = com_transform(F_X_Y_half, F_Y_X_half_inv, grid, range_flow)
            F_Y_X = com_transform(F_Y_X_half, F_X_Y_half_inv, grid, range_flow)

            seg1_predicted = seg_net(X).softmax(dim=1)
            seg2_predicted = seg_net(Y).softmax(dim=1)

            # Below we compute the following:
            # loss_supervised: supervised segmentation loss; compares ground truth seg with predicted seg
            # loss_anatomy: anatomy loss; compares warped seg of moving image to seg of target image
            # loss_metric: a single supervised seg loss, as a metric to track the progress of training

            if 'seg1' in batch.keys() and 'seg2' in batch.keys():
                seg1 = monai.networks.one_hot(batch['seg1'].to(device), num_segmentation_classes)
                seg2 = monai.networks.one_hot(batch['seg2'].to(device), num_segmentation_classes)
                loss_metric = dice_loss2(seg2_predicted, seg2)
                loss_supervised = dice_loss2(seg1_predicted, seg1) + loss_metric
                loss_consistency = 0.0

            elif 'seg1' in batch.keys():  # seg1 available, but no seg2
                seg1 = monai.networks.one_hot(batch['seg1'].to(device), num_segmentation_classes)
                loss_metric = dice_loss2(seg1_predicted, seg1)
                loss_supervised = loss_metric
                seg2 = seg2_predicted
                seg1_warped = transform(seg1, F_X_Y.permute(0, 2, 3, 4, 1) * range_flow, grid)
                loss_consistency = torch.mean((seg1_warped - seg2) ** 2)
                # seg2_warped = transform(seg2, F_Y_X.permute(0, 2, 3, 4, 1) * range_flow, grid)
                # loss_consistency = torch.mean((seg2_warped - seg1) ** 2)

            elif 'seg2' in batch.keys():  # seg2 available, but no seg1
                assert 'seg2' in batch.keys()
                seg2 = monai.networks.one_hot(batch['seg2'].to(device), num_segmentation_classes)
                loss_metric = dice_loss2(seg2_predicted, seg2)
                loss_supervised = loss_metric
                seg1 = seg1_predicted
                seg2_warped = transform(seg2, F_Y_X.permute(0, 2, 3, 4, 1) * range_flow, grid)
                loss_consistency = torch.mean((seg2_warped - seg1) ** 2)
                # seg1_warped = transform(seg1, F_X_Y.permute(0, 2, 3, 4, 1) * range_flow, grid)
                # loss_consistency = torch.mean((seg1_warped - seg2) ** 2)

            else: # seg1 and seg2 are not available
                seg1 = seg1_predicted
                seg2 = seg2_predicted
                seg1_warped = transform(seg1, F_X_Y.permute(0, 2, 3, 4, 1) * range_flow, grid)
                seg2_warped = transform(seg2, F_Y_X.permute(0, 2, 3, 4, 1) * range_flow, grid)
                loss_metric = dice_loss2(seg1_warped, seg2)
                loss_supervised = 0.0
                loss_consistency = torch.mean((seg1_warped - seg2) ** 2) + torch.mean((seg2_warped - seg1) ** 2)


            consistency_weight = get_current_consistency_weight(seg_step // 150)

            loss = lambda_sp * loss_supervised + consistency_weight * loss_consistency
            loss.backward()
            optimizer_seg.step()

            losses.append(loss_metric.item())
            seg_step += 1

        training_loss = np.mean(losses)
        logging.info(f"\tepoch-{epoch_number+1} seg training loss: {training_loss}")
        training_losses_seg.append([epoch_number, training_loss])

        if (epoch_number+1) % val_interval == 0:
            seg_net.eval()
            losses = []
            with torch.no_grad():
                for img, true_seg in dataloader_seg_available_valid:
                    img, true_seg = img.to(device), true_seg.to(device)
                    predicted_segs = seg_net(img)
                    loss = dice_loss(predicted_segs, true_seg)
                    losses.append(loss.item())

            validation_loss = np.mean(losses)
            logging.info(f"\tepoch-{epoch_number+1} seg validation loss: {validation_loss}")
            validation_losses_seg.append([epoch_number, validation_loss])

            if validation_loss < best_seg_validation_loss:
                best_seg_validation_loss = validation_loss
                save_path = os.path.join(snapshot_path, 'seg_net_best.pth')
                torch.save(seg_net.state_dict(), save_path)

        # Free up memory
        del X, Y, F_xy, F_yx, F_X_Y_half, F_Y_X_half, F_X_Y_half_inv, F_Y_X_half_inv, F_X_Y, F_Y_X, \
            seg1_predicted, seg2_predicted, loss, loss_metric, loss_supervised, loss_consistency, seg1, \
            seg2

        torch.cuda.empty_cache()

    logging.info(f"\n\nBest reg_net validation loss : {best_reg_validation_loss}")
    logging.info(f"Best seg_net validation loss : {best_seg_validation_loss}")

    # Plot
    plot_against_epoch_numbers(training_losses_reg, label="training")
    plot_against_epoch_numbers(validation_losses_reg, label="validation")
    plt.legend()
    plt.ylabel('loss')
    plt.title('Alternating training: registration loss')
    plt.savefig(snapshot_path + 'reg_net_losses.png')

    plot_against_epoch_numbers(training_losses_seg, label="training")
    plt.ylabel('training loss')
    plt.title('Alternating training: segmentation loss (training)')
    plt.savefig(snapshot_path + 'seg_net_training_losses.png')

    plot_against_epoch_numbers(validation_losses_seg, label="validation", color='orange')
    plt.ylabel('validation loss')
    plt.title('Alternating training: segmentation loss (validation)')
    plt.savefig(snapshot_path + 'seg_net_validation_losses.png')

    reg_save_path = os.path.join(snapshot_path + 'reg_net.pth')
    seg_save_path = os.path.join(snapshot_path + 'seg_net.pth')
    torch.save(seg_net.state_dict(), seg_save_path)
    torch.save(reg_net.state_dict(), reg_save_path)