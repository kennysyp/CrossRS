import numpy as np
import re
import matplotlib.pyplot as plt
import monai
import itertools

import torch
import torch.nn as nn
import torch.utils.data as Data

import nibabel as nib
import SimpleITK as sitk
sitk.ProcessObject.SetGlobalWarningDisplay(False)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = []
        self.std = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vals.append(val)
        self.std = np.std(self.vals)


def plot_2D_vector_field(vector_field, downsampling):
    """Plot a 2D vector field given as a tensor of shape (2,H,W).
    The plot origin will be in the lower left.
    Using "x" and "y" for the rightward and upward directions respectively,
      the vector at location (x,y) in the plot image will have
      vector_field[1,y,x] as its x-component and
      vector_field[0,y,x] as its y-component.
    """
    downsample2D = monai.networks.layers.factories.Pool['AVG', 2](
        kernel_size=downsampling)
    vf_downsampled = downsample2D(vector_field.unsqueeze(0))[0]
    plt.quiver(
        vf_downsampled[1, :, :], vf_downsampled[0, :, :],
        angles='xy', scale_units='xy', scale=downsampling,
        headwidth=4.
    )


def preview_3D_vector_field(vector_field, downsampling=None):
    """
        Display three orthogonal slices of the given 3D vector field.
        vector_field should be a tensor of shape (3,H,W,D)
        Vectors are projected into the viewing plane, so you are only seeing
        their components in the viewing plane.
    """

    if downsampling is None:
        # guess a reasonable downsampling value to make a nice plot
        downsampling = max(1, int(max(vector_field.shape[1:])) >> 5)

    x, y, z = np.array(vector_field.shape[1:])//2  # half-way slices
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.axis('off')
    plot_2D_vector_field(vector_field[[1, 2], x, :, :], downsampling)
    plt.subplot(1, 3, 2)
    plt.axis('off')
    plot_2D_vector_field(vector_field[[0, 2], :, y, :], downsampling)
    plt.subplot(1, 3, 3)
    plt.axis('off')
    plot_2D_vector_field(vector_field[[0, 1], :, :, z], downsampling)
    plt.show()


def plot_2D_deformation(vector_field, grid_spacing, **kwargs):
    """
        Interpret vector_field as a displacement vector field defining a deformation,
        and plot an x-y grid warped by this deformation.
        vector_field should be a tensor of shape (2,H,W)
    """
    _, H, W = vector_field.shape
    grid_img = np.zeros((H,W))
    grid_img[np.arange(0, H, grid_spacing),:]=1
    grid_img[:,np.arange(0, W, grid_spacing)]=1
    grid_img = torch.tensor(grid_img, dtype=vector_field.dtype).unsqueeze(0) # adds channel dimension, now (C,H,W)
    warp = monai.networks.blocks.Warp(mode="bilinear", padding_mode="zeros")
    grid_img_warped = warp(grid_img.unsqueeze(0), vector_field.unsqueeze(0))[0]
    plt.imshow(grid_img_warped[0], origin='lower', cmap='gist_gray')


def preview_image(image_array, normalize_by="volume", cmap=None, figsize=(12, 12), threshold=None):
    """
    Display three orthogonal slices of the given 3D image.

    image_array os assumed to be of shape (H,W,D)

    if a number is provided for threshold, then pixels for which the value
    is below the threshold will be shown in red
    """
    if normalize_by == "slice":
        vmin = None
        vmax = None
    elif normalize_by == "volume":
        vmin = 0
        vmax = image_array.max().item()
    else:
        raise(ValueError(
            f"Invalid value '{normalize_by}' given for normalize_by"))

    # half-way slices
    x, y, z = np.array(image_array.shape) // 2
    imgs = (image_array[x, :, :], image_array[:, y, :], image_array[:, :, z])

    fig, axs = plt.subplots(1, 3, figsize=figsize)
    for ax, im in zip(axs, imgs):
        ax.axis('off')
        ax.imshow(im, origin='lower', vmin=vmin, vmax=vmax, cmap=cmap)

        # threshold will be useful when displaying jacobian determinant images;
        # we will want to clearly see where the jacobian determinant is negative
        if threshold is not None:
            red = np.zeros(im.shape + (4, ))  # RGBA array
            red[im <= threshold] = [1, 0, 0, 1]
            ax.imshow(red, origin='lower')

    # plt.savefig("img_save_folder/Picture.png") 
    plt.show()


def preview_3D_deformation(vector_field, grid_spacing, **kwargs):
    """
        Interpret vector_field as a displacement vector field defining a deformation,
        and plot warped grids along three orthogonal slices.
        vector_field should be a tensor of shape (3,H,W,D)
        kwargs are passed to matplotlib plotting
        Deformations are projected into the viewing plane, so you are only seeing
        their components in the viewing plane.
    """
    x, y, z = np.array(vector_field.shape[1:])//2  # half-way slices
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.axis('off')
    plot_2D_deformation(vector_field[[1, 2], x, :, :], grid_spacing, **kwargs)
    plt.subplot(1, 3, 2)
    plt.axis('off')
    plot_2D_deformation(vector_field[[0, 2], :, y, :], grid_spacing, **kwargs)
    plt.subplot(1, 3, 3)
    plt.axis('off')
    plot_2D_deformation(vector_field[[0, 1], :, :, z], grid_spacing, **kwargs)
    plt.show()

def plot_against_epoch_numbers(epoch_and_value_pairs, **kwargs):
    """
        Helper to reduce code duplication when plotting quantities that vary over training epochs
        epoch_and_value_pairs: An array_like consisting of pairs of the form (<epoch number>, <value of thing to plot>)
        kwargs are forwarded to matplotlib.pyplot.plot
    """
    array = np.array(epoch_and_value_pairs)
    plt.plot(array[:, 0], array[:, 1], **kwargs)
    plt.xlabel("epochs")


def csv_writter(line, name):
    with open(name + '.csv', 'a') as file:
        file.write(line)
        file.write('\n')


def generate_grid(imgshape):
    x = np.arange(imgshape[0])
    y = np.arange(imgshape[1])
    z = np.arange(imgshape[2])
    grid = np.rollaxis(np.array(np.meshgrid(z, y, x)), 0, 4)
    grid = np.swapaxes(grid, 0, 2)
    grid = np.swapaxes(grid, 1, 2)
    return grid


def load_4D_with_crop(name, cropx, cropy, cropz):
    X = nib.load(name)
    X = X.get_fdata()

    x, y, z = X.shape
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    startz = z // 2 - cropz // 2

    X = X[startx:startx+cropx, starty:starty+cropy, startz:startz+cropz]

    X = np.reshape(X, (1, ) + X.shape)
    return X


def imgnorm(img):
    i_max = np.max(img)
    i_min = np.min(img)
    norm = (img - i_min) / (i_max - i_min)
    return norm


class Dataset_epoch_crop_seg(Data.Dataset):
    def __init__(self, names, norm=False):
        'Initialization'
        super(Dataset_epoch_crop_seg, self).__init__()
        self.names = names
        self.norm = norm

    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, step):
        # Select sample
        img = load_4D_with_crop(self.names[step]['img'], cropx=160, cropy=144, cropz=192)
        seg = load_4D_with_crop(self.names[step]['seg'], cropx=160, cropy=144, cropz=192)

        if self.norm:
            return torch.from_numpy(imgnorm(img)).float(), torch.from_numpy(seg).float()
        else:
            return torch.from_numpy(img).float(), torch.from_numpy(seg).float()


class Dataset_epoch_crop(Data.Dataset):
    def __init__(self, names, norm=False):
        'Initialization'
        super(Dataset_epoch_crop, self).__init__()
        self.names = names
        self.norm = norm
        self.index_pair = list(itertools.permutations(names, 2))

    def __len__(self):
        return len(self.index_pair)
    
    def __getitem__(self, step):
        # Select sample
        img_A = load_4D_with_crop(self.index_pair[step][0], cropx=160, cropy=144, cropz=192)
        img_B = load_4D_with_crop(self.index_pair[step][1], cropx=160, cropy=144, cropz=192)

        if self.norm:
            return torch.from_numpy(imgnorm(img_A)).float(), torch.from_numpy(imgnorm(img_B)).float()
        else:
            return torch.from_numpy(img_A).float(), torch.from_numpy(img_B).float()
        

class Predict_dataset_crop(Data.Dataset):
    def __init__(self, fixed_names, moving_names, norm=False):
        super(Predict_dataset_crop, self).__init__()
        self.fixed_names = fixed_names
        self.moving_names = moving_names
        self.norm = norm
        self.index_pair = list(itertools.product(moving_names, fixed_names))

    def __len__(self):
        return len(self.index_pair)
    
    def __getitem__(self, step):
        # Select sample
        img_A = load_4D_with_crop(self.index_pair[step][0]['img'], cropx=160, cropy=144, cropz=192)
        img_B = load_4D_with_crop(self.index_pair[step][1]['img'], cropx=160, cropy=144, cropz=192)
        label_A = load_4D_with_crop(self.index_pair[step][0]['seg'], cropx=160, cropy=144, cropz=192)
        label_B = load_4D_with_crop(self.index_pair[step][1]['seg'], cropx=160, cropy=144, cropz=192)

        if self.norm:
            img_A = imgnorm(img_A)
            img_B = imgnorm(img_B)

        img_A = torch.from_numpy(img_A)
        img_B = torch.from_numpy(img_B)
        label_A = torch.from_numpy(label_A)
        label_B = torch.from_numpy(label_B)

        output = {'fixed': img_A.float(), 'move': img_B.float(),
                  'fixed_label': label_A.float(), 'move_label': label_B.float()}
        
        return output
    

class Dataset_pairs_crop(Data.Dataset):
    def __init__(self, data_pairs, norm=False):
        super(Dataset_pairs_crop, self).__init__()
        self.data_pairs = data_pairs
        self.norm = norm
    
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, step):
        # Select sample
        img_A = load_4D_with_crop(self.data_pairs[step]['img1'], cropx=160, cropy=144, cropz=192)
        img_B = load_4D_with_crop(self.data_pairs[step]['img2'], cropx=160, cropy=144, cropz=192)

        if "seg1" in self.data_pairs[step].keys():
            seg_A = load_4D_with_crop(self.data_pairs[step]['seg1'], cropx=160, cropy=144, cropz=192)
            seg_A = torch.from_numpy(seg_A)
        else:
            seg_A = None
        if "seg2" in self.data_pairs[step].keys():
            seg_B = load_4D_with_crop(self.data_pairs[step]['seg2'], cropx=160, cropy=144, cropz=192)
            seg_B = torch.from_numpy(seg_B)
        else:
            seg_B = None
        
        if self.norm:
            img_A = imgnorm(img_A)
            img_B = imgnorm(img_B)

        img_A = torch.from_numpy(img_A)
        img_B = torch.from_numpy(img_B)

        if seg_A  is not None and seg_B  is not None:
            output = {'img1': img_A.float(), 'img2': img_B.float(),
                      'seg1': seg_A.float(), 'seg2': seg_B.float()}
        elif seg_A is not None:
            output = {'img1': img_A.float(), 'img2': img_B.float(),
                      'seg1': seg_A.float()}
        elif seg_B is not None:
            output = {'img1': img_A.float(), 'img2': img_B.float(),
                      'seg2': seg_B.float()}
        else:
            output = {'img1': img_A.float(), 'img2': img_B.float()}
            
        return output
    
    
def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))