import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
import re
import glob
import csv
import itertools
from torch.utils.data.sampler import Sampler

# 28:2:10 for training, validation and testing

seed = 19260817
random.seed(seed)
np.random.seed(seed)

data_dir = "/home/gdut-627/huang/dataset/LPBA40/lpba40/"

def path_to_id(path):
    return path.split('/')[7].split('.')[0]


def take_data_pairs_ants(fixed, moving):
    """Given a list of dicts that have keys for an image and maybe a segmentation,
    return a list of dicts corresponding to *pairs* of images and maybe segmentations.
    Pairs consisting of a repeated image are not included.
    If symmetric is set to True, then for each pair that is included, its reverse is also included"""
    data_pairs = []
    for i in range(len(fixed)):
        for j in range(len(moving)):
            d1 = fixed[i]
            d2 = moving[j]
            pair = {"img1": d1["img"], "img2": d2["img"], "seg1": d1["seg"], "seg2": d2["seg"]}
            data_pairs.append(pair)
    return data_pairs


def take_data_pairs(data, symmetric=True):
    """Given a list of dicts that have keys for an image and maybe a segmentation,
    return a list of dicts corresponding to *pairs* of images and maybe segmentations.
    Pairs consisting of a repeated image are not included.
    If symmetric is set to True, then for each pair that is included, its reverse is also included"""
    data_pairs = []
    for i in range(len(data)):
        j_limit = len(data) if symmetric else i
        for j in range(j_limit):
            if j == i:
                continue
            d1 = data[i]
            d2 = data[j]
            pair = {"img1": d1["img"], "img2": d2["img"]}
            if "seg" in d1.keys():
                pair["seg1"] = d1["seg"]
            if "seg" in d2.keys():
                pair["seg2"] = d2["seg"]
            data_pairs.append(pair)
    return data_pairs


def subdivide_list_of_data_pairs(data_pairs_list):
    out_dict = {"00": [], "01": [], "10": [], "11": []}
    for d in data_pairs_list:
        if "seg1" in d.keys() and "seg2" in d.keys():
            out_dict["11"].append(d)
        elif "seg1" in d.keys():
            out_dict["10"].append(d)
        elif "seg2" in d.keys():
            out_dict["01"].append(d)
        else:
            out_dict["00"].append(d)
    return out_dict


seg_availabilities = ['00', '01', '10', '11']


def create_batch_generator(dataloader_subdivided, weights=None):
    """
    Create a batch generator that samples data pairs with various segmentation availabilities.

    Arguments:
        dataloader_subdivided : a mapping from the labels in seg_availabilities to dataloaders
        weights : a list of probabilities, one for each label in seg_availabilities;
                  if not provided then we weight by the number of data items of each type,
                  effectively sampling uniformly over the union of the datasets

    Returns: batch_generator
        A function that accepts a number of batches to sample and that returns a generator.
        The generator will weighted-randomly pick one of the seg_availabilities and
        yield the next batch from the corresponding dataloader.
    """
    if weights is None:
        weights = np.array([len(dataloader_subdivided[s]) for s in seg_availabilities])
    weights = np.array(weights)
    weights = weights / weights.sum()
    dataloader_subdivided_as_iterators = {s: iter(d) for s, d in dataloader_subdivided.items()}

    def batch_generator(num_batches_to_sample):
        for _ in range(num_batches_to_sample):
            seg_availability = np.random.choice(seg_availabilities, p=weights)
            try:
                yield next(dataloader_subdivided_as_iterators[seg_availability])
            except StopIteration:  # If dataloader runs out, restart it
                dataloader_subdivided_as_iterators[seg_availability] =\
                    iter(dataloader_subdivided[seg_availability])
                yield next(dataloader_subdivided_as_iterators[seg_availability])
    return batch_generator



def csvwriter(data, filename, headers):
    with open(filename, 'w', encoding='utf-8', newline='') as f:
        write = csv.DictWriter(f, headers)
        write.writeheader()
        for data_item in data:
            write.writerow(data_item)


def create_data_csv(data_dir, use_label_num=5):
    image_paths = sorted(glob.glob(data_dir + '*.skullstripped.img.gz'))
    segmentation_paths = sorted(glob.glob(data_dir + '*.label.img.gz'))

    print(len(image_paths))

    seg_ids = list(map(path_to_id, segmentation_paths))
    img_ids = map(path_to_id, image_paths)
    data = []
    for img_index, img_id in enumerate(img_ids):
        data_item = {'img': image_paths[img_index]}
        if img_id in seg_ids:
            data_item['seg'] = segmentation_paths[seg_ids.index(img_id)]
        data.append(data_item)

    train_set, test_set = train_test_split(data, test_size=10, random_state=seed)
    train_set, val_set = train_test_split(train_set, test_size=2, random_state=seed)
    print(
        f"train set: {len(train_set)}, val set: {len(val_set)}, test set: {len(test_set)}")
    print(
        f"use {use_label_num} label and {len(train_set) - use_label_num} unlabel data")

    headers = ('img', 'seg')
    csvwriter(train_set, "./data/train.csv", headers)
    csvwriter(val_set, "./data/val.csv", headers)
    csvwriter(test_set, "./data/test.csv", headers)


def read_data_csv(data_csv) -> list:
    data = []
    with open(data_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for data_item in reader:
            for k in list(data_item.keys()):
                if not data_item[k]:
                    del data_item[k]
            data.append(data_item)

    return data


def process_label(label_dir="labels.txt"):
    seg_table = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 
                 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]

    file1 = open(label_dir, 'r')
    Lines = file1.readlines()
    dict = {}
    seg_i = 0
    seg_look_up = []
    for seg_label in seg_table:
        for line in Lines:
            line = re.sub('[\\t\\n]', ' ',line).split(' ')

            if int(line[0]) == seg_label:
                seg_look_up.append([seg_i, int(line[0]), line[1]])
                dict[seg_i] = line[1]
        seg_i += 1
    return dict


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices
    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


def create_train_data_csv(data_dir):
    train_data_dir = "/home/gdut-627/huang/RRS-LPBA/data/"

    train_path = read_data_csv(train_data_dir + "train.csv")
    train_path_25 = train_path

    # for index in random.sample(train_path_10, int(len(train_path) * 0.9)):
    #     del index['seg']

    # for index in random.sample(train_path_40, int(len(train_path) * 0.6)):
    #     del index['seg']

    for index in random.sample(train_path_25, int(len(train_path) * 0.75)):
        del index['seg']

    headers = ('img', 'seg')
    csvwriter(train_path_25, "./data/train_25per.csv", headers)

if __name__ == "__main__":
    create_data_csv(data_dir=data_dir)
    create_train_data_csv(data_dir=data_dir)