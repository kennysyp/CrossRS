import csv
import glob
import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
import re

# 28:2:10 for training, validation and testing

seed = 19260817
random.seed(seed)
np.random.seed(seed)

data_dir = "/home/gdut-627/huang/dataset/neurite-oasis.v1.0/" 

def path_to_id(path):
    return path.split('/')[6].split('_')[2]


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


def take_data_pairs_atlas(fixed_data, moving_data):
    data_pairs = []
    for i in range(len(fixed_data)):
        d1 = fixed_data[i]
        for j in range(len(moving_data)):
            d2 = moving_data[j]
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


def create_data_csv(data_dir, use_label_num=21):
    image_paths = sorted(glob.glob(data_dir + 'OASIS_OAS1_*_MR1/aligned_norm.nii.gz'))
    segmentation_paths = sorted(glob.glob(data_dir + 'OASIS_OAS1_*_MR1/aligned_seg35.nii.gz'))

    seg_ids = list(map(path_to_id, segmentation_paths))
    img_ids = map(path_to_id, image_paths)
    data = []
    for img_index, img_id in enumerate(img_ids):
        data_item = {'img': image_paths[img_index]}
        if img_id in seg_ids:
            data_item['seg'] = segmentation_paths[seg_ids.index(img_id)]
        data.append(data_item)

    # data = random.sample(data, 85)

    train_set, test_set = train_test_split(data, test_size=150, random_state=seed)
    train_set, val_set = train_test_split(train_set, test_size=20, random_state=seed)
    print(
        f"train set: {len(train_set)}, val set: {len(val_set)}, test set: {len(test_set)}")
    print(
        f"use {use_label_num} label and {len(train_set) - use_label_num} unlabel data")

    headers = ('img', 'seg')
    csvwriter(train_set, "./data/train.csv", headers)

    # fixed_set, test_set = train_test_split(test_set, test_size=8, random_state=seed)

    # for index in random.sample(train_set, len(train_set) - use_label_num):
    #     del index['seg']

    # csvwriter(train_set, "./data/train_21labels.csv", headers)
    csvwriter(val_set, "./data/val.csv", headers)
    csvwriter(test_set, "./data/test.csv", headers)
    # csvwriter(fixed_set, "./data/atlas.csv", headers)


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


def process_label(label_dir="seg35_labels.txt"):
    seg_table = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]

    file1 = open(label_dir, 'r')
    Lines = file1.readlines()
    dict = {}
    seg_i = 0
    seg_look_up = []
    for seg_label in seg_table:
        for line in Lines:
            line = re.sub(' +', ' ', line).split(' ')
            try:
                int(line[0])
            except:
                continue
            if int(line[0]) == seg_label:
                seg_look_up.append([seg_i, int(line[0]), line[1]])
                dict[seg_i] = line[1]
        seg_i += 1
    return dict


def test_for_ants():
    test_data_dir = "/home/gdut-627/huang/dataset/LPBA40/pre-processing/"
    image_paths = sorted(glob.glob(test_data_dir + '*.skullstripped_trans.nii.gz'))
    segmentation_paths = sorted(glob.glob(test_data_dir + '*.label_trans.nii.gz'))

    test_set = []
    for img_index in range(len(image_paths)):
        data_item = {'img': image_paths[img_index]}
        data_item['seg'] = segmentation_paths[img_index]
        test_set.append(data_item)

    headers = ('img', 'seg')

    csvwriter(test_set, "./data/test_ants.csv", headers)


def create_train_data_csv(data_dir):
    train_data_dir = "/home/gdut-627/huang/RRS-OASIS/data/"

    train_path = read_data_csv(train_data_dir + "train.csv")
    train_path_10 = train_path
    train_path_40 = train_path
    train_path_70 = train_path

    # for index in random.sample(train_path_10, int(len(train_path) * 0.9)):
    #     del index['seg']

    # for index in random.sample(train_path_40, int(len(train_path) * 0.6)):
    #     del index['seg']

    for index in random.sample(train_path_70, int(len(train_path) * 0.3)):
        del index['seg']

    headers = ('img', 'seg')
    # csvwriter(train_path_10, "./data/train_10per.csv", headers)
    # csvwriter(train_path_40, "./data/train_40per.csv", headers)
    csvwriter(train_path_70, "./data/train_70per.csv", headers)


if __name__ == "__main__":
    # create_data_csv(data_dir=data_dir)
    print("hello")

    # create_train_data_csv(data_dir=data_dir)

    # test_for_ants()
