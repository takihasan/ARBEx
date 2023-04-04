#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from config import BaseConfig


def get_all_ext_in_dir(path, ext=('jpg',)):
    """
    Get all files with certain extensions in directory

    Args:
        path: str
            root directory
        ext: tuple or list
            tuple or list of all extensions
    """

    ext = tuple(set(ext))

    path = Path(path)
    files = [str(f) for f in path.rglob("*")]
    files = [str(f) for f in files if f.endswith(ext)]

    return files


class AffWildDataset(Dataset):
    """
    Dataset for AffWild2
    """

    # index to class mapping
    index2class = {
            0: 'neutral',
            1: 'anger',
            2: 'disgust',
            3: 'fear',
            4: 'happiness',
            5: 'sadness',
            6: 'surprise',
            7: 'contempt',
            }

    # class to index mapping
    class2index = {
             'neutral': 0,
             'anger': 1,
             'disgust': 2,
             'fear': 3,
             'happiness': 4,
             'sadness': 5,
             'surprise': 6,
             'contempt': 7,
            }


    def __init__(self, dir_img, dir_txt, transform=None, verbose=False,
                 items_to_keep=None):
        """
        Args:
            dir_img: str
                base image directory
            dir_txt: str
                base text annotations directory
            transforms: callable (optional)
                transform to apply to images
            verbose: bool
                whether to print info to cli
            items_to_keep: iterable
                iterable of names of items to keep, if None, keep all
        """
        super().__init__()
        self.dir_img = dir_img
        self.dir_txt = dir_txt
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225)),
                ]
                    )
        else:
            self.transform = transform

        if verbose:
            print("Getting text files")
        # get all text files
        files_txt = get_all_ext_in_dir(self.dir_txt, ('txt', ))
        # base filename
        files_txt_filename = [f.split('/')[-1][:-4] for f in files_txt]

        xs = []  # images
        ys = []  # labels
        base_files = []  # base files

        # for each file
        z = zip(files_txt, files_txt_filename)
        if verbose:
            bar = tqdm(z, total=len(files_txt))
            bar.set_description('Reading labels')
        else:
            bar = z

        for f, f_name in bar:
            if items_to_keep is not None and f_name not in items_to_keep:
                continue
            # load labels
            y = np.loadtxt(f, skiprows=1, dtype=int)  # skip first row
            # load images
            x = get_all_ext_in_dir(os.path.join(self.dir_img, f_name), ('jpg', ))
            x = np.array(x)
            # keep only frames for which images exist
            # frame {i} is named {i:05d}.jpg, starting from 1
            x_int = np.array([int(i.split('/')[-1].split('.')[-2]) - 1
                              for i in x])
            y = y[x_int]

            index = y != -1  # ignore frames with label -1
            x = x[index]
            y = y[index]

            xs.append(x)
            ys.append(y)
            base_files.extend([f_name] * len(x))

        self.x = np.concatenate(xs)  # array of all paths to images
        self.y = np.concatenate(ys)  # array of all labels
        self.names = np.array(base_files)  # array of all filenames

        if verbose:
            print(f"Data from {self.dir_txt} loaded")
            print(f"Total items: {len(self)}")


    def __getitem__(self, index):
        # get
        x = self.x[index]
        y = self.y[index]

        # open the image
        x = Image.open(x)
        # transform
        x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.x)

    def get_x(self):
        """
        get all images
        """
        return self.xs

    def get_y(self):
        """
        get all labels
        """
        return self.ys

    def get_some_examples(self, xs=None, ys=None, shuffle=False):
        """
        Get all examples that satisfy the condition
        If {xs} is not None, return all examples whose filename is in {xs}
        If {ys} is not None, return all examples with label in {ys}

        Args:
            xs: iterable (optional)
                iterable of all filenames to return
            ys: iterable (optional)
                iterable of all labels to return
            shuffle: bool
                whether to shuffle the return data

        Return:
            x: array of paths to images
            y: array of labels

        Example:
            self.get_some_examples(ys=[0, 1])
                returns all examples with labels 0 or 1
            self.get_some_examples(xs=['title1', 'title2'])
                returns all examples of files with title 'title1' and 'title2'
        """
        if xs is None and ys is None:
            return None, None

        # filter on x
        if xs is not None:
            index_x = []
            for x in xs:
                index_x.append(np.where(self.names == x))
            index_x = np.concatenate(index_x).reshape(-1)
        else:
            index_x = np.ones(len(self.x), dtype=bool)

        # filter on y
        if ys is not None:
            index_y = []
            for y in ys:
                index_y.append(np.where(self.y == y))
            index_y = np.concatenate(index_y).reshape(-1)
        else:
            index_y = np.ones(len(self.y), dtype=bool)

        index = list(set.union(set(index_x), set(index_y)))
        index = np.array(index)
        np.sort(index)

        # shuffle
        if shuffle:
            np.random.shuffle(index)

        x = self.x[index]
        y = self.y[index]

        return x, y

    def get_random_example(self):
        files_all = np.array(list(set(self.names)))
        np.random.shuffle(files_all)
        file = files_all[0]
        return self.get_some_examples(xs={file})

    def get_weights(self):
        """
        get weight per class
        """
        labels = sorted(set(self.y))
        total = len(self) / len(labels)
        counts = [total/(self.y == l).sum() for l in labels]
        weights = torch.tensor(counts)
        return weights

class AffWildDatasetDev(Dataset):
    """
    Dataset for AffWild2
    """

    # index to class mapping
    index2class = {
            0: 'neutral',
            1: 'anger',
            2: 'disgust',
            3: 'fear',
            4: 'happiness',
            5: 'sadness',
            6: 'surprise',
            7: 'contempt',
            }

    # class to index mapping
    class2index = {
             'neutral': 0,
             'anger': 1,
             'disgust': 2,
             'fear': 3,
             'happiness': 4,
             'sadness': 5,
             'surprise': 6,
             'contempt': 7,
            }


    def __init__(self, dir_img, dir_txt, transform=None, verbose=False,
                 items_to_keep=None):
        """
        Args:
            dir_img: str
                base image directory
            dir_txt: str
                base text annotations directory
            transforms: callable (optional)
                transform to apply to images
            verbose: bool
                whether to print info to cli
            items_to_keep: iterable
                iterable of names of items to keep, if None, keep all
        """
        super().__init__()
        self.dir_img = dir_img
        self.dir_txt = dir_txt
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225)),
                ]
                    )
        else:
            self.transform = transform

        if verbose:
            print("Getting text files")
        # get all text files
        files_txt = get_all_ext_in_dir(self.dir_txt, ('txt', ))
        # base filename
        files_txt_filename = [f.split('/')[-1][:-4] for f in files_txt]

        xs = []  # images
        ys = []  # labels
        base_files = []  # base files

        # for each file
        z = zip(files_txt, files_txt_filename)
        if verbose:
            bar = tqdm(z, total=len(files_txt))
            bar.set_description('Reading labels')
        else:
            bar = z

        for f, f_name in bar:
            if items_to_keep is not None and f_name not in items_to_keep:
                continue
            # load labels
            y = np.loadtxt(f, skiprows=1, dtype=int)  # skip first row
            # load images
            x = get_all_ext_in_dir(os.path.join(self.dir_img, f_name), ('jpg', ))
            x = np.array(x)
            # keep only frames for which images exist
            # frame {i} is named {i:05d}.jpg, starting from 1
            x_int = np.array([int(i.split('/')[-1].split('.')[-2]) - 1
                              for i in x])
            y = y[x_int]

            index = y != -1  # ignore frames with label -1
            x = x[index]
            y = y[index]

            xs.append(x)
            ys.append(y)
            base_files.extend([f_name] * len(x))

        self.x = np.concatenate(xs)  # array of all paths to images
        self.y = np.concatenate(ys)  # array of all labels
        self.names = np.array(base_files)  # array of all filenames

        if verbose:
            print(f"Data from {self.dir_txt} loaded")
            print(f"Total items: {len(self)}")


    def __getitem__(self, index):
        # get
        img = self.x[index]
        y = self.y[index]

        # open the image
        x = Image.open(img)
        # transform
        x = self.transform(x)

        return x, y, img

    def __len__(self):
        return len(self.x)

    def get_x(self):
        """
        get all images
        """
        return self.xs

    def get_y(self):
        """
        get all labels
        """
        return self.ys

    def get_some_examples(self, xs=None, ys=None, shuffle=False):
        """
        Get all examples that satisfy the condition
        If {xs} is not None, return all examples whose filename is in {xs}
        If {ys} is not None, return all examples with label in {ys}

        Args:
            xs: iterable (optional)
                iterable of all filenames to return
            ys: iterable (optional)
                iterable of all labels to return
            shuffle: bool
                whether to shuffle the return data

        Return:
            x: array of paths to images
            y: array of labels

        Example:
            self.get_some_examples(ys=[0, 1])
                returns all examples with labels 0 or 1
            self.get_some_examples(xs=['title1', 'title2'])
                returns all examples of files with title 'title1' and 'title2'
        """
        if xs is None and ys is None:
            return None, None

        # filter on x
        if xs is not None:
            index_x = []
            for x in xs:
                index_x.append(np.where(self.names == x))
            index_x = np.concatenate(index_x).reshape(-1)
        else:
            index_x = np.ones(len(self.x), dtype=bool)

        # filter on y
        if ys is not None:
            index_y = []
            for y in ys:
                index_y.append(np.where(self.y == y))
            index_y = np.concatenate(index_y).reshape(-1)
        else:
            index_y = np.ones(len(self.y), dtype=bool)

        index = list(set.union(set(index_x), set(index_y)))
        index = np.array(index)
        np.sort(index)

        # shuffle
        if shuffle:
            np.random.shuffle(index)

        x = self.x[index]
        y = self.y[index]

        return x, y

    def get_random_example(self):
        files_all = np.array(list(set(self.names)))
        np.random.shuffle(files_all)
        file = files_all[0]
        return self.get_some_examples(xs={file})


class AffWildDatasetTest(Dataset):
    """
    Dataset for AffWild2
    """

    # index to class mapping
    index2class = {
            0: 'neutral',
            1: 'anger',
            2: 'disgust',
            3: 'fear',
            4: 'happiness',
            5: 'sadness',
            6: 'surprise',
            7: 'contempt',
            }

    # class to index mapping
    class2index = {
             'neutral': 0,
             'anger': 1,
             'disgust': 2,
             'fear': 3,
             'happiness': 4,
             'sadness': 5,
             'surprise': 6,
             'contempt': 7,
            }


    def __init__(self, dir_img, file_txt, transform=None, verbose=False,
                 items_to_keep=None):
        """
        Args:
            dir_img: str
                base image directory
            file_txt: str
                text file containing names of videos
            transforms: callable (optional)
                transform to apply to images
            verbose: bool
                whether to print info to cli
            items_to_keep: iterable
                iterable of names of items to keep, if None, keep all
        """
        super().__init__()
        self.dir_img = dir_img
        self.file_txt = file_txt
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225)),
                ]
                    )
        else:
            self.transform = transform

        if verbose:
            print("Getting text files")
        # get all text files
        with open(file_txt, 'r') as f:
            files_txt_filename = [l.strip() for l in f.readlines()]

        xs = []  # images
        base_files = []  # base files

        # for each file
        if verbose:
            bar = tqdm(files_txt_filename, total=len(files_txt_filename))
            bar.set_description('Reading labels')
        else:
            bar = files_txt_filename

        for f_name in bar:
            if items_to_keep is not None and f_name not in items_to_keep:
                continue
            # load images
            x = get_all_ext_in_dir(os.path.join(self.dir_img, f_name), ('jpg', ))
            x = np.array(x)
            xs.append(x)
            base_files += ['/'.join(i.split('/')[-2:]) for i in x]

        self.x = np.concatenate(xs)  # array of all paths to images
        self.names = np.array(base_files)  # array of all filenames

        if verbose:
            print(f"Data from {self.dir_txt} loaded")
            print(f"Total items: {len(self)}")


    def __getitem__(self, index):
        # get
        x = self.x[index]
        base_file = self.names[index]

        # open the image
        x = Image.open(x)
        # transform
        x = self.transform(x)

        return x, base_file

    def __len__(self):
        return len(self.x)

    def get_x(self):
        """
        get all images
        """
        return self.xs

    def get_y(self):
        """
        get all labels
        """
        return self.ys

    def get_some_examples(self, xs=None, ys=None, shuffle=False):
        """
        Get all examples that satisfy the condition
        If {xs} is not None, return all examples whose filename is in {xs}
        If {ys} is not None, return all examples with label in {ys}

        Args:
            xs: iterable (optional)
                iterable of all filenames to return
            ys: iterable (optional)
                iterable of all labels to return
            shuffle: bool
                whether to shuffle the return data

        Return:
            x: array of paths to images
            y: array of labels

        Example:
            self.get_some_examples(ys=[0, 1])
                returns all examples with labels 0 or 1
            self.get_some_examples(xs=['title1', 'title2'])
                returns all examples of files with title 'title1' and 'title2'
        """
        if xs is None and ys is None:
            return None, None

        # filter on x
        if xs is not None:
            index_x = []
            for x in xs:
                index_x.append(np.where(self.names == x))
            index_x = np.concatenate(index_x).reshape(-1)
        else:
            index_x = np.ones(len(self.x), dtype=bool)

        # filter on y
        if ys is not None:
            index_y = []
            for y in ys:
                index_y.append(np.where(self.y == y))
            index_y = np.concatenate(index_y).reshape(-1)
        else:
            index_y = np.ones(len(self.y), dtype=bool)

        index = list(set.union(set(index_x), set(index_y)))
        index = np.array(index)
        np.sort(index)

        # shuffle
        if shuffle:
            np.random.shuffle(index)

        x = self.x[index]
        y = self.y[index]

        return x, y

    def get_random_example(self):
        files_all = np.array(list(set(self.names)))
        np.random.shuffle(files_all)
        file = files_all[0]
        return self.get_some_examples(xs={file})


class DatasetXY(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x  # images
        self.y = y  # labels

        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.ColorJitter(),
                transforms.ToTensor(),
                transforms.RandomErasing(p=1, scale=(0.05, 0.05)),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225)),
                ])

        self.transform = transform

    def __getitem__(self, index):
        img = self.x[index]
        label = self.y[index]

        img = Image.open(img)
        img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.x)

    def get_weights(self):
        """
        get weight per class
        """
        labels = sorted(set(self.y))
        total = len(self) / len(labels)
        counts = [total/(self.y == l).sum() for l in labels]
        weights = torch.tensor(counts)
        return weights


def get_dataset_train(config=BaseConfig, transform=None):
    """
    get training dataset

    Args:
        config: Config
    """
    return AffWildDataset(config.DIR_IMG, config.DIR_ANN_TRAIN, transform=transform)


def get_dataset_dev(config=BaseConfig, transform=None, img=False):
    """
    get dev dataset

    Args:
        config: Config
            config dictionary
        transform: callable
            transform to apply to image
        img: bool
            if True, also return path to img
    """
    if img:
        return AffWildDatasetDev(config.DIR_IMG, config.DIR_ANN_DEV, transform=transform)
    else:
        return AffWildDataset(config.DIR_IMG, config.DIR_ANN_DEV, transform=transform)


def get_dataloader_train(config=BaseConfig, batch_size=32, shuffle=True,
                         transform=None, image_size=224):
    temp_image_resize_size = int(image_size * 1.1)
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(temp_image_resize_size),
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225)),
            ]
                )

    ds = get_dataset_train(config=config, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return loader


def get_dataloader_dev(config=BaseConfig, batch_size=32, shuffle=True,
                       transform=None, image_size=224, img=False):
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225)),
            ]
                )

    ds = get_dataset_dev(config=config, transform=transform, img=img)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return loader


def dir_to_df(dir_img, dir_ann):
    """
    Make a DataFrame with paths to images, labels and original video name

    Args:
        dir_img: str
            path to directory containing all images
        dir_ann: str
            path to directory containing all annotations
    """
    # get all files
    files_txt = get_all_ext_in_dir(dir_ann, ('txt', ))
    # pure filenames
    files_txt_filename = [f.split('/')[-1][:-4] for f in files_txt]

    xs = []  # images
    ys = []  # labels
    base_files = []  # video filename

    # for each file
    bar = tqdm(zip(files_txt, files_txt_filename), total=len(files_txt))

    for f, f_name in bar:
        # load labels
        y = np.loadtxt(f, skiprows=1, dtype=int)
        # path to image dir
        path_img = os.path.join(dir_img, f_name)
        # get all images
        x = np.array(get_all_ext_in_dir(path_img, ('jpg', )))
        # keep only frames for which images exist
        # frame {i} is named {i:05d}.jpg, starting from 1
        index = np.array([int(i.split('/')[-1].split('.')[-2]) - 1
                          for i in x])
        y = y[index]

        # ignore frames with label -1
        index = y != -1
        x = x[index]
        y = y[index]

        xs.append(x)
        ys.append(y)
        base_files.extend([f_name] * len(x))

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    names = np.array(base_files)

    df = pd.DataFrame({
        'img': xs,
        'label': ys,
        'file': names,
        })

    return df
    df.to_csv(save_file)


def make_label_file(dir_img, dir_ann, save_file):
    """
    Make a .csv file containing all paths to images and their labels

    Args:
        dir_img: str
            path to directory containing all images
        dir_ann: str
            path to directory containing all annotations
        save_file: str
            where to save the .csv file
    """
    df = dir_to_df(dir_img, dir_ann)
    if os.path.exists(save_file):
        print(f'File "{save_file}" already exits. Not overwriting.')
    else:
        df.to_csv(save_file)


class DataMaker():
    """
    DataMaker instance.
    Provides methods for creating fresh pytorch data loaders.
    """
    def __init__(self, df, max_frames_per_vid=512,
                 col_img='img', col_label='label', col_file='file'):
        """
        Args:
            df: pd.DataFrame
                DataFrame containing paths to images, labels and filenames
            max_frames_per_vid: int
                maximum frames to keep for each video
            col_img: str
                name of image column
            col_label: str
                name of label column
            col_file: str
                name of file column
        """
        self.df = df
        self.max_frames_per_vid = max_frames_per_vid

        self.col_img = col_img
        self.col_label = col_label
        self.col_file = col_file

        self.all_labels = self.df[self.col_label].unique()
        self.all_files = self.df[col_file].unique()

    def get_dataset_full(self, transform=None):
        x = self.df[self.col_img]
        y = self.df[self.col_label]
        ds = DatasetXY(x, y, transform=transform)
        return ds

    def get_dataset(self, balance_classes=False, examples_per_class=1024,
                    transform=None):
        """
        Args:
            balance_classes: bool
                keep same number of examples for each class
            examples_per_class: int
                number of examples to keep per class
            transform: callable
                transform to apply to images
        """
        imgs = self.df[self.col_img].values
        labels = self.df[self.col_label].values
        files = self.df[self.col_file].values
        index = np.arange(len(imgs))

        index_to_keep = []
        for file in self.all_files:
            index_file = []
            for label in self.all_labels:
                filter_file = files == file
                filter_label = labels == label
                filter_all = filter_file & filter_label
                filter_all = index[filter_all].copy()
                np.random.shuffle(filter_all)
                filter_all = filter_all[:self.max_frames_per_vid]
                index_file.append(filter_all)
            index_file = np.concatenate(index_file)
            np.random.shuffle(index_file)
            index_file = index_file[:self.max_frames_per_vid]
            index_to_keep.append(index_file)

        index_to_keep = np.concatenate(index_to_keep)
        x = imgs[index_to_keep]
        y = labels[index_to_keep]

        if balance_classes:
            index = np.arange(len(x))
            index_to_keep = []
            for label in self.all_labels:
                filter_label = y == label
                index_to_keep.append(index[filter_label][:examples_per_class])
            index_to_keep = np.concatenate(index_to_keep)

            x = x[index_to_keep]
            y = y[index_to_keep]

        ds = DatasetXY(x, y, transform)
        return ds

    def get_dataloader(self,
                       batch_size,
                       shuffle=True,
                       balance_classes=True,
                       examples_per_class=1000,
                       transform=None):

        ds = self.get_dataset(balance_classes=balance_classes,
                              examples_per_class=examples_per_class,
                              transform=transform)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

        return dl

    def get_dataloader_full(self, batch_size, shuffle=False, transform=None):
        ds = self.get_dataset_full(transform=transform)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

        return dl


def get_datamaker(config=BaseConfig):
    dir_img = config.DIR_IMG
    dir_ann = config.DIR_ANN_TRAIN
    df = dir_to_df(dir_img, dir_ann)
    data_maker = DataMaker(df)
    return data_maker
