# -*- coding: utf-8 -*-

import numpy as np
import torch.utils.data as data
from sklearn.model_selection import train_test_split

import utils.datasets as ds
from dataloader.TSC_data_loader import TSC_data_loader

class MTL(data.Dataset):
    def __init__(self, data, targets, K, transform, totensor_transform, transform_cuts):
        self.data = np.asarray(data, dtype=np.float32)
        self.targets = np.asarray(targets, dtype=np.int16)
        self.K = K  # tot number of augmentations
        self.transform = transform
        self.totensor_transform = totensor_transform
        self.transform_cuts = transform_cuts

    def __getitem__(self, index):
        # print("### {}".format(index))
        img, target = self.data[index], self.targets[index]
        img_list0 = list()
        img_list1 = list()

        if self.transform is not None:
            for _ in range(self.K):
                img_transformed = self.transform(img.copy())
                img_cut0, img_cut1 = self.transform_cuts(img_transformed)

                img_list0.append(self.totensor_transform(img_cut0))
                img_list1.append(self.totensor_transform(img_cut1))

        return img_list0, img_list1, target

    def __len__(self):
        return self.data.shape[0]


class UCR2018(data.Dataset):

    def __init__(self, data, targets, transform):
        self.data = np.asarray(data, dtype=np.float32)
        self.targets = np.asarray(targets, dtype=np.int64)
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        if self.transform is not None:
            img_transformed = self.transform(img.copy())
        else:
            img_transformed = img

        return img_transformed, target

    def __len__(self):
        return self.data.shape[0]


class traindataset(data.Dataset):

    def __init__(self, data, targets, transform, label):
        self.data = np.asarray(data, dtype=np.float32)
        self.targets = np.asarray(targets, dtype=np.int64)
        self.transform = transform
        self.label = label

    def __getitem__(self, index):
        img_o, target = self.data[index], self.targets[index]
        if self.transform is not None:
            img_a = self.transform(img_o.copy())
        else:
            img_a = img_o
        img_o, img_a = self.colorful_spectrum_mix(img_o, img_a)

        if self.label:
            return img_o, img_a, target
        else:
            return img_o, img_a

    def __len__(self):
        return self.data.shape[0]

    def colorful_spectrum_mix(self, img1, img2, alpha=1.0, ratio=1.0):
        """Input image size: ndarray of [H, W, C]"""
        lam = np.random.uniform(0, alpha)

        assert img1.shape == img2.shape
        t, c = img1.shape
        t_crop = int(t * np.sqrt(ratio))

        t_start = t // 2 - t_crop // 2

        img1_fft = np.fft.fft(img1)
        img2_fft = np.fft.fft(img2)
        img1_abs, img1_pha = np.abs(img1_fft), np.angle(img1_fft)  # shibu xubu
        img2_abs, img2_pha = np.abs(img2_fft), np.angle(img2_fft)

        img1_abs = np.fft.fftshift(img1_abs, axes=(0))
        img2_abs = np.fft.fftshift(img2_abs, axes=(0))

        img1_abs_ = np.copy(img1_abs)
        img2_abs_ = np.copy(img2_abs)
        img1_abs[t_start:t_start + t_crop] = \
            lam * img2_abs_[t_start:t_start + t_crop] + (1 - lam) * img1_abs_[
                                                                    t_start:t_start + t_crop]
        img2_abs[t_start:t_start + t_crop] = \
            lam * img1_abs_[t_start:t_start + t_crop] + (1 - lam) * img2_abs_[
                                                                    t_start:t_start + t_crop]

        img1_abs = np.fft.ifftshift(img1_abs, axes=(0))
        img2_abs = np.fft.ifftshift(img2_abs, axes=(0))

        img21 = img1_abs * (np.e ** (1j * img1_pha))
        img12 = img2_abs * (np.e ** (1j * img2_pha))
        img21 = np.real(np.fft.ifft(img21))
        img12 = np.real(np.fft.ifft(img12))
        return img21, img12


class MultiUCR2018_Intra(data.Dataset):

    def __init__(self, data, targets, K, transform, transform_cut, totensor_transform):
        self.data = np.asarray(data, dtype=np.float32)
        self.targets = np.asarray(targets, dtype=np.int16)
        self.K = K  # tot number of augmentations
        self.transform = transform
        self.transform_cut = transform_cut
        self.totensor_transform = totensor_transform

    def __getitem__(self, index):
        # print("### {}".format(index))
        img, target = self.data[index], self.targets[index]
        img_list0 = list()
        img_list1 = list()
        label_list = list()

        if self.transform is not None:
            for _ in range(self.K):
                img_transformed = self.transform(img.copy())
                img_cut0, img_cut1, label = self.transform_cut(img_transformed)
                img_list0.append(self.totensor_transform(img_cut0))
                img_list1.append(self.totensor_transform(img_cut1))
                label_list.append(label)

        return img_list0, img_list1, label_list, target

    def __len__(self):
        return self.data.shape[0]


class MultiUCR2018_InterIntra(data.Dataset):

    def __init__(self, data, targets, K, transform, transform_cut, totensor_transform):
        self.data = np.asarray(data, dtype=np.float32)
        self.targets = np.asarray(targets, dtype=np.int16)
        self.K = K  # tot number of augmentations
        self.transform = transform
        self.transform_cut = transform_cut
        self.totensor_transform = totensor_transform

    def __getitem__(self, index):
        # print("### {}".format(index))
        img, target = self.data[index], self.targets[index]
        img_list = list()
        img_list0 = list()
        img_list1 = list()
        label_list = list()

        if self.transform is not None:
            for _ in range(self.K):
                img_transformed = self.transform(img.copy())
                img_cut0, img_cut1, label = self.transform_cut(img_transformed)
                img_list.append(self.totensor_transform(img_transformed))
                img_list0.append(self.totensor_transform(img_cut0))
                img_list1.append(self.totensor_transform(img_cut1))
                label_list.append(label)

        return img_list, img_list0, img_list1, label_list, target

    def __len__(self):
        return self.data.shape[0]


class MultiUCR2018_Forecast(data.Dataset):
    """Override torchvision CIFAR10 for multi-image management.
    Similar class can be defined for other datasets (e.g. CIFAR100).
    Given K total augmentations, it returns a list of lenght K with
    different augmentations of the input mini-batch.
    """

    def __init__(self, data, targets, K, transform, transform_cut, totensor_transform):
        self.data = np.asarray(data, dtype=np.float32)
        self.targets = np.asarray(targets, dtype=np.int16)
        self.K = K  # tot number of augmentations
        self.transform = transform
        self.transform_cut = transform_cut
        self.totensor_transform = totensor_transform

    def __getitem__(self, index):
        # print("### {}".format(index))
        img, target = self.data[index], self.targets[index]
        img_list0 = list()
        img_list1 = list()

        if self.transform is not None:
            for _ in range(self.K):
                img_transformed = self.transform(img.copy())
                img_cut0, img_cut1 = self.transform_cut(img_transformed)

                img_list0.extend(self.totensor_transform(img_cut0))
                img_list1.extend(self.totensor_transform(img_cut1))

        return img_list0, img_list1, target

    def __len__(self):
        return self.data.shape[0]


class MultiUCR2018_PF(data.Dataset):
    """Override torchvision CIFAR10 for multi-image management.
    Similar class can be defined for other datasets (e.g. CIFAR100).
    Given K total augmentations, it returns a list of lenght K with
    different augmentations of the input mini-batch.
    """

    def __init__(self, data, targets, K, transform, totensor_transform, transform_cuts):
        self.data = np.asarray(data, dtype=np.float32)
        self.targets = np.asarray(targets, dtype=np.int16)
        self.K = K  # tot number of augmentations
        self.transform = transform
        self.totensor_transform = totensor_transform
        self.transform_cuts = transform_cuts

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img_list = list()
        img_list_past = list()
        img_list_future = list()

        if self.transform is not None:
            for _ in range(self.K):
                img_transformed = self.transform(img.copy())
                img_list.append(self.totensor_transform(img_transformed))
                (img_transformed_P, img_transformed_F) = self.transform_cuts(img_transformed.copy())

                img_list_past.append(self.totensor_transform(img_transformed_P))
                img_list_future.append(self.totensor_transform(img_transformed_F))

        else:
            img_list = img

        return img_list, img_list_past, img_list_future, target

    def __len__(self):
        return self.data.shape[0]


class MultiUCR2018(data.Dataset):

    def __init__(self, data, targets, K, transform):
        self.data = np.asarray(data, dtype=np.float32)
        self.targets = np.asarray(targets, dtype=np.int16)
        self.K = K  # tot number of augmentations
        self.transform = transform

    def __getitem__(self, index):
        # print("### {}".format(index))
        img, target = self.data[index], self.targets[index]
        img_list = list()
        if self.transform is not None:
            for _ in range(self.K):
                img_transformed = self.transform(img.copy())
                img_list.append(img_transformed)
        else:
            img_list = img

        return img_list, target

    def __len__(self):
        return self.data.shape[0]


def get_EpilepticSeizure(dataset_path, dataset_name):
    data = []
    data_x = []
    data_y = []
    f = open('{}/{}/data.csv'.format(dataset_path, dataset_name), 'r')
    for line in range(0, 11501):
        if line == 0:
            f.readline()
            continue
        else:
            data.append(f.readline().strip())
    for i in range(0, 11500):
        tmp = data[i].split(",")
        del tmp[0]
        del tmp[178]
        data_x.append(tmp)
        data_y.append(data[i][-1])
    data_x = np.asfarray(data_x, dtype=np.float32)
    data_y = np.asarray([int(x) - 1 for x in data_y], dtype=np.int64)
    return data_x, data_y


def load_ucr2018(dataset_path, dataset_name,config):
    ##################
    # load raw data
    ##################
    nb_class = ds.nb_classes(dataset_name)
    nb_dims = ds.nb_dims(dataset_name)

    if dataset_name in ['MFPT', 'XJTU']:
        x = np.load("{}/{}/{}_data.npy".format(dataset_path, dataset_name, dataset_name))
        y = np.load("{}/{}/{}_label.npy".format(dataset_path, dataset_name, dataset_name))

        (x_train, x_test) = (x[:100], x[100:])
        (y_train, y_test) = (y[:100], y[100:])

    elif dataset_name in ['EpilepticSeizure']:
        data_x, data_y = get_EpilepticSeizure(dataset_path, dataset_name)

        (x_train, x_test) = (data_x[:int(0.5 * data_x.shape[0])], data_x[int(0.5 * data_x.shape[0]):])
        (y_train, y_test) = (data_y[:int(0.5 * data_x.shape[0])], data_y[int(0.5 * data_x.shape[0]):])

    else:
        x_train, y_train, x_test, y_test = TSC_data_loader(dataset_path, dataset_name)

    config['model_params']['seq_len'] = x_train.shape[1]
    config['model_params']['in_channel'] = 1 if len(x_train.shape) == 2 else x_train.shape[2]

    if dataset_name in ['EpilepticSeizure','MFPT', 'XJTU']:
        nb_timesteps = int(x_train.shape[1] / nb_dims)
        input_shape = (nb_timesteps, nb_dims)

        ############################################
        # Combine all train and test data for resample
        ############################################

        x_all = np.concatenate((x_train, x_test), axis=0)
        y_all = np.concatenate((y_train, y_test), axis=0)
        ts_idx = list(range(x_all.shape[0]))
        # np.random.shuffle(ts_idx)
        x_all = x_all[ts_idx]
        y_all = y_all[ts_idx]

        label_idxs = np.unique(y_all)
        class_stat_all = {}
        for idx in label_idxs:
            class_stat_all[idx] = len(np.where(y_all == idx)[0])
        print("[Stat] All class: {}".format(class_stat_all))

        test_idx = []
        val_idx = []
        train_idx = []
        for idx in label_idxs:
            target = list(np.where(y_all == idx)[0])
            nb_samp = int(len(target))
            test_idx += target[:int(nb_samp * 0.2)]
            val_idx += target[int(nb_samp * 0.2):int(nb_samp * 0.4)]
            train_idx += target[int(nb_samp * 0.4):]

        x_test = x_all[test_idx]
        y_test = y_all[test_idx]
        x_val = x_all[val_idx]
        y_val = y_all[val_idx]
        x_train = x_all[train_idx]
        y_train = y_all[train_idx]

        label_idxs = np.unique(y_train)
        class_stat = {}
        for idx in label_idxs:
            class_stat[idx] = len(np.where(y_train == idx)[0])
        # print("[Stat] Train class: {}".format(class_stat))
        print("[Stat] Train class: mean={}, std={}".format(np.mean(list(class_stat.values())),
                                                           np.std(list(class_stat.values()))))

        label_idxs = np.unique(y_val)
        class_stat = {}
        for idx in label_idxs:
            class_stat[idx] = len(np.where(y_val == idx)[0])
        # print("[Stat] Test class: {}".format(class_stat))
        print("[Stat] Val class: mean={}, std={}".format(np.mean(list(class_stat.values())),
                                                         np.std(list(class_stat.values()))))

        label_idxs = np.unique(y_test)
        class_stat = {}
        for idx in label_idxs:
            class_stat[idx] = len(np.where(y_test == idx)[0])
        # print("[Stat] Test class: {}".format(class_stat))
        print("[Stat] Test class: mean={}, std={}".format(np.mean(list(class_stat.values())),
                                                          np.std(list(class_stat.values()))))

        ########################################
        # Data Split End
        ########################################

        # Process data
        label_idxs = np.unique(y_train)
        x_test = x_test.reshape((-1, input_shape[0], input_shape[1]))
        x_val = x_val.reshape((-1, input_shape[0], input_shape[1]))
        x_train = x_train.reshape((-1, input_shape[0], input_shape[1]))

        print("Train:{}, Test:{}, Class:{}".format(x_train.shape, x_test.shape, nb_class))

        # Normalize
        x_train_max = np.max(x_train)
        x_train_min = np.min(x_train)
        x_train = 2. * (x_train - x_train_min) / (x_train_max - x_train_min) - 1.
        # Test is secret
        x_val = 2. * (x_val - x_train_min) / (x_train_max - x_train_min) - 1.
        x_test = 2. * (x_test - x_train_min) / (x_train_max - x_train_min) - 1.

        return x_train, y_train, x_val, y_val, x_test, y_test, nb_class, class_stat_all
    else:
        nb_timesteps = int(x_train.shape[1] / nb_dims)
        input_shape = (nb_timesteps, nb_dims)
        x_val = x_test
        y_val = y_test
        x_test = x_test.reshape((-1, input_shape[0], input_shape[1]))
        x_val = x_val.reshape((-1, input_shape[0], input_shape[1]))
        x_train = x_train.reshape((-1, input_shape[0], input_shape[1]))

        x_train_max = np.max(x_train)
        x_train_min = np.min(x_train)
        x_train = 2. * (x_train - x_train_min) / (x_train_max - x_train_min) - 1.
        # Test is secret
        x_val = 2. * (x_val - x_train_min) / (x_train_max - x_train_min) - 1.
        x_test = 2. * (x_test - x_train_min) / (x_train_max - x_train_min) - 1.
        return x_train, y_train, x_val, y_val, x_test, y_test, nb_class, None


def load_ucr2018_without_resplit(dataset_path, dataset_name):
    ##################
    # load raw data
    ##################
    nb_class = ds.nb_classes(dataset_name)
    nb_dims = ds.nb_dims(dataset_name)

    if dataset_name in ['MFPT', 'XJTU']:
        x = np.load("{}/{}/{}_data.npy".format(dataset_path, dataset_name, dataset_name))
        y = np.load("{}/{}/{}_label.npy".format(dataset_path, dataset_name, dataset_name))

        (x_train, x_test) = (x[:100], x[100:])
        (y_train, y_test) = (y[:100], y[100:])

    elif dataset_name in ['EpilepticSeizure']:
        data_x, data_y = get_EpilepticSeizure(dataset_path, dataset_name)

        (x_train, x_test) = (data_x[:int(0.5 * data_x.shape[0])], data_x[int(0.5 * data_x.shape[0]):])
        (y_train, y_test) = (data_y[:int(0.5 * data_x.shape[0])], data_y[int(0.5 * data_x.shape[0]):])

    else:
        x_train, y_train, x_test, y_test = TSC_data_loader(dataset_path, dataset_name)

    nb_timesteps = int(x_train.shape[1] / nb_dims)
    input_shape = (nb_timesteps, nb_dims)

    ############################################
    # Combine all train and test data for resample
    ############################################

    x_all = np.concatenate((x_train, x_test), axis=0)
    y_all = np.concatenate((y_train, y_test), axis=0)
    ts_idx = list(range(x_all.shape[0]))

    label_idxs = np.unique(y_all)
    class_stat_all = {}
    for idx in label_idxs:
        class_stat_all[idx] = len(np.where(y_all == idx)[0])
    print("[Stat] All class: {}".format(class_stat_all))

    X_train, X_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=666)

    return X_train, y_train, None, None, X_test, y_test, nb_class, class_stat_all
