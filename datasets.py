import torch
import pickle
import os
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler


class MiniImagenet(Dataset):
    """
    The dataset contains 64 train classes, 16 validation classes and 20 novel classes in test
    Since we are not doing any kind of few-shot learning, we will combine all 3 sets and split them into
    training and validation part only.

    For first time usage, first download the dataset through https://www.kaggle.com/whitemoon/miniimagenet

    """

    def __init__(self, root, train=True,
                 transform=None, target_transform=None):
        """
        :param root: root path of mini-imagenet
        :param mode: train or test
        """
        self.train = train
        # First we check whether the raw mini-imagenet data is already processed
        # If not, we do the process first
        mode = 'train' if train else 'test'
        if not os.path.exists(root + f"/processed_{mode}.pkl"):
            print("Mini-Imagenet dataset is not processed, processing now!")
            self.processing_raw_data(root)

        # If yes, we directly load the processed data
        with open(root + f"/processed_{mode}.pkl", 'rb') as f:
            pkl_file = pickle.load(f)
            self.data = pkl_file[f"{mode}_data"]
            self.targets = pkl_file[f"{mode}_label"]

        self.num_classes = 100
        self.transform = transform
        self.target_transform = target_transform

        self.index_data = self.data
        self.index_labels = self.targets

    def __len__(self):
        return len(self.index_data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.index_data[index], self.index_labels[index]


        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)


        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def processing_raw_data(self, root):
        """
        load train val test pkl all together and combine them to create a 100 class dataset with
        50000 training samples and 10000 validation samples
        :param root: root directory where the downloaded mini-imagenet dataset stored
        """

        if not os.path.exists(root + "/mini-imagenet-cache-train.pkl"):
            print("File not exists")
            print("Please make sure the data directory entered is correct, or the downloaded zip is unzipped!")

        with open(root + "/mini-imagenet-cache-train.pkl", 'rb') as f:
            train_data = pickle.load(f)
            train_image = train_data['image_data'].reshape([64, 600, 84, 84, 3])
            train_label = train_data['class_dict']
        with open(root + "/mini-imagenet-cache-val.pkl", 'rb') as f:
            val_data = pickle.load(f)
            val_image = val_data['image_data'].reshape([16, 600, 84, 84, 3])
            val_label = val_data['class_dict']
        with open(root + "/mini-imagenet-cache-test.pkl", 'rb') as f:
            test_data = pickle.load(f)
            test_image = test_data['image_data'].reshape([20, 600, 84, 84, 3])
            test_label = test_data['class_dict']

        # Now Combine all 60000 images and split into 50000 training and 10000 validation images
        processed_train_data = np.concatenate([train_image[:, :500], val_image[:, :500], test_image[:, :500]],
                                              0).reshape([50000, 84, 84, 3])
        processed_test_data = np.concatenate([train_image[:, 500:], val_image[:, 500:], test_image[:, 500:]],
                                             0).reshape([10000, 84, 84, 3])

        processed_train_label = []
        [processed_train_label.extend([i] * 500) for i in range(100)]
        processed_train_label = np.array(processed_train_label)
        processed_test_label = []
        [processed_test_label.extend([i] * 100) for i in range(100)]
        processed_test_label = np.array(processed_test_label)

        processed_train_dict = {"train_data": processed_train_data,
                                "train_label": processed_train_label}

        processed_test_dict = {"test_data": processed_test_data,
                               "test_label": processed_test_label}

        # Save the processed data and labels to new pickle files
        with open(root + "/processed_train.pkl", 'wb') as f:
            pickle.dump(processed_train_dict, f)
        with open(root + "/processed_test.pkl", 'wb') as f:
            pickle.dump(processed_test_dict, f)

        return "Processing raw data done!"

def make_dataset(dataset_name, data_dir, batch_size=128, sample_size=None, SOTA=False):

    if dataset_name == 'cifar10':
        print('Dataset: CIFAR10.')
        if SOTA:
            trainset = CIFAR10(root=data_dir, train=True, download=True, transform=transforms.Compose([
                transforms.RandomCrop(size=32, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),

                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                ]))
        else:
            trainset = CIFAR10(root=data_dir, train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ]))

        testset = CIFAR10(root=data_dir, train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ]))
        num_classes = 10
    elif dataset_name == 'mini_imagenet':
        print('Dataset: Mini-ImageNet.')
        if SOTA:
            trainset = MiniImagenet(root=data_dir, train=True, transform=transforms.Compose([
                transforms.RandomCrop(84, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]))
        else:
            trainset = MiniImagenet(root=data_dir, train=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]))

        testset = MiniImagenet(root=data_dir, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]))
        num_classes = 100
    else:
        raise ValueError

    if sample_size is not None:
        total_sample_size = num_classes * sample_size
        cnt_dict = dict()
        total_cnt = 0
        indices = []
        for i in range(len(trainset)):

            if total_cnt == total_sample_size:
                break

            label = trainset[i][1]
            if label not in cnt_dict:
                cnt_dict[label] = 1
                total_cnt += 1
                indices.append(i)
            else:
                if cnt_dict[label] == sample_size:
                    continue
                else:
                    cnt_dict[label] += 1
                    total_cnt += 1
                    indices.append(i)

        train_indices = torch.tensor(indices)
        trainloader = DataLoader(
            trainset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices), num_workers=1)

    else:
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    return trainloader, testloader, num_classes


