import torch
from .mnist import MNIST_Dataset
from .cifar10 import CIFAR10_Dataset
from .mydata import mydata_Dataset


def load_dataset(dataset_name, data_path, normal_class):
    """Loads the dataset."""

    implemented_datasets = ('mnist', 'cifar10', 'mydata')
    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name == 'mnist':
        dataset = MNIST_Dataset(root=data_path, normal_class=normal_class)

    if dataset_name == 'cifar10':
        dataset = CIFAR10_Dataset(root=data_path, normal_class=normal_class)
    if dataset_name == 'mydata':
        # data_path = data_path + '/train.txt'
        # jf_dataset = JFDetDataset(data_path, 64, 64)
        # dataset = torch.utils.data.DataLoader(dataset=jf_dataset,
        #                                     batch_size=1,
        #                                     # num_workers =num_workers,
        #                                     shuffle=False)
        dataset = mydata_Dataset(root=data_path)
    return dataset
