
from .dset import MNISTDataset, USPSDataset


def get_dataset(dataset_name):
    return {
        'mnist': MNISTDataset,
        'usps': USPSDataset,
    }[dataset_name]
