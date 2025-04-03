import timm
import torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision.datasets import ImageFolder,DatasetFolder


def get_validation_loader(dataset_name, model, batch_size, num_workers=8):
    if dataset_name == 'imagenet':
        data_path = '/gpfs/gibbs/project/panda/shared/imagenet'
    else:
        raise NotImplementedError
    config = resolve_data_config({}, model=model)
    test_transform = create_transform(**config)
    val_set = ImageFolder(data_path, test_transform)
    val_loader = torch.utils.data.DataLoader(val_set,
                                             batch_size=batch_size,
                                             num_workers=num_workers,
                                             shuffle=False)
    return val_loader


def get_calibration_loader(dataset_name, model, num_data, num_workers=8):
    if dataset_name == 'imagenet':
        data_path = '/gpfs/gibbs/project/panda/shared/imagenet_2012/train'
    else:
        raise NotImplementedError
    config = resolve_data_config({}, model=model)
    train_transform = create_transform(**config, is_training=False)
    calib_set = ImageFolder(data_path, train_transform)
    calib_loader = torch.utils.data.DataLoader(calib_set,
                                             batch_size=num_data,
                                             num_workers=num_workers,
                                             shuffle=True)

    return calib_loader


def get_calibration_data(dataset_name, model, num_data=128):
    calib_loader = get_calibration_loader(dataset_name, model, num_data)
    calib_loader = iter(calib_loader)
    image, label = next(calib_loader)
    image = image.cuda()
    return image
