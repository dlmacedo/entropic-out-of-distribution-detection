import torch
from torchvision import datasets, transforms
import os


def getSVHN(batch_size, TF, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'svhn'))
    kwargs.pop('input_size', None)

    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.SVHN(root=data_root, split='train', download=True, transform=TF),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.SVHN(root=data_root, split='test', download=True, transform=TF,),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def getCIFAR10(batch_size, TF, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar10'))
    kwargs.pop('input_size', None)

    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=data_root, train=True, download=True, transform=TF),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=data_root, train=False, download=True, transform=TF),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def getCIFAR100(batch_size, TF, data_root='/tmp/public_dataset/pytorch', TTF=None, train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar100'))
    kwargs.pop('input_size', None)

    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root=data_root, train=True, download=True, transform=TF, target_transform=TTF),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root=data_root, train=False, download=True, transform=TF, target_transform=TTF),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def getTargetDataSet(data_type, batch_size, input_TF, dataroot):
    if data_type == 'cifar10':
        train_loader, test_loader = getCIFAR10(batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1)
    elif data_type == 'cifar100':
        train_loader, test_loader = getCIFAR100(batch_size=batch_size, TF=input_TF, data_root=dataroot, TTF=None, num_workers=1)
    elif data_type == 'svhn':
        train_loader, test_loader = getSVHN(batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1)
    return train_loader, test_loader


def getNonTargetDataSet(data_type, batch_size, input_TF, dataroot):
    if data_type == 'cifar10':
        _, test_loader = getCIFAR10(batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1)
    elif data_type == 'svhn':
        _, test_loader = getSVHN(batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1)
    elif data_type == 'cifar100':
        _, test_loader = getCIFAR100(batch_size=batch_size, TF=input_TF, data_root=dataroot, TTF=lambda x: 0, num_workers=1)
    elif data_type == 'imagenet_resize':
        dataroot = os.path.expanduser(os.path.join(dataroot, 'Imagenet_resize'))
        testsetout = datasets.ImageFolder(dataroot, transform=input_TF)
        test_loader = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False, num_workers=1)
    elif data_type == 'lsun_resize':
        dataroot = os.path.expanduser(os.path.join(dataroot, 'LSUN_resize'))
        testsetout = datasets.ImageFolder(dataroot, transform=input_TF)
        test_loader = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False, num_workers=1)
    elif data_type == 'fooling_images':
        dataroot = os.path.expanduser(os.path.join(dataroot, 'fooling_images'))
        testsetout = datasets.ImageFolder(dataroot, transform=transforms.Compose([transforms.Resize((32, 32)), input_TF]))
        test_loader = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False, num_workers=1)
    elif data_type == 'gaussian_noise':
        dataroot = os.path.expanduser(os.path.join(dataroot, 'gaussian_noise'))
        testsetout = datasets.ImageFolder(dataroot, transform=input_TF)
        test_loader = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False, num_workers=1)
    elif data_type == 'uniform_noise':
        dataroot = os.path.expanduser(os.path.join(dataroot, 'uniform_noise'))
        testsetout = datasets.ImageFolder(dataroot, transform=input_TF)
        test_loader = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False, num_workers=1)
    return test_loader
