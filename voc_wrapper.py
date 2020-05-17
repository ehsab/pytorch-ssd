import os

import torch
from pathlib import Path
from torch.hub import load_state_dict_from_url
from torch.utils.data import ConcatDataset

__all__ = [
    'get_voc_dataset'
]

model_urls = {
    'vgg16_ssd': 'http://download.deeplite.ai/zoo/models/vgg16-ssd-voc-mp-0_7726-b1264e8beec69cbc.pth',
}

from vision.datasets.voc_dataset import VOCDataset, VOC_CLASS_NAMES
from vision.ssd.config import vgg_ssd_config
from vision.ssd.data_preprocessing import TrainAugmentation
from vision.ssd.ssd import MatchPrior
from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor


def get_voc_dataset(data_root_07="", data_root_12="", net='vgg16-ssd', batch_size=128, num_torch_workers=4):
    if net == 'vgg16-ssd':
        config = vgg_ssd_config
    else:
        raise ValueError

    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance, config.size_variance, 0.5)

    datasets = []
    if data_root_07 != "":
        train_dataset_07 = VOCDataset(root=data_root_07,
                                      is_test=False,
                                      transform=train_transform,
                                      target_transform=target_transform)
        datasets.append(train_dataset_07)

    if data_root_12 != "":
        train_dataset_12 = VOCDataset(root=data_root_12,
                                      is_test=False,
                                      transform=train_transform,
                                      target_transform=target_transform)
        datasets.append(train_dataset_12)

    train_dataset = ConcatDataset(datasets)

    test_dataset = VOCDataset(root=data_root_07, is_test=True)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=num_torch_workers)

    # test_loader = torch.utils.data.DataLoader(test_dataset,
    #                                           batch_size=batch_size,
    #                                           shuffle=False,
    #                                           pin_memory=True,
    #                                           num_workers=num_torch_workers)

    return {'train': train_loader, 'test': test_dataset}


def get_vgg16_ssd(pretrained=False, progress=True, is_test=False):
    model = create_vgg_ssd(len(VOC_CLASS_NAMES), is_test)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['vgg16_ssd'], progress=progress, check_hash=True)
        model.load_state_dict(state_dict)

    return model


def vgg16_ssd_eval_func(model, data_loader, device):
    eval_path = os.path.join(os.getcwd(), ".neutrino-torch-zoo/voc/eval_results")
    Path(eval_path).mkdir(parents=True, exist_ok=True)

    net = create_vgg_ssd(len(VOC_CLASS_NAMES), is_test=True)
    net.load_state_dict(model.state_dict())
    predictor = create_vgg_ssd_predictor(net, nms_method='hard', device=device)
    from eval_ssd import ssd_eval
    return ssd_eval(predictor=predictor, dataset=data_loader, data_path=eval_path, class_names=VOC_CLASS_NAMES)
