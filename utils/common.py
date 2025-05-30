import torch
from dataset import UAVSegmDataset
from torch.utils.data import DataLoader
import numpy as np
import os
import segmentation_models_pytorch as smp
from omegaconf import OmegaConf

def get_loss_function(loss_fn):
    if loss_fn == 'CrossEntropyLoss':
        return torch.nn.CrossEntropyLoss()
    elif loss_fn == 'BCEWithLogitsLoss':
        return torch.nn.BCEWithLogitsLoss()
    elif loss_fn == 'DiceLoss':
        return smp.losses.DiceLoss(mode='binary')
    elif loss_fn == 'IoULoss':
        return smp.losses.JaccardLoss(mode='binary')
    else:
        raise ValueError(f'Loss function {loss_fn} not supported')
    

def get_optimizer(optimizer, model, lr):
    if optimizer == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer == 'AdamW':
        return torch.optim.AdamW(model.parameters(), lr=lr)
    elif optimizer == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.8)
    else:
        raise ValueError(f'Optimizer {optimizer} not supported')
                         

def get_dataloaders(config, tsfm, tsfm_aug=None, test=False):
    if test:
        test_dataset = UAVSegmDataset(
            config.root,
            2,
            tsfm,
            "test"
        )
        print('Test dataset size:', len(test_dataset))

        test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=custom_collate_fn)

        return test_dataloader

    train_dataset = UAVSegmDataset(
        config.root,
        2,
        tsfm,
        "train"
    )
    print('Train dataset size:', len(train_dataset))

    val_dataset = UAVSegmDataset(
        config.root,
        2,
        tsfm,
        "val"
    )
    print('Val dataset size:', len(val_dataset))

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=config.num_workers, collate_fn=custom_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=custom_collate_fn)

    # Iterate over the train dataloader
    for images, masks in train_dataloader:
        print('Train batch size:', images.size())
        break

    # Iterate over the val dataloader
    for images, masks in val_dataloader:
        print('Val batch size:', images.size())
        break

    # for idx, (image, mask) in enumerate(train_dataset):
    #     print('Image shape:', image.shape)
    #     print('Mask shape:', mask.shape)
    #     break

    return train_dataloader, val_dataloader

# TODO: create write config function
def write_config(ckpt_dir, config):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    with open(os.path.join(ckpt_dir, 'config.yaml'), 'w') as f:
        OmegaConf.save(config, f)

    print('Write config.yaml file to:', ckpt_dir)

def pixel_accuracy(pred_mask, true_mask):
    correct = (pred_mask == true_mask).sum().item()
    total = true_mask.numel()
    return correct / total


def seg_miou(pred_mask, true_mask):
    pred_mask = pred_mask.bool()
    true_mask = true_mask.bool()
    intersection = (pred_mask & true_mask).float().sum((1, 2, 3))
    union = (pred_mask | true_mask).float().sum((1, 2, 3))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()


def dice_coeff(pred_mask, true_mask):
    pred_mask = pred_mask.bool()
    true_mask = true_mask.bool()
    intersection = (pred_mask & true_mask).float().sum((1, 2, 3))
    dice = (2. * intersection + 1e-6) / (pred_mask.float().sum((1, 2, 3)) + true_mask.float().sum((1, 2, 3)) + 1e-6)
    return dice.mean().item()

def custom_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    
    if len(batch) == 0:
        return None, None
    
    data, masks = zip(*batch)
    return torch.stack(data), torch.stack(masks)