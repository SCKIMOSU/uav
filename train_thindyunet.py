
import os

from omegaconf import OmegaConf

import torch
from torchvision import transforms as T

from trainer import train_one_epoch
from eval import evaluate
from utils import EarlyStopper
from utils import save_model
from utils.common import get_loss_function, get_optimizer, get_dataloaders, write_config
from model.ThinDyUNet import ThinDyUNet

config = OmegaConf.load('config_thindyunet.yaml')
dataset_cfg = config.dataset
trainer_cfg = config.trainer
model_cfg = config.model
ckpt_dir = config.trainer.checkpoint.save_dir

BEST_TRAIN_LOSS = float('inf')
BEST_VAL_LOSS = float('inf')

# Get data loaders
transform = T.Compose([
        T.Resize(dataset_cfg.image_size),
        T.ToTensor()
    ])

train_dataloader, val_dataloader = get_dataloaders(dataset_cfg, transform)

# Define the model
model = ThinDyUNet(in_channels=model_cfg.in_channels, start_out_channels=model_cfg.start_out_channels, num_class=model_cfg.num_classes, size=model_cfg.num_blocks, padding=model_cfg.num_padding, upsample=model_cfg.is_upsample)

# Load pre-trained model weights for transfer learning
if trainer_cfg.pretrained_path:
    checkpoint = torch.load(trainer_cfg.pretrained_path,weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(f"Loaded pre-trained model weights from {trainer_cfg.pretrained_path}")

criterion = get_loss_function(trainer_cfg.loss_fn)
optimizer = get_optimizer(trainer_cfg.optimizer, model, trainer_cfg.lr)

if trainer_cfg.scheduler.type == 'ReduceLROnPlateau':
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=trainer_cfg.scheduler.factor, patience=trainer_cfg.scheduler.patience, cooldown=trainer_cfg.scheduler.cooldown)

device = torch.device(f'cuda:{trainer_cfg.gpu_id}' if torch.cuda.is_available() else 'cpu')
early_stopper = EarlyStopper(
    patience = int(trainer_cfg.early_stop.patience), 
    min_delta = float(trainer_cfg.early_stop.min_delta)
)

model.to(device)

# Training
if __name__ == '__main__':
    # write model configuration
    print(f'Running experiment: {ckpt_dir}')
    write_config(ckpt_dir, config)
    

    for epoch in range(trainer_cfg.epochs):
        print(f'Epoch {epoch + 1}/{trainer_cfg.epochs}')
        
        train_loss = train_one_epoch(model, train_dataloader, criterion, optimizer, device)
        print(f'Train Loss: {train_loss}')
        
        val_loss, acc, mIoU, dice = evaluate(model, val_dataloader, criterion, device)
        print(f'Val Loss: {val_loss}')
        
        lr_scheduler.step(val_loss)

        # Save the model
        if train_loss < BEST_TRAIN_LOSS:
            save_model(
                model, optimizer, lr_scheduler, epoch, train_loss,
                f'{ckpt_dir}/{config.model.name}-best-train.pth'
            )
            BEST_TRAIN_LOSS = train_loss
        
        if val_loss < BEST_VAL_LOSS:
            print('Saving validation best model: ', val_loss)
            save_model(
                model, optimizer, lr_scheduler, epoch, val_loss, 
                f'{ckpt_dir}/{config.model.name}-best-val.pth'
            )
            BEST_VAL_LOSS = val_loss

        if early_stopper.early_stop(val_loss):
            print('Early stopping')
            break