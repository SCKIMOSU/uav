import time

import torch
from torchvision import transforms
from torchvision.utils import save_image
from torchinfo import summary

from omegaconf import OmegaConf
import segmentation_models_pytorch as smp
from tqdm import tqdm

from utils.common import get_dataloaders
from utils.common import pixel_accuracy, seg_miou, dice_coeff

config = OmegaConf.load('config_unet.yaml')
dataset_cfg = config.dataset
trainer_cfg = config.trainer
ckpt_dir = config.trainer.checkpoint.save_dir

BEST_TRAIN_LOSS = float('inf')
BEST_VAL_LOSS = float('inf')

# Get data loaders
transform = transforms.Compose([
        transforms.Resize(dataset_cfg.image_size),
        transforms.ToTensor()
    ])

test_dataloader = get_dataloaders(dataset_cfg, transform, test=True)

# Define the model
model = smp.Unet(encoder_name='resnet34', classes=1, in_channels=3)

model_path = f'{config.trainer.checkpoint.save_dir}/{config.model.name}-best-val.pth'

device = torch.device(f'cuda:{trainer_cfg.gpu_id}' if torch.cuda.is_available() else 'cpu')

model.load_state_dict(torch.load(
    model_path,
    weights_only=True,
)['model_state_dict'])
model.to(device)
model.eval()

# Training
if __name__ == '__main__':
    print('Test model: ', model_path)

    total_acc = 0.0
    total_miou = 0.0
    total_dice = 0.0
    total_infr_time = 0.0
    total_batches = len(test_dataloader)
    
    with torch.no_grad():
        for images, true_masks in tqdm(test_dataloader, desc='Model Testing'):
            start_time = time.time()
            images = images.to(device)
            true_masks = true_masks.to(device)            
            outputs = model(images)

            # Decode the predictions logits
            pred_masks = outputs.sigmoid()
            pred_masks = (pred_masks > 0.9).float()
            end_time = time.time()
            total_infr_time += (end_time - start_time)
            total_acc += pixel_accuracy(pred_masks, true_masks)
            total_miou += seg_miou(pred_masks, true_masks)
            total_dice += dice_coeff(pred_masks, true_masks)
            
            images = images.cpu()
            true_masks = true_masks.cpu()
            pred_masks = pred_masks * 255
            pred_masks = pred_masks.cpu()
        
        avg_acc = total_acc / total_batches
        avg_miou = total_miou / total_batches
        avg_dice = total_dice / total_batches
        avg_infr_time = total_infr_time / total_batches
        
        print('TEST METRICS: ')
        print(f"Pixel Accuracy: {avg_acc:.4f} |Mean IoU: {avg_miou:.4f} | Dice Coeff: {avg_dice:.4f} | Inference Time: {avg_infr_time*100:.2f} ms")
            