import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
from joint_transforms import Transform2
from models.Dhruv import DnCNN, DnCNN_v0, initialize_weights
from dataset import Agulhas2, Agulhas3
from utils.utils import AdjustLearningRate, save_checkpoint, save_examples2, save_examples3


def val_loop(dataloader, transform_params, model, saving_path):

    model.eval()
    with torch.no_grad():
        for counter, (ssh, it, bm) in enumerate(dataloader, 1):

            # GPU deployment
            ssh = ssh.cuda()
            it = it.cuda()
            bm = bm.cuda()

            # Compute prediction and loss
            bm_ = model(ssh)
            it_ = ssh - bm_
            
            y = torch.cat([it, bm], dim=1)
            y_ = torch.cat([it_, bm_], dim=1)

            save_examples3(ssh, y, y_, transform_params, counter, saving_path)
            
            if counter == 5:
                break


def psnr(original, reconstructed):
    
    # R is the maximum fluctuation in the input image data type. [-1, +1] -> 2.0
    mse = F.mse_loss(original, reconstructed)
    psnr = 20 * torch.log10(2.0 / torch.sqrt(mse))

    return psnr

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.get_device_name(device))

LEARNING_RATE = 2.0E-4
BATCH_SIZE = 2
IMAGE_SIZE = 256
CHANNELS_IMG = 1
NUM_EPOCHS = 101
FEATURES = 16

joint_transforms = Transform2()

dataset = Agulhas3(split='train', joint_transform=joint_transforms)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = Agulhas3(split='val', joint_transform=joint_transforms)
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=True)

transform_params = dataset.transform_dict
    
model = DnCNN_v0(CHANNELS_IMG, FEATURES).to(device)
initialize_weights(model)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

criterion = nn.L1Loss()
# criterion = nn.MSELoss()

# max_iter = NUM_EPOCHS * len(dataloader.dataset)
# scheduler = AdjustLearningRate(optimizer, LEARNING_RATE, max_iter, 0.9)

cudnn.enabled = True
cudnn.benchmark = True
model.train()

logger = {'loss': list(), 'psnr': list()}

for epoch in range(NUM_EPOCHS):
    print('Epoch:', epoch,'LR:', scheduler.get_last_lr())
    
    batch_loss = 0.0
    batch_psnr = 0.0
    for batch_idx, (ssh, it, bm) in enumerate(dataloader):

        ssh = ssh.to(device)
        # it = it.to(device)
        bm = bm.to(device)

        bm_ = model(ssh)
        
        loss = criterion(bm_, bm)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # scheduler.num_of_iterations += ssh.size(0)
        # lr = scheduler(scheduler.num_of_iterations)
        
        loss = loss.detach()
        batch_loss += loss.item()
        batch_psnr += psnr(bm.detach(), bm_.detach()).cpu().numpy()

        if batch_idx % 100 == 0:
            print(f"Epoch: [{epoch+1}/{NUM_EPOCHS}] Batch: {batch_idx:>2}/{len(dataloader)} Loss: {loss.item():.4f},")
    
    scheduler.step()
    
    logger['loss'].append(batch_loss / len(dataloader))
    logger['psnr'].append(batch_psnr / len(dataloader))

    current_dir = os.path.join('outputs', 'DnCNN', 'T08', f'epoch-{epoch:003d}')
    
    try:
        os.makedirs(current_dir)
    except FileExistsError:
        pass
    
    if epoch % 10 == 0:
        save_checkpoint(model, optimizer,  os.path.join(current_dir, "model.pth.tar"))
    
    val_loop(val_dataloader, transform_params, model, current_dir)


with open(os.path.join(current_dir, "logger.npy"), mode = 'wb') as f:
    np.save(f, np.array(logger['loss']))
    np.save(f, np.array(logger['psnr']))