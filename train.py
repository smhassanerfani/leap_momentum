import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from joint_transforms import Transform2
from models.Dhruv import DnCNN, initialize_weights
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
            bm_, it_ = model(ssh)
            
            y = torch.cat([it, bm], dim=1)
            y_ = torch.cat([it_, bm_], dim=1)

            save_examples3(ssh, y, y_, transform_params, counter, saving_path)
            
            if counter == 5:
                break

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.get_device_name(device))

LEARNING_RATE = 2.0E-4
BATCH_SIZE = 2
IMAGE_SIZE = 256
CHANNELS_IMG = 1
NUM_EPOCHS = 101
FEATURES = 64


joint_transforms = Transform2()

dataset = Agulhas3(split='train', joint_transform=joint_transforms)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = Agulhas3(split='val', joint_transform=joint_transforms)
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=True)

transform_params = dataset.transform_dict
    
model = DnCNN(CHANNELS_IMG, FEATURES).to(device)
initialize_weights(model)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

# max_iter = NUM_EPOCHS * len(dataloader.dataset)
# scheduler = AdjustLearningRate(optimizer, LEARNING_RATE, max_iter, 0.9)

model.train()
loss_list = list()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (ssh, it, bm) in enumerate(dataloader):

        ssh = ssh.to(device)
        it = it.to(device)
        bm = bm.to(device)

        bm_, it_ = model(ssh)
        
        bm_loss = criterion(bm_, bm)
        it_loss = criterion(it_, it)
        loss = (bm_loss + it_loss) / 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # scheduler.num_of_iterations += ssh.size(0)
        # lr = scheduler(scheduler.num_of_iterations)

        if batch_idx % 100 == 0:
            print(f"Epoch: [{epoch}/{NUM_EPOCHS}] Batch: {batch_idx:>2}/{len(dataloader)} Loss: {loss.item():.4f},")
            loss_list.append([bm_loss.item(), it_loss.item()])
    
    current_dir = os.path.join('outputs', 'DnCNN', 'T06', f'epoch-{epoch:003d}')
    
    try:
        os.makedirs(current_dir)
    except FileExistsError:
        pass
    
    if epoch % 20 == 0:
        save_checkpoint(model, optimizer,  os.path.join(current_dir, "model.pth.tar"))
    
    val_loop(val_dataloader, transform_params, model, current_dir)

with open(os.path.join(current_dir, "loss.npy"), mode = 'wb') as f:
    np.save(f, np.array(loss_list))