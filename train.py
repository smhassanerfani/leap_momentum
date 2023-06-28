import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from joint_transforms import Transform2
from models.Dhruv import Generator, DnCNN, initialize_weights
from dataset import Agulhas2
from utils.utils import AdjustLearningRate, save_examples2


def val_loop(dataloader, transform_params, model, saving_path):

    model.eval()
    with torch.no_grad():
        for counter, (x, y_it, y_bm) in enumerate(dataloader, 1):

            # GPU deployment
            x = x.cuda()
            y_it = y_it.cuda()
            y_bm = y_bm.cuda()

            # Compute prediction and loss
            y_bm_fake = model(x)
            y_it_fake = x - y_bm_fake
            
            y_fake = torch.cat([y_it_fake, y_bm_fake], dim=1)
            y = torch.cat([y_it, y_bm], dim=1)

            save_examples2(x, y, y_fake, transform_params, counter, saving_path)
            if counter == 5:
                break

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.get_device_name(device))

LEARNING_RATE = 2.0E-4
BATCH_SIZE = 2
IMAGE_SIZE = 256
CHANNELS_IMG = 1
NUM_EPOCHS = 100
FEATURES = 64


joint_transforms = Transform2()

dataset = Agulhas2(split='train', joint_transform=joint_transforms)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = Agulhas2(split='val', joint_transform=joint_transforms)
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=True)

transform_params = dict()
transform_params['inputs_mean'] = dataset.inps_mean_std[0]
transform_params['inputs_std'] = dataset.inps_mean_std[1]
transform_params['targets_mean'] = dataset.tars_mean_std[0]
transform_params['targets_std'] = dataset.tars_mean_std[1]
transform_params['targets_bm_mean'] = dataset.tars_bm_mean_std[0]
transform_params['targets_bm_std'] = dataset.tars_bm_mean_std[1]
    

model = DnCNN(CHANNELS_IMG, FEATURES).to(device)
initialize_weights(model)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
criterion1 = nn.MSELoss()
criterion2 = nn.L1Loss()

# max_iter = NUM_EPOCHS * len(dataloader.dataset)
# scheduler = AdjustLearningRate(optimizer, LEARNING_RATE, max_iter, 0.9)

model.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (ssh, it, bm) in enumerate(dataloader):

        ssh = ssh.to(device)
        it = it.to(device)
        bm = bm.to(device)

        out = model(ssh)
        it_out = ssh - out
        
        # loss1 = criterion1(out, bm)
        loss2 = criterion2(out, bm)

        loss3 = criterion1(it_out, it)
        # loss4 = criterion2(it_out, it)

        loss = (loss2 + loss3)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # scheduler.num_of_iterations += ssh.size(0)
        # lr = scheduler(scheduler.num_of_iterations)

        if batch_idx % 100 == 0:
            print(f"Epoch: [{epoch}/{NUM_EPOCHS}] Batch: {batch_idx:>2}/{len(dataloader)} "
                  f"Loss: {loss.item():.4f},")# LR:{lr:#.4E}")
    
    current_dir = os.path.join('outputs', 'Dhruv', 'trial11', f'epoch-{epoch:02d}')
    
    try:
        os.makedirs(current_dir)
    except FileExistsError:
        pass
    
    val_loop(val_dataloader, transform_params, model, current_dir)