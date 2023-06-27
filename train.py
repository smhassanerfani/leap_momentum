import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from joint_transforms import Transform
from torch.utils.tensorboard import SummaryWriter
from models.Dhruv import Generator, initialize_weights
from dataset import Agulhas
from utils.utils import AdjustLearningRate


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.get_device_name(device))

LEARNING_RATE = 2.0E-5
BATCH_SIZE = 4
IMAGE_SIZE = 256
CHANNELS_IMG = 1
NUM_EPOCHS = 30
FEATURES = 4


joint_transforms = Transform()

dataset = Agulhas(split='train', joint_transform=joint_transforms)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = Generator(CHANNELS_IMG, FEATURES).to(device)
initialize_weights(model)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
criterion = nn.MSELoss()


max_iter = NUM_EPOCHS * len(dataloader.dataset)
scheduler = AdjustLearningRate(optimizer, LEARNING_RATE, max_iter, 0.9)

writer_real = SummaryWriter('logs/LEAP/real')
writer_fake = SummaryWriter('logs/LEAP/fake')
writer_loss = SummaryWriter('logs/LEAP/loss')
step = 0

model.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (ssh, it) in enumerate(dataloader):

        ssh = ssh.to(device)
        it = it.to(device)

        git = model(ssh)
        
        loss = criterion(git, it)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scheduler.num_of_iterations += ssh.size(0)
        lr = scheduler(scheduler.num_of_iterations)

        if batch_idx % 100 == 0:
            print(f"Epoch: [{epoch}/{NUM_EPOCHS}]\tBatch: {batch_idx:>2}/{len(dataloader)} "
                  f"Loss: {loss.item():.4f}, LR:{lr:#.4E}")

            with torch.no_grad():

                writer_loss.add_scalar('Model Loss', loss.item(), global_step=step)

                git = model(ssh)
                
                it = it * dataset.tars_mean_std[0] + dataset.tars_mean_std[0]
                git = git * dataset.tars_mean_std[0] + dataset.tars_mean_std[0]
                
                img_grid_fake = torchvision.utils.make_grid(it, normalize=False)
                img_grid_real = torchvision.utils.make_grid(git, normalize=False)

                writer_fake.add_image("Fake Images", img_grid_fake, global_step=step)
                writer_real.add_image("Real Images", img_grid_real, global_step=step)
                step += 1