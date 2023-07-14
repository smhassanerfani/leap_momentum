import torch
from utils import save_examples, save_examples3

def train_loop(discriminator, generator, dataloader, disc_optimizer, gen_optimizer, bce_loss, l1_loss, l1_lambda):
    
    generator.train()
    
    logger = dict()
    running_disc_loss = 0.0
    running_gen_loss = 0.0
    for batch, (x, y, _) in enumerate(dataloader, 1):
        
        # GPU deployment
        x = x.cuda()
        y = y.cuda()

        # Training Discriminator            
        y_fake = generator(x)
        D_real = discriminator(x, y)
        D_fake = discriminator(x, y_fake.detach())

        # Compute Loss Function
        D_real_loss = bce_loss(D_real, torch.ones_like(D_real))
        D_fake_loss = bce_loss(D_fake, torch.zeros_like(D_fake))
        D_loss = (D_real_loss + D_fake_loss) / 2

        # Backpropagation
        disc_optimizer.zero_grad()
        D_loss.backward()
        disc_optimizer.step()

        # Training Generator
        D_fake = discriminator(x, y_fake)
        G_fake_loss = bce_loss(D_fake, torch.ones_like(D_fake))
        L1 = l1_loss(y_fake, y) * l1_lambda
        G_loss = G_fake_loss + L1

        # Backpropagation
        gen_optimizer.zero_grad()
        G_loss.backward()
        gen_optimizer.step()

        # Statistics
        D_loss = D_loss.detach()
        G_loss = G_loss.detach()
        running_disc_loss += D_loss.item() * x.size(0)
        running_gen_loss += G_loss.item() * x.size(0)

        if batch % 200 == 0:
            disc_loss, gen_loss = D_loss.item(), G_loss.item()
            print(f"Discriminator Loss: {disc_loss:.5f}, Generator Loss: {gen_loss:.5f}")


    logger['disc_loss'] = running_disc_loss / len(dataloader.dataset)
    logger['gen_loss'] = running_gen_loss / len(dataloader.dataset)

    return logger

def val_loop(dataloader, transform_params, model, saving_path):

    model.eval()
    with torch.no_grad():
        for counter, (ssh, it, bm) in enumerate(dataloader, 1):

            # GPU deployment
            ssh = ssh.cuda()
            it = it.cuda()
            bm = bm.cuda()

            # Compute prediction and loss
            it_ = model(ssh)
            bm_ = ssh - it_
            
            y = torch.cat([it, bm], dim=1)
            y_ = torch.cat([it_, bm_], dim=1)

            save_examples3(ssh, y, y_, transform_params, counter, saving_path)
            
            if counter == 5:
                break