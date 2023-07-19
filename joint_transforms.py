import torch
import random
import torchvision.transforms as T
import torchvision.transforms.functional as TF

class RandomChoice(torch.nn.Module):
    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms

    def __call__(self, imgs):
        t = random.choice(self.transforms)
        return [t(img) for img in imgs]

class Transform(torch.nn.Module):
    
    def __init__(self, resize=None, crop=None):
        super().__init__()
        self.resize = resize
        self.crop = crop

    def __call__(self, image, mask):    
        
        # Resize
        if self.resize is not None:
            resize = T.Resize(size=(self.resize, self.resize))
            image = resize(image)
            mask = resize(mask)

        # Random crop
        if self.crop is not None:
            i, j, h, w = T.RandomCrop.get_params(
                image, output_size=(self.crop, self.crop))
            
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        
        return image, mask

class Transform2(torch.nn.Module):
    
    def __init__(self, resize=None, crop=None):
        super().__init__()
        self.resize = resize
        self.crop = crop

    def __call__(self, image, mask1, mask2):    
        
        # Resize
        if self.resize is not None:
            resize = T.Resize(size=(self.resize, self.resize))
            image = resize(image)
            mask1 = resize(mask1)
            mask2 = resize(mask2)

        # Random crop
        if self.crop is not None:
            i, j, h, w = T.RandomCrop.get_params(
                image, output_size=(self.crop, self.crop))
            
            image = TF.crop(image, i, j, h, w)
            mask1 = TF.crop(mask1, i, j, h, w)
            mask2 = TF.crop(mask2, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask1 = TF.hflip(mask1)
            mask2 = TF.hflip(mask2)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask1 = TF.vflip(mask1)
            mask2 = TF.vflip(mask2)
        
        return image, mask1, mask2