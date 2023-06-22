import torch
import torchvision.transforms.functional as TF

class RandomChoice(torch.nn.Module):
    def __init__(self, transforms):
       super().__init__()
       self.transforms = transforms

    def __call__(self, imgs):
        t = random.choice(self.transforms)
        return [t(img) for img in imgs]

class Transform(torch.nn.Module):
    
    def __init__(self):
       super().__init__()

    def __call__(self, image, mask):    
        # Resize
#         resize = T.Resize(size=(512, 512))
#         image = resize(image)
#         mask = resize(mask)

#         # Random crop
#         i, j, h, w = T.RandomCrop.get_params(
#             image, output_size=(256, 256))
#         image = TF.crop(image, i, j, h, w)
#         mask = TF.crop(mask, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        
        return image, mask
