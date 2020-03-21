import os
from datasets import *
from torchvision import transforms
from torch.utils.data import DataLoader
from models import *
import torch
from torchvision.utils import save_image
val_transforms = [
            transforms.CenterCrop((170,170)),
            transforms.Resize((128,128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

val_dataloader = DataLoader(CelebADataset('../stargan/CelebA/img_align_celeba/', transforms_=val_transforms, mode="val", attributes=["Bald","Bangs","Black_Hair", "Blond_Hair", "Brown_Hair","Bushy_Eyebrows","Eyeglasses", "Male","Mouth_Slightly_Open","Mustache","No_Beard","Pale_Skin", "Young"]),
            batch_size=2000,shuffle=False)
cuda = torch.cuda.is_available()
device =torch.device('cuda' if cuda else 'cpu',1)
os.makedirs('./path/imgs',exist_ok=True)
net =GeneratorResNet((3,128,128),13)
net=net.to(device)
net.load_state_dict(torch.load('./saved_models/generator_19.pth',map_location='cuda'))
def sample_images():
        net.eval()
        label_changes = [
            ((0, 1),),  # Set to black hair
            ((1, 1),),  # Set to blonde hair
            ((2, 1), (3, 0), (4, 0)),  # Set to brown hair
            ((2, 0),(3,1),(4,0)),  # Flip gender
            ((2,0),(3,0),(4,1)), # Age flip
            ((5,1),),
            ((6,1),),
            ((7,-1),),
            ((8,1),),
            ((9,1),(10,0)),
            ((9,0),(10,1)),
            ((11,1),),
            ((12,-1),),
        ]

        val_imgs, val_labels = next(iter(val_dataloader))
        val_imgs = val_imgs.to(device)
        val_labels = val_labels.to(device)
        img_samples = None
        for i in range(2000):
            img, label = val_imgs[i], val_labels[i]

            # Repeat for number of label changes
            imgs = img.repeat(13, 1, 1, 1)
            labels0 = label.repeat(13, 1)
            labels = label.repeat(13, 1)
            # Make changes to labels
            for sample_i, changes in enumerate(label_changes):
                for col, val in changes:
                    labels[sample_i, col] = 1 - labels[sample_i, col] if val == -1 else val
            # Generate translations
            labels=labels-labels0
            rec,gen_imgs,_ = net(imgs, labels)
            gen_img = torch.cat([x for x in gen_imgs.data], -1)
            save_image(gen_img,os.path.join('path/imgs',str(i)+'.jpg'),normalize=True)
sample_images()
print('done')

