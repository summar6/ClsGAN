import os
from datasets import *
from torchvision import transforms
from torch.utils.data import DataLoader
from models import *
import torch
from torchvision.utils import save_image

def sample_images(opt):
        val_transforms = [
           transforms.CenterCrop((170, 170)),
           transforms.Resize((128, 128)),
           transforms.ToTensor(),
           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        val_dataloader = DataLoader(
            CelebADataset(opt.root, transforms_=val_transforms, mode="test",
                      attributes=opt.selected_attrs),batch_size=2000, shuffle=False)
        cuda = torch.cuda.is_available()
        device = torch.device('cuda' if cuda else 'cpu', opt.gpu)
        os.makedirs(opt.test, exist_ok=True)
        net = GeneratorResNet((3, 128, 128), 13)
        net = net.to(device)
        net.load_state_dict(torch.load('%s/generator_19.pth'%opt.model_dir, map_location='cuda'))

        net.eval()
        label_changes = [
            ((0, 1),),
            ((1, 1),),
            ((2, 1), (3, 0), (4, 0)),
            ((2, 0),(3,1),(4,0)),
            ((2,0),(3,0),(4,1)),
            ((5,1),),
            ((6,1),),
            ((7,-1),),
            ((8,1),),
            ((9,1),(10,0)),
            ((9,0),(10,1)),
            ((11,1),),
            ((12,-1),),]

        val_imgs, val_labels = next(iter(val_dataloader))
        val_imgs,val_labels= val_imgs.to(device),val_labels.to(device)
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
            gen_imgs,_ = net(imgs, labels)
            gen_img = torch.cat([x for x in gen_imgs.data], -1)
            img_sample = torch.cat((img.data, gen_img), -1)
            
            save_image(img_sample,os.path.join(opt.test_dir,str(i)+'.jpg'),normalize=True)





