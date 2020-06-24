import os
from datasets import *
from torchvision import transforms
from torch.utils.data import DataLoader
from models import *
import torch
import numpy as np
from torchvision.utils import save_image

def sample_images(opt):
        torch.no_grad()
        val_transforms = [
           transforms.CenterCrop((170, 170)),
           transforms.Resize((128, 128)),
           transforms.ToTensor(),
           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        val_dataloader = DataLoader(
            CelebADataset(opt.dataset_dir, transforms_=val_transforms, mode="test",
                      attributes=opt.selected_attrs),batch_size=2000, shuffle=False)
        cuda = torch.cuda.is_available()
        device = torch.device('cuda' if cuda else 'cpu', opt.gpu)
        os.makedirs(opt.test_dir, exist_ok=True)
        os.makedirs('FID/1',exist_ok=True)
        os.makedirs('accuracy/bald',exist_ok=True)
        os.makedirs('accuracy/bangs',exist_ok=True)
        os.makedirs('accuracy/blackh',exist_ok=True)
        os.makedirs('accuracy/blondh',exist_ok=True)
        os.makedirs('accuracy/brownh',exist_ok=True)
        os.makedirs('accuracy/eyebrows',exist_ok=True)
        os.makedirs('accuracy/eyeglasses',exist_ok=True)
        os.makedirs('accuracy/gender',exist_ok=True)
        os.makedirs('accuracy/mouthopen',exist_ok=True)
        os.makedirs('accuracy/mustache',exist_ok=True)
        os.makedirs('accuracy/nobeard',exist_ok=True)
        os.makedirs('accuracy/pale',exist_ok=True)
        os.makedirs('accuracy/young',exist_ok=True)
        os.makedirs('FID/rec',exist_ok=True)

        net = GeneratorResNet((3, 128, 128), 13)
        net = net.to(device)
        net.load_state_dict(torch.load('%s/generator_16.pth'%opt.model_dir, map_location='cuda'))

        net.eval()
        label_changes = [
            ((0, 1),),
            ((1, 1),),
            ((2, 1), (3, 0), (4, 0)),
            ((2, 0),(3,1),(4,0)),
            ((2,0),(3,0),(4,1)),
            ((5,1),),
            ((6,1),),
            ((7,1),),
            ((8,1),),
            ((9,1),(10,0)),
            ((9,0),(10,1)),
            ((11,1),),
            ((12,1),),]

        val_imgs, val_labels = next(iter(val_dataloader))
        val_imgs,val_labels= val_imgs.to(device),val_labels.to(device)
        for i in range(2000):
            img, label = val_imgs[i], val_labels[i]

            # Repeat for number of label changes
            imgs = img.repeat(14, 1, 1, 1)
            labels0 = label.repeat(14, 1)
            labels = label.repeat(14, 1)
            # Make changes to labels
            for sample_i, changes in enumerate(label_changes):
                for col, val in changes:
                    labels[sample_i, col] = 1 - labels[sample_i, col] if val == -1 else val
            # Generate translations
            labels=labels-labels0
            gen_imgs,_ = net(imgs, labels)
            gen_img = torch.cat([x for x in gen_imgs.data], -1)
            img_sample = torch.cat((img.data, gen_img), -1)
            #basic generated images
            save_image(img_sample,os.path.join(opt.test_dir,str(i)+'.jpg'),normalize=True)
            #Fid compute
            a=np.random.randint(0,13,(5))
            save_image(gen_imgs[a[0]], os.path.join('FID', str(1),str(i*5) + '.jpg'), normalize=True)
            save_image(gen_imgs[a[1]], os.path.join('FID',str(1), str(i*5+1) + '.jpg'), normalize=True)
            save_image(gen_imgs[a[2]], os.path.join('FID',str(1), str(i*5+2) + '.jpg'), normalize=True)
            save_image(gen_imgs[a[3]], os.path.join('FID',str(1), str(i*5+3) + '.jpg'), normalize=True)
            save_image(gen_imgs[a[4]], os.path.join('FID',str(1), str(i*5+4) + '.jpg'), normalize=True)
            #accuracy
            
            save_image(gen_imgs[0],os.path.join('accuracy/bald',str(i)+'.jpg'),normalize=True)
            save_image(gen_imgs[1],os.path.join('accuracy/bangs',str(i)+'.jpg'),normalize=True)
            save_image(gen_imgs[2],os.path.join('accuracy/blackh',str(i)+'.jpg'),normalize=True)
            save_image(gen_imgs[3],os.path.join('accuracy/blondh',str(i)+'.jpg'),normalize=True)
            save_image(gen_imgs[4],os.path.join('accuracy/brownh',str(i)+'.jpg'),normalize=True)
            save_image(gen_imgs[5],os.path.join('accuracy/eyebrows',str(i)+'.jpg'),normalize=True)
            save_image(gen_imgs[6],os.path.join('accuracy/eyeglasses',str(i)+'.jpg'),normalize=True)
            save_image(gen_imgs[7],os.path.join('accuracy/gender',str(i)+'.jpg'),normalize=True)
            save_image(gen_imgs[8],os.path.join('accuracy/mouthopen',str(i)+'.jpg'),normalize=True)
            save_image(gen_imgs[9],os.path.join('accuracy/mustache',str(i)+'.jpg'),normalize=True)
            save_image(gen_imgs[10],os.path.join('accuracy/nobeard',str(i)+'.jpg'),normalize=True)
            save_image(gen_imgs[11],os.path.join('accuracy/pale',str(i)+'.jpg'),normalize=True)
            save_image(gen_imgs[12],os.path.join('accuracy/young',str(i)+'.jpg'),normalize=True)
            save_image(gen_imgs[13],os.path.join('FID/rec',str(i)+'.jpg'),normalize=True)
        print('finished')
         




