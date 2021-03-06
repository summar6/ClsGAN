from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import sys
from torchvision.datasets import ImageFolder
from torchvision import transforms 
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.autograd as autograd
from models import *
from datasets import *
import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

#training net
class network(object):
    def __init__(self,opt):

        #parameters
        self.c_dim = len(opt.selected_attrs)
        self.img_shape = (opt.channels, opt.img_height, opt.img_width)
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu',opt.gpu)
        self.img_height=opt.img_height
        self.dataset_name=opt.dataset_name
        self.selected_attrs=opt.selected_attrs
        self.batch_size=opt.batch_size
        self.epoch=opt.epoch
        self.decay_epoch=opt.decay_epoch
        self.n_epochs=opt.n_epochs
        self.n_critic=opt.n_critic
        self.sample_interval=opt.sample_interval
        self.checkpoint_interval=opt.checkpoint_interval
        self.image_dir=opt.image_dir
        self.loss_dir=opt.loss_dir
        self.model_dir=opt.model_dir
        # Loss functions
        self.criterion_att = torch.nn.L1Loss()
        self.criterion_cls=torch.nn.BCEWithLogitsLoss(size_average=False)
        # Loss weights
        self.lambda_cls = 3
        self.lambda_gp1=30
        self.lambda_rec = 20
        self.lambda_gp = 10
        self.lambda_s=1
        self.lambda_dadv=4
        self.label_changes = [
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
           ((12,-1),),
        ]

        # Initialize generator and discriminator
        self.generator=GeneratorResNet(img_shape=self.img_shape,c_dim=self.c_dim)
        self.discriminator = Discriminator(img_shape=self.img_shape, c_dim=self.c_dim)
        
        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)
        self.criterion_att.to(self.device)
        self.criterion_cls.to(self.device)
       # self.generator=nn.DataParallel(self.generator,device_ids=[0,1])
       # self.discriminator=nn.DataParallel(self.discriminator,device_ids=[0,1])
        # Load pretrained models
        if opt.epoch != 0:
          self.generator.load_state_dict(torch.load("saved_models/generator_%d.pth" % opt.epoch))
          self.discriminator.load_state_dict(torch.load("saved_models/discriminator_%d.pth" % opt.epoch))
        else:
          self.generator.apply(weights_init_normal)
          self.discriminator.apply(weights_init_normal)

        # Optimizers
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        self.schedule_G=LambdaLR(self.optimizer_G,lr_lambda=Lambda(self.n_epochs,self.epoch,self.decay_epoch).step,last_epoch=-1)
        self.schedule_D=LambdaLR(self.optimizer_D,lr_lambda=Lambda(self.n_epochs,self.epoch,self.decay_epoch).step,last_epoch=-1)
        #dataloader
        self.dataloader,self.val_dataloader= self.dataloader()
        #paralel
    def dataloader(self):
        train_transforms = [
            transforms.CenterCrop((170,170)),
            transforms.Resize((128,128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        dataloader = DataLoader(
            CelebADataset("../stargan/%s" % self.dataset_name, transforms_=train_transforms, mode="train", attributes=self.selected_attrs),
            batch_size=self.batch_size,shuffle=True,num_workers=8)

        val_dataloader = DataLoader(
            CelebADataset("../stargan/%s" % self.dataset_name, transforms_=train_transforms, mode="val", attributes=self.selected_attrs),
            batch_size=10,shuffle=True)
        return dataloader,val_dataloader

    # Calculates the gradient penalty loss
    def compute_gradient_penalty(self,D, real_samples, fake_samples):
        # Random weight term for interpolation between real and fake samples
        alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(self.device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates,c_inter= D(interpolates)
        fake = Variable(torch.Tensor(np.ones(d_interpolates.shape)), requires_grad=False).to(self.device)
        fake1 = Variable(torch.Tensor(np.ones(c_inter.shape)), requires_grad=False).to(self.device)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients_c = autograd.grad(
            outputs=c_inter,
            inputs=interpolates,
            grad_outputs=fake1,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        gradients_c = gradients_c.view(gradients_c.size(0), -1)
        gradient_penalty_c = ((gradients_c.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty,gradient_penalty_c

    def sample_images(self,epoch,number):
        """Saves a generated sample of domain translations"""
        self.generator.eval()
        val_imgs, val_labels = next(iter(self.val_dataloader))
        val_imgs = val_imgs.to(self.device)
        val_labels = val_labels.to(self.device)
        img_samples = None
        for i in range(10):
            img, label = val_imgs[i], val_labels[i]
            imgs = img.repeat(self.c_dim, 1, 1, 1)
            labels = label.repeat(self.c_dim, 1)
            labels0 = label.repeat(self.c_dim, 1)
            # Make changes to labels
            for sample_i, changes in enumerate(self.label_changes):
                for col, val in changes:
                    labels[sample_i, col] = 1 - labels[sample_i, col] if val == -1 else val
            labels=labels-labels0
            # Generate translations
            gen_imgs,_ = self.generator(imgs, labels)
            # Concatenate images by width
            gen_imgs = torch.cat([x for x in gen_imgs.data], -1)
            img_sample = torch.cat((img.data, gen_imgs), -1)
            img_samples = img_sample if img_samples is None else torch.cat((img_samples, img_sample), -2)

        save_image(img_samples.view(1, *img_samples.shape), "%s/%d_%d.png" % (self.image_dir,epoch,number), normalize=True)


    def loss_plot(self,epoch,i):
        x = range(len(self.loss['loss_Grec']))

        y1 = self.loss['loss_Gadv']
        y2 = self.loss['loss_Dadv']
        y3 = self.loss['loss_Dcls']
        y4 = self.loss['loss_Gcls']
        y5 = self.loss['loss_Grec']
        y6=self.loss['loss_att']
        y7=self.loss['loss_D']
        y8=self.loss['loss_G']
      
        plt.plot(x, y1, label='loss_Gadv')
        plt.plot(x, y2, label='loss_Dadv')
        plt.plot(x, y3, label='loss_Dcls')
        plt.plot(x, y4, label='loss_Gcls')
        plt.plot(x, y5, label='loss_Grec')
        plt.plot(x,y6,label='loss_att')
        plt.plot(x,y7,label='loss_D')
        plt.plot(x,y8,label='loss_G')
       
        plt.xlabel('Iter')
        plt.ylabel('Loss')

        plt.legend(loc=0)
        plt.grid(True)
        plt.tight_layout()

        plt.savefig('%s/%d_%d_loss.png'%(self.loss_dir,epoch,i))

        plt.close()     
# ----------
#  Training
# ----------
    def train(self):
        self.loss= {}
        self.loss['loss_Gadv']=[]
        self.loss['loss_Dadv']=[]
        self.loss['loss_Dcls']=[]
        self.loss['loss_Gcls']=[]
        self.loss['loss_Grec']=[]
        self.loss['loss_G'] = []
        self.loss['loss_D'] = []
        self.loss['loss_att']=[]

        for epoch in range(self.epoch, self.n_epochs):
            for i, (imgs, labels) in enumerate(self.dataloader):
                # Model inputs
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                label=torch.ones(labels.size(0),labels.size(1)+1).to(self.device)
                label[:,1:14]=labels
                labels1=torch.zeros(imgs.size(0)).to(self.device)
                # ---------------------
                #  Train Discriminator
                # ---------------------
                self.optimizer_D.zero_grad()
                trg=torch.randperm(labels.size(0))
                sampled = labels[trg].to(self.device)
                sampled1=sampled-labels
                sampled_c=torch.ones(labels.size(0),labels.size(1)+1).to(self.device)
                sampled_c[:,1:14]=sampled
                fake_imgs,la = self.generator(imgs, sampled1)
            
                # Real images
                real_validity, pred_cls = self.discriminator(imgs)
                # Fake images
                fake_validity, pred= self.discriminator(fake_imgs.detach())
                # Gradient penalty
                gradient_penalty,c_gra = self.compute_gradient_penalty(self.discriminator, imgs.data, fake_imgs.data)
                # Adversarial loss
                loss_D_adv =-torch.mean(real_validity) +torch.mean(fake_validity)+self.lambda_gp * gradient_penalty 
                # Classification loss
                a=self.criterion_cls(pred_cls, label)/pred_cls.size(0)
                b=self.criterion_cls(pred[:,0],labels1)/pred.size(0)
             #   loss_D_cls = self.criterion_cls(pred_cls, label)/pred_cls.size(0)+self.criterion_cls(pred[:,0],labels1)/pred.size(0)+15*c_gra
                # Total loss
                loss_D_cls=a+b+self.lambda_gp1*c_gra
                loss_D = self.lambda_dadv*loss_D_adv + self.lambda_cls * loss_D_cls
            
                loss_D.backward()
                self.optimizer_D.step()

                # Every n_critic times update generator
                self.optimizer_G.zero_grad()
                if i % self.n_critic == 0:

                    # -----------------
                    #  Train Generator
                    # -----------------
                    la1=la-labels
                    rec_imgs,_ = self.generator(imgs,la1)
                    # Discriminator evaluates translated image
                    fake_validity, pred_cls = self.discriminator(fake_imgs)
                    # Adversarial loss
                    loss_G_adv =-torch.mean(fake_validity)
                    # Classification loss
                    loss_G_cls = self.criterion_cls(pred_cls, sampled_c)/pred_cls.size(0)
                    # Reconstruction loss
                    loss_G_rec = self.criterion_att(rec_imgs, imgs)
                    #attribute continuity loss
                    loss_G_att=self.criterion_att(la,labels)
                    # Total loss
                    loss_G = loss_G_adv + loss_G_cls + self.lambda_rec * loss_G_rec+self.lambda_s*loss_G_att

                    loss_G.backward()
                    self.optimizer_G.step()

                    # Print log
                    sys.stdout.write(
                        "\r[Epoch %d/%d][Batch %d/%d][D_adv:%f,g_pen:%f,D_cls:%f,real:%f,fake:%f,c_pen:%f] [G_adv:%f,G_cls:%f,G_rec:%f,G_att:%f]" % (epoch,self.n_epochs,i,len(self.dataloader),loss_D_adv.item(),gradient_penalty.item(),(self.lambda_cls*loss_D_cls).item(),a.item(),b.item(),self.lambda_cls*self.lambda_gp1*c_gra.item(), loss_G_adv.item(),loss_G_cls.item(),(self.lambda_rec*loss_G_rec).item(),(loss_G_att).item()))

                    # If at sample interval sample and save image
                    if i % self.sample_interval == 0 and epoch!=0 and epoch!=1:#and i!=0 and epoch!=0:
                       self.loss['loss_Gadv'].append(loss_G_adv.item())
                       self.loss['loss_Dadv'].append(loss_D_adv.item())
                       self.loss['loss_Dcls'].append(self.lambda_cls*loss_D_cls.item())
                       self.loss['loss_Gcls'].append(loss_G_cls.item())
                       self.loss['loss_Grec'].append((self.lambda_rec * loss_G_rec).item())
                       self.loss['loss_att'].append(loss_G_att.item())
                       self.loss['loss_G'].append(loss_G.item())
                       self.loss['loss_D'].append(loss_D.item())

                       self.sample_images(epoch,i)
                       self.loss_plot(epoch,i)
                      
            self.schedule_D.step()
            self.schedule_G.step()         
            if epoch % self.checkpoint_interval == 0:
                 # Save model checkpoints
                torch.save(self.generator.state_dict(), "%s/generator_%d.pth" % (self.model_dir,epoch))
                torch.save(self.discriminator.state_dict(), "%s/discriminator_%d.pth" % (self.model_dir,epoch))
   

