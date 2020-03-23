import torch.nn as nn
import torch.nn.functional as F
import torch

# learning rate attenuation mode
class Lambda:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch
    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

# Tr-resnet block
class ResidualBlock(nn.Module):
    def __init__(self,inchannel,outchannel,kernel=3,stride=1,shortcut=None):
        super(ResidualBlock,self).__init__()
        self.left=nn.Sequential(nn.ConvTranspose2d(inchannel,outchannel,kernel,stride,1,bias=False),
                               nn.InstanceNorm2d(outchannel, affine=True, track_running_stats=True),
                               nn.ReLU(inplace=True),
                               nn.ConvTranspose2d(outchannel,outchannel,3,1,1,bias=False),
                               nn.InstanceNorm2d(outchannel, affine=True, track_running_stats=True))
        self.right=shortcut
        self.gamma = nn.Parameter(torch.rand((outchannel,1,1)))
    def forward(self,x):
        out=self.left(x)
        residual=x if self.right is None else self.right(x)
        out=out*self.gamma+residual*(1-self.gamma)
        return F.relu(out)

#Tr-resnet
class ResNet(nn.Module):
    def __init__(self,d,s):
        super(ResNet,self).__init__()
        self.layer1=self._make_layer(d,s,3,4,stride=2)
    def _make_layer(self,inchannel,outchannel,block_num,kernel,stride=1):
        shortcut=nn.Sequential(nn.ConvTranspose2d(inchannel,outchannel,2,2,bias=False),
                              nn.InstanceNorm2d(outchannel, affine=True, track_running_stats=True))
        layers=[]
        layers.append(ResidualBlock(inchannel,outchannel,kernel,stride,shortcut))
        for i in range(1,block_num):
            layers.append(ResidualBlock(outchannel,outchannel))
        return nn.Sequential(*layers)
    def forward(self,x):
        x=self.layer1(x)
        return x

#initialize parameters
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

#content encoder
class Encoder(nn.Module):
    def __init__(self,channels=3):
        super(Encoder,self).__init__()
        model = [
            nn.Conv2d(channels,64, 7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(64, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        ]
        curr_dim = 64
        model += [
                nn.Conv2d(curr_dim, curr_dim * 2, 4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(curr_dim*2, curr_dim * 4, 4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(curr_dim * 4, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True)]
        model1=[
                nn.Conv2d(curr_dim*4, curr_dim * 8, 4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(curr_dim * 8, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True)]

        self.model=nn.Sequential(*model)
        self.model1=nn.Sequential(*model1)
    def forward(self,x):
        x1=self.model(x)
        return x1,self.model1(x1)

#style encoder
class Style(nn.Module):
    def __init__(self,c_dim=5):
        super(Style,self).__init__()
        model = [
            nn.Conv2d(3, 1, 7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)]
        
        for _ in range(4):
            model += [
                nn.Conv2d(1,1, 4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True)]
         
        model1=[nn.Linear(8*8,c_dim),nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*model)
        self.model1=nn.Sequential(*model1)
    def forward(self,x):
        style=self.model(x)
        style=style.view(style.size(0),-1)        
        style=self.model1(style)
        return style

#generator
class GeneratorResNet(nn.Module):
    def __init__(self, img_shape=(3, 128, 128), c_dim=5):
        super(GeneratorResNet, self).__init__()
        channels, img_size, _ = img_shape

        self.content=Encoder(channels)
        self.style=Style(c_dim)
        # Residual blocks
        model=[]
        curr_dim=512
        model += [ResNet(curr_dim+c_dim,curr_dim//2)]
        model1=[ResNet(curr_dim//2,curr_dim//4)]
        model1+= [ResNet(curr_dim//4,curr_dim//8)]

        # Output layer
        model1 += [nn.Conv2d(curr_dim//8, channels, 7, stride=1, padding=3), nn.Tanh()]
        self.a = nn.Parameter(torch.rand((curr_dim//2,1,1)))
        self.model = nn.Sequential(*model)
        self.model1 = nn.Sequential(*model1)
    def forward(self, x,label):
        la,content=self.content(x)
        style=self.style(x)
        label=label.view(label.size(0),label.size(1),1,1)
        label=label.repeat(1,1,content.size(2),content.size(3))
        x=torch.cat((content,label),1)
        x= self.model(x)
        x=x+la*self.a   #skip-connection

        return self.model1(x),style

#discriminator
class Discriminator(nn.Module):
    def __init__(self, img_shape=(3, 128, 128), c_dim=5, n_strided=6):
        super(Discriminator, self).__init__()
        channels, img_size, _ = img_shape

        def discriminator_block(in_filters, out_filters):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1), nn.LeakyReLU(0.01)]
            return layers

        layers = discriminator_block(channels, 64)
        curr_dim = 64
        for _ in range(n_strided - 1):
            layers.extend(discriminator_block(curr_dim, curr_dim * 2))
            curr_dim *= 2

        self.model = nn.Sequential(*layers)

        # Output 1: PatchGAN
        self.out1 = nn.Conv2d(curr_dim, 1, 3, padding=1, bias=False)
        # Output 2: Class prediction
        kernel_size = img_size // 2 ** n_strided
        self.out2 = nn.Conv2d(curr_dim, c_dim+1, kernel_size, bias=False)

    def forward(self, img):
        feature_repr = self.model(img)
        out_adv = self.out1(feature_repr)
        out_cls = self.out2(feature_repr)
        return out_adv, out_cls.view(out_cls.size(0), -1)
