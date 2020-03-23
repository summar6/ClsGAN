
import argparse
import os
from train import network
from test_single import sample_images
from train import network
#parameter assign
def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--mode", type=str, default='train', help="the eecution mode")
    parser.add_argument("--gpu", type=int, default=0, help="the cuda number")
    parser.add_argument("--n_epochs", type=int, default=20, help="number of epochs of training")
    parser.add_argument("--dataset_name", type=str, default="CelebA", help="name of the dataset")
    parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=10, help="epoch from which to start lr decay")
    parser.add_argument("--img_height", type=int, default=128, help="size of image height")
    parser.add_argument("--img_width", type=int, default=128, help="size of image width")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--image_dir", type=str, default='output/images', help="the dir of saved images")
    parser.add_argument("--loss_dir", type=str, default='output/loss', help="the dir of saved loss ")
    parser.add_argument("--model_dir", type=str, default='output/models/', help="the dir of saved models")
    parser.add_argument("--test_dir", type=str, default='output/test/', help="the dir of saved test images")
    parser.add_argument("--sample_interval", type=int, default=2000, help="interval between saving generator samples")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between model checkpoints")
    parser.add_argument("--selected_attrs","--list",nargs="+",help="selected attributes for the CelebA dataset",
        default=["Bald","Bangs","Black_Hair", "Blond_Hair", "Brown_Hair","Bushy_Eyebrows","Eyeglasses", "Male","Mouth_Slightly_Open","Mustache","No_Beard","Pale_Skin", "Young"])
    parser.add_argument("--n_critic", type=int, default=5, help="number of training iterations for WGAN discriminator")
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = args()
    os.makedirs(opt.image_dir, exist_ok=True)
    os.makedirs(opt.loss_dir, exist_ok=True)
    os.makedirs(opt.model_dir, exist_ok=True)
    if opt.mode=='train':
        net = network(opt)
        net.train()
    else :
        sample_images(opt)
        print('done')
