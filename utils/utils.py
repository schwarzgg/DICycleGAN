import os
import torch.nn.init

from options.config import *


def save_networks(model, optimizer, epoch, name, file_dir="./checkpoints"):
    save_name = '%s_%s' % (epoch+1, name)
    save_path = os.path.join(file_dir, save_name)
    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, save_path)


def load_networks(model, name, file_dir='./checkpoints'):
    save_path = os.path.join(file_dir, name)
    checkpoint = torch.load(save_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.cuda()

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
