from torchviz import make_dot
from torchinfo import summary
import torch

from models.Generator import Generator
from models.Discriminator import Discriminator

gen = Generator(in_channel=3,out_channel=3,dim=64)
disc = Discriminator(in_channel=3)

# summary(gen,(1,3,256,256))
summary(disc,(1,3,256,256))