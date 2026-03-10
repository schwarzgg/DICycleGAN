import itertools
import os

import torch.nn
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm

from models.Discriminator import Discriminator
from models.Generator import Generator
from models.Loss import PerceptualLoss, GANLoss
from options.config import *
from utils.dataset import ImageSet
from utils.image_pool import ImagePool
from utils.utils import save_networks, weights_init_normal


def train():
    writer = SummaryWriter(log_dir="logs/train")

    dataset = ImageSet(root="./data", model="train", transforms=transform)
    loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    val = ImageSet(root="./data", model="val", transforms=val_transform)
    val_loader = DataLoader(dataset=val, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # Generator init
    netG_A2B = Generator(in_channel=3).to(DEVICE)  # G
    netG_B2A = Generator(in_channel=3).to(DEVICE)  # F

    # Discriminator init
    netD_A = Discriminator(in_channel=3).to(DEVICE)
    netD_B = Discriminator(in_channel=3).to(DEVICE)

    netG_A2B.apply(weights_init_normal)
    netG_B2A.apply(weights_init_normal)
    netD_A.apply(weights_init_normal)
    netD_B.apply(weights_init_normal)

    # Image Pool init
    fake_A_pool = ImagePool(POOL_SIZE)
    fake_B_pool = ImagePool(POOL_SIZE)

    # Loss init
    cycle_loss = nn.L1Loss().to(DEVICE)
    identity_loss = nn.L1Loss().to(DEVICE)
    GAN_loss = GANLoss(use_ls=True).to(DEVICE)
    Perceptual_loss = PerceptualLoss(torch.nn.MSELoss())

    # Optimizer init
    optim_G = optim.AdamW(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), lr=LEARNING_RATE,
                          betas=(0.5, 0.999), weight_decay=0.001)
    optim_D = optim.AdamW(itertools.chain(netD_A.parameters(), netD_B.parameters()), lr=LEARNING_RATE,
                          betas=(0.5, 0.999), weight_decay=0.001)

    lambda_lr = lambda step: (1.0 - max(0, step - DECAY_START_EPOCH) / (NUM_EPOCHS - DECAY_START_EPOCH))

    # Scheduler init
    scheduler_G = optim.lr_scheduler.LambdaLR(optimizer=optim_G, lr_lambda=lambda_lr, last_epoch=-1)
    scheduler_D = optim.lr_scheduler.LambdaLR(optimizer=optim_D, lr_lambda=lambda_lr, last_epoch=-1)

    total_iter = 0
    for epoch in range(NUM_EPOCHS):
        epoch_iter = 0

        loop = tqdm(enumerate(loader), total=len(loader), leave=True)
        loop.set_description(f"Epoch:{epoch + 1}/{NUM_EPOCHS}")

        for i, data in loop:
            total_iter += BATCH_SIZE
            epoch_iter += BATCH_SIZE

            real_A = Variable(data['A']).to(DEVICE)
            real_B = Variable(data['B']).to(DEVICE)

            optim_G.zero_grad()

            lambda_idt = LAMBDA_IDENTITY
            lambda_perceptual = LAMBDA_PERCEPTUAL
            lambda_A = LAMBDA_A
            lambda_B = LAMBDA_B

            # ----------------------------------
            # Identity Loss
            idt_B = netG_A2B(real_B)  # 由A2B生成B，判断生成的B与真实B的差距
            loss_idt_B = identity_loss(idt_B, real_B) * lambda_idt  # 身份损失计算

            idt_A = netG_B2A(real_A)  # B2A生成A，判断生成的A与真实A的差距
            loss_idt_A = identity_loss(idt_A, real_A) * lambda_idt  # 身份损失计算

            # ----------------------------------
            # General Loss
            fake_B = netG_A2B(real_A)  # 由真实图A生成假的B
            pre_fake_B = netD_B(fake_B)  # 判别器B判断生成B的真假
            loss_G_A2B = GAN_loss(pre_fake_B, True)  # 对抗损失计算

            fake_A = netG_B2A(real_B)  # 有真实图B生成假的A
            pre_fake_A = netD_A(fake_A)  # 判别器A判断生成A的真假
            loss_G_B2A = GAN_loss(pre_fake_A, True)  # 对抗损失计算

            # ----------------------------------
            # Cycle Loss
            rec_A = netG_B2A(fake_B.detach())  # 由生成的B生成A
            loss_cycle_A2B2A = cycle_loss(rec_A, real_A) * lambda_A  # 计算A->B->A的循环损失

            rec_B = netG_A2B(fake_A.detach())  # 由生成的A生成B
            loss_cycle_B2A2B = cycle_loss(rec_B, real_B) * lambda_B  # 计算A->B->A的循环损失

            # ----------------------------------
            # Perceptual Loss
            loss_perceptual_A = Perceptual_loss.get_loss(real_A, fake_A) * lambda_perceptual  # A的感知损失
            loss_perceptual_B = Perceptual_loss.get_loss(real_B, fake_B) * lambda_perceptual  # B的感知损失

            # ----------------------------------
            # Generator Loss
            loss_G = loss_G_A2B + loss_G_B2A + loss_cycle_A2B2A + loss_cycle_B2A2B + loss_idt_A + loss_idt_B + loss_perceptual_A + loss_perceptual_B

            # loss_G = loss_G_A2B/loss_G_A2B.detach() + loss_G_B2A/loss_G_B2A.detach() \
            #          + loss_cycle_A2B2A/loss_cycle_A2B2A.detach() + loss_cycle_B2A2B/loss_cycle_B2A2B.detach()\
            #          +loss_idt_A/loss_idt_A.detach() + loss_idt_B/loss_idt_B.detach() \
            #          + loss_perceptual_A/loss_perceptual_A.detach() + loss_perceptual_B/loss_perceptual_B.detach()

            loss_G.backward()
            optim_G.step()

            # A
            optim_D.zero_grad()
            pre_real_A = netD_A(real_A)
            loss_D_real = GAN_loss(pre_real_A, True)

            fake_A = fake_A_pool.query(fake_A)
            pre_fake_A = netD_A(fake_A.detach())
            loss_D_fake = GAN_loss(pre_fake_A, False)

            loss_DA = (loss_D_real + loss_D_fake) * 0.5

            # B
            pre_real_B = netD_B(real_B)
            loss_D_real = GAN_loss(pre_real_B, True)

            fake_B = fake_B_pool.query(fake_B)
            pre_fake_B = netD_B(fake_B.detach())
            loss_D_fake = GAN_loss(pre_fake_B, False)

            loss_DB = (loss_D_real + loss_D_fake) * 0.5

            loss_D = loss_DA + loss_DB

            loss_D.backward()
            optim_D.step()

            # tensorboard writer
            loop.set_postfix({'loss_DA': '{0:1.5f}'.format(loss_DA), 'loss_DB': '{0:1.5f}'.format(loss_DB),
                              'loss_G': '{0:1.5f}'.format(loss_G)})
            loop.update()
            if epoch_iter % FREQ_STEP == 0:
                writer.add_scalars("loss_GD", {"loss_netD": loss_D, "loss_netG": loss_G}, total_iter)
                writer.add_scalars("cycle_loss", {"loss_A": loss_cycle_A2B2A, "loss_B": loss_cycle_B2A2B}, total_iter)
                writer.add_scalars("identity_loss", {"loss_A": loss_idt_A, "loss_B": loss_idt_B}, total_iter)

        scheduler_D.step()
        scheduler_G.step()

        if JUST_EVAL:
            val_loop = tqdm(enumerate(val_loader))
            val_loop.set_description(f"Validation Epoch:{epoch + 1}")

            if not os.path.exists(f"./data/val/outputs/epoch{epoch + 1}"):
                os.makedirs(f"./data/val/outputs/epoch{epoch + 1}/A")
                os.makedirs(f"./data/val/outputs/epoch{epoch + 1}/B")

            for i, data in val_loop:
                real_A = Variable(data['A']).to(DEVICE)
                real_B = Variable(data['B']).to(DEVICE)

                fake_A = netG_B2A(real_B.detach())
                fake_B = netG_A2B(real_A.detach())

                save_image(fake_A, f"./data/val/outputs/epoch{epoch + 1}/A/{i + 1}.jpg")
                save_image(fake_B, f"./data/val/outputs/epoch{epoch + 1}/B/{i + 1}.jpg")

        if SAVE_MODEL:
            save_networks(netD_A, optim_D, epoch, CHECKPOINT_DISC_A)
            save_networks(netD_B, optim_D, epoch, CHECKPOINT_DISC_B)
            save_networks(netG_A2B, optim_G, epoch, CHECKPOINT_GEN_A)
            save_networks(netG_B2A, optim_G, epoch, CHECKPOINT_GEN_B)
            print("\n Saving Success.\n ")

    writer.close()


if __name__ == '__main__':
    train()
