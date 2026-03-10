import torch
from PIL import Image
from torchvision import transforms

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/train"
VAL_DIR = "data/test"
BATCH_SIZE = 1
POOL_SIZE = 50
LEARNING_RATE = 0.0002
LAMBDA_A = 10
LAMBDA_B = 10
LAMBDA_IDENTITY = 5
LAMBDA_PERCEPTUAL = 0.0001*0.5


NUM_WORKERS = 6
NUM_EPOCHS = 80
SAVE_MODEL = True
JUST_EVAL = True
CHECKPOINT_GEN_A = "net_GA.pth"
CHECKPOINT_GEN_B = "net_GB.pth"
CHECKPOINT_DISC_A = "net_DA.pth"
CHECKPOINT_DISC_B = "net_DB.pth"
FREQ_STEP = 100
DECAY_START_EPOCH = NUM_EPOCHS // 2

transform = transforms.Compose([
    transforms.Resize(int(256 * 1.2), Image.BILINEAR),
    transforms.RandomCrop(256),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
