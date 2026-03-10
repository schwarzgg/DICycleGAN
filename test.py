import os,cv2
from pathlib import Path

from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.utils import save_image
from tqdm import tqdm

from models.Generator import Generator
from options.config import *
from utils.metrics import PSNR, SSIM
from utils.utils import load_networks


def test():
    A2B = Generator(in_channel=3)

    trans = Compose([
        ToTensor(),
        Resize((256, 256))
    ])

    load_networks(A2B, CHECKPOINT_GEN_A)
    testA2B_data = [str(i) for i in Path('./data/test/A').glob('*') if str(i).split(".")[-1] in ['jpg', 'png', 'jpeg']]

    if not os.path.exists("./data/test/outputs"):
        os.makedirs("./data/test/outputs")

    PSNR_list,SSIM_list = [],[]

    with torch.no_grad():
        for img_path in tqdm(testA2B_data):
            img_name = img_path.split("\\")[-1]
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype('float32') / 255.
            real = ToTensor()(img).to(DEVICE)
            H, W = img.shape[:2]
            real = real.view([1, 3, H, W])
            img = trans(img).unsqueeze(0).to(DEVICE)

            out = 0.5 * (A2B(img) + 1.)
            out = Resize((H, W))(out)
            out = out.clamp(0, 1).to(DEVICE)

            psnr = PSNR(real, out)
            ssim = SSIM(real, out, window_size=7)

            PSNR_list.append(psnr)
            SSIM_list.append(ssim)

            print(f"{img_name},psnr:{psnr},ssim:{ssim}")

            save_image(out, f'./data/test/outputs/{img_name}')

    print(f"PSNR:{sum(PSNR_list) / len(PSNR_list)},Max:{max(PSNR_list)}")
    print(f"SSIM:{sum(SSIM_list) / len(SSIM_list)},Max:{max(SSIM_list)}")


if __name__ == '__main__':
    test()
