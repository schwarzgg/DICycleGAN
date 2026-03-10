import torch
import random
from torch.autograd import Variable


class ImagePool:
    def __init__(self,pool_size) -> None:
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self,images):
        if self.pool_size == 0:
            return images

        new_images = []
        for image in images.data:
            image = torch.unsqueeze(image,0)
            if self.num_imgs < self.pool_size:
                self.num_imgs+=1
                self.images.append(image)
                new_images.append(image)
            else:
                d = random.uniform(0,1)
                if d > 0.5:
                    id = random.randint(0,self.pool_size - 1)
                    tmp = self.images[id].clone()
                    self.images[id] = image
                    new_images.append(tmp)
                else :
                    new_images.append(image)
        
        new_images = Variable(torch.cat(new_images,0))
        return new_images
