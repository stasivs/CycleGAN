import os
from PIL import Image

import torch.utils as utils
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        self.img = [f for f in os.listdir(main_dir)]

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.img[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image

def return_dataloader(path, transforms, batch_size=2, num_workers=0):
    dataset = CustomDataset(path, transforms)
    dataloader = utils.data.DataLoader(dataset, shuffle=True,
                                       batch_size=batch_size, num_workers=num_workers)
    return dataloader


def return_const_photo(dataloader):
    return next(iter(dataloader))[0]


def return_img_from_tensor(img):
    img = img.view((3, 256, 256))
    img = img * 0.5 + 0.5
    img = img.permute(1, 2, 0).numpy()
    return img.clip(0, 1)