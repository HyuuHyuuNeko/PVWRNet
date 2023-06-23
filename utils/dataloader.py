import os
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image


def image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])


class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, img_options=None):
        super(DataLoaderTrain, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'syn')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'target')))

        self.inp_filenames = [os.path.join(rgb_dir, 'syn', x) for x in inp_files if image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'target', x) for x in tar_files if image_file(x)]

        self.img_options = img_options
        self.sizex = len(self.tar_filenames)

        self.ps = self.img_options['patch_size']

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex

        inp_img = Image.open(self.inp_filenames[index_])
        tar_img = Image.open(self.tar_filenames[index_])

        w, h = tar_img.size
        padw = self.ps - w if w <= self.ps else 0
        padh = self.ps - h if h <= self.ps else 0

        # Padding
        if padw != 0 or padh != 0:
            inp_img = TF.pad(inp_img, (0, 0, padw, padh), padding_mode='reflect')
            tar_img = TF.pad(tar_img, (0, 0, padw, padh), padding_mode='reflect')

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)

        if h - self.ps != 0:
            rr = random.randint(0, h - self.ps)
            cc = random.randint(0, w - self.ps)

            # Crop patch
            inp_img = inp_img[:, rr:rr + self.ps, cc:cc + self.ps]
            tar_img = tar_img[:, rr:rr + self.ps, cc:cc + self.ps]

        # Data Augmentations
        aug = random.randint(0, 8)
        if aug == 1:
            inp_img = inp_img.flip(1)
            tar_img = tar_img.flip(1)
        elif aug == 2:
            inp_img = inp_img.flip(2)
            tar_img = tar_img.flip(2)
        elif aug == 3:
            inp_img = torch.rot90(inp_img, dims=(1, 2))
            tar_img = torch.rot90(tar_img, dims=(1, 2))
        elif aug == 4:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=2)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=2)
        elif aug == 5:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=3)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=3)
        elif aug == 6:
            inp_img = torch.rot90(inp_img.flip(1), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(1), dims=(1, 2))
        elif aug == 7:
            inp_img = torch.rot90(inp_img.flip(2), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(2), dims=(1, 2))

        filename = os.path.splitext(os.path.split(self.tar_filenames[index_])[-1])[0]


        return tar_img, inp_img, filename


class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, img_options=None):
        super(DataLoaderVal, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'syn')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'target')))

        self.inp_filenames = [os.path.join(rgb_dir, 'syn', x) for x in inp_files if image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'target', x) for x in tar_files if image_file(x)]

        self.img_options = img_options
        self.sizex = len(self.tar_filenames)

        self.ps = self.img_options['patch_size']

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex

        inp_img = Image.open(self.inp_filenames[index_])
        tar_img = Image.open(self.tar_filenames[index_])

        # Validate on center crop
        w, h = tar_img.size
        if w - self.ps != 0:
            if self.ps is not None:
                inp_img = TF.center_crop(inp_img, (self.ps, self.ps))
                tar_img = TF.center_crop(tar_img, (self.ps, self.ps))

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)

        filename = os.path.splitext(os.path.split(self.tar_filenames[index_])[-1])[0]

        return tar_img, inp_img, filename


class DataLoaderTest(Dataset):
    def __init__(self, inp_dir, img_options):
        super(DataLoaderTest, self).__init__()

        inp_files = sorted(os.listdir(inp_dir))
        self.inp_filenames = [os.path.join(inp_dir, x) for x in inp_files if image_file(x)]

        self.inp_size = len(self.inp_filenames)
        self.img_options = img_options

    def __len__(self):
        return self.inp_size

    def __getitem__(self, index):
        path_inp = self.inp_filenames[index]
        filename = os.path.splitext(os.path.split(path_inp)[-1])[0]
        inp = Image.open(path_inp).convert('RGB')
        inp = TF.to_tensor(inp)
        return inp, filename
