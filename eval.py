import os
import torch
import argparse
from tqdm import tqdm
import torch.nn as nn
from pathlib import Path
from skimage import img_as_ubyte
from torch.utils.data import DataLoader

from utils.dataloader import DataLoaderTest
from utils.basic_utils import load_checkpoint, save_img
from module.PWRNet import PWRNet

parser = argparse.ArgumentParser(description='PWMRNet')
parser.add_argument('--input_dir', default='', type=str,
                    help='Directory of validation images')
parser.add_argument('--result_dir', default='', type=str,
                    help='Directory for results')
parser.add_argument('--weights', default='BestWeight.pth', type=str,
                    help='Path to weights')
parser.add_argument('--gpus', default=[1], type=list,
                    help='CUDA_VISIBLE_DEVICES')
args = parser.parse_args()

gpus = ','.join([str(i) for i in args.gpus])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
    print('=' * 50)
    print("\t\t\t", torch.cuda.device_count(), "GPUs will be used!")
    print('=' * 50)


if __name__ == '__main__':
    model_restoration = PWRNet()
    load_checkpoint(model_restoration, args.weights)

    model_restoration.cuda()
    model_restoration = nn.DataParallel(model_restoration)
    model_restoration.eval()

    image_paths = list(Path(args.input_dir).glob("*.*"))

    test_dataset = DataLoaderTest(args.input_dir, img_options={})
    test_loader = DataLoader(dataset=test_dataset, batch_size=5, shuffle=False, num_workers=0, drop_last=False,
                             pin_memory=True)

    result_dir = args.result_dir

    with torch.no_grad():
        for ii, data_test in enumerate(tqdm(test_loader), 0):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            input_ = data_test[0].cuda()
            filenames = data_test[1]

            restored = model_restoration(input_)
            restored = torch.clamp(restored[0], 0, 1)

            restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()

            for batch in range(len(restored)):
                restored_img = img_as_ubyte(restored[batch])
                save_img((os.path.join(result_dir, filenames[batch] + '.png')), restored_img)

