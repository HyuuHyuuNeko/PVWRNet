import os
import random
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils import losses
from utils.basic_utils import mkdir, load_checkpoint, load_optim, load_start_epoch
from utils.image_metric import PSNR
from utils.dataloader import DataLoaderTrain, DataLoaderVal
from utils.warmup_scheduler import GradualWarmupScheduler
from module.PWRNet import PWRNet

from skimage import img_as_ubyte
from utils.basic_utils import load_checkpoint, save_img

# ----------------------------------------------------------------------------------------------------
# Set seeds
random.seed(12345)
np.random.seed(12345)
torch.manual_seed(12345)
torch.cuda.manual_seed_all(12345)
# ----------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='PWMRNet')
parser.add_argument('--train_dir', default='', type=str,
                    help='Directory of training images')
parser.add_argument('--val_dir', default='', type=str,
                    help='Directory for testing images')
parser.add_argument('--save_dir', default='checkpoints/', type=str,
                    help='Directory for weight saving')
parser.add_argument('--weight', default='./BestWeight.pth', type=str,
                    help='weight file')
parser.add_argument('--gpus', default=[0, 1], type=list,
                    help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--batch_size', default=6, type=int,
                    help='batch_size. ATTENTION: this value cannot be lower than 4!')
parser.add_argument('--epochs', default=100, type=int,
                    help='epochs')
parser.add_argument('--val_epoch_every', default=2, type=int,
                    help='The interval between epochs for validation.')
parser.add_argument('--train_patch', default=256, type=int,
                    help='The patch size used for training.')
parser.add_argument('--val_patch', default=512, type=int,
                    help='The patch size used for validation.')
parser.add_argument('--lr_init', default=1e-3, type=float,
                    help='The initial value of the learning rate.')
parser.add_argument('--lr_min', default=1e-6, type=float,
                    help='The minimum value of the learning rate.')
# ----------------------------------------------------------------------------------------------------
# torch.backends.cudnn.benchmark = True
torch.cuda.ipc_collect()
torch.cuda.empty_cache()
# ----------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    args = parser.parse_args()
    gpus = ','.join([str(i) for i in args.gpus])

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    device_ids = [i for i in range(torch.cuda.device_count())]
    if torch.cuda.device_count() > 1:
        print('*' * 50)
        print("\t\t\t", "GPU [", gpus, "] will be used!")
        print('*' * 50)

    model_dir = os.path.join(args.save_dir, 'weights')
    mkdir(model_dir)

    # Folders only named "syn" and "target" in the "train_dir" / "val_dir"!
    train_dir = args.train_dir
    val_dir = args.val_dir

    model = PWRNet()
    model.cuda()
    # ----------------------------------------------------------------------------------------------------
    # Initial
    start_epoch = 1
    # new_lr = args.lr_init
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr_init,
                           betas=(0.9, 0.999),
                           eps=1e-8)
    # ----------------------------------------------------------------------------------------------------
    # Scheduler initial
    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            args.epochs - warmup_epochs,
                                                            eta_min=args.lr_min)
    scheduler = GradualWarmupScheduler(optimizer,
                                       multiplier=1,
                                       total_epoch=warmup_epochs,
                                       after_scheduler=scheduler_cosine)
    scheduler.step()
    # ----------------------------------------------------------------------------------------------------
    # Loss
    loss_func_char = losses.CharbonnierLoss()
    loss_func_edge = losses.LaplaLoss()
    # ----------------------------------------------------------------------------------------------------
    # DataLoaders
    train_dataset = DataLoaderTrain(train_dir, {'patch_size': args.train_patch})
    val_dataset = DataLoaderVal(val_dir, {'patch_size': args.val_patch})
    # ↓ Tips: "num_workers" only working on Linux. (Windows: num_workers=1) ↓
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=1,
                              drop_last=False,
                              pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=1,
                            drop_last=False,
                            pin_memory=True)
    # ----------------------------------------------------------------------------------------------------
    # Load checkpoint
    print("-" * 120)
    if args.weight:
        load_checkpoint(model, args.weight)
        print("===> \"", args.weight.split("/")[-1], "\" was loading! <===")

        if args.load_opt_lr:
            start_epoch = load_start_epoch(args.weight) + 1
            load_optim(optimizer, args.weight)

            for i in range(1, start_epoch):
                scheduler.step()
            print("===> lr: {:.8f} <===".format(scheduler.get_last_lr()[0]))

    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
    # ----------------------------------------------------------------------------------------------------
    print("===> Epochs Start from {} to {} <===".format(start_epoch, args.epochs + 1))
    print("===> Start at: {} <===".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))))
    print("-" * 120)
    # ----------------------------------------------------------------------------------------------------
    best_psnr = 0
    best_epoch = 0

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.time()
        epoch_loss = 0
        # ----------------------------------------------------------------------------------------------------
        # Training
        model.train()
        psnr_train = []
        for i, data in enumerate(tqdm(train_loader), 0):
            target = data[0].cuda()
            input_ = data[1].cuda()
            pred = model(input_)

            # Some versions of NumPy may have bugs.
            # loss_char = np.sum([loss_func_char(pred[j], target) for j in range(len(pred))])
            # loss_edge = np.sum([loss_func_edge(pred[j], target) for j in range(len(pred))])
            loss_char = 0
            loss_edge = 0
            for j in range(len(pred)):
                loss_char += loss_func_char(pred[j], target)
                loss_edge += loss_func_edge(pred[j], target)
            loss = loss_char + (0.05 * loss_edge)

            with torch.no_grad():
                for res, tar in zip(pred[0], target):
                    psnr_train.append(PSNR(res, tar))

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            model.zero_grad()
        psnr_train = torch.stack(psnr_train).mean().item()
        # ----------------------------------------------------------------------------------------------------
        scheduler.step()
        # ----------------------------------------------------------------------------------------------------
        # Print train info
        tqdm.write("[ TRAIN_INFO:    Epoch: {}    LastTime: {}    TotalLoss: {:.4f}    PSNR: {:.4f} ]"
                   .format(epoch,
                           time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),
                           epoch_loss,
                           # scheduler.get_last_lr()[0]))
                           psnr_train))
        # ----------------------------------------------------------------------------------------------------
        # Evaluation
        if epoch % args.val_epoch_every == 0:
            model.eval()
            psnr_val = []

            for ii, data_val in enumerate(tqdm(val_loader), 0):
                target = data_val[0].cuda()
                input_ = data_val[1].cuda()

                with torch.no_grad():
                    pred = model(input_)

                    restored = torch.clamp(pred[0], 0, 1)
                    restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
                    restored_img = img_as_ubyte(restored[0])
                    save_img((os.path.join('checkpoints', str(epoch) + '.png')), restored_img)

                for res, tar in zip(pred[0], target):
                    psnr_val.append(PSNR(res, tar))

            psnr_val = torch.stack(psnr_val).mean().item()
            # ----------------------------------------------------------------------------------------------------
            # print val info
            tqdm.write("[ VAL_INFO:    Epoch: {}    LastTime: {}    PSNR: {:.4f} ]"
                       .format(epoch,
                               time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),
                               psnr_val))
            # ----------------------------------------------------------------------------------------------------
            # Save best weights
            if psnr_val > best_psnr:
                best_psnr = psnr_val
                best_epoch = epoch
                torch.save({'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict()},
                           os.path.join(model_dir, "BestPSNR_{:.4f}_epoch_{}.pth".format(best_psnr,
                                                                                         best_epoch)))
            # ----------------------------------------------------------------------------------------------------
            tqdm.write(
                "[ CHECKPOINT_INFO:    NowEpoch: %d    BestEpoch: %d    BestPSNR: %.4f ]" % (epoch,
                                                                                             best_epoch,
                                                                                             best_psnr))
            # ----------------------------------------------------------------------------------------------------
            # Save each weights
            # torch.save({'epoch': epoch,
            #             'state_dict': model.state_dict(),
            #             'optimizer': optimizer.state_dict()},
            #            os.path.join(model_dir, f"epoch_{epoch}_PSNR_{psnr_val}.pth"))
