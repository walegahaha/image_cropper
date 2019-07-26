import datetime
import os
import numpy as np
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
#from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from network import Unet
from dataset import PairDataset

cfg = {'PicaNet': "GGLLL",
       'Size': [28, 28, 28, 56, 112, 224],
       'Channel': [1024, 512, 512, 256, 128, 64],
       'loss_ratio': [0.5, 0.5, 0.5, 0.8, 0.8, 1]}

if __name__ == '__main__':
    torch.cuda.manual_seed_all(1234)
    torch.manual_seed(1234)

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    print("Default path : " + os.getcwd())
    parser.add_argument("--load",
                        help="Directory of pre-trained model, you can download at \n"
                             "https://drive.google.com/file/d/109a0hLftRZ5at5hwpteRfO1A6xLzf8Na/view?usp=sharing\n"
                             "None --> Do not use pre-trained model. Training will start from random initialized model\n"
                             "default = None",
                        default=None)
    parser.add_argument('--dataset', help='Directory of your Dataset',
                        default='./datasets')
    parser.add_argument('--cuda', help="'cuda' for cuda, 'cpu' for cpu, default = cuda",
                        default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--batch_size', help="batchsize, default = 5", default=5, type=int)
    parser.add_argument('--epoch', help='# of epochs. default = 30', default=30, type=int)
    parser.add_argument('-lr', '--learning_rate', help='learning_rate. default = 0.001', default=0.01, type=float)
    parser.add_argument('--lr_decay', help='Learning rate decrease by lr_decay time per decay_step, default = 0.1',
                        default=1, type=float)
    parser.add_argument('--decay_step', help='Learning rate decrease by lr_decay time per decay_step,  default = 7000',
                        default=7000, type=int)
    parser.add_argument('--display_freq', help='display_freq to display result image on Tensorboard',
                        default=1000, type=int)


    args = parser.parse_args()
    # TODO : Add multiGPU Model
    device = torch.device(args.cuda)
    batch_size = args.batch_size
    epoch = args.epoch
    train_dataset = PairDataset(args.dataset)
    load = args.load
    start_iter = 0
    model = Unet(cfg).cuda()
    vgg = torchvision.models.vgg16(pretrained=True)
    model.encoder.seq.load_state_dict(vgg.features.state_dict())
    now = datetime.datetime.now()
    start_epo = 0
    del vgg

    if load is not None:
        state_dict = torch.load(load, map_location=args.cuda)

        start_iter = int(load.split('epo_')[1].strip('step.ckpt')) 
        start_epo = int(load.split('/')[3].split('epo')[0])
        now = datetime.datetime.strptime(load.split('/')[2], '%m%d%H%M')

        print("Loading Model from {}".format(load))
        print("Start_iter : {}".format(start_iter))
        print("now : {}".format(now.strftime('%m%d%H%M')))
        model.load_state_dict(state_dict)
        for cell in model.decoder:
            if cell.mode == 'G':
                cell.picanet.renet.vertical.flatten_parameters()
                cell.picanet.renet.horizontal.flatten_parameters()
        print('Loading_Complete')

    # Optimizer Setup
    learning_rate = args.learning_rate
    lr_decay = args.lr_decay
    decay_step = args.decay_step  # from 50000 step
    learning_rate = learning_rate * (lr_decay ** (start_iter // decay_step))
    opt_en = torch.optim.SGD(model.encoder.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    opt_dec = torch.optim.SGD(model.decoder.parameters(), lr=learning_rate * 10, momentum=0.9, weight_decay=0.0005)
    # Dataloader Setup
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=0)
    
    # Logger Setup
    os.makedirs(os.path.join('log', now.strftime('%m%d%H%M')), exist_ok=True)
    weight_save_dir = os.path.join('models', 'state_dict', now.strftime('%m%d%H%M'))
    os.makedirs(os.path.join(weight_save_dir), exist_ok=True)
    writer = SummaryWriter(os.path.join('log', now.strftime('%m%d%H%M')))
    iterate = start_iter
    for epo in range(start_epo, epoch):
        print("\nEpoch : {}".format(epo))
        for i, batch in enumerate(tqdm(train_dataloader)):
            #if i > 10:
            #    break
            opt_dec.zero_grad()
            opt_en.zero_grad()
            img = batch['image'].to(device)
            mask = batch['mask'].to(device)
            pred, loss = model(img, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
            opt_dec.step()
            opt_en.step()
            writer.add_scalar('loss', float(loss), global_step=iterate)
            if iterate % args.display_freq == 0:
                writer.add_image('224', torchvision.utils.make_grid(pred[5]), global_step=iterate)
                writer.add_image('GT', torchvision.utils.make_grid(mask), iterate)
                writer.add_image('Image', torchvision.utils.make_grid(img), iterate)
               

            if iterate % 1000 == 0 and i != 0:
                torch.save(model.state_dict(), os.path.join(weight_save_dir, '{}epo_{:07d}step.ckpt'.format(iterate//(8791*5), iterate)))
            if iterate % 10000 == 0 and i != 0:
                for file in os.listdir(weight_save_dir):
                    remove_iter = int(file.split('_')[1][:-9])
                    if remove_iter % 10000 != 0:
                        os.remove(os.path.join(weight_save_dir, file))
                writer.close()
                exit()
            if i + epo * len(train_dataloader) % decay_step == 0 and i != 0:
                learning_rate *= lr_decay
                opt_en = torch.optim.SGD(model.encoder.parameters(), lr=learning_rate, momentum=0.9,
                                         weight_decay=0.0005)
                opt_dec = torch.optim.SGD(model.decoder.parameters(), lr=learning_rate * 10, momentum=0.9,
                                          weight_decay=0.0005)
            iterate += args.batch_size
            del loss
        start_iter = 0
