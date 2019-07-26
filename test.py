from __future__ import division
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
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    print("Default path : " + os.getcwd())
    parser.add_argument("--load",
                        help="Directory of pre-trained model, you can download at \n"
                             "https://drive.google.com/file/d/109a0hLftRZ5at5hwpteRfO1A6xLzf8Na/view?usp=sharing\n"
                             "None --> Do not use pre-trained model. Training will start from random initialized model\n"
                             "default = None",
                        default=None)
    parser.add_argument('--dataset', help='Directory of your Dataset',
                        default='./datasets/DUTS_TEST')
    parser.add_argument('--cuda', help="'cuda' for cuda, 'cpu' for cpu, default = cuda",
                        default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--batch_size', help="batchsize, default = 8", default=8, type=int)


    args = parser.parse_args()
    # TODO : Add multiGPU Model
    device = torch.device(args.cuda)
    batch_size = args.batch_size
    test_dataset = PairDataset(args.dataset, train=False, data_augmentation=False)
    load = args.load
    model = Unet(cfg).to(device)

    assert load is not None
    state_dict = torch.load(load, map_location=device)

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

    model.eval()

    # Dataloader Setup
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=0)
    
    # Logger Setup
    os.makedirs(os.path.join('log', now.strftime('%m%d%H%M')+'_test'), exist_ok=True)
    writer = SummaryWriter(os.path.join('log', now.strftime('%m%d%H%M')+'_test'))
    model_iter = start_iter
    
    display_image = True
    loss = 0
    mae = 0
    preds = []
    masks = []
    precs = []
    recalls = []
    for i, batch in enumerate(tqdm(test_dataloader)):
        #if i > 10:
        #    break
        img_ = batch['image'].to(device)
        mask_ = batch['mask'].to(device)
        with torch.no_grad():
            predict, loss_batch = model(img_, mask_)
        loss += loss_batch.detach()
        pred = predict[5].data
        mae += torch.mean(torch.abs(pred - mask_))
        pred = pred.requires_grad_(False)
        preds.append(pred.cpu())
        masks.append(mask_.cpu())
        prec, recall = torch.zeros(mask_.shape[0], 256), torch.zeros(mask_.shape[0], 256)
        pred = pred.squeeze(dim=1).cpu()
        mask = mask_.squeeze(dim=1).cpu()
        thlist = torch.linspace(0, 1 - 1e-10, 256)
        for j in range(256):
            y_temp = (pred >= thlist[j]).float()
            tp = (y_temp * mask).sum(dim=-1).sum(dim=-1)
            # avoid prec becomes 0
            prec[:, j], recall[:, j] = (tp + 1e-10) / (y_temp.sum(dim=-1).sum(dim=-1) + 1e-10), (tp + 1e-10) / (mask.sum(dim=-1).sum(dim=-1) + 1e-10)
        # (batch, threshold)
        precs.append(prec)
        recalls.append(recall)

        if display_image == True:
            if np.random.random() < 0.01:
            	writer.add_image('224', torchvision.utils.make_grid(predict[5]), global_step=model_iter)
            	writer.add_image('GT', torchvision.utils.make_grid(mask_),model_iter)
            	writer.add_image('Image', torchvision.utils.make_grid(img_),model_iter)
            	display_image = False
    
    writer.add_scalar('loss', loss/i, global_step=model_iter)
    prec = torch.cat(precs, dim=0).mean(dim=0)
    recall = torch.cat(recalls, dim=0).mean(dim=0)
    beta_square = 0.3
    f_score = (1 + beta_square) * prec * recall / (beta_square * prec + recall)
    thlist = torch.linspace(0, 1 - 1e-10, 256)
    writer.add_scalar("Max F_score", torch.max(f_score), global_step=model_iter)
    writer.add_scalar("Max_F_threshold", thlist[torch.argmax(f_score)], global_step=model_iter)
    pred = torch.cat(preds, 0)
    mask = torch.cat(masks, 0).round().float()
    writer.add_pr_curve('PR_curve', mask, pred, global_step=model_iter)
    writer.add_scalar('MAE', torch.mean(torch.abs(pred - mask)), global_step=model_iter)
    writer.close()

