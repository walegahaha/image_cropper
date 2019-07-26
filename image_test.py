from network import Unet
from dataset import CustomDataset
import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse

torch.set_printoptions(profile='full')
if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    print("Default path : " + os.getcwd())
    parser.add_argument("--model_dir",
                        help="Directory of pre-trained model, you can download at \n"
                             "https://drive.google.com/drive/folders/1s4M-_SnCPMj_2rsMkSy3pLnLQcgRakAe?usp=sharing",
                        default="models/36epo_383000step.ckpt")
    parser.add_argument('--dataset', help='Directory of your test_image ""folder""',
                        default='test_images')
    parser.add_argument('--cuda', help="cuda for cuda, cpu for cpu, default = cuda", default='cuda')
    parser.add_argument('--batch_size', help="batchsize, default = 4", default=4, type=int)
    parser.add_argument('--logdir', help="logdir, log on tensorboard", default=None)
    parser.add_argument('--save_dir', help="save result images as .jpg file. If None -> Not save",
                        default='result')

    args = parser.parse_args()

    if args.logdir is None and args.save_dir is None:
        print("You should specify either --logdir or --save_dir to save results!")
        assert 0

    print(args)
    print(os.getcwd())
    device = torch.device(args.cuda)
    state_dict = torch.load(args.model_dir, map_location=args.cuda)
    model = Unet().to(device)
    model.load_state_dict(state_dict)
    custom_dataset = CustomDataset(root_dir=args.dataset)
    dataloader = DataLoader(custom_dataset, args.batch_size, shuffle=False)
    os.makedirs(os.path.join(args.save_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'masks'), exist_ok=True)
    if args.logdir is not None:
        writer = SummaryWriter(args.logdir)
    model.eval()
    for i, (batch, batchnames) in enumerate(tqdm(dataloader)):
        img = batch.to(device)
        with torch.no_grad():
            pred, loss = model(img)
        pred = pred[5].data.cpu()
        pred.requires_grad_(False)
        if args.logdir is not None:
            writer.add_image(args.model_dir + ', img', img, i)
            writer.add_image(args.model_dir + ', mask', pred, i)
        if args.save_dir is not None:
            for j in range(img.shape[0]):
                torchvision.utils.save_image(img[j], os.path.join(args.save_dir, 'images', '{}'.format(batchnames[j])))
                torchvision.utils.save_image(pred[j], os.path.join(args.save_dir, 'masks', '{}'.format(batchnames[j])))
    if args.logdir is not None:
        writer.close()
