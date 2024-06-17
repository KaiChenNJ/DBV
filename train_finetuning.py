import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from monai.losses.dice import DiceCELoss
from monai.networks.utils import one_hot

from dataset.finetuning_dataset import finetuning_Dataset
from model.UNet import Unet
from utils.metric import dice_coeff_1label
from utils.tools import get_logger, _set_random, seed_worker


def create_directories():
    if not os.path.exists('logs/'):
        os.makedirs('logs/')
    if not os.path.exists('checkpoints/'):
        os.makedirs('checkpoints/')


def initialize_model(device):
    model = Unet()
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    return model


def save_best_model(model, s, best_metric, now_metric):
    if now_metric > best_metric:
        best_metric = now_metric
        torch.save(model.state_dict(), f"checkpoints/{s}_best_model.pth")
    return best_metric


def log_epoch_results(train_log, epoch, epoch_loss, train_dice, test_dice, best_metric):
    log_message = (
        f'--epoch:{epoch} --train_Dice_CE_loss:{round(epoch_loss, 6)} --train Mean Dice:{round(np.mean(train_dice), 6)}'
        f' --test Mean Dice:{round(np.mean(test_dice), 6)} --best_dice:{best_metric}')
    train_log.write('\n' + '*********************' + '\n')
    train_log.write(log_message + '\n')
    print(log_message)


def train_single_site(model, s, trainloader, testloader, epochs, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    loss_fn = DiceCELoss(reduction="mean")
    best_metric = -1
    train_log = open(f'logs/{s}.log', 'w')

    for epoch in range(epochs):
        model.train()
        running_loss, train_dice = 0, []

        for x, y in tqdm(trainloader):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            y_one_hot = one_hot(y[:, None, ...], num_classes=2)
            loss = loss_fn(y_pred, y_one_hot).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                running_loss += loss.item()
                y_pred = torch.argmax(y_pred, dim=1)[:, None, ...]
                Dice = dice_coeff_1label(y_pred.cpu().numpy(), y[:, None, ...])
                train_dice.append(Dice)

        scheduler.step()
        epoch_loss = round(running_loss / len(trainloader), 6)

        model.eval()
        test_dice = []
        with torch.no_grad():
            for x, y in tqdm(testloader):
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                y_pred = torch.argmax(y_pred, dim=1)[:, None, ...]
                test_Dice = dice_coeff_1label(y_pred.cpu().numpy(), y[:, None, ...])
                test_dice.append(test_Dice)

        now_metric = round(np.mean(test_dice), 6)
        best_metric = save_best_model(model, s, best_metric, now_metric)

        log_epoch_results(train_log, epoch, epoch_loss, train_dice, test_dice, best_metric)

    train_log.close()


def train(root, site, epochs):
    device = torch.device('cuda')
    model = initialize_model(device)
    create_directories()

    for s in site:
        print(f'\r\r====================== {s} =======================')
        train_Dataset = finetuning_Dataset(root, s, train=True, ratio=0.8)
        test_Dataset = finetuning_Dataset(root, s, train=False, ratio=0.8)
        trainloader = DataLoader(dataset=train_Dataset, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(dataset=test_Dataset, batch_size=batch_size, shuffle=True)

        train_single_site(model, s, trainloader, testloader, epochs, device)


if __name__ == '__main__':
    seed = 2023
    _set_random(seed)
    ROOT = '/data/ck/continual_seg/processed'
    site_name = ["RUNMC", "BMC", "I2CVB", "UCL", "BIDMC", "HK"]
    batch_size = 8
    Epochs = 300
    train(ROOT, site_name, Epochs)
