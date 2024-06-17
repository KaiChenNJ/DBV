import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from monai.losses.dice import DiceCELoss
from monai.networks.utils import one_hot

from buffer.base_dataset import base_Dataset
from buffer.tiny_buffer import save_buffer_Dataset, buffer_Dataset
from model.UNet import Unet
from utils.metric import dice_coeff_1label
from utils.tools import get_logger, _set_random, seed_worker


def create_directories():
    if not os.path.exists('logs/random_buffer/'):
        os.makedirs('logs/random_buffer/')
    if not os.path.exists('checkpoints/random_buffer/'):
        os.makedirs('checkpoints/random_buffer/')


def initialize_model(device):
    model = Unet()
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    return model


def save_best_model(model, s, best_metric, now_metric):
    if now_metric > best_metric:
        best_metric = now_metric
        torch.save(model.state_dict(), f"checkpoints/random_buffer/{s}_rb_best_model.pth")
    return best_metric


def log_epoch_results(train_log, epoch, epoch_loss, train_dice, test_dice, best_metric, buffer_dice=None):
    log_message = (
        f'--epoch:{epoch} --train_Dice_CE_loss:{round(epoch_loss, 6)} --train Mean Dice:{round(np.mean(train_dice), 6)}'
        f' --test Mean Dice:{round(np.mean(test_dice), 6)} --best_dice:{best_metric}')

    if buffer_dice is not None:
        log_message += f" --Buffer Dice:{round(np.mean(buffer_dice), 6)}"

    train_log.write('\n' + '*********************' + '\n')
    train_log.write(log_message + '\n')
    print(log_message)


def train_first_site(model, s, trainloader, testloader, epochs, device, root, i_list, t_list):
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    loss_fn = DiceCELoss(reduction="mean")
    best_metric = -1
    train_log = open(f'logs/random_buffer/{s}_random_buffer.log', 'w')

    for epoch in range(epochs):
        model.train()
        running_loss, train_dice = 0, []
        for x, y, img_buffer, lab_buffer in tqdm(trainloader):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            y_one_hot = one_hot(y[:, None, ...], num_classes=2)
            loss = loss_fn(y_pred, y_one_hot).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            save_buffer_Dataset(root='./replay_buffer_data', sitename=s, finish=False, max_memory_size=128,
                                total_domain=6, img_path=img_buffer, target_path=lab_buffer, img_list=i_list,
                                target_list=t_list).pipeline()

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
            for x, y, _, _ in tqdm(testloader):
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                y_pred = torch.argmax(y_pred, dim=1)[:, None, ...]
                test_Dice = dice_coeff_1label(y_pred.cpu().numpy(), y[:, None, ...])
                test_dice.append(test_Dice)

        now_metric = round(np.mean(test_dice), 6)
        best_metric = save_best_model(model, s, best_metric, now_metric)

        log_epoch_results(train_log, epoch, epoch_loss, train_dice, test_dice, best_metric)

    train_log.close()


def train_other_sites(model, s, trainloader, testloader, bufferloader, epochs, device, root, i_list, t_list):
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    loss_fn = DiceCELoss(reduction="mean")
    best_metric = -1
    train_log = open(f'logs/random_buffer/{s}_random_buffer.log', 'w')

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        train_dice, buffer_dice = [], []

        for x, y, img_buffer, lab_buffer in tqdm(trainloader):
            x, y = x.to(device), y.to(device)
            x_old, y_old, _, _ = next(iter(bufferloader))
            x_old, y_old = x_old.to(device), y_old.to(device)

            y_pred = model(x)
            y_old_pred = model(x_old)

            y_one_hot = one_hot(y[:, None, ...], num_classes=2)
            y_old_one_hot = one_hot(y_old[:, None, ...], num_classes=2)
            loss1 = loss_fn(y_pred, y_one_hot).mean()
            loss2 = loss_fn(y_old_pred, y_old_one_hot).mean()
            loss = loss1 + loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            save_buffer_Dataset(root='./replay_buffer_data', sitename=s, finish=False, max_memory_size=128,
                                total_domain=6, img_path=img_buffer, target_path=lab_buffer, img_list=i_list,
                                target_list=t_list).pipeline()

            with torch.no_grad():
                running_loss += loss.item()
                y_pred = torch.argmax(y_pred, dim=1)[:, None, ...]
                Dice1 = dice_coeff_1label(y_pred.cpu().numpy(), y[:, None, ...])
                train_dice.append(Dice1)

                y_old_pred = torch.argmax(y_old_pred, dim=1)[:, None, ...]
                Dice2 = dice_coeff_1label(y_old_pred.cpu().numpy(), y_old[:, None, ...])
                buffer_dice.append(Dice2)

        scheduler.step()
        epoch_loss = round(running_loss / len(trainloader), 6)

        model.eval()
        test_dice = []
        with torch.no_grad():
            for x, y, _, _ in tqdm(testloader):
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                y_pred = torch.argmax(y_pred, dim=1)[:, None, ...]
                test_Dice = dice_coeff_1label(y_pred.cpu().numpy(), y[:, None, ...])
                test_dice.append(test_Dice)

        now_metric = round(np.mean(test_dice), 6)
        best_metric = save_best_model(model, s, best_metric, now_metric)

        log_epoch_results(train_log, epoch, epoch_loss, train_dice, test_dice, best_metric, buffer_dice)

    train_log.close()


def train(root, site, epochs):
    device = torch.device('cuda')
    model = initialize_model(device)
    create_directories()

    i_list, t_list = [], []

    for idx, s in enumerate(site):
        print(f'\r\r====================== {s} {"First" if idx == 0 else "Not First"} =======================')
        train_Dataset = base_Dataset(root, s, train=True, ratio=0.8)
        test_Dataset = base_Dataset(root, s, train=False, ratio=0.8)
        trainloader = DataLoader(dataset=train_Dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        testloader = DataLoader(dataset=test_Dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        if idx == 0:
            train_first_site(model, s, trainloader, testloader, epochs, device, root, i_list, t_list)
        else:
            buffer_set = buffer_Dataset(root='./replay_buffer_data', site_list=site, cur_site=s, Augmentation=True)
            bufferloader = DataLoader(dataset=buffer_set, batch_size=batch_size, shuffle=True, drop_last=True)
            train_other_sites(model, s, trainloader, testloader, bufferloader, epochs, device, root, i_list, t_list)


if __name__ == '__main__':
    _set_random(2023)
    ROOT = '/data/ck/continual_seg/processed'
    site_name = ["RUNMC", "BMC", "I2CVB", "UCL", "BIDMC", "HK"]
    batch_size = 8
    Epochs = 300
    train(ROOT, site_name, Epochs)
