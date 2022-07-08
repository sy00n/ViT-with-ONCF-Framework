import os
import re
import sys
import glob
import json
import shutil
from pathlib import Path
from collections import namedtuple
from importlib import import_module
from turtle import pos

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from utils import * 

from tqdm import tqdm
from datetime import datetime
import numpy as np; np.set_printoptions(threshold=np.inf, linewidth=np.inf)

cur_date = datetime.today().strftime("%y%m%d")

def increment_path(path, exist_ok=False):
    """
    Automatically increment path
    
    Args:
        path (str or pathlib.Path): f"{saved_dir}/{run_name}"
        exist_ok (bool): whether to increment path (increment path if False)
    """

    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs] 
        # path.stem은 그 path에서 파일 이름에서 확장자빼고 가져옴.
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def save_checkpoint(epoch, model, hr, ndcg, optimizer, saved_dir, scheduler, file_name):
    """Save checkpoint in save_dir

    Args:
        epoch (int): number of epoch
        model (torch model): torch model to save
        hr (float): hr to save
        ndcg (float) : ndcg to save
        optimizer (torch optimizer): optimizer to save
        saved_dir (str): path to save the file
        scheduler (torch scheduler): scheduler to save
        file_name (str): file name
    """
    check_point = {'epoch': epoch,
                   'model': model.state_dict(),
                   'optimizer_state_dict': optimizer.state_dict(),
                   'hr': hr,
                   'ndcg' : ndcg}
    if scheduler:
        check_point['scheduler_state_dict'] = scheduler.state_dict()
    output_path = os.path.join(saved_dir, file_name)
    torch.save(check_point, output_path)


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, mode):
    """Load checkpoint if resume_from is set

    Args:
        checkpoint_path (str): path to load checkpoint
        model (torch model): torch model to load
        optimizer (torch optimizer): optimizer to load
        scheduler (torch scheduler): scheduler to load
        mode (str or None): If all, the optimizer and scheduler are loaded

    Returns:
        state loaded model, optimizer, scheduler, etc
    """
    # load model if resume_from is set
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    if mode =="all":
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    start_hr = checkpoint['hr']
    start_ndcg = checkpoint['ndcg']

    return model, optimizer, scheduler, start_epoch, start_hr, start_ndcg


def train(num_epochs, model, train_loader, val_loader, criterion, optimizer, top_k,
          saved_dir, val_every, save_mode, resume_from, resume_mode, checkpoint_path, 
          num_to_remain, device, scheduler = None, fp16 = False):

    print(f'Start training..')
    start_epoch = 0
    best_hr = 0
    best_ndcg = 0

    num_to_remain = 3 # remain 3 files

    if resume_from:
        model, optimizer, scheduler, start_epoch, best_hr, best_ndcg = load_checkpoint(checkpoint_path, model, optimizer, scheduler, resume_mode)
    
    if fp16:
        print("Mixed precision is applied")
        scaler = GradScaler()

    for epoch in range(start_epoch, num_epochs):
        model.train()

        running_loss = 0
        sum_loss = 0

        pbar = tqdm(enumerate(train_loader), total = len(train_loader))
        train_loader.dataset.ng_sample()
        for step, input in pbar:
            user = input['user_id'].to(device)
            if model.model_name == 'ONCF':
                pos_item = input['item_id'].to(device)
                neg_item = input['neg_item'].to(device)
            else:
                item = input['item_id'].to(device)
            label = input['target_main'].to(device)
            
            user_aux = input['target_user_aux'].to(device)
            item_aux = input['target_item_aux'].to(device)
            
            optimizer.zero_grad()
            if fp16:
                if model.model_name == 'ONCF':
                    with autocast():
                        if epoch < 3:
                            pos_preds = model(user, pos_item, True)
                            neg_preds = model(user, neg_item, True)
                        else:
                            pos_preds = model(user, pos_item, False)
                            neg_preds = model(user, neg_item, False)
                        loss = criterion(pos_preds, neg_preds)
                else:
                    with autocast():
                        outputs = model(user, item)
                        loss = criterion(outputs['main'], outputs['user'], outputs['item'],
                                        label, user_aux, item_aux)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                if model.model_name == 'ONCF':
                    if epoch < 3:
                        pos_preds = model(user, pos_item, True)
                        neg_preds = model(user, neg_item, True)
                    else:
                        pos_preds = model(user, pos_item, False)
                        neg_preds = model(user, neg_item, False)
                    loss = criterion(pos_preds, neg_preds)
                else:
                    outputs = model(user, item)
                    loss = criterion(outputs['main'], outputs['user'], outputs['item'],
                                 label, user_aux, item_aux)
                loss.backward()
                optimizer.step()

            sum_loss += loss.item()
            running_loss = sum_loss / (step + 1)
          
            description =  f'Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(train_loader)}]: ' 
            description += f'running Loss: {round(running_loss,4)}'
            pbar.set_description(description)
             
        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % val_every == 0:
            hr, ndcg = validation(epoch+1, num_epochs, model, val_loader, top_k, device)
            
            if save_mode == 'hr':
                if hr > best_hr:
                    print(f"Best performance at epoch: {epoch + 1}")
                    print(f"Save model in {saved_dir}")
                    best_hr = hr
                    save_checkpoint(epoch, model, best_hr, best_ndcg, optimizer, saved_dir, scheduler, file_name=f"{model.model_name}_{round(best_hr,3)}_{cur_date}.pt")
            elif save_mode == 'ndcg':
                if ndcg > best_ndcg:
                    print(f"Best performance at epoch: {epoch + 1}")
                    print(f"Save model in {saved_dir}")
                    best_hr = hr
                    save_checkpoint(epoch, model, best_hr, best_ndcg, optimizer, saved_dir, scheduler, file_name=f"{model.model_name}_{round(best_ndcg,3)}_{cur_date}.pt")

            if len(os.listdir(saved_dir)) > num_to_remain:
                remove_old_files(saved_dir, thres=num_to_remain)
            
            # lr 조정
            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if save_mode == 'hr':
                        scheduler.step(hr)
                    if save_mode == 'ndcg':
                        scheduler.step(ndcg)
                else:
                    scheduler.step()


def validation(epoch, num_epochs, model, data_loader, top_k, device):
    model.eval()

    with torch.no_grad():

        HR = []
        NDCG = []
    
        for step, input in tqdm(enumerate(data_loader), total = len(data_loader)):
            user = input['user_id'].to(device)
            item = input['item_id'].to(device)
            
            #user_aux = input['target_user_aux'].to(device)
            #item_aux = input['target_item_aux'].to(device)
            
            if model.model_name == 'ONCF':
                outputs = model(user, item, False)
                predictions = outputs.view(-1)
            else:
                outputs = model(user, item)
                predictions = outputs['main'].view(-1)

            _, indices = torch.topk(predictions, top_k)
            recommends = torch.take(item, indices).cpu().numpy().tolist()

            gt_item = item[0].item()
            HR.append(hit(gt_item, recommends))
            NDCG.append(ndcg(gt_item, recommends))
        
        print(f"Epoch [{epoch + 1} / {num_epochs}], HR: {np.mean(HR)}, NDCG: {np.mean(NDCG)}")
    return np.mean(HR), np.mean(NDCG)


def main():
    args = arg_parse()

    with open(args.cfg, 'r') as f:
        cfgs = json.load(f, object_hook=lambda d: namedtuple('x', d.keys())(*d.values()))

    # fix seed
    fix_seed(cfgs.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataset & data loader
    train_dataset_module = getattr(import_module("data_utils"), cfgs.train_dataset.name)
    train_dataset = train_dataset_module(**cfgs.train_dataset.args._asdict())
    train_dataloader = DataLoader(train_dataset, **cfgs.train_dataloader.args._asdict())
    
    val_dataset_module = getattr(import_module("data_utils"), cfgs.val_dataset.name)
    val_dataset = val_dataset_module(**cfgs.val_dataset.args._asdict())
    val_dataloader = DataLoader(val_dataset, **cfgs.val_dataloader.args._asdict())

    # model
    model_module = getattr(import_module("model"), cfgs.model.name)
    model = model_module(**cfgs.model.args._asdict()).to(device)

    # criterion
    if hasattr(import_module("criterions"), cfgs.criterion.name):
        criterion_module = getattr(import_module("criterions"), cfgs.criterion.name)
    else:
        criterion_module = getattr(import_module("torch.nn"), cfgs.criterion.name)
    criterion = criterion_module(**cfgs.criterion.args._asdict())

    # optimizer
    optimizer_module = getattr(import_module("torch.optim"), cfgs.optimizer.name)
    optimizer = optimizer_module(model.parameters(), **cfgs.optimizer.args._asdict())

    # scheduler
    try:
        if hasattr(import_module("scheduler"), cfgs.scheduler.name):
            scheduler_module = getattr(import_module("scheduler"), cfgs.scheduler.name)
            scheduler = scheduler_module(optimizer, **cfgs.scheduler.args._asdict())
        else:
            scheduler_module = getattr(import_module("torch.optim.lr_scheduler"), cfgs.scheduler.name)
            scheduler = scheduler_module(optimizer, **cfgs.scheduler.args._asdict())
    except AttributeError :
            print('There is no Scheduler!')
            scheduler = None
    
    # get a path to save checkpoints and config
    saved_dir = increment_path(f"{cfgs.saved_dir}/{cfgs.run_name}")
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)

    # save a config.json before training
    shutil.copy(args.cfg, f"{saved_dir}/config.json")

    # call train
    train_args = {
        'num_epochs': cfgs.num_epochs, 
        'model': model, 
        'train_loader': train_dataloader, 
        'val_loader': val_dataloader, 
        'criterion': criterion, 
        'optimizer': optimizer, 
        'saved_dir': saved_dir, 
        'val_every': cfgs.val_every, 
        'save_mode': cfgs.save_mode, 
        'resume_from': cfgs.resume_from,
        'resume_mode': cfgs.resume_mode, 
        'checkpoint_path': cfgs.checkpoint_path,
        'num_to_remain': cfgs.num_to_remain,
        'device': device,
        'scheduler': scheduler,
        'top_k' : cfgs.top_k,
        'fp16': cfgs.fp16
    }

    train(**train_args)

if __name__ == "__main__":
    main()