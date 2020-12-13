import math
import sys
import time
import torch
import transforms as T
import utils
from model import save_checkpoint, load_checkpoint
from dataloader import collate_fn, split_dataset, FLIRPseudoDataset, convert_subset
from torch.utils.data import DataLoader
import numpy as np
from coco_evaluate import coco_evaluate
import pickle

all_training_loss = []
all_evaluation_loss = []

def self_training(model, labeled_dataset, unlabeled_dataset, optimizer, scheduler=None, batch_size=4, train_ratio=0.7, score_threshold=0.7, unlabeled_loss_weight=0.1, relabel_step=None,
                  device='cpu', max_epochs=100, print_freq=10, save_path=None, checkpoint=None):
    model.to(device)
    metric_logger = utils.MetricLogger(delimiter=" ")
    last_loss = 1e9

    cur_epoch = 0
    # train_labeled_dataset, val_labeled_dataset = split_dataset(labeled_dataset, train_ratio)
    # train_unlabeled_dataset, val_unlabeled_dataset = split_dataset(unlabeled_dataset, train_ratio)
    dataset_path = os.path.join(save_path, 'dataset')

    if checkpoint is not None:
        print("loading checkpoint:" + checkpoint)
        model, optimizer, scheduler, cur_epoch = load_checkpoint(model, optimizer, scheduler, device, checkpoint)
    

    for epoch in range(cur_epoch, max_epochs):
        print("epoch {} / {}".format(epoch + 1, max_epochs))
        with open(os.path.join(dataset_path, 'train_labeled_dataset.pickle'), 'rb') as handle:
          train_labeled_dataset = pickle.load(handle)
        with open(os.path.join(dataset_path, 'val_labeled_dataset.pickle'), 'rb') as handle:
          val_labeled_dataset = pickle.load(handle)
        with open(os.path.join(dataset_path, 'train_unlabeled_dataset.pickle'), 'rb') as handle:
          train_unlabeled_dataset = pickle.load(handle)
        with open(os.path.join(dataset_path, 'val_unlabeled_dataset.pickle'), 'rb') as handle:
          val_unlabeled_dataset = pickle.load(handle)
          
        train_unlabeled_dataset = convert_subset(train_unlabeled_dataset)
        val_unlabeled_dataset = convert_subset(val_unlabeled_dataset)

        labeled_train_loader = DataLoader(train_labeled_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True)
        labeled_vld_loader= DataLoader(val_labeled_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=False)
        pseudo_train = FLIRPseudoDataset(model, train_unlabeled_dataset, batch_size=batch_size, device=device, score_threshold=score_threshold)
        pseudo_val = FLIRPseudoDataset(model, val_unlabeled_dataset, batch_size=batch_size, device=device, score_threshold=score_threshold)
        unlabeled_train_loader = DataLoader(pseudo_train, collate_fn=collate_fn, batch_size=batch_size, shuffle=True)
        unlabeled_vld_loader= DataLoader(pseudo_val, collate_fn=collate_fn, batch_size=batch_size, shuffle=False)
        
        train_label_loss = train_one_epoch_self_training(model, optimizer, labeled_train_loader, 1, device, epoch, print_freq)
        train_loss = train_one_epoch_self_training(model, optimizer, unlabeled_train_loader, unlabeled_loss_weight, device, epoch, print_freq)
        train_loss = train_label_loss + unlabeled_loss_weight * train_loss
        all_training_loss.append(train_loss)

        coco_evaluate(model, labeled_vld_loader, device)
        # labeled_loss = evaluate(model, vld_loader, device, epoch, print_freq)
        coco_evaluate(model, unlabeled_vld_loader, device)
        # unlabeled_loss = evaluate(model, vld_loader, device, epoch, print_freq)

        # loss = labeled_loss + unlabeled_loss_weight * unlabeled_loss
        loss = 0
        all_evaluation_loss.append(loss)

        if save_path is not None:
            save_checkpoint(model, optimizer, scheduler, epoch + 1, device, save_path)
            last_loss = loss
        print("epoch {}, train loss {}, validation loss {}".format(epoch + 1, train_loss, loss))
            
        if scheduler is not None:
            scheduler.step()
        # if relabel_step != None:
        #     if epoch % relabel_step == 0 and epoch != 0:
        #         pseudo_train = FLIRPseudoDataset(model, train_unlabeled_dataset, batch_size=batch_size, device=device, score_threshold=score_threshold)
        #         pseudo_val = FLIRPseudoDataset(model, val_unlabeled_dataset, batch_size=batch_size, device=device, score_threshold=score_threshold)
        #         unlabeled_train_loader = DataLoader(pseudo_train, collate_fn=collate_fn, batch_size=batch_size, shuffle=True)
        #         unlabeled_vld_loader= DataLoader(pseudo_val, collate_fn=collate_fn, batch_size=batch_size, shuffle=False)

def train_one_epoch_self_training(model, optimizer, data_loader, weight, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    all_loss = []

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        # for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values()) * weight
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()
        all_loss.append(losses.item())
        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        optimizer.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if device == 'cuda':
            torch.cuda.empty_cache()
            del images
            del targets
            del losses_reduced
            del losses
            del loss_dict
            del loss_dict_reduced
    return np.mean(all_loss)
