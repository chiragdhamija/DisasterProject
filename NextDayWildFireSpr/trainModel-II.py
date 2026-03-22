import os
from datetime import datetime
import argparse
import json
import torch.multiprocessing as mp
import torch
import torch.nn as nn
import torch.distributed as dist
from models import *
from datasets import *
import platform
from metrics import*
import copy
import numpy as np
from torch.utils.data import Dataset, IterableDataset, DataLoader
from milesial_unet_model import UNet, APAU_Net
from leejunhyun_unet_models import U_Net, R2U_Net, AttU_Net, R2AttU_Net, AttU_Net_S
import pickle

# Import custom loss and evaluation functions


TRAIN = 'train'
VAL = 'validation'
MASTER_RANK = 0
SAVE_INTERVAL = 1

# Canonical hazard-model dataset (original NDWS bands + location/time metadata).
DATASET_PATH = 'data/next-day-wildfire-spread-ca-hazard'
CHANNELS_METADATA_PATH = 'data/next-day-wildfire-spread-ca-hazard/channels_metadata.json'
SAVE_MODEL_PATH = 'savedModels'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--master', default='sardine',
                        help='master node')
    parser.add_argument('-p', '--port', default='30437',
                         help = 'master node')
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=10, type=int,
                        metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--dataset_path', default=DATASET_PATH,
                        help='Path to pickled dataset directory')
    parser.add_argument('--channels_metadata', default=CHANNELS_METADATA_PATH,
                        help='JSON file containing channel_names list')
    parser.add_argument('--learning_rate', default=1e-3, type=float,
                        help='Optimizer learning rate')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='L2 regularization strength for AdamW')
    parser.add_argument('--pos_weight', default=5.0, type=float,
                        help='Positive class weight for BCEWithLogitsLoss')
    parser.add_argument('--grad_clip', default=1.0, type=float,
                        help='Gradient clipping max norm')
    parser.add_argument('--early_stop_patience', default=3, type=int,
                        help='Stop training when validation F1 does not improve')
    parser.add_argument('--min_delta', default=1e-4, type=float,
                        help='Minimum F1 improvement to reset early stopping')
    parser.add_argument('--lr_factor', default=0.5, type=float,
                        help='ReduceLROnPlateau multiplicative factor')
    parser.add_argument('--lr_patience', default=1, type=int,
                        help='Number of bad epochs before LR reduction')
    parser.add_argument('--random_flip', action='store_true',
                        help='Enable random H/V flips in training augmentation')
    parser.add_argument('--rotation_factor', default=4, type=int,
                        help='How many rotation variants per sample (1,2,3,4)')
    parser.add_argument('--max_train_samples', default=0, type=int,
                        help='Cap training base samples before rotation (0 = all)')
    parser.add_argument('--max_val_samples', default=0, type=int,
                        help='Cap validation samples (0 = all)')
    parser.add_argument('--amp', action='store_true',
                        help='Enable mixed precision training on CUDA')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size for train/validation loaders')
    parser.add_argument('--num_workers', default=0, type=int,
                        help='DataLoader workers')
    args = parser.parse_args()
    print(f'initializing training on single GPU')
    train(0, args)

def _load_channel_names(channels_metadata, num_input_channels):
    fallback = [f"feature_{i}" for i in range(num_input_channels)]

    if channels_metadata is None:
        return fallback

    if not os.path.exists(channels_metadata):
        print(f"channels_metadata not found: {channels_metadata}; using generic channel names.")
        return fallback

    try:
        with open(channels_metadata, "r", encoding="utf-8") as f:
            payload = json.load(f)
        channel_names = payload.get("channel_names", [])
        if isinstance(channel_names, list) and len(channel_names) == num_input_channels:
            return channel_names
        print(
            "channels_metadata has mismatched channel_names length; "
            "using generic channel names."
        )
    except Exception as exc:
        print(f"failed to read channels_metadata ({channels_metadata}): {exc}")

    return fallback


def create_data_loaders(
    rank,
    gpu,
    world_size,
    dataset_path,
    channels_metadata=None,
    random_flip=False,
    rotation_factor=4,
    max_train_samples=0,
    max_val_samples=0,
    batch_size=64,
    num_workers=0,
):
    batch_size = max(1, int(batch_size))
    num_workers = max(0, int(num_workers))
    pin_memory = torch.cuda.is_available()

    datasets = {
        TRAIN: RotatedWildfireDataset(
            f"{dataset_path}/{TRAIN}.data",
            f"{dataset_path}/{TRAIN}.labels",
            features=None,
            crop_size=64,
            random_flip=random_flip,
            rotation_factor=rotation_factor,
            max_samples=max_train_samples,
        ),
        VAL: WildfireDataset(
            f"{dataset_path}/{VAL}.data",
            f"{dataset_path}/{VAL}.labels",
            features=None,
            crop_size=64,
            max_samples=max_val_samples,
        )
    }

    dataLoaders = {
        TRAIN: torch.utils.data.DataLoader(
            dataset=datasets[TRAIN],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        VAL: torch.utils.data.DataLoader(
            dataset=datasets[VAL],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    }

    num_input_channels = datasets[TRAIN].data.shape[1]
    channel_names = _load_channel_names(channels_metadata, num_input_channels)
    print(f"\nUsing all channels ({num_input_channels}):")
    print(list(zip(channel_names, range(num_input_channels))))
    print(f"dataset_path={dataset_path}")
    print(f"num_input_channels={num_input_channels}")
    print(f"batch_size={batch_size}, num_workers={num_workers}, pin_memory={pin_memory}")
    print(f"train_len={len(datasets[TRAIN])}, val_len={len(datasets[VAL])}")

    return dataLoaders, num_input_channels

def perform_validation(model, loader, device, use_amp=False):
    model.eval()

    total_loss = 0
    total_iou = 0
    total_accuracy = 0
    total_f1 = 0
    total_auc = 0
    valid_auc_count = 0
    total_dice = 0
    total_precision = 0
    total_recall = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(device, non_blocking=torch.cuda.is_available()).float()
            labels = labels.to(device, non_blocking=torch.cuda.is_available()).float()

             # Forward pass
            with torch.autocast(device_type="cuda", enabled=use_amp):
                outputs = model(images)

            labels = torch.flatten(labels)
            outputs = torch.flatten(outputs)

            
            # Compute metrics
            total_loss += loss(labels, outputs).item()
            total_iou += float(mean_iou(labels, outputs))
            total_accuracy += float(accuracy(labels, outputs))
            total_f1 += float(f1_score(labels, outputs))
            auc_val = auc_score(labels, outputs)
            if not np.isnan(auc_val):
                total_auc += float(auc_val)
                valid_auc_count += 1
            total_dice += float(dice_score(labels, outputs))
            
            precision, recall = precision_recall(labels, outputs)
            total_precision += float(precision)
            total_recall += float(recall)

    avg_loss = total_loss / len(loader)
    avg_iou = total_iou / len(loader)
    avg_accuracy = total_accuracy / len(loader)
    avg_f1 = total_f1 / len(loader)
    avg_auc = total_auc / valid_auc_count if valid_auc_count > 0 else float("nan")
    avg_dice = total_dice / len(loader)
    avg_precision = total_precision / len(loader)
    avg_recall = total_recall / len(loader)

    print(f"Validation - Loss: {avg_loss:.4f}, IoU: {avg_iou:.4f}, Accuracy: {avg_accuracy:.4f}")
    print(f"F1 Score: {avg_f1:.4f}, AUC: {avg_auc:.4f}, Dice: {avg_dice:.4f}")
    print(f"Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}")

    return avg_loss, avg_iou, avg_accuracy, avg_f1, avg_auc, avg_dice, avg_precision, avg_recall


def train(gpu, args):
    rank = args.nr * args.gpus + gpu
    validate = True
    print("Current GPU", gpu, "\n RANK: ", rank)

    dataLoaders, num_input_channels = create_data_loaders(
        rank,
        gpu,
        args.gpus * args.nodes,
        dataset_path=args.dataset_path,
        channels_metadata=args.channels_metadata,
        random_flip=args.random_flip,
        rotation_factor=args.rotation_factor,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    torch.manual_seed(0)

    model = U_Net(num_input_channels, 1)
    use_cuda = torch.cuda.is_available()
    use_amp = bool(args.amp and use_cuda)
    if use_cuda:
        torch.cuda.set_device(gpu)
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([args.pos_weight], dtype=torch.float32, device=device)
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.lr_factor,
        patience=args.lr_patience,
        threshold=args.min_delta,
        min_lr=1e-6,
    )
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start = datetime.now()
    print(f'TRAINING ON: {platform.node()}, Starting at: {datetime.now()}')
    print(f"random_flip={args.random_flip}")
    print(f"rotation_factor={max(1, min(4, int(args.rotation_factor)))}, max_train_samples={args.max_train_samples}, max_val_samples={args.max_val_samples}")
    print(f"amp={use_amp}")

    total_step = len(dataLoaders[TRAIN])
    best_epoch = 0
    best_f1_score = -float("inf")

    train_loss_history = []
    val_metrics_history = []
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        model.train()

        loss_train = 0

        for i, (images, labels) in enumerate(dataLoaders[TRAIN]):
            images = images.to(device, non_blocking=torch.cuda.is_available()).float()
            labels = labels.to(device, non_blocking=torch.cuda.is_available()).float()

            # Forward pass
            with torch.autocast(device_type="cuda", enabled=use_amp):
                outputs = model(images)


            # Not entirely sure if this flattening is required or not
            labels = torch.flatten(labels)
            outputs = torch.flatten(outputs)

            loss = criterion(outputs, labels)
            #loss = torchvision.ops.sigmoid_focal_loss(outputs, labels, alpha=0.85, gamma=2, reduction="mean")

            loss_train += loss.item()


            # Backward and optimize
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                optimizer.step()

            if i % 20 == 0:
                print('Epoch [{}/{}], Steps [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1,
                    args.epochs,
                    i,
                    total_step,
                    loss.item())
                )

        train_loss_history.append(loss_train / len(dataLoaders[TRAIN]))

        if validate:
            metrics = perform_validation(model, dataLoaders[VAL], device, use_amp=use_amp)
            val_metrics_history.append(metrics)

            curr_avg_loss_val, _, _, curr_f1_score, _, _, _, _ = metrics

            scheduler.step(curr_f1_score)
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch [{epoch + 1}/{args.epochs}] LR: {current_lr:.6g}")

            if curr_f1_score > (best_f1_score + args.min_delta):
                print("Saving model...")
                best_epoch = epoch
                best_f1_score = curr_f1_score
                filename = f'model-{model.__class__.__name__}-bestF1Score-Rank-{rank}.weights'
                torch.save(model.state_dict(), f'{SAVE_MODEL_PATH}/{filename}')
                print("Model has been saved!")
                epochs_without_improvement = 0
            else:
                print("Model is not being saved")
                epochs_without_improvement += 1

            if epochs_without_improvement >= args.early_stop_patience:
                print(
                    f"Early stopping triggered at epoch {epoch + 1} "
                    f"(no F1 improvement for {epochs_without_improvement} epoch(s))."
                )
                break

    pickle.dump(train_loss_history, open(f"{SAVE_MODEL_PATH}/train_loss_history.pkl", "wb"))
    pickle.dump(val_metrics_history, open(f"{SAVE_MODEL_PATH}/val_metrics_history.pkl", "wb"))

    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start))
        print(f"Endtime: {datetime.now()}")
        print(f"Best epoch: {best_epoch}")
        print(f"Best F1 score: {best_f1_score}")


if __name__ == '__main__':
    main()
