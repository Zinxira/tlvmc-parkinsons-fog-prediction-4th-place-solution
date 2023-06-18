import datetime
import logging
import os
import sys
import time
import json

from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm


def get_logger(model_path: str, log_fname: str, verbose=True):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_fpath = os.path.join(model_path, log_fname)
    formatter = logging.Formatter("%(message)s")

    # file handler
    f_handler = logging.FileHandler(log_fpath)
    f_handler.setFormatter(formatter)
    logger.addHandler(f_handler)

    # console handler
    if verbose:
        s_handler = logging.StreamHandler(sys.stdout)
        s_handler.setFormatter(formatter)
        logger.addHandler(s_handler)
    return logger


def close_logger(logger):
    handlers = logger.handlers[:]
    for handler in handlers:
        logger.removeHandler(handler)
        handler.close()


def plot_history(history, model_path=".", show=True):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Training Loss")
    plt.plot(epochs, history["valid_loss"], label="Validation Loss")
    plt.title("Loss evolution")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(model_path, "loss_evo.png"))
    if show:
        plt.show()
    plt.close()

    plt.figure()
    plt.plot(epochs, history["valid_mAP"])
    plt.title("Validation mAP evolution")
    plt.xlabel("Epochs")
    plt.ylabel("mAP")
    plt.savefig(os.path.join(model_path, "mAP_evo.png"))
    if show:
        plt.show()
    plt.close()

    plt.figure()
    plt.plot(epochs, history["lr"])
    plt.title("Learning Rate evolution")
    plt.xlabel("Epochs")
    plt.ylabel("LR")
    plt.savefig(os.path.join(model_path, "lr_evo.png"))
    if show:
        plt.show()
    plt.close()


def get_model_id(model, extra_info="", timestamp=""):
    model_id = f"{type(model).__name__}"
    if extra_info != "":
        model_id += f"_{extra_info}"
    if timestamp != "":
        model_id += f"_{timestamp}"
    else:
        now = datetime.datetime.now().strftime("%y%m%d%H%M%S")
        model_id += f"_{now}"
    return model_id


def evaluate(
    model: nn.Module, max_chunk_size: int, loader: DataLoader, device, criterion
):
    model.eval()
    valid_loss = 0.0
    y_true_full = torch.FloatTensor([])
    y_pred_full = torch.FloatTensor([])
    with torch.no_grad():
        for X_batch, y_batch, mask in tqdm(loader, desc="Eval", unit="batch"):
            # as some of the sequences we are dealing with are pretty long, we
            # use a chunk-based approach

            # (re)initialize model's hidden state(s)
            h = None

            # number of chunks for this sequence (we assume batch size = 1)
            seq_len = X_batch.shape[1]

            for i in range(0, seq_len, max_chunk_size):
                X_chunk = X_batch[:, i : i + max_chunk_size]
                y_chunk = y_batch[:, i : i + max_chunk_size]
                m_chunk = mask[:, i : i + max_chunk_size]

                X_chunk = X_chunk.to(device, non_blocking=True)
                y_chunk = torch.stack([y[m] for y, m in zip(y_chunk, m_chunk)])
                y_chunk = y_chunk.to(device, non_blocking=True)
                m_chunk = m_chunk.to(device, non_blocking=True)

                # sometimes the masking can result in an empty chunk
                if y_chunk.shape[1] == 0:
                    break

                # with autocast():  # mixed precision
                y_pred, h = model(X_chunk, h)

                # detaching the hidden states from the current chunk
                h = [hi.detach() for hi in h]

                y_pred = torch.stack([y[m] for y, m in zip(y_pred, m_chunk)])

                loss = criterion(
                    y_pred.reshape(-1, y_pred.shape[-1]),
                    y_chunk.reshape(-1, y_chunk.shape[-1]).argmax(dim=1),
                )

                # normalize the loss using the number of true labels in m_chunk
                loss = loss * (m_chunk.sum() / mask.sum())
                valid_loss += loss.item()

                y_true_full = torch.cat((y_true_full, y_chunk.cpu()), axis=1)
                y_pred_full = torch.cat((y_pred_full, y_pred.cpu()), axis=1)

        valid_loss /= len(loader)

        y_true_full = y_true_full.squeeze(0)
        y_pred_full = y_pred_full.squeeze(0)
        y_pred_full = torch.nn.functional.softmax(y_pred_full, dim=1)

        # we do not consider the "no-activity" class when evaluating metrics
        y_true_full = y_true_full[:, :3]
        y_pred_full = y_pred_full[:, :3]

        mAP_score = average_precision_score(y_true_full, y_pred_full)
    model.train()
    return valid_loss, mAP_score


def train(
    train_loader: DataLoader,
    valid_loader: DataLoader,
    model: nn.Module,
    max_chunk_size: int,
    nepochs: int,
    valid_inter: int,
    criterion,
    opt,
    scheduler,
    cos_epoch,
    device,
    model_path: str,
    verbose=True,
    log_fname="trace.log",
    metrics_evo_fname="metrics_evo.png",
) -> dict:
    logger = get_logger(model_path, log_fname, verbose)

    history = {
        "train_loss": [],
        "valid_loss": [],
        "valid_mAP": [],
        "lr": [],
    }

    best_valid_loss = np.inf

    scaler = GradScaler()

    model.train()
    dt = time.time()
    for epoch in range(1, nepochs + 1):
        train_loss = 0.0
        n_tot_chunks = 0

        for X_batch, y_batch, mask in tqdm(
            train_loader, desc="Training", unit="batch"
        ):
            # (re)initialize model's hidden state(s)
            h = None

            seq_len = X_batch.shape[1]
            for i in range(0, seq_len, max_chunk_size):
                X_chunk = X_batch[:, i : i + max_chunk_size]
                y_chunk = y_batch[:, i : i + max_chunk_size]
                m_chunk = mask[:, i : i + max_chunk_size]

                X_chunk = X_chunk.to(device, non_blocking=True)
                y_chunk = torch.stack([y[m] for y, m in zip(y_chunk, m_chunk)])
                y_chunk = y_chunk.to(device, non_blocking=True)
                m_chunk = m_chunk.to(device, non_blocking=True)

                # sometimes the masking can result in an empty chunk
                if y_chunk.shape[1] == 0:
                    break

                with autocast():  # mixed precision
                    y_pred, h = model(X_chunk, h)

                    # detaching the hidden states from the current chunk
                    h = [hi.detach() for hi in h]

                    y_pred = torch.stack(
                        [y[m] for y, m in zip(y_pred, m_chunk)]
                    )

                    loss = criterion(
                        y_pred.reshape(-1, y_pred.shape[-1]),
                        y_chunk.reshape(-1, y_chunk.shape[-1]).argmax(dim=1),
                    )

                    # normalize the loss using the number of true labels in m_chunk
                    loss = loss * (m_chunk.sum() / mask.sum())

                # accumulating gradients
                scaler.scale(loss).backward()

                train_loss += loss.item()

                if epoch > cos_epoch:
                    scheduler.step()

            # unscale gradients and clip them
            scaler.unscale_(opt)
            clip_grad_norm_(model.parameters(), max_norm=2.0)

            # optim step after all the chunks have been processed
            # ie update model's params once per sequence
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()

        train_loss /= len(train_loader)  # per seq

        if epoch % valid_inter == 0:
            valid_loss, valid_mAP = evaluate(
                model, max_chunk_size, valid_loader, device, criterion
            )

            history["train_loss"].append(train_loss)
            history["valid_loss"].append(valid_loss)
            history["valid_mAP"].append(valid_mAP)
            history["lr"].append(opt.param_groups[0]["lr"])

            # if the current validation loss is the best one so far
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss

                # save the model
                torch.save(
                    model.state_dict(),
                    os.path.join(model_path, "model.pth"),
                )

            # save the evolution of the metrics over time
            plot_history(history, model_path=model_path, show=False)

            dt = time.time() - dt
            logger.info(
                f"{epoch}/{nepochs} -- "
                f"train_loss = {train_loss:.6f} -- "
                f"valid_loss = {valid_loss:.6f} -- "
                f"valid_mAP = {valid_mAP:.6f} -- "
                f"time = {dt:.6f}s"
            )
            dt = time.time()

    # save and plot the evolution of the metrics over time
    plot_history(history, model_path=model_path)

    history_path = os.path.join(model_path, "history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=4)

    close_logger(logger)

    return history
