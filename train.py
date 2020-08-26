from comet_ml import Experiment
import sys
import os
import pickle
from collections import Counter
from tqdm import tqdm
import argparse

import numpy as np
import torch
from torch import nn
from torch import optim

from dataset import get_dataloader
from utils import load_pretrained_embedings
from model import MACNetwork
from config import get_default_cfg, config_to_comet

ON_SLURM = "SLURM_JOBID" in os.environ

device = None

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def train(epoch, train_dataloader):
    dataloader = iter(train_dataloader)
    if not ON_SLURM:
        pbar = tqdm(dataloader)
    else:
        pbar = dataloader
    moving_accuracy = None
    moving_steps = None

    net.train(True)
    for image, question, q_len, answer, _, _ in pbar:
        image, question, answer = (
            image.to(device),
            question.to(device),
            answer.to(device),
        )

        net.zero_grad()
        output, ponder_cost, steps = net(image, question, q_len)
        if cfg.MAC.USE_ACT:
            loss = criterion(output, answer) + \
                (cfg.ACT.PENALTY_COEF * torch.mean(ponder_cost))
        else:
            loss = criterion(output, answer)

        loss.backward()

        if cfg.SOLVER.GRAD_CLIP:
            nn.utils.clip_grad_norm_(net.parameters(), cfg.SOLVER.GRAD_CLIP)
        optimizer.step()

        # train metrics
        correct = output.detach().argmax(1) == answer
        accuracy = correct.float().mean().item()
        steps = torch.mean(steps).item()
        if moving_accuracy is None:
            moving_accuracy = accuracy
            moving_steps = steps
        else:
            moving_accuracy = moving_accuracy * 0.99 + accuracy * 0.01
            moving_steps = moving_steps * 0.99 + steps * 0.01

        if not ON_SLURM:
            pbar.set_description(
                'Epoch: {}; Loss: {:.5f}; Acc: {:.5f}'.format(
                    epoch, loss.item(), moving_accuracy
                )
            )

        accumulate(net_running, net)

    experiment.log_metrics({
        "train_precision": moving_accuracy,
        "train_steps": moving_steps,
        })


def valid(epoch, val_dataloader):
    dataloader = iter(val_dataloader)
    if not ON_SLURM:
        pbar = tqdm(dataloader)
    else:
        pbar = dataloader

    net_running.train(False)
    with torch.no_grad():
        all_corrects = 0
        acc_steps = 0

        for i, (image, question, q_len, answer, _, _) in enumerate(pbar):
            image, question = image.to(device), question.to(device)

            output, ponder_cost, steps  = net_running(image, question, q_len)
            correct = output.detach().argmax(1) == answer.to(device)

            all_corrects += correct.float().mean().item()
            acc_steps += torch.mean(steps).item()

            if not ON_SLURM:
                pbar.set_description(
                    'Val Epoch: {}; Acc: {:.5f}; Steps: {:.2f}'.format(
                        epoch, all_corrects / (i + 1), acc_steps / (i + 1)
                    )
                )

        experiment.log_metrics({
                "valid_precision": all_corrects / len(dataloader),
                "valid_steps": acc_steps / len(dataloader),
                })

        experiment.log_epoch_end(epoch)

        if scheduler:
            scheduler.step(all_corrects / len(dataloader))


def generate_cfg():
    parser = argparse.ArgumentParser(description="Train for DACT-Mac")
    parser.add_argument(
        "--mode",
        default="clevr",
        choices=["clevr", "gqa"],
        help="whether to use clevr or gqa default parameters",
    )
    parser.add_argument(
        "--config-file",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    # get default configuration params
    cfg = get_default_cfg(mode=args.mode)

    # update config with configuration file
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    return cfg, args.mode


if __name__ == '__main__':
    cfg, mode = generate_cfg()
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    print("using {}".format(device))

    experiment = Experiment(
        api_key = os.environ["COMET_KEY"],
        project_name=cfg.COMET.PROJECT_NAME,
        workspace = os.environ["COMET_USER"],
    )

    experiment.log_parameters(config_to_comet(cfg))
    experiment.add_tag("NEW_MAC")
    if cfg.COMET.EXPERIMENT_NAME:
        experiment.set_name(cfg.COMET.EXPERIMENT_NAME)

    net = MACNetwork(cfg).to(device)
    if cfg.MAC.TRAINED_EMBD_PATH:
        load_pretrained_embedings(cfg, net.embed, device)
    net_running = MACNetwork(cfg).to(device)

    if cfg.LOAD:
        with open(cfg.LOAD_PATH, 'rb') as f:
            state = torch.load(f, map_location=device)
        net.load_state_dict(state, strict=False)
    accumulate(net_running, net, 0)

    if cfg.MAC.USE_ACT and cfg.ACT.SMOOTH:
        _criterion = nn.NLLLoss()
        criterion = lambda x, y: _criterion(torch.log(x.clamp(min=1e-8)), y)
    else:
        criterion = nn.CrossEntropyLoss()

    if cfg.SOLVER.GATE_LR:
        filtered = ['actmac.gate.weight', 'actmac.gate.bias']
        gate_params = map(lambda kv: kv[1], (filter(lambda kv: kv[0] in filtered,
                    net.named_parameters())))
        base_params = map(lambda kv: kv[1], (filter(lambda kv: kv[0] not in filtered,
                    net.named_parameters())))
        optimizer = optim.Adam([{
            'params': base_params
        }, {
            'params': gate_params,
            'lr': cfg.SOLVER.GATE_LR
        }],
        lr=cfg.SOLVER.LR)
    else:
        optimizer = optim.Adam(net.parameters(), lr=cfg.SOLVER.LR)

    # LR scheduler
    scheduler = None
    if cfg.SOLVER.USE_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=0, threshold=0.001, threshold_mode='rel', verbose=True)

    train_dataloader, val_dataloader = get_dataloader(cfg, mode=mode)

    for epoch in range(1, cfg.SOLVER.EPOCHS + 1):
        train(epoch, train_dataloader)
        valid(epoch, val_dataloader)

        with open(
            '{}_{}.model'.format(
                cfg.SAVE_PATH,
                str(epoch).zfill(2)),
            'wb') as f:
            torch.save(net_running.state_dict(), f)
