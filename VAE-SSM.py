import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.utils.data import Dataset, DataLoader
import torchvision as tv
from torchvision import transforms
from torchvision.utils import make_grid
import tensorflow as tf  # for Progbar
from torch.optim.lr_scheduler import _LRScheduler
import pickle

# importing from the Cartesia repo
from models.s4.s4d import S4D


# Hyperparameters

INPUT_SHAPE          = (3, 64, 64)
H, W                 = INPUT_SHAPE[1], INPUT_SHAPE[2]
C                    = INPUT_SHAPE[0]
LATENT_DIM           = 256
D_MODEL              = 512
N_LAYERS             = 4
DROPOUT              = 0.1

BATCH_SIZE           = 8
TOTAL_TRAINING_STEPS = 20000
EVAL_EVERY_N_STEPS   = 2000
VALIDATION_STEPS     = 50

INITIAL_LR           = 1e-3
DECAY_STEPS          = 50000
DECAY_RATE           = 0.5
WARMUP_STEPS         = 1000

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# Dataset + Dataloaders

class CreateDataset(Dataset):
    def __init__(self, filepaths):
        self.filepaths = filepaths
        self.transform = transforms.Compose([
            transforms.Lambda(lambda p: tv.io.read_image(p).float()/255.),
            transforms.Resize((80,80)),
            transforms.CenterCrop((64,64)),
        ])

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img = self.transform(self.filepaths[idx])
        return img, img


def make_filepaths(csv_path, images_dir):
    df        = pd.read_csv(csv_path)
    train_ids = df[df['partition']==0]['image_id'].values
    val_ids   = df[df['partition']==1]['image_id'].values
    train_paths = [os.path.join(images_dir, i) for i in train_ids]
    val_paths   = [os.path.join(images_dir, i) for i in val_ids]
    return train_paths, val_paths


train_paths, val_paths = make_filepaths(
    '/cs/cs153/datasets/celeb_a_dataset/list_eval_partition.csv',
    '/cs/cs153/datasets/celeb_a_dataset/img_align_celeba/'
)
train_dl = DataLoader(CreateDataset(train_paths), BATCH_SIZE, shuffle=True,
                      pin_memory=True, num_workers=2)
val_dl   = DataLoader(CreateDataset(val_paths),   BATCH_SIZE, shuffle=True,
                      pin_memory=True, num_workers=2)



# SSM‐based VAE

class SSMEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Conv2d(C, D_MODEL, kernel_size=1)
        self.pos_emb    = nn.Parameter(torch.randn(1, H*W, D_MODEL))
        self.s4_layers  = nn.ModuleList([S4D(D_MODEL, dropout=DROPOUT, transposed=True)
                                         for _ in range(N_LAYERS)])
        self.norms      = nn.ModuleList([nn.LayerNorm(D_MODEL) for _ in range(N_LAYERS)])
        self.dropouts   = nn.ModuleList([nn.Dropout1d(DROPOUT) for _ in range(N_LAYERS)])
        self.to_stats   = nn.Linear(D_MODEL, LATENT_DIM*2)

    def forward(self, x):
        B = x.size(0)
        x = self.input_proj(x)
        x = x.flatten(2).transpose(1,2)
        x = x + self.pos_emb
        x = x.transpose(-1,-2)
        for s4, ln, dp in zip(self.s4_layers, self.norms, self.dropouts):
            z, _ = s4(x)
            z    = dp(z)
            x    = x + z
            x    = ln(x.transpose(-1,-2)).transpose(-1,-2)
        x = x.mean(-1)
        stats = self.to_stats(x)
        mu, log_sig = stats.chunk(2, dim=-1)
        return mu, log_sig


class SSMDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_proj = nn.Linear(LATENT_DIM, D_MODEL)
        self.pos_emb     = nn.Parameter(torch.randn(1, H*W, D_MODEL))
        self.s4_layers   = nn.ModuleList([S4D(D_MODEL, dropout=DROPOUT, transposed=True)
                                          for _ in range(N_LAYERS)])
        self.norms       = nn.ModuleList([nn.LayerNorm(D_MODEL) for _ in range(N_LAYERS)])
        self.dropouts    = nn.ModuleList([nn.Dropout1d(DROPOUT) for _ in range(N_LAYERS)])
        self.to_pixels   = nn.Linear(D_MODEL, C*2)

    def forward(self, z):
        B = z.size(0)
        x = self.latent_proj(z)
        x = x.unsqueeze(1).repeat(1, H*W, 1)
        x = x + self.pos_emb
        x = x.transpose(-1,-2)
        for s4, ln, dp in zip(self.s4_layers, self.norms, self.dropouts):
            z2, _ = s4(x)
            z2    = dp(z2)
            x     = x + z2
            x     = ln(x.transpose(-1,-2)).transpose(-1,-2)
        x     = x.transpose(-1,-2)
        stats = self.to_pixels(x)
        stats = stats.transpose(1,2).view(B, C*2, H, W)
        mu, log_sig = stats.chunk(2, dim=1)
        return mu, log_sig


class VAEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SSMEncoder()
        self.decoder = SSMDecoder()

    def forward(self, x):
        enc_mean, enc_logstd = self.encoder(x)
        eps = torch.randn_like(enc_mean).to(device)
        z   = enc_mean + torch.exp(enc_logstd) * eps
        dec_mean, dec_logstd = self.decoder(z)
        return enc_mean, enc_logstd, dec_mean, dec_logstd

    def generate(self, z_temp=1., x_temp=1.):
        z = torch.randn(BATCH_SIZE, LATENT_DIM, device=device) * z_temp
        dec_mean, dec_logstd = self.decoder(z)
        return Normal(dec_mean, torch.exp(dec_logstd) * x_temp).sample()



# Loss

class NelboLoss(nn.Module):
    def __init__(self, kl_slope: float = 1e-4, free_bits: float = 0.0):
        super().__init__()
        zeros = torch.zeros((BATCH_SIZE, LATENT_DIM), device=device)
        ones  = torch.ones_like(zeros)
        self.prior     = Normal(zeros, ones)
        self.kl_slope  = kl_slope     # how fast to ramp KL
        self.free_bits = free_bits    # minimum kl per sample in nats

    def forward(self,
                dec_mean, dec_logstd,
                enc_mean, enc_logstd,
                targets, step):
        # reconstruction term
        likelihood = Normal(
            dec_mean,
            torch.exp(torch.clamp(dec_logstd, min=-10.))
        ).log_prob(targets)
        rec_loss = -likelihood.sum() / likelihood.numel()

        # KL term per dimension
        q  = Normal(enc_mean, torch.exp(torch.clamp(enc_logstd, min=-10.)))
        kl = torch.distributions.kl.kl_divergence(q, self.prior)  # shape [B, D]

        # sum over latent dims → [B]
        kl_per_sample = kl.sum(dim=1)


        raw_kl = kl_per_sample.mean()  # scalar

        # annealing weight
        step_t    = torch.tensor(step, dtype=torch.float32, device=dec_mean.device)
        kl_weight = torch.clamp(step_t * self.kl_slope, max=1.0)

        kl_loss   = raw_kl * kl_weight

        return rec_loss + kl_loss, rec_loss, kl_loss



# Scheduler

class WarmupExponentialDecay(_LRScheduler):
    def __init__(self, optimizer, decay_steps, decay_rate, warmup_steps=500,
                 last_epoch=-1, verbose=False):
        self.warmup_steps = warmup_steps
        self.decay_steps  = decay_steps
        self.decay_rate   = decay_rate
        super().__init__(optimizer=optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        if self.last_epoch <= self.warmup_steps:
            return [base_lr * min(self.last_epoch / self.warmup_steps, 1.0)
                    for base_lr in self.base_lrs]
        return [base_lr * self.decay_rate ** ((self.last_epoch - self.warmup_steps) / self.decay_steps)
                for base_lr in self.base_lrs]



# Instantiating everything

model       = VAEModel().to(device)
optimizer   = torch.optim.Adamax(model.parameters(), lr=INITIAL_LR)
lr_schedule = WarmupExponentialDecay(optimizer, DECAY_STEPS, DECAY_RATE, WARMUP_STEPS)
loss        = NelboLoss()



# Checkpoint helpers

MODEL_NAME = 'vae_celeba64_dense256'
path       = f'pytorch_checkpoints/{MODEL_NAME}'
os.makedirs(os.path.dirname(path), exist_ok=True)

def save_checkpoint(step, model, optimizer):
    torch.save({
        'step':                   step,
        'model_state_dict':       model.state_dict(),
        'optimizer_state_dict':   optimizer.state_dict(),
        'lr_schedule_state_dict': lr_schedule.state_dict(),
    }, path)

def load_checkpoint():
    if os.path.exists(path):
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        lr_schedule.load_state_dict(ckpt['lr_schedule_state_dict'])
        print(f'Restored checkpoint at step {ckpt["step"]}')
        return ckpt['step']
    return 0


# Training / Validation functions

def trainval_step(inputs, targets, step, training=True):
    if training:
        model.train()
        optimizer.zero_grad()
        enc_m, enc_s, dec_m, dec_s = model(inputs)
        loss_val, rec_l, kl_l    = loss(dec_m, dec_s, enc_m, enc_s, targets, step)
        with torch.no_grad():
            loss_val.backward()
            optimizer.step()
            lr_schedule.step()
        return loss_val.item(), rec_l.item(), kl_l.item()
    else:
        model.eval()
        with torch.no_grad():
            enc_m, enc_s, dec_m, dec_s = model(inputs)
            loss_val, rec_l, kl_l    = loss(dec_m, dec_s, enc_m, enc_s, targets, step)
        return loss_val.item(), rec_l.item(), kl_l.item(), dec_m, dec_s


def validate(dataset, step):
    avg_loss, avg_rec, avg_kl = 0., 0., 0.
    for i, (x, y) in zip(range(VALIDATION_STEPS), dataset):
        lv, rl, kl, *_ = trainval_step(x.to(device), y.to(device), step, training=False)
        avg_loss += lv; avg_rec += rl; avg_kl += kl
    n = VALIDATION_STEPS
    return avg_loss/n, avg_rec/n, avg_kl/n


def validate_and_plot(train_dl, val_dl, step):
    t_loss, t_rec, t_kl = validate(train_dl, step)
    v_loss, v_rec, v_kl = validate(val_dl, step)
    print(f"\nstep {step}/{TOTAL_TRAINING_STEPS}  train: nelbo={t_loss:.4f}, rec={t_rec:.4f}, kl={t_kl:.4f}  "
          f"val: nelbo={v_loss:.4f}, rec={v_rec:.4f}, kl={v_kl:.4f}")
    # (you can insert plot_samples here if you like)



# Training Loop

def train_loop():
    step = load_checkpoint()
    pbar = tf.keras.utils.Progbar(TOTAL_TRAINING_STEPS, stateful_metrics=['nelbo','rec','kl'])
    results = {'nelbo':[], 'rec':[], 'kl':[]}

    try:
        for x, y in train_dl:
            step += 1
            if step == 50000:
                loss.kl_slope  = 5e-5    # slow the KL‐ramp by a factor of two  
                loss.free_bits = 0.1 
            lv, rl, kl = trainval_step(x.to(device), y.to(device), step)
            results['nelbo'].append(lv)
            results['rec'].append(rl) 
            results['kl'].append(kl)
            pbar.update(step, [('nelbo', lv), ('rec', rl), ('kl', kl)])

            if step % EVAL_EVERY_N_STEPS == 0:
                validate_and_plot(train_dl, val_dl, step)
                save_checkpoint(step, model, optimizer)
                print(f'Saved checkpoint at step {step}')

            if step >= TOTAL_TRAINING_STEPS:
                break

    except Exception:
        save_checkpoint(step, model, optimizer)
        print(f' Interrupted at step {step}, checkpoint saved.')
        raise

    with open('training_results.pkl','wb') as f:
        pickle.dump(results, f)
    return results


# Putting it all together

if __name__ == "__main__":
    start = time.time()
    results = train_loop()
    end = time.time()
    print(f"\nTotal training time: {end - start:.2f}s")

    # Saving losses
    with open('training_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("Saved training_results.pkl")

    # Plotting the losses
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # NELBO
    axes[0].plot(results['nelbo'], label='train nelbo')
    if 'val_nelbo' in results:
        axes[0].plot(results['nelbo'],   label='val nelbo')
    axes[0].set_title("NELBO Loss")
    axes[0].legend()

    # Reconstruction
    axes[1].plot(results['rec'], label='train rec')
    if 'rec' in results:
        axes[1].plot(results['rec'],     label='val rec')
    axes[1].set_title("Reconstruction Loss")
    axes[1].legend()

    # KL
    axes[2].plot(results['kl'],  label='train kl')
    if 'kl' in results:
        axes[2].plot(results['kl'],      label='val kl')
    axes[2].set_title("KL Loss")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig('loss_curves.png', bbox_inches='tight')
    plt.show()





