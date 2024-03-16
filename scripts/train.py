import os
import time
import torch
import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint

# Custom packages
from dataloader import Dataset
from train_configs import get_cfg
from trainer import TrainingModule

def parse_config():
    # Get arguments
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify path config file')

    args = parser.parse_args()

    # Load default configs and merge with args
    config = get_cfg(args.cfg_file)

    return config

def main():
    configs = parse_config()

    print('configs:\n', configs)

    pl.seed_everything(configs.SEED, workers=True)

    # Image transform
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor()])

    train_dataset = Dataset(configs, configs.DATASET.TRAIN_DATA, transform)
    valid_dataset = Dataset(configs, configs.DATASET.VALID_DATA, transform, train_dataset.weights)

    train_loader = DataLoader(train_dataset, batch_size=configs.TRAINING.BATCHSIZE, shuffle=True, num_workers=configs.TRAINING.N_WORKERS, drop_last=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=configs.TRAINING.BATCHSIZE, shuffle=False, num_workers=configs.TRAINING.N_WORKERS, drop_last=True, pin_memory=True)

    model = TrainingModule(configs)

    # Load a previously trained network
    if configs.MODEL.LOAD_NETWORK is not None:
        print('Loading saved network from {}'.format(configs.MODEL.LOAD_NETWORK))
        pretrained_dict = torch.load(configs.MODEL.LOAD_NETWORK, map_location='cpu')['state_dict']
        model.load_state_dict(pretrained_dict)

    save_dir    = os.path.join('runs', configs.TAG + "_" + time.strftime('%d-%m-%Y_%H-%M-%S'))
    tb_logger   = pl.loggers.TensorBoardLogger(save_dir=save_dir)
    checkpoint_callback = ModelCheckpoint(
        filename            = 'checkpoint_{epoch}-{valid_loss:.4f}',
        monitor             = "valid_trav_loss",
        save_weights_only   = True,
        save_last           = True,
        save_top_k          = 1)

    trainer = pl.Trainer(
        devices                 = 2,
        num_nodes               = 1,
        gradient_clip_val       = 10,
        sync_batchnorm          = True,
        enable_model_summary    = True,
        accelerator             = 'gpu',
        # profiler                = 'pytorch', #'simple',
        logger                  = tb_logger,
        default_root_dir        = 'checkpoints',
        max_epochs              = configs.TRAINING.EPOCHS,
        precision               = configs.TRAINING.PRECISION,
        strategy                = DDPStrategy(find_unused_parameters=False),
        callbacks               = [checkpoint_callback])

    trainer.fit(model, train_loader, valid_loader)

if __name__ == "__main__":
    main()
