import torch
import argparse
import pytorch_lightning as pl
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# Custom packages
from train.dataloader import Dataset
from train.train_configs import get_cfg
from train.trainer import TrainingModule

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

    train_dataset = Dataset(configs, configs.DATASET.TRAIN_DATA, transform=transform, train=True)
    valid_dataset = Dataset(configs, configs.DATASET.VALID_DATA, weights=train_dataset.weights)

    train_loader = DataLoader(
        train_dataset,
        batch_size  = configs.TRAINING.BATCHSIZE,
        num_workers = configs.TRAINING.WORKERS,
        shuffle     = True,
        drop_last   = True,
        pin_memory  = True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size  = configs.TRAINING.BATCHSIZE,
        shuffle     = False,
        num_workers = configs.TRAINING.WORKERS,
        drop_last   = True,
        pin_memory  = True
    )

    model = TrainingModule(configs)

    # Load a previously trained network
    if configs.MODEL.LOAD_NETWORK is not None:
        print('Loading saved network from {}'.format(configs.MODEL.LOAD_NETWORK))
        pretrained_dict = torch.load(configs.MODEL.LOAD_NETWORK, map_location='cpu')['state_dict']
        model.load_state_dict(pretrained_dict)

    # Initialize logger
    wandb_logger = WandbLogger(
        project="WayFASTER",
        log_model="all",
    )

    # Set up checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath             = 'checkpoints',
        filename            = 'checkpoint_{epoch}-{valid_loss:.4f}',
        monitor             = "valid_loss",
        save_weights_only   = True,
        save_last           = True,
        save_top_k          = 1)
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        devices                 = -1,
        num_nodes               = 1,
        gradient_clip_val       = 10,
        sync_batchnorm          = True,
        enable_model_summary    = True,
        accelerator             = 'gpu',
        logger                  = wandb_logger,
        default_root_dir        = 'checkpoints',
        max_epochs              = configs.TRAINING.EPOCHS,
        precision               = configs.TRAINING.PRECISION,
        strategy                = DDPStrategy(find_unused_parameters=False),
        callbacks               = [checkpoint_callback, lr_monitor],
    )

    trainer.fit(model, train_loader, valid_loader)

if __name__ == "__main__":
    main()
