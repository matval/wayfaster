import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .utils import path_to_map
from models.traversability_net import TravNet

class TrainingModule(pl.LightningModule):
    def __init__(self, configs):
        super().__init__()
        # Save hyperparamters to hparams.yaml
        self.save_hyperparameters()
        # Get training params
        self.eps = 1e-6
        gamma = torch.tensor([configs.TRAINING.GAMMA**i for i in range(configs.TRAINING.HORIZON)])
        self.gamma = nn.Parameter(gamma, requires_grad=False)
        self.learning_rate = configs.OPTIMIZER.LR
        self.weight_decay = configs.OPTIMIZER.WEIGHT_DECAY
        # Depth training
        self.train_depth = configs.MODEL.TRAIN_DEPTH
        # Depth prediction
        self.predict_depth = configs.MODEL.PREDICT_DEPTH
        # MPPI
        # self.mppi = MPPI(self.configs)
        # Mapping Model
        self.model = TravNet(
            configs.MODEL.GRID_BOUNDS,
            configs.MODEL.INPUT_SIZE,
            downsample=configs.MODEL.DOWNSAMPLE,
            image_dim=configs.MODEL.LATENT_DIM,
            temporal_length=configs.MODEL.TIME_LENGTH,
            predict_depth=configs.MODEL.PREDICT_DEPTH,
            fuse_pcloud=configs.MODEL.FUSE_PCLOUD
        )
        # torch.autograd.set_detect_anomaly(True)
        # self.automatic_optimization = False
        self.grid_bounds = configs.MODEL.GRID_BOUNDS
        self.map_size = (
            int((self.grid_bounds['xbound'][1] - self.grid_bounds['xbound'][0])/self.grid_bounds['xbound'][2]),
            int((self.grid_bounds['ybound'][1] - self.grid_bounds['ybound'][0])/self.grid_bounds['ybound'][2]))

        self.map_origin = (
            int((self.grid_bounds['xbound'][1])/self.grid_bounds['xbound'][2]),
            int((self.grid_bounds['ybound'][1])/self.grid_bounds['ybound'][2]))

        self.map_resolution = (
            self.grid_bounds['xbound'][2],
            self.grid_bounds['ybound'][2])

        # Visualization rate
        self.vis_interval = configs.TRAINING.VIS_INTERVAL

    def training_step(self, batch, batch_idx):
        # data = (item.type(self.configs.dtype) for item in batch)
        color_img, pcloud, inv_intrinsics, extrinsics, path, target_trav, trav_weights, depth_target, depth_mask, label_img, label_mask = batch
        # color_img, pcloud, inv_intrinsics, extrinsics, path, goal, depth_target, depth_mask = batch

        trav_map, pred_depth, debug = self.model(color_img, pcloud, inv_intrinsics, extrinsics, depth_target)
        # trav_map, anomaly_map, pred_depth, debug = self.model(color_img, pcloud, inv_intrinsics, extrinsics)

        # # Reset previous solution to sample around optimal actions
        # self.mppi.reset(path[...,-2:])
        # # Run MPPI
        # with torch.no_grad():
        #     a_hat, trajs, pred_states, probs, costs = self.mppi(goal, anomaly_map, trav_map)
        # # Compute state visitation for expert path
        executed_path = path_to_map(path.unsqueeze(1), torch.ones_like(path[...,0,0]).unsqueeze(1), self.map_size, self.map_resolution, self.map_origin)
        # # Compute state visitation for MPPI trajectories
        # Dl = SVF_gaussian2(pred_states, probs, self.configs.map_size, self.configs.map_resolution, self.configs.map_origin)
        
        # Compute gradients
        # optimizer = self.optimizers()
        # optimizer.zero_grad()
        # costmap.backward(De-Dl, retain_graph=True)

        # pred_states = self.mppi.dyn_model(path[...,-2:].unsqueeze(1), trav_map)
        # Calculate traversability loss
        trav_loss, _ = self.trav_criterion(path, trav_map, target_trav, trav_weights)
        # Calculate states loss
        # states_loss = self.states_criterion(pred_states, path[...,:3].unsqueeze(1))
        # Calculate anomaly loss
        # anom_loss = self.model.get_anomaly_loss(anomaly_map, path)
        # Calculate depth classification loss
        depth_target = depth_target.view(-1, *depth_target.shape[2:])
        depth_mask = depth_mask.view(-1, *depth_mask.shape[2:])
        depth_loss = self.depth_criterion(pred_depth, depth_target, depth_mask)
        # supervised_loss = torch.mean(F.l1_loss(trav_map[:,0], label_img, reduction='none') * label_mask)

        if self.train_depth:
            loss = trav_loss + 0.1*depth_loss #+ supervised_loss #+ states_loss #+ anom_loss
        else:
            loss = trav_loss

        # # Backward gradients
        # loss.backward()
        # # clip gradients
        # self.clip_gradients(optimizer, gradient_clip_val=10, gradient_clip_algorithm="norm")
        # optimizer.step()

        # Visualize results
        if (batch_idx % self.vis_interval) == 0:
            self.visualize(color_img, pcloud, trav_map, pred_depth, depth_target, depth_mask, debug, executed_path, prefix='train')
            # self.visualize(color_img, pcloud, trav_map, pred_depth, depth_target, depth_mask, debug, executed_path, label_img, label_mask, prefix='train') #, prefix='train') # Dl, De
            # self.visualize(color_img, pcloud, trav_map, anomaly_map, pred_depth, depth_target, depth_mask, debug, executed_path, prefix='train') # Dl, De, prefix='train')

        # Logging to TensorBoard by default
        self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_trav_loss", trav_loss, on_step=True, on_epoch=True, sync_dist=True)
        # self.log("train_states_loss", states_loss, on_step=True, on_epoch=True, sync_dist=True)
        # self.log("train_anom_loss", anom_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_depth_loss", depth_loss, on_step=True, on_epoch=True, sync_dist=True)
        # self.log("train_supervised_loss", supervised_loss, on_step=True, on_epoch=True, sync_dist=True)
        # self.log("train_error", torch.mean((path[...,:3].unsqueeze(1)-trajs)**2), on_step=True, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        # data = (item.type(self.configs.dtype) for item in batch)
        color_img, pcloud, inv_intrinsics, extrinsics, path, target_trav, trav_weights, depth_target, depth_mask, label_img, label_mask = batch
        # color_img, pcloud, inv_intrinsics, extrinsics, path, goal, depth_target, depth_mask = batch

        trav_map, pred_depth, debug = self.model(color_img, pcloud, inv_intrinsics, extrinsics, depth_target)
        # trav_map, anomaly_map, pred_depth, debug = self.model(color_img, pcloud, inv_intrinsics, extrinsics)

        # # Reset previous solution to sample around optimal actions
        # self.mppi.reset(path[...,-2:])
        # # Run MPPI
        # a_hat, trajs, pred_states, probs, costs = self.mppi(goal, costmap, travmap)
        # # Compute state visitation for expert path
        executed_path = path_to_map(path.unsqueeze(1), torch.ones_like(path[...,0,0]).unsqueeze(1), self.map_size, self.map_resolution, self.map_origin)
        # # Compute state visitation for MPPI trajectories
        # Dl = SVF_gaussian2(pred_states, probs, self.configs.map_size, self.configs.map_resolution, self.configs.map_origin)
        # # Compute loss
        # # loss = self.criterion(Dl, De)

        # pred_states = self.mppi.dyn_model(path[...,-2:].unsqueeze(1), trav_map)
        # Calculate traversability loss
        trav_loss, trav_error = self.trav_criterion(path, trav_map, target_trav, trav_weights)
        # Calculate states loss
        # states_loss = self.states_criterion(pred_states, path[...,:3].unsqueeze(1))
        # Calculate anomaly loss
        # anom_loss = self.model.get_anomaly_loss(anomaly_map, path)
        # Calculate depth classification loss
        depth_target = depth_target.view(-1, *depth_target.shape[2:])
        depth_mask = depth_mask.view(-1, *depth_mask.shape[2:])
        depth_loss = self.depth_criterion(pred_depth, depth_target, depth_mask)
        # supervised_loss = torch.mean(F.l1_loss(trav_map[:,0], label_img, reduction='none') * label_mask)

        if self.train_depth:
            loss = trav_loss + 0.1*depth_loss #+ supervised_loss #+ states_loss #+ anom_loss
        else:
            loss = trav_loss

        # Visualize results
        if (batch_idx % self.vis_interval) == 0:
            self.visualize(color_img, pcloud, trav_map, pred_depth, depth_target, depth_mask, debug, executed_path, prefix='valid')
            # self.visualize(color_img, pcloud, trav_map, pred_depth, depth_target, depth_mask, debug, executed_path, label_img, label_mask, prefix='valid') #, executed_path, prefix='valid') # Dl, De, prefix='valid')
            # self.visualize(color_img, pcloud, trav_map, anomaly_map, pred_depth, depth_target, depth_mask, debug, executed_path, prefix='valid') # Dl, De, prefix='valid')

        # Logging to TensorBoard by default
        self.log("valid_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log("valid_trav_loss", trav_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log("valid_trav_error", trav_error, on_step=True, on_epoch=True, sync_dist=True)
        # self.log("valid_states_loss", states_loss, on_step=True, on_epoch=True, sync_dist=True)
        # self.log("valid_anom_loss", anom_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log("valid_depth_loss", depth_loss, on_step=True, on_epoch=True, sync_dist=True)
        # self.log("valid_supervised_loss", supervised_loss, on_step=True, on_epoch=True, sync_dist=True)
        # self.log("valid_error", torch.mean((path[...,:3].unsqueeze(1)-trajs)**2), on_step=True, on_epoch=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        model_params = self.model.parameters()
        optimizer = torch.optim.AdamW(model_params, lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer
    
    # Define criterion
    def trav_criterion(self, states, trav_map, target_trav, target_weights):
        # Calculate traversability
        idxu = 2 * (states[...,0] - self.grid_bounds['xbound'][0]) / (self.grid_bounds['xbound'][1] - self.grid_bounds['xbound'][0]) - 1
        idxv = 2 * (states[...,1] - self.grid_bounds['ybound'][0]) / (self.grid_bounds['ybound'][1] - self.grid_bounds['ybound'][0]) - 1
        grid = torch.stack((-idxv, -idxu), -1).unsqueeze(1)
        mask = (grid[...,0] >= -1) * (grid[...,0] <= 1) * (grid[...,1] >= -1) * (grid[...,1] <= 1)
        # Sample from map
        traversability = F.grid_sample(trav_map, grid, align_corners=True)
        traversability = traversability.permute((0,2,3,1)).squeeze(1)
        # Calculate error
        # error = (traversability - target_trav)**2 * target_weights
        gamma = self.gamma[:traversability.shape[1]].view(1,-1,1).expand(traversability.shape[0],-1,2)
        error = F.l1_loss(traversability, target_trav, reduction='none')
        loss = error * target_weights * gamma
        return torch.mean(loss[mask.squeeze(1),:]), torch.mean(error[mask.squeeze(1),:])

    def states_criterion(self, prediction, target):
        mask = (target[...,0] >= self.grid_bounds['xbound'][0]) & \
            (target[...,0] <= self.grid_bounds['xbound'][1]) & \
            (target[...,1] >= self.grid_bounds['ybound'][0]) & \
            (target[...,1] <= self.grid_bounds['ybound'][1])
        error = target - prediction
        error[...,2] = 1 - torch.cos(target[...,2] - prediction[...,2])
        return torch.mean(error[mask]**2)
    
    def depth_criterion(self, prediction, target, mask):
        loss = F.cross_entropy(prediction, target, reduction='none') * mask
        return torch.mean(loss)

    # def visualize(self, image, pcloud, trav_map, anom_map, pred_depth, depth_target, depth_mask, debug, executed_path, prefix='train'): # trajs, label, prefix='train'):
    # def visualize(self, image, pcloud, trav_map, pred_depth, depth_target, depth_mask, debug, executed_path, label_img, label_mask, prefix='train'): # , executed_path
    def visualize(self, image, pcloud, trav_map, pred_depth, depth_target, depth_mask, debug, executed_path, prefix='train'):
        self.logger.experiment.add_image(prefix + '_image', image.view(-1,*image.shape[2:]), self.current_epoch, dataformats='NCHW')
        pcloud = torch.mean(pcloud, dim=2, keepdim=True)
        pcloud = pcloud.view(-1,*pcloud.shape[2:])
        self.logger.experiment.add_image(prefix + '_pcloud', pcloud, self.current_epoch, dataformats='NCHW')
        self.logger.experiment.add_image(prefix + '_mu', trav_map[:,:1], self.current_epoch, dataformats='NCHW')
        self.logger.experiment.add_image(prefix + '_nu', trav_map[:,1:], self.current_epoch, dataformats='NCHW')
        # self.logger.experiment.add_image(prefix + '_anomaly', torch.exp(anom_map), self.current_epoch, dataformats='NCHW')
        n_d = (self.grid_bounds['dbound'][1] - self.grid_bounds['dbound'][0])/self.grid_bounds['dbound'][2]
        self.logger.experiment.add_image(prefix + '_depth_pred', torch.argmax(pred_depth, dim=1, keepdim=True)/(n_d-1), self.current_epoch, dataformats='NCHW')
        if self.predict_depth:
            self.logger.experiment.add_image(prefix + '_depth_target', depth_target.unsqueeze(1)/(n_d-1), self.current_epoch, dataformats='NCHW')
        else:
            depth_target = depth_target.argmax(1).unsqueeze(1)/(n_d-1)
            self.logger.experiment.add_image(prefix + '_depth_target', depth_target, self.current_epoch, dataformats='NCHW')
        self.logger.experiment.add_image(prefix + '_depth_mask', depth_mask.unsqueeze(1), self.current_epoch, dataformats='NCHW')
        temp = torch.sum(debug, dim=1, keepdim=True)
        temp = (temp-torch.min(temp))/(torch.max(temp)-torch.min(temp))
        self.logger.experiment.add_image(prefix + '_debug', temp, self.current_epoch, dataformats='NCHW')
        self.logger.experiment.add_image(prefix + '_path', executed_path/(torch.amax(executed_path, (1,2,3), keepdim=True)+self.eps), self.current_epoch, dataformats='NCHW')
        # self.logger.experiment.add_image(prefix + '_label', label_img.unsqueeze(1), self.current_epoch, dataformats='NCHW')
        # self.logger.experiment.add_image(prefix + '_label_mask', label_mask.unsqueeze(1), self.current_epoch, dataformats='NCHW')
        # self.logger.experiment.add_image(prefix + '_trajs', trajs/(torch.amax(trajs, (1,2,3), keepdim=True)+self.eps), self.current_epoch, dataformats='NCHW')
        # self.logger.experiment.add_image(prefix + '_label', label/(torch.amax(label, (1,2,3), keepdim=True)+self.eps), self.current_epoch, dataformats='NCHW')
