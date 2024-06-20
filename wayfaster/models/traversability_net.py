import torch
import torch.nn as nn
import torch.nn.functional as F

from .temporal_fusion import TemporalModel
from .encoder_decoder import Encoder, Decoder

class TravNet(nn.Module):
    """
    TravNet: A neural network for traversability prediction and navigation.

    Attributes:
        grid_bounds (dict): Dictionary containing the boundaries for the grid.
        input_size (tuple): Size of the input image.
        downsample (int): Downsampling factor.
        image_dim (int): Dimension of the image.
        temporal_length (int): Length of the temporal sequence.
        predict_depth (bool): Whether to predict depth or not.
        fuse_pcloud (bool): Whether to fuse point cloud data or not.
    """
    def __init__(
            self, grid_bounds, input_size, downsample=8,
            image_dim=64, temporal_length=3, predict_depth=True,
            fuse_pcloud=True
        ):
        super(TravNet, self).__init__()

        self.grid_bounds    = grid_bounds
        self.input_size     = (input_size[1], input_size[0])
        self.camC           = image_dim
        self.downsample     = downsample     # Due to the encoder
        self.predict_depth  = predict_depth
        self.fuse_pcloud    = fuse_pcloud
        self.eps            = 1e-6
        
        dx = torch.Tensor([row[2] for row in [grid_bounds['xbound'], grid_bounds['ybound'], grid_bounds['zbound']]])
        bx = torch.Tensor([row[0] + row[2]/2.0 for row in [grid_bounds['xbound'], grid_bounds['ybound'], grid_bounds['zbound']]])
        nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [grid_bounds['xbound'], grid_bounds['ybound'], grid_bounds['zbound']]])

        self.int_nx = nx.cpu().detach().numpy()

        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        # Images mean and std
        mean = torch.as_tensor([0.485, 0.456, 0.406]).reshape(1,3,1,1).float()
        std = torch.as_tensor([0.229, 0.224, 0.225]).reshape(1,3,1,1).float()
        self.mean = nn.Parameter(mean, requires_grad=False)
        self.std = nn.Parameter(std, requires_grad=False)

        # Create 3D voxels
        self.voxels = self.create_voxels()

        # Define predicted depth dimension
        if self.predict_depth:
            self.D = int((grid_bounds['dbound'][1] - grid_bounds['dbound'][0]) / grid_bounds['dbound'][2])
        else:
            self.D = 0

        self.latent_dim = image_dim

        # Define pcloud fusion dimension
        if self.fuse_pcloud:
            pcloud_dim = self.nx[2]
        else:
            pcloud_dim = 0
        
        # Image encoder
        self.encoder = Encoder(self.D + self.camC, downsample=self.downsample)

        # Bird's eye view compressor        
        self.bev_compressor = nn.Sequential(
            nn.Conv2d(self.camC * self.nx[2] + pcloud_dim, self.camC, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.camC),
            nn.ReLU(inplace=True))

        # Temporal Model
        self.temporal_model = TemporalModel(
            channels=self.camC,
            temporal_length=temporal_length,
            input_shape=(nx[0], nx[1]))

        # Bird's eye view decoder
        self.decoder = Decoder(in_channels=self.latent_dim)
        
        # Traversability map head
        self.travmap_head = nn.Sequential(
            nn.Conv2d(self.latent_dim, self.latent_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.latent_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.latent_dim, 2, kernel_size=1, padding=0),
            nn.Sigmoid())
    
    def create_voxels(self):
        """
        Create a 3D grid in the map space.

        Returns:
            torch.nn.Parameter: 3D grid of voxels.
        """
        grid_z = torch.arange(*self.grid_bounds['zbound'], dtype=torch.float).flip(0)
        grid_z = torch.reshape(grid_z, [self.nx[2], 1, 1])
        grid_z = grid_z.repeat(1, self.nx[0], self.nx[1])

        grid_y = torch.arange(*self.grid_bounds['ybound'], dtype=torch.float).flip(0)
        grid_y = torch.reshape(grid_y, [1, 1, self.nx[1]])
        grid_y = grid_y.repeat(self.nx[2], self.nx[0], 1)

        grid_x = torch.arange(*self.grid_bounds['xbound'], dtype=torch.float).flip(0)
        grid_x = torch.reshape(grid_x, [1, self.nx[0], 1])
        grid_x = grid_x.repeat(self.nx[2], 1, self.nx[1])

        # Z x X x Y x 3
        voxels = torch.stack((grid_x, grid_y, grid_z), -1)
        return nn.Parameter(voxels, requires_grad=False)
    
    def get_inv_geometry(self, intrinsics, extrinsics):
        """
        Calculate the mapping from 3D map voxels to camera frustum.

        Args:
            intrinsics (torch.Tensor): Intrinsics matrix (3x3) for projection.
            extrinsics (torch.Tensor): Extrinsics matrix (4x4) with rotation and translation.

        Returns:
            torch.Tensor: Transformed points in the camera reference frame (u, v, depth).
        """
        # separate rotation and translation from extrinsics matrix
        rotation, translation = extrinsics[..., :3, :3], extrinsics[..., :3, 3]
        # get batch and number of cams dimensions
        B, N, _ = translation.shape
        # Add batch, camera dimension, and a dummy dimension at the end
        points = self.voxels.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        points = points.expand(B, -1, -1, -1, -1, -1, -1)
        # Voxels to camera frame
        points = points - translation.view(B, N, 1, 1, 1, 3, 1)
        combined_transformation = intrinsics.matmul(torch.inverse(rotation))
        points = combined_transformation.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)

        points = torch.cat((points[..., :2] / (points[..., 2:3] + self.eps), points[..., 2:3]), -1)

        # The 3 dimensions in the camera reference frame (u, v, depth)
        return points
    
    def sample2bev(self, geometry, x):
        """
        Sample from the frustum to bird's eye view.

        Args:
            geometry (torch.Tensor): Geometry of the frustum.
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Bird's eye view representation.
        """
        # batch*T, depth, height, width, channels
        _, d, h, w, c = x.shape

        batch, T, Z, X, Y, _ = geometry.shape
        geometry = geometry.view(batch*T, Z, X, Y, 3)

        # Normalize grid
        u = 2 * geometry[..., 0] / (self.input_size[1]-1) - 1
        v = 2 * geometry[..., 1] / (self.input_size[0]-1) - 1
        depth = 2 * (geometry[...,2] - self.grid_bounds['dbound'][0]) / (self.grid_bounds['dbound'][1] - self.grid_bounds['dbound'][0]) - 1
        grid = torch.stack((u, v, depth), -1)

        # Sample from frustum
        x = F.grid_sample(x, grid, align_corners=False)
        x = x.view(batch, T, *x.shape[1:])
        return x
        
    def forward(self, color_img, pcloud, intrinsics, extrinsics, depth_img=None):
        """
        Forward pass of the TravNet model.

        Args:
            color_img (torch.Tensor): Color image tensor of shape (B, T, C, H, W).
            pcloud (torch.Tensor): Point cloud tensor of shape (B, T, Z, Y, X).
            intrinsics (torch.Tensor): Intrinsics matrix tensor of shape (B, T, 3, 3).
            extrinsics (torch.Tensor): Extrinsics matrix tensor of shape (B, T, 4, 4).
            depth_img (torch.Tensor, optional): Depth image tensor. Defaults to None.

        Returns:
            tuple: A tuple containing:
                - trav_map (torch.Tensor): Traversability map of shape (B, 2, X, Y).
                - depth_logits (torch.Tensor): Predicted depth logits of shape (B*T, D, H, W).
                - debug (torch.Tensor): Debug information tensor.
        """
        B, T, C, imH, imW = color_img.shape
        assert(C==3)

        color_img = color_img.view(B*T, C, imH, imW)

        # Normalize image according to statistics
        x = (color_img - self.mean) / self.std
        # Pass image throught encoder
        x = self.encoder(x)

        # Depth is B x N x D x H/downsample x W/downsample
        if self.predict_depth:
            depth_logits = x[:, :self.D]
        else:
            depth_logits = depth_img.view(B*T, *depth_img.shape[2:])
        depth_context = x[:, self.D:(self.D + self.camC)]

        # depth_logits is B*N x D x H/downsample x W/downsample
        # depth_context is B*N x C x H/downsample x W/downsample
        x = depth_logits.softmax(dim=1).unsqueeze(1) * depth_context.unsqueeze(2)

        # Create 3D frustrum
        geom = self.get_inv_geometry(intrinsics, extrinsics)
        x = self.sample2bev(geom, x)

        # X is B x T x C x Z x X x Y ->  B*T x C*Z x X x Y
        x = x.view(B*T, -1, *x.shape[4:])
        pcloud = pcloud.view(B*T, *pcloud.shape[2:])
        debug = x

        if self.fuse_pcloud:
            # Concatenate with pointcloud
            x = torch.cat([x, pcloud], dim=1)
        
        # And compress X to B x T x C x X x Y
        x = self.bev_compressor(x)

        # Recover temporal component: B x T x C x X x Y
        x = x.view(B, T, *x.shape[1:])

        # Temporal fusion with 3D conv layers
        x = self.temporal_model(x)
        
        # Now, x is a grid of shape (B x C x X x Y)
        x = x.view(B, -1, self.int_nx[0], self.int_nx[1])
        
        # BEV decoder
        bev_features = self.decoder(x)
        
        # Calculate traversability map
        trav_map = self.travmap_head(bev_features)

        return trav_map, depth_logits, debug
