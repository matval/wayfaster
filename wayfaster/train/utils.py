import torch
import kornia

EPS = 1e-6

def path_to_map(trajs, probs, map_shape, map_resolution, map_origin, std=0.1):
    '''
    Project path to map
    
    trajs: B,K,T,3
    probs: B,K
    '''
    B,K,T,_ = trajs.shape
    D = torch.zeros((B, map_shape[0], map_shape[1]), dtype=trajs.dtype, device=trajs.device)
    u = torch.round(map_origin[0] - trajs[...,0]/map_resolution[0]).long()
    v = torch.round(map_origin[1] - trajs[...,1]/map_resolution[1]).long()
    _, idx = torch.max(probs, dim=1)
    probs = probs.unsqueeze(-1).expand(-1,-1,T)
    for i in range(B):
        bu = u[i]
        bv = v[i]
        idxs = (bu >= 0) * (bu < map_shape[0]) * (bv >= 0) * (bv < map_shape[1])
        D[i] = D[i].index_put_((bu[idxs], bv[idxs]), probs[i,idxs], accumulate=True)
        D[i] = D[i] / (torch.sum(D[i]) + EPS)

    D = kornia.filters.gaussian_blur2d(D.unsqueeze(1), (7, 7), (std/map_resolution[0], std/map_resolution[1]))
    return D