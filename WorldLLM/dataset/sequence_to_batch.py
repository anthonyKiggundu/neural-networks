from rasterizer import SparseToDense
import torch

class SequenceBatchBuilder:
    def __init__(self, H=128, W=128):
        self.raster = SparseToDense(H, W)

    def build_pair(self, sequence):
        """
        sequence length = T
        returns training pairs:
            S_t , S_t+1
        """
        grids = []

        for timestep in sequence:
            coords, feats = timestep
            grid = self.raster(coords, feats)
            grids.append(grid)

        grids = torch.stack(grids)   # T C H W

        return grids[:-1], grids[1:]  # inputs, targets

