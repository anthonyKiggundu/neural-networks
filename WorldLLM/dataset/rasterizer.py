import torch

class SparseToDense:
    """
    Converts sparse telecom observations into dense grid tensor.
    Output channels:
        0 download
        1 upload
        2 latency
        3 measurement_density
    """

    def __init__(self, H=128, W=128):
        self.H = H
        self.W = W
        self.C = 4

    def __call__(self, coords, feats):
        grid = torch.zeros(self.C, self.H, self.W)

        if len(coords) == 0:
            return grid

        xs = coords[:, 0].clamp(0, self.W-1)
        ys = coords[:, 1].clamp(0, self.H-1)

        grid[0, ys, xs] += feats[:, -3]   # download
        grid[1, ys, xs] += feats[:, -2]   # upload
        grid[2, ys, xs] += feats[:, -1]   # latency
        grid[3, ys, xs] += 1.0            # density

        # normalize density cells
        mask = grid[3] > 0
        grid[0][mask] /= grid[3][mask]
        grid[1][mask] /= grid[3][mask]
        grid[2][mask] /= grid[3][mask]

        # normalize ranges
        grid[0] /= 200.0   # Mbps
        grid[1] /= 50.0
        grid[2] /= 200.0

        return grid

