import torch


class CutRandomDataPieces(torch.nn.Module):
    """Cut random end or start data pieces"""
    def __init__(self, max_cut=10):
        super(CutRandomDataPieces, self).__init__()
        self.max_cut = max_cut

    def forward(self, x):
        side = torch.randint(0, 2, (1,))
        cut = torch.randint(1, self.max_cut, (1,))
        if side == 0:
            return x[:, :, :-cut]
        elif side == 1:
            return x[:, :, cut:]


print(CutRandomDataPieces())