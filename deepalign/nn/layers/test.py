import sys
sys.path.append("/home/springer/PycharmProjects/research/deep-align2")
import torch
from experiments.utils import count_parameters
from layers import *
from layers import DownSampleCannibalLayer
from shared_weight import WeightSharedBlock

if __name__ == "__main__":
    d0, d1, d2, d3, d4, d5 = 32, 10, 20, 30, 40, 16
    matrices = (
        torch.randn(4, d0, d1, 12),
        torch.randn(4, d1, d2, 12),
        torch.randn(4, d2, d3, 12),
        torch.randn(4, d3, d4, 12),
        torch.randn(4, d4, d5, 12),
    )
    print(len(matrices))
    shared_weight_block = WeightSharedBlock(
        in_features=12, out_features=24, shapes=tuple(m.shape[1:3] for m in matrices)
    )
    weight_block = WeightToWeightBlock(
        in_features=12, out_features=24, shapes=tuple(m.shape[1:3] for m in matrices), hnp_setup=False
    )
    print(count_parameters(weight_block))
    print(count_parameters(shared_weight_block))
    # shape test
    siamese_out = weight_block(matrices)
    shared_out = shared_weight_block(matrices, siamese_out)
    print([o.shape for o in out])

    # perm test
    perm1 = torch.randperm(d1)
    perm2 = torch.randperm(d2)
    perm3 = torch.randperm(d3)
    perm4 = torch.randperm(d4)
    out_perm = weight_block(
        (
            matrices[0][:, :, perm1, :],
            matrices[1][:, perm1, :, :][:, :, perm2, :],
            matrices[2][:, perm2, :, :][:, :, perm3, :],
            matrices[3][:, perm3, :, :][:, :, perm4, :],
            matrices[4][:, perm4, :, :],
        )
    )

    assert torch.allclose(out[0][:, :, perm1, :], out_perm[0], atol=1e-5, rtol=0)
    assert torch.allclose(
        out[1][:, perm1, :, :][:, :, perm2, :], out_perm[1], atol=1e-5, rtol=0
    )
    assert torch.allclose(
        out[2][:, perm2, :, :][:, :, perm3, :], out_perm[2], atol=1e-5, rtol=0
    )
    assert torch.allclose(
        out[3][:, perm3, :, :][:, :, perm4, :], out_perm[3], atol=1e-5, rtol=0
    )
    assert torch.allclose(out[4][:, perm4, :, :], out_perm[4], atol=1e-5, rtol=0)
