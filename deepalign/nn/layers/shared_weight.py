from typing import Optional, Tuple

import torch
from torch.nn import ModuleDict

from deepalign.nn.layers.base import BaseLayer, GeneralSetLayer, MatrixLayer, ScalarLayer


class SharedWeightLayer(BaseLayer):
    """Mapping L(W1_1,W1_2) -> L(W1_1,W1_2)"""

    def __init__(
            self,
            in_features,
            out_features,
            in_shape,
            out_shape,
            bias: bool = True,
            reduction: str = "max",
            n_fc_layers: int = 1,
            last_dim_is_output=False,
            first_dim_is_output=False,

    ):
        """
        :param in_features: input feature dim
        :param out_features:
        :param in_shape:
        :param out_shape:
        :param bias:
        :param reduction:
        :param n_fc_layers:
        """
        super().__init__(
            in_features,
            out_features,
            in_shape=in_shape,
            out_shape=out_shape,
            bias=bias,
            reduction=reduction,
            n_fc_layers=n_fc_layers,
        )
        self.last_dim_is_output = last_dim_is_output
        self.first_dim_is_output = first_dim_is_output

        if self.first_dim_is_output:
            # w1 -> w1 # d0
            in_shape = self.in_shape[0]
            out_shape = self.out_shape[0]
            self.layer = MatrixLayer(
                in_shape=in_shape, out_shape=out_shape,
                in_features=in_features, out_features=out_features, bias=bias
            )
        elif self.last_dim_is_output:
            # wL -> wL  # dL-1
            self.layer = MatrixLayer(
                in_shape=self.in_shape[1], out_shape=self.out_shape[1],
                in_features=in_features, out_features=out_features, bias=bias
            )
        else:
            # wi -> wi
            in_features = self.in_features
            out_features = self.out_features
            self.layer = ScalarLayer(
                in_features=in_features,
                out_features=out_features,
            )

    def forward(self, x):
        # (bs, k, d0, d1, in_features)
        if self.first_dim_is_output:
            # w is d1*d0
            # v_fixed is constant columns (rows are the same). project to v_fixed by col sum
            # (bs, k, d0, d1, in_features)
            x = self._reduction(x, dim=3).unsqueeze(3).repeat(1, 1, 1, self.out_shape[1], 1)
            # sum all different weights
            # (bs, d0, d1, in_features)
            x = x.mean(dim=1)
            # apply params (d1*d0)(d0*d0) -> (d1*d0)
            # (bs, d0, d1, out_features)
            x = x.transpose(-2, -3)
            x = self.layer(x, from_right=True)
            # (bs, k, d0, d1, out_features)
            x = x.reshape(x.shape[0], *self.out_shape, self.out_features)
            # ( i think we can save extra dimension computation here since rows are the same. not implemented)
        elif self.last_dim_is_output:
            # v_fixed is constant rows (columns are the same). project to v_fixed be row sum
            # (bs, k, dL-1, dL, in_features)
            x = self._reduction(x, dim=2).unsqueeze(2).repeat(1, 1, self.out_shape[0], 1, 1)
            # sum all different weights
            # (bs, d0, d1, in_features)
            x = x.mean(dim=1)
            # apply params (dL-1*dL-1)(dL-1*dL) -> (dL-1*dL)
            # (bs, d0, d1, out_features)
            x = self.layer(x)
            # (bs, k, d0, d1, out_features)
        else:
            # v fixed is scalar matrices. project to v_fixed by summing all elements
            # (bs, k, d1, in_features)
            x = self._reduction(x, dim=2)
            # (bs, k, in_features)
            x = self._reduction(x, dim=2)
            # sum all different weights
            # (bs, in_features)
            x = x.mean(dim=1)
            # apply params
            # (bs, out_features)
            x = self.layer(x)
            # repeat to original size
            # (bs, dL-1, dL, out_features)
            x = x.unsqueeze(1).unsqueeze(1).repeat(1, *self.out_shape, 1)
            # (bs, k, dL-1, dL, out_features)
        return x


class WeightSharedBlock(BaseLayer):
    def __init__(
            self,
            in_features,
            out_features,
            shapes,
            bias: bool = True,
            reduction: str = "max",
            n_fc_layers: int = 1,
            num_heads: int = 8,
            set_layer: str = "sab",
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            reduction=reduction,
            n_fc_layers=n_fc_layers,
            num_heads=num_heads,
            set_layer=set_layer,
        )
        assert all([len(shape) == 2 for shape in shapes])
        assert len(shapes) > 2
        self.shapes = shapes
        self.n_layers = len(shapes)
        self.layers = ModuleDict()
        first_dim_is_output = False
        last_dim_is_output = False
        # construct layers:
        for i in range(self.n_layers):
            if i == 0:
                first_dim_is_output = True
            if i == self.n_layers - 1:
                last_dim_is_output = True
            self.layers[f"{i}_{i}"] = SharedWeightLayer(
                in_features=in_features,
                out_features=out_features,
                in_shape=shapes[i],
                out_shape=shapes[i],
                reduction=reduction,
                bias=bias,
                n_fc_layers=n_fc_layers,
                last_dim_is_output=last_dim_is_output,
                first_dim_is_output=first_dim_is_output)

            first_dim_is_output = False
            last_dim_is_output = False

    def forward(self, x: Tuple[torch.tensor], siamese_weights):
        out_wights = [[None for _ in range(self.n_layers)] for _ in range(2)]
        for i in range(self.n_layers):
            out_weight = self.layers[f"{i}_{i}"](x[i])
            for j in range(2):
                out_wights[j][i] = siamese_weights[j][i] + out_weight

        return out_wights
