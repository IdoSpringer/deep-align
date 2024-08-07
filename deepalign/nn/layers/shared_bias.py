from typing import Optional, Tuple

import torch
from torch.nn import ModuleDict

from deepalign.nn.layers.base import BaseLayer, GeneralSetLayer, ScalarLayer


class BiasSharedLayer(BaseLayer):
    """Mapping non-siamese layers L(b1,b2) -> (b1,b2)"""

    def __init__(
            self,
            in_features,
            out_features,
            in_shape,
            out_shape,
            bias: bool = True,
            reduction: str = "sum",
            n_fc_layers: int = 1,
            num_heads: int = 8,
            set_layer: str = "sab",
            is_output_layer=False,
    ):
        """
        :param in_features: input feature dim
        :param out_features:
        :param in_shape:
        :param out_shape:
        :param bias:
        :param reduction:
        :param n_fc_layers:
        :param num_heads:
        :param set_layer:
        :param is_output_layer: indicates that the bias is that of the last layer.
        :param num_weights: number of weights to align
        """
        super().__init__(
            in_features,
            out_features,
            in_shape=in_shape,
            out_shape=out_shape,
            bias=bias,
            reduction=reduction,
            n_fc_layers=n_fc_layers,
            num_heads=num_heads,
            set_layer=set_layer,
        )
        self.is_output_layer = is_output_layer
        if is_output_layer:
            # i=L-1
            assert in_shape == out_shape
            self.layer = self._get_mlp(
                in_features=in_shape[0] * in_features,
                out_features=in_shape[0] * out_features,
                bias=bias,
            )
        else:
            self.layer = ScalarLayer(
                in_features=in_features,
                out_features=out_features,
            )

    def forward(self, x):
        # (bs, k, d{i+1}, in_features)
        num_weights = x.shape[1]
        if self.is_output_layer:
            # sum all different weights
            # (bs, d{i+1}, in_features)
            x = self._reduction(x, dim=1)
            # (bs, d{i+1} * out_features)
            x = self.layer(x.flatten(start_dim=1))
            # (bs, k, d{i+1}, out_features)
            x = x.reshape(x.shape[0], self.out_shape[0], self.out_features).unsqueeze(1).repeat(1, num_weights, 1, 1)
        else:
            # (bs, k, d{i+1}, in_features)
            # project to trivial irreps
            x = x.mean(dim=2).unsqueeze(dim=2).repeat(1, 1, self.out_shape[0], 1)
            # sum all different weights
            # (bs, d{i+1}, in_features)
            x = self._reduction(x, dim=1)
            # (bs, d{i+1}, out_features)
            x = self.layer(x)
            # (bs, k, d{i+1}, out_features)
            x = x.unsqueeze(1).repeat(1, num_weights, 1, 1)
        return x


class BiasSharedBlock(BaseLayer):
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
            hnp_setup=True,
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
        assert all([len(shape) == 1 for shape in shapes])

        self.shapes = shapes
        self.n_layers = len(shapes)

        self.layers = ModuleDict()
        # construct layers:
        for i in range(self.n_layers):
            self.layers[f"{i}_{i}"] = BiasSharedLayer(
                in_features=in_features,
                out_features=out_features,
                in_shape=shapes[i],
                out_shape=shapes[i],
                reduction=reduction,
                bias=bias,
                num_heads=num_heads,
                set_layer=set_layer,
                n_fc_layers=n_fc_layers,
                is_output_layer=(i == self.n_layers - 1) and hnp_setup,
            )

    def forward(self, x: Tuple[torch.tensor]):
        out_biases = [
                         0.0,
                     ] * len(x)
        for i in range(self.n_layers):
            out_biases[i] = self.layers[f"{i}_{i}"](x[i])

        return tuple(out_biases)
