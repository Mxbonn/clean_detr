from torch import nn


class MLP(nn.Sequential):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation_layer: type[nn.Module] = nn.ReLU,
        sigmoid_output: bool = False,
    ):
        layers = []
        hidden_dims = [hidden_dim] * (num_layers - 1)
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation_layer())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        if sigmoid_output:
            layers.append(nn.Sigmoid())
        super().__init__(*layers)
