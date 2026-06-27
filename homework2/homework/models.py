"""
Implement the following models for classification.

Feel free to modify the arguments for each of model's __init__ function.
This will be useful for tuning model hyperparameters such as hidden_dim, num_layers, etc,
but remember that the grader will assume the default constructor!
"""

from pathlib import Path

import torch
import torch.nn as nn


class ClassificationLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.LongTensor) -> torch.Tensor:
        """
        Multi-class classification loss
        Hint: simple one-liner

        Args:
            logits: tensor (b, c) logits, where c is the number of classes
            target: tensor (b,) labels

        Returns:
            tensor, scalar loss
        """
        
        return nn.functional.cross_entropy(logits, target)


class LinearClassifier(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
    ):
        """
        Args:
            h: int, height of the input image
            w: int, width of the input image
            num_classes: int, number of classes
        """
        super().__init__()

        
        c = 3*h*w
        self.linear = nn.Linear(c, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        
        x_flat = x.view(x.size(0), -1)
        logits = self.linear(x_flat)
        
        return logits


class MLPClassifier(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
        hidden_dim: int = 128,
    ):
        """
        An MLP with a single hidden layer

        Args:
            h: int, height of the input image
            w: int, width of the input image
            num_classes: int, number of classes
            hidden_dim: int, the dimensions of the hidden layer
        """
        super().__init__()

        c = 3*h*w
        
        # self.mlp = nn.Sequential(
        #   nn.Linear(c, hidden_dim),
        #   nn.ReLU(),
        #   nn.Linear(hidden_dim, num_classes)
        # )

        layers = []
        layers.append(torch.nn.Flatten())
        
        layers.append(torch.nn.Linear(c, hidden_dim))
        layers.append(torch.nn.ReLU())

        layers.append(torch.nn.Linear(hidden_dim, num_classes))
        self.model = torch.nn.Sequential(*layers)

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        
        # x_flat = x.view(x.size(0), -1)
        # logits = self.mlp(x_flat)
        
        # return logits

        return self.model(x)


class MLPClassifierDeep(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
        hidden_dim: int = 128,
        num_layers: int = 4,
    ):
        """
        An MLP with multiple hidden layers

        Args:
            h: int, height of image
            w: int, width of image
            num_classes: int

        Hint - you can add more arguments to the constructor such as:
            hidden_dim: int, size of hidden layers
            num_layers: int, number of hidden layers
        """
        super().__init__()

        # input_dim = 3*h*w
        # layers = []

        # layers.append(nn.Linear(input_dim, hidden_dim))
        
        # layers.append(nn.ReLU())
        # # layers.append(nn.BatchNorm1d(hidden_dim))

        # for _ in range(num_layers - 2):
        #     layers.append(nn.Linear(hidden_dim, hidden_dim))
            
        #     layers.append(nn.ReLU())
        #     #layers.append(nn.BatchNorm1d(hidden_dim))

        # layers.append(nn.Linear(hidden_dim, num_classes))

        # self.deepmlp = nn.Sequential(*layers)

        # self.mlp = nn.Sequential(
        #   for _ in range(num_layers):
        #   nn.Linear(input_dim, hidden_dim),
        #   nn.ReLU(),
        #   nn.Linear(hidden_dim, num_classes)
        # )

        c = 3*h*w
        layers = []
        layers.append(torch.nn.Flatten())
        
        for _ in range(num_layers-1):
          layers.append(torch.nn.Linear(c, hidden_dim))
          layers.append(torch.nn.ReLU())
          c = hidden_dim

        layers.append(torch.nn.Linear(hidden_dim, num_classes))
        self.model = torch.nn.Sequential(*layers)

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        
        # x_flat = x.view(x.size(0), -1)
        
        # # Pass through our deep network
        # return self.deepmlp(x_flat)
        return self.model(x)

class MLPClassifierDeepResidual(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
        hidden_dim: int = 124,
        num_layers: int = 4,
    ):
        """
        Args:
            h: int, height of image
            w: int, width of image
            num_classes: int

        Hint - you can add more arguments to the constructor such as:
            hidden_dim: int, size of hidden layers
            num_layers: int, number of hidden layers
        """
        super().__init__()

        input_dim = 3 * h * w
        
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.input_bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()

        self.hidden_layers = nn.ModuleList()
        self.hidden_bns = nn.ModuleList()
        
        for _ in range(num_layers - 2):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.hidden_bns.append(nn.BatchNorm1d(hidden_dim))

        self.output_layer = nn.Linear(hidden_dim, num_classes)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        
        # x = self.input_norm(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.relu(self.input_bn(self.input_layer(x)))

        for layer, bn in zip(self.hidden_layers, self.hidden_bns):
            shortcut = x 
            residual = bn(layer(x))
            x = self.relu(residual + shortcut)

        logits = self.output_layer(x)
        
        return logits


model_factory = {
    "linear": LinearClassifier,
    "mlp": MLPClassifier,
    "mlp_deep": MLPClassifierDeep,
    "mlp_deep_residual": MLPClassifierDeepResidual,
}


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def save_model(model):
    """
    Use this function to save your model in train.py
    """
    for n, m in model_factory.items():
        if isinstance(model, m):
            return torch.save(model.state_dict(), Path(__file__).resolve().parent / f"{n}.th")
    raise ValueError(f"Model type '{str(type(model))}' not supported")


def load_model(model_name: str, with_weights: bool = False, **model_kwargs):
    """
    Called by the grader to load a pre-trained model by name
    """
    r = model_factory[model_name](**model_kwargs)
    if with_weights:
        model_path = Path(__file__).resolve().parent / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"
        try:
            r.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # Limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(r)
    if model_size_mb > 10:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")
    print(f"Model size: {model_size_mb:.2f} MB")

    return r
