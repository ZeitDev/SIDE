import torch
from utils.loader import load

def test_load_optimizer():
    """
    Tests if the loader can correctly load a class from the torch.optim module.
    """
    optimizer_class = load("torch.optim.AdamW")
    assert optimizer_class == torch.optim.AdamW

def test_load_custom_model():
    """
    Tests if the loader can correctly load a custom model from the project.
    """
    combiner_class = load("models.combiner.Combiner")
    assert hasattr(combiner_class, 'forward')

def test_load_with_kwargs():
    """
    Tests if the loader can instantiate a class with constructor arguments.
    """
    double_conv = load("models.simple_unet.DoubleConv", in_ch=3, out_ch=64)
    assert isinstance(double_conv, torch.nn.Module)