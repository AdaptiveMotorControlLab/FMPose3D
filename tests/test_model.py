import pytest
import torch
from fmpose3d.models import Model

class Args:
    """Mock args for model configuration."""
    channel = 512
    d_hid = 1024
    token_dim = 256
    layers = 5
    n_joints = 17

@pytest.fixture
def device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def model_args():
    return Args()

@pytest.fixture
def model(model_args, device):
    return Model(model_args).to(device)


def test_model_instantiation(model):
    """Test that the model can be instantiated."""
    assert model is not None

def test_model_forward_shape(model, device):
    """Test that the model output has the correct shape."""
    batch_size = 1
    x = torch.randn(batch_size, 17, 17, 2, device=device)
    y_t = torch.randn(batch_size, 17, 17, 3, device=device)
    t = torch.randn(batch_size, 1, 1, 1, device=device)

    output = model(x, y_t, t)

    assert output.shape == (batch_size, 17, 17, 3), f"Expected shape {(batch_size, 17, 17, 3)}, got {output.shape}"


def test_model_forward_different_batch_sizes(model, device):
    """Test that the model works with different batch sizes."""
    for batch_size in [1, 4, 8, 16]:
        x = torch.randn(batch_size, 17, 17, 2, device=device)
        y_t = torch.randn(batch_size, 17, 17, 3, device=device)
        t = torch.randn(batch_size, 1, 1, 1, device=device)

        output = model(x, y_t, t)

        assert output.shape == (batch_size, 17, 17, 3), f"Batch size {batch_size}: Expected shape {(batch_size, 17, 17, 3)}, got {output.shape}"