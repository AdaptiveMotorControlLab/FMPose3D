import pytest
import torch
import torch.optim as optim
from fmpose.models import Model


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


@pytest.fixture
def optimizer(model):
    return optim.Adam(model.parameters(), lr=1e-3)


@pytest.fixture
def sample_batch(device):
    """Create a sample batch for training.
    
    Shapes based on FMPose_main.py train function:
    - input_2D: (B, F, J, 2) - 2D pose input
    - gt_3D: (B, F, J, 3) - 3D pose ground truth
    """
    batch_size = 4
    frames = 1
    joints = 17
    
    input_2D = torch.randn(batch_size, frames, joints, 2, device=device)
    gt_3D = torch.randn(batch_size, frames, joints, 3, device=device)
    
    return input_2D, gt_3D


def test_training_step_forward(model, sample_batch, device):
    """Test that a single forward pass works correctly."""
    input_2D, gt_3D = sample_batch
    B, F, J, _ = input_2D.shape
    
    # Create noise and interpolated sample (CFM training)
    x0 = torch.randn(B, F, J, 3, device=device)
    t = torch.rand(B, 1, 1, 1, device=device)
    y_t = (1.0 - t) * x0 + t * gt_3D
    
    # Forward pass
    v_pred = model(input_2D, y_t, t)
    
    # Check output shape
    assert v_pred.shape == (B, F, J, 3), f"Expected shape {(B, F, J, 3)}, got {v_pred.shape}"


def test_training_step_backward(model, optimizer, sample_batch, device):
    """Test that backward pass and gradient computation works."""
    input_2D, gt_3D = sample_batch
    B, F, J, _ = input_2D.shape
    
    # Create noise and interpolated sample
    x0 = torch.randn(B, F, J, 3, device=device)
    t = torch.rand(B, 1, 1, 1, device=device)
    y_t = (1.0 - t) * x0 + t * gt_3D
    v_target = gt_3D - x0
    
    # Forward pass
    v_pred = model(input_2D, y_t, t)
    
    # Compute loss (MSE)
    loss = ((v_pred - v_target) ** 2).mean()
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Check that gradients exist
    has_grad = False
    for param in model.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
    
    assert has_grad, "Model should have non-zero gradients after backward pass"


def test_training_step_parameter_update(model, optimizer, sample_batch, device):
    """Test that optimizer updates model parameters."""
    input_2D, gt_3D = sample_batch
    B, F, J, _ = input_2D.shape
    
    # Store initial parameter values
    initial_params = {name: param.clone() for name, param in model.named_parameters()}
    
    # Create noise and interpolated sample
    x0 = torch.randn(B, F, J, 3, device=device)
    t = torch.rand(B, 1, 1, 1, device=device)
    y_t = (1.0 - t) * x0 + t * gt_3D
    v_target = gt_3D - x0
    
    # Forward pass
    v_pred = model(input_2D, y_t, t)
    loss = ((v_pred - v_target) ** 2).mean()
    
    # Backward and update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Check that at least some parameters changed
    params_changed = False
    for name, param in model.named_parameters():
        if not torch.allclose(param, initial_params[name]):
            params_changed = True
            break
    
    assert params_changed, "Parameters should change after optimizer step"


def test_loss_decreases_over_steps(model, optimizer, sample_batch, device):
    """Test that loss decreases over multiple training steps."""
    input_2D, gt_3D = sample_batch
    B, F, J, _ = input_2D.shape
    
    losses = []
    num_steps = 10
    
    for _ in range(num_steps):
        # Create noise and interpolated sample
        x0 = torch.randn(B, F, J, 3, device=device)
        t = torch.rand(B, 1, 1, 1, device=device)
        y_t = (1.0 - t) * x0 + t * gt_3D
        v_target = gt_3D - x0
        
        # Forward pass
        v_pred = model(input_2D, y_t, t)
        loss = ((v_pred - v_target) ** 2).mean()
        losses.append(loss.item())
        
        # Backward and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Check that loss generally decreases (last < first)
    # Note: May not always be strictly decreasing due to stochasticity
    assert losses[-1] < losses[0], f"Loss should decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"


def test_model_train_eval_modes(model, sample_batch, device):
    """Test that model can switch between train and eval modes."""
    input_2D, gt_3D = sample_batch
    B, F, J, _ = input_2D.shape
    
    x0 = torch.randn(B, F, J, 3, device=device)
    t = torch.rand(B, 1, 1, 1, device=device)
    y_t = (1.0 - t) * x0 + t * gt_3D
    
    # Test train mode
    model.train()
    assert model.training, "Model should be in training mode"
    v_pred_train = model(input_2D, y_t, t)
    assert v_pred_train.shape == (B, F, J, 3)
    
    # Test eval mode
    model.eval()
    assert not model.training, "Model should be in eval mode"
    with torch.no_grad():
        v_pred_eval = model(input_2D, y_t, t)
    assert v_pred_eval.shape == (B, F, J, 3)