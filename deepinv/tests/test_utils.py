import deepinv
import torch
import pytest
from deepinv.utils.decorators import _deprecated_alias
import warnings


@pytest.fixture
def tensorlist():
    x = torch.ones((1, 1, 2, 2))
    y = torch.ones((1, 1, 2, 2))
    x = deepinv.utils.TensorList([x, x])
    y = deepinv.utils.TensorList([y, y])
    return x, y


def test_tensordict_sum(tensorlist):
    x, y = tensorlist
    z = torch.ones((1, 1, 2, 2)) * 2
    z1 = deepinv.utils.TensorList([z, z])
    z = x + y
    assert (z1[0] == z[0]).all() and (z1[1] == z[1]).all()
    z = y + x
    assert (z1[0] == z[0]).all() and (z1[1] == z[1]).all()


def test_tensordict_mul(tensorlist):
    x, y = tensorlist
    alpha = 1.0
    z = torch.ones((1, 1, 2, 2))
    z1 = deepinv.utils.TensorList([z, z])
    z = x * alpha
    assert (z1[0] == z[0]).all() and (z1[1] == z[1]).all()
    z = alpha * x
    assert (z1[0] == z[0]).all() and (z1[1] == z[1]).all()


def test_tensordict_scalar_mul(tensorlist):
    x, y = tensorlist
    z = torch.ones((1, 1, 2, 2))
    z1 = deepinv.utils.TensorList([z, z])
    z = x * y
    assert (z1[0] == z[0]).all() and (z1[1] == z[1]).all()


def test_tensordict_div(tensorlist):
    x, y = tensorlist
    z = torch.ones((1, 1, 2, 2))
    z1 = deepinv.utils.TensorList([z, z])
    z = x / y
    assert (z1[0] == z[0]).all() and (z1[1] == z[1]).all()


def test_tensordict_sub(tensorlist):
    x, y = tensorlist
    z = torch.zeros((1, 1, 2, 2))
    z1 = deepinv.utils.TensorList([z, z])
    z = x - y
    assert (z1[0] == z[0]).all() and (z1[1] == z[1]).all()


def test_tensordict_neg(tensorlist):
    x, y = tensorlist
    z = -torch.ones((1, 1, 2, 2))
    z1 = deepinv.utils.TensorList([z, z])
    z = -x
    assert (z1[0] == z[0]).all() and (z1[1] == z[1]).all()


def test_tensordict_append(tensorlist):
    x, y = tensorlist
    z = torch.ones((1, 1, 2, 2))
    z1 = deepinv.utils.TensorList([z, z, z, z])
    z = x.append(y)
    assert (z1[0] == z[0]).all() and (z1[-1] == z[-1]).all()


def test_plot():
    for c in range(1, 5):
        x = torch.ones((1, c, 2, 2))
        titles, imgs = ["a", "b"], [x, x]
        deepinv.utils.plot(imgs, titles=titles, show=False)
        deepinv.utils.plot(x, titles="a", show=False)
        deepinv.utils.plot(imgs, show=False)
        deepinv.utils.plot({k: v for k, v in zip(titles, imgs)}, show=False)


def test_plot_inset():
    # Plots a batch of images with a checkboard pattern, with different inset locations
    x = torch.ones(2, 1, 100, 100)

    for i in range(0, 100, 10):
        x[:, :, :, i : i + 5] = 0
        x[:, :, i : i + 5, :] = 0

    deepinv.utils.plot_inset(
        [x],
        titles=["a"],
        labels=["a"],
        inset_loc=((0, 0.5), (0.5, 0.5)),
        show=False,
        save_fn="temp.png",
    )


def test_plot_videos():
    x = torch.rand((1, 3, 5, 8, 8))  # B,C,T,H,W image sequence
    y = torch.rand((1, 3, 5, 16, 16))
    deepinv.utils.plot_videos(
        [x, y], display=True
    )  # this should generate warning without IPython installed
    deepinv.utils.plot_videos([x, y], save_fn="vid.gif")


def test_save_videos():
    x = torch.rand((1, 3, 5, 8, 8))  # B,C,T,H,W image sequence
    y = torch.rand((1, 3, 5, 16, 16))
    deepinv.utils.save_videos([x, y], time_dim=2, save_fn="vid.gif")


def test_plot_ortho3D():
    for c in range(1, 5):
        x = torch.ones((1, c, 2, 2, 2))
        imgs = [x, x]
        deepinv.utils.plot_ortho3D(imgs, titles=["a", "b"], show=False)
        deepinv.utils.plot_ortho3D(x, titles="a", show=False)
        deepinv.utils.plot_ortho3D(imgs, show=False)


# -------------- Test deprecated_alias --------------
class DummyModule(torch.nn.Module):
    @_deprecated_alias(old_lr="lr")
    def __init__(self, lr=0.1):
        super().__init__()
        self.lr = lr


@_deprecated_alias(old_arg="new_arg")
def dummy_function(new_arg=0.1):
    return new_arg**2


def test_deprecated_alias():
    # --- Class (torch.nn.Module) tests ---
    with pytest.warns(DeprecationWarning, match="old_lr.*deprecated"):
        m1 = DummyModule(old_lr=0.01)
        assert m1.lr == 0.01

    m2 = DummyModule(lr=0.02)
    assert m2.lr == 0.02

    with pytest.raises(TypeError, match="Cannot specify both 'old_lr' and 'lr'"):
        DummyModule(old_lr=0.01, lr=0.02)

    # Test no warning with correct parameter
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        DummyModule(lr=0.3)
        assert len(record) == 0

    # --- Function tests ---
    with pytest.warns(DeprecationWarning, match="old_arg.*deprecated"):
        result1 = dummy_function(old_arg=0.1)
        assert result1 == 0.1**2

    result2 = dummy_function(new_arg=0.2)
    assert result2 == 0.2**2
    with pytest.raises(TypeError, match="Cannot specify both 'old_arg' and 'new_arg'"):
        dummy_function(old_arg=0.1, new_arg=0.2)
    # Test no warning with correct parameter
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        dummy_function(new_arg=0.3)
        assert len(record) == 0
