import numpy as np
import pytest
import torch

from loka.data import load_breakthrough_states, save_breakthrough_states
from loka.models import BreakthroughCNN, BreakthroughResNet, board_features, get_breakthrough_cnn


def n_params(model):
    return sum(p.numel() for p in model.parameters())


@pytest.mark.parametrize("board_size, n_extra", [(5, 0), (5, 2), (6, 0), (6, 2)])
@pytest.mark.parametrize("make", [
    lambda n_extra, board_size: BreakthroughCNN(n_extra=n_extra, board_size=board_size),
    lambda n_extra, board_size: BreakthroughResNet(n_extra=n_extra, board_size=board_size),
])
def test_forward_shape(make, board_size, n_extra):
    model = make(n_extra, board_size).eval()
    x = torch.rand(7, board_features(board_size) + n_extra)
    assert model(x).shape == (7,)


def test_get_breakthrough_cnn_sizes_channels_from_x():
    X6 = torch.rand(4, 72)
    assert get_breakthrough_cnn(X6, board_size=6)(X6).shape == (4,)
    X5 = torch.rand(4, 52)   # default board_size=5, as before
    assert get_breakthrough_cnn(X5)(X5).shape == (4,)


def test_5x5_parameter_counts_unchanged():
    assert n_params(BreakthroughCNN()) == 224_161
    assert n_params(BreakthroughResNet()) == 224_167


def test_resnet_hidden_119_matches_cnn_on_6x6():
    cnn = n_params(BreakthroughCNN(board_size=6))
    res = n_params(BreakthroughResNet(board_size=6, hidden=119))
    assert cnn == 314_273
    assert abs(res / cnn - 1) < 0.001


def test_6x6_states_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    cells = rng.integers(0, 3, size=(20, 36))            # 0 empty, 1 white, 2 black
    X = np.concatenate([cells == 1, cells == 2], axis=1).astype(np.uint8)
    Y = rng.choice([-1, 1], size=20).astype(np.int8)
    path = tmp_path / "states.txt"
    save_breakthrough_states(path, X, Y)
    X2, Y2 = load_breakthrough_states(path)
    assert X2.shape == (20, 72)
    assert np.array_equal(X, X2) and np.array_equal(Y, Y2)
    X3, _ = load_breakthrough_states(path, two_channel=True)
    assert X3.shape == (20, 2, 6, 6)
