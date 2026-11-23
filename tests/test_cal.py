import pytest
import torch

from loka.models import BreakthroughCNN, BreakthroughResNet
from loka.training_methods import CAL_scores, CAL_select


def reference_scores(model, X_label, X_pool, k):
    """Straightforward O(L * P) version of the CAL score in the raw space."""
    model.eval()
    with torch.no_grad():
        pl, pp = model(X_label), model(X_pool)
        d = torch.cdist(X_pool, X_label)
        idx = d.topk(min(k, len(X_label)), largest=False).indices
        return (pl[idx] - pp[:, None]).abs().mean(dim=1)


@pytest.fixture
def data():
    g = torch.Generator().manual_seed(0)
    X_label = (torch.rand(40, 50, generator=g) > 0.5).float()
    X_pool = (torch.rand(300, 50, generator=g) > 0.5).float()
    return X_label, X_pool


def test_raw_scores_match_reference_and_chunking_is_irrelevant(data):
    torch.manual_seed(0)
    model = BreakthroughCNN()
    X_label, X_pool = data
    ref = reference_scores(model, X_label, X_pool, 5)
    assert torch.allclose(CAL_scores(model, X_label, X_pool, 5), ref, atol=1e-6)
    assert torch.allclose(CAL_scores(model, X_label, X_pool, 5, chunk=7, bs=13), ref, atol=1e-6)


def test_select_is_top_of_scores(data):
    torch.manual_seed(0)
    model = BreakthroughCNN()
    X_label, X_pool = data
    scores = CAL_scores(model, X_label, X_pool, 5)
    assert torch.equal(CAL_select(model, X_label, X_pool, 10, 5), scores.topk(10).indices)


@pytest.mark.parametrize("make", [BreakthroughCNN, BreakthroughResNet])
def test_penultimate_space_shapes_and_validation(make, data):
    torch.manual_seed(0)
    model = make()
    X_label, X_pool = data
    scores = CAL_scores(model, X_label, X_pool, 5, space="penultimate")
    assert scores.shape == (len(X_pool),) and torch.isfinite(scores).all() and (scores >= 0).all()
    with pytest.raises(ValueError):
        CAL_scores(model, X_label, X_pool, 5, space="nope")


def test_k_larger_than_labeled_set_is_clamped(data):
    model = BreakthroughCNN()
    X_label, X_pool = data
    assert CAL_scores(model, X_label[:3], X_pool, 20).shape == (len(X_pool),)
