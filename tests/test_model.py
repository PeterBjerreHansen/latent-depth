import torch
from torch.nn import functional as F

from nanogpt import GPT, GPTConfig


def test_model_shapes_and_hidden_states():
    cfg = GPTConfig(vocab_size=32, block_size=7, n_layer=3, n_head=4, n_embd=64)
    model = GPT(cfg).eval()
    x = torch.randint(0, 32, (5, 7))
    targets = torch.randint(0, 32, (5, 7))
    logits, loss, hidden, final = model(x, targets=targets, return_hidden=True)

    assert logits.shape == (5, 7, 32)
    assert loss is not None
    assert len(hidden) == 4  # embedding stream + post-block streams
    assert all(h.shape == (5, 7, 64) for h in hidden)
    assert final.shape == (5, 7, 64)
    assert model.transformer.wte.weight is model.lm_head.weight


def test_inference_only_forward_returns_last_position():
    cfg = GPTConfig(vocab_size=16, block_size=6, n_layer=2, n_head=4, n_embd=32)
    model = GPT(cfg).eval()
    x = torch.randint(0, 16, (3, 6))

    logits, loss = model(x)

    assert logits.shape == (3, 1, 16)
    assert loss is None


def test_shifted_targets_use_full_sequence_context():
    cfg = GPTConfig(vocab_size=16, block_size=6, n_layer=2, n_head=4, n_embd=32)
    model = GPT(cfg).eval()
    x = torch.randint(0, 16, (3, 6))
    targets = x[:, 1:]

    logits, loss = model(x, targets=targets)

    assert logits.shape == (3, 6, 16)
    assert loss is not None
    assert torch.isfinite(loss)


def test_shifted_loss_matches_independent_cross_entropy_reference():
    torch.manual_seed(11)
    cfg = GPTConfig(vocab_size=13, block_size=5, n_layer=1, n_head=1, n_embd=16)
    model = GPT(cfg).eval()
    x = torch.tensor([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]])
    logits, loss = model(x, targets=x[:, 1:])

    expected = F.cross_entropy(
        logits[:, :-1].reshape(-1, cfg.vocab_size),
        x[:, 1:].reshape(-1),
    )
    assert loss is not None
    torch.testing.assert_close(loss, expected, atol=0.0, rtol=0.0)


def test_attention_is_causal():
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=16, block_size=6, n_layer=2, n_head=4, n_embd=32)
    model = GPT(cfg).eval()

    a = torch.tensor([[1, 2, 3, 4, 5, 6]])
    b = torch.tensor([[1, 2, 3, 9, 10, 11]])
    logits_a, _ = model(a, targets=a)
    logits_b, _ = model(b, targets=b)

    # Positions 0,1,2 cannot depend on positions 3,4,5.
    torch.testing.assert_close(logits_a[:, :3], logits_b[:, :3], atol=1e-7, rtol=1e-6)
