"""Numerical tests run in the ML environment, never import ML packages in HTTP."""
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from nekaise_loop.workers.batched_training import backward_update, collate, step_update


@pytest.mark.parametrize("size", [1, 2, 4, 6, 8])
def test_padded_causal_loss_and_gradients_match_serial_full_sequence(size):
    torch.manual_seed(11)
    torch.set_num_threads(1)
    model = transformers.LlamaForCausalLM(transformers.LlamaConfig(vocab_size=32, hidden_size=32,
        intermediate_size=48, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        attention_dropout=0., pad_token_id=1, use_cache=False))
    rows = [{"input_ids": ids} for ids in ([0, 5, 1], [0, 9, 8, 7, 1], [0, 4], [0, 2, 3, 4, 5, 1],
        [0, 6, 1], [0, 7, 9, 1], [0, 8, 9, 4, 1], [0, 3], [0, 4, 1])]
    inputs, count = collate(rows, 1, "cpu")
    assert count == 24
    assert int((inputs["labels"][:, 1:] != -100).sum()) == count
    assert inputs["labels"][0, 2].item() == 1  # EOS is trained despite pad == EOS
    assert inputs["labels"][0, 3].item() == -100
    expected = 0.
    for row in rows:
        ids = torch.tensor([row["input_ids"]])
        loss = model(input_ids=ids, labels=ids).loss * ((len(row["input_ids"])-1)/count)
        loss.backward()
        expected += loss.detach()
    grads = {n: p.grad.clone() for n, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    actual = backward_update(model, rows, 1, size, "cpu")
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    for name, p in model.named_parameters():
        torch.testing.assert_close(p.grad, grads[name], rtol=2e-5, atol=2e-7)


def test_nonfinite_update_never_changes_parameters_or_adam():
    from types import SimpleNamespace
    class Broken(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(3))
        def forward(self, input_ids, **kwargs):
            return SimpleNamespace(logits=self.weight.expand(*input_ids.shape, 3) * float("nan"))
    model = Broken()
    optimizer = torch.optim.AdamW(model.parameters())
    before = model.weight.detach().clone()
    with pytest.raises(ValueError, match="Non-finite"):
        step_update(model, optimizer, [{"input_ids": [0, 1, 2]}],
            {"learning_rate": 1e-5, "warmup_tokens": 0, "training_microbatch_size": 2}, 1, "cpu", 0)
    assert torch.equal(model.weight, before)
    assert not optimizer.state


def test_checkpoint_round_trip_preserves_adam_and_rejects_layout_change(tmp_path):
    import json
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from nekaise_loop.workers.batched_training import train, load_training
    base = tmp_path / "base"
    model = transformers.LlamaForCausalLM(transformers.LlamaConfig(vocab_size=32, hidden_size=32,
        intermediate_size=48, num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        attention_dropout=0., pad_token_id=1, eos_token_id=1, use_cache=False))
    model.save_pretrained(base)
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
        {"<bos>": 0, "<eos>": 1, "<unk>": 2}, unk_token="<unk>")), bos_token="<bos>", eos_token="<eos>", unk_token="<unk>", pad_token="<eos>")
    tokenizer.save_pretrained(base)
    config = {"seed": 42, "training_execution": "batched_v1", "training_microbatch_size": 2,
        "training_activation_checkpointing": False, "inherit_optimizer": False,
        "learning_rate": 2e-5, "tokens_per_update": 64, "max_seq_len": 64,
        "warmup_tokens": 0, "train_epochs": 1, "train_steps": 0}
    samples = [{"input_ids": [0, 3, 1], "stream": "corpus"}, {"input_ids": [0, 4, 5, 1], "stream": "teacher"}]
    output = tmp_path / "trained"
    payload = {"checkpoint": str(base), "dataset": {"samples": samples, "ledger": {"total_tokens": 5}},
        "config": config, "dataset_hash": "test-only", "output": str(output), "allow_cpu": True}
    train(payload)
    manifest = json.loads((output / "checkpoint.json").read_text())
    assert manifest["tokens"] == 5 and manifest["global_step"] == 1
    assert manifest["stream_tokens"] == {"corpus": 2, "teacher": 3}
    loaded = load_training({**payload, "checkpoint": str(output), "config": {**config, "inherit_optimizer": True}})
    assert loaded[-1]["optimizer_origin"] == "inherited"
    assert loaded[-1]["optimizer_transition"]["loaded_state_equal"] is True
    assert all(s["step"].device.type == "cpu" and s["step"].item() == 1 for s in loaded[2].state.values())
    manifest["parameter_layout_sha256"] = "wrong-order"
    (output / "checkpoint.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="ordering"):
        load_training({**payload, "checkpoint": str(output), "config": {**config, "inherit_optimizer": True}})
