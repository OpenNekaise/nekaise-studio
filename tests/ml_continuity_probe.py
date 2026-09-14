"""CPU-only tiny-model integration probe, never a live provider or quality result."""
import contextlib
import io
import json
from pathlib import Path
import sys

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from nekaise_loop.workers.model import main


def probe(root):
    torch.manual_seed(7)
    base = root/"base"
    model = GPT2LMHeadModel(GPT2Config(vocab_size=8, n_positions=64, n_embd=16, n_layer=1, n_head=2, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0, bos_token_id=1, eos_token_id=2))
    model.save_pretrained(base)
    tok = Tokenizer(WordLevel({"<unk>":0,"<bos>":1,"<eos>":2,"a":3,"b":4,"c":5,"d":6,"e":7}, unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", bos_token="<bos>", eos_token="<eos>", pad_token="<eos>").save_pretrained(base)
    config = {"seed":7,"learning_rate":1e-4,"warmup_tokens":32,"tokens_per_update":64,"max_seq_len":64,"train_steps":2,"train_epochs":1,"inherit_optimizer":True}
    dataset = {"samples":[{"row_id":"one","stream":"teacher","input_ids":[1,3,4,5,2]}],"ledger":{"total_tokens":4}}
    def train(parent, name, **updates):
        data = {"config":{**config, **updates},"checkpoint":str(parent),"dataset":dataset,"dataset_hash":"tiny-cpu-fixture","output":str(root/name),"allow_cpu":True}
        path = root/f"{name}.json"
        path.write_text(json.dumps(data))
        sys.argv = ["model.py","train",str(path)]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            main()
        records = [json.loads(line[5:]) for line in output.getvalue().splitlines() if line.startswith("LOOP ")]
        return records[-1]["data"], [r["data"] for r in records if r["type"] == "metric"]
    first, _ = train(base, "first")
    second, metrics = train(root/"first", "second")
    whole, _ = train(base, "whole", train_steps=4)
    assert first["manifest"]["optimizer_origin"] == "initialized"
    assert second["manifest"]["optimizer_origin"] == "inherited"
    assert second["manifest"]["global_step"] == whole["manifest"]["global_step"] == 4
    assert second["manifest"]["global_tokens"] == 16
    assert metrics[0]["learning_rate"] == config["learning_rate"]*12/32
    a = GPT2LMHeadModel.from_pretrained(root/"second")
    b = GPT2LMHeadModel.from_pretrained(root/"whole")
    maximum = max((x-y).abs().max().item() for x,y in zip(a.parameters(), b.parameters()))
    assert maximum < 1e-7, maximum
    assert all(p.dtype == torch.float32 for p in a.parameters())
    state = torch.load(root/"second/training_state.pt", weights_only=True)
    assert state["optimizer"]["state"]
    try:
        train(root/"second", "incompatible", learning_rate=2e-4)
    except ValueError as error:
        assert "recipe/code changed" in str(error)
    else:
        raise AssertionError("Recipe change was silently accepted")
    reset, _ = train(root/"second", "reset", learning_rate=2e-4, inherit_optimizer=False)
    assert reset["manifest"]["optimizer_origin"] == "explicit_reset"
    assert reset["manifest"]["global_step"] == 2
    # Unequal examples: the reported batch loss is weighted by target tokens.
    dataset["samples"] = [{"row_id":"short","stream":"teacher","input_ids":[1,3]}, {"row_id":"long","stream":"corpus","input_ids":[1,4,5,2]}]
    model.eval()
    with torch.no_grad():
        expected = sum(model(input_ids=(ids := torch.tensor([r["input_ids"]])), labels=ids).loss.item()*(len(r["input_ids"])-1)/4 for r in dataset["samples"])
    weighted, steps = train(base, "weighted", train_steps=1)
    assert abs(steps[0]["loss"]-expected) < 1e-5
    assert weighted["manifest"]["stream_tokens"] == {"teacher":1,"corpus":3}
    generation = root/"generate.json"
    generation.write_text(json.dumps({"config":{**config,"max_seq_len":16,"max_new_tokens":2},"checkpoint":str(base),"rows":[{"id":"context","prompt":"a "*80}]}))
    sys.argv = ["model.py","generate",str(generation)]
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        main()
    answers = [json.loads(line[5:])["data"] for line in output.getvalue().splitlines() if line.startswith('LOOP ') and json.loads(line[5:])["type"] == "answer"]
    assert answers[0]["prompt_truncated"] is True
    assert answers[0]["prompt_tokens"] == 16
    assert answers[0]["effective_prompt"]
    print(json.dumps({"device":"cpu","global_steps":4,"max_parameter_difference":maximum,"weighted_loss_verified":True,"prompt_observations_verified":True}))


if __name__ == "__main__":
    probe(Path(sys.argv[1]))
