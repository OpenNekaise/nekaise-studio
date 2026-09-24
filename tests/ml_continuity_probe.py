"""CPU-only tiny-model integration probe, never a live provider or quality result."""
import contextlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from nekaise_loop.workers.model import audit_generation, main
from nekaise_loop.material_portfolio import portfolio, completed_portfolio


def probe(root):
    torch.manual_seed(7)
    base = root/"base"
    model = GPT2LMHeadModel(GPT2Config(vocab_size=8, n_positions=64, n_embd=16, n_layer=1, n_head=2, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0, bos_token_id=1, eos_token_id=2))
    model.save_pretrained(base)
    from nekaise_loop.workers.scoring import score_sample
    model.eval()
    sample = {"sample_index": 0, "row_id": "fixture", "stream": "teacher", "input_ids": [1, 3, 4, 5, 2], "prompt_tokens": 3}
    parameters = [p.detach().clone() for p in model.parameters()]
    score = score_sample(model, sample, [2, 7], "cpu")
    tokens = torch.tensor([sample["input_ids"]])
    with torch.inference_mode():
        expected_loss = model(input_ids=tokens, labels=tokens).loss.item()
        expected_probs = model(input_ids=tokens).logits[0, 2].softmax(-1)
    assert abs(score["mean_nll"] - expected_loss) < 1e-6
    assert score["parts"]["prompt"]["tokens"] == score["parts"]["continuation"]["tokens"] == 2
    assert score["parts"]["unassigned"]["tokens"] == 0
    first_continuation = score["first_continuation"]
    assert first_continuation["position"] == 3 and first_continuation["target_id"] == 5
    assert abs(first_continuation["target_probability"] - expected_probs[5].item()) < 1e-6
    assert abs(first_continuation["eos_probability"] - expected_probs[[2, 7]].sum().item()) < 1e-6
    assert score["token_scores"][-1]["target_id"] == 2
    unknown = score_sample(model, {**sample, "prompt_tokens": None}, [2], "cpu")
    assert unknown["first_continuation"] is None and unknown["parts"]["unassigned"]["tokens"] == 4
    assert all(torch.equal(a, b) and b.grad is None for a, b in zip(parameters, model.parameters()))
    tok = Tokenizer(WordLevel({"<unk>":0,"<bos>":1,"<eos>":2,"a":3,"b":4,"c":5,"d":6,"e":7}, unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", bos_token="<bos>", eos_token="<eos>", pad_token="<eos>").save_pretrained(base)
    from nekaise_loop.workers.scoring import main as scoring_main
    before_files = {p.name: p.read_bytes() for p in base.iterdir()}
    score_input = root/"score.json"
    score_input.write_text(json.dumps({"checkpoint": str(base), "samples": [sample], "dataset_hash": "tiny-fixture"}))
    sys.argv = ["scoring.py", "score", str(score_input)]
    score_output = io.StringIO()
    with contextlib.redirect_stdout(score_output):
        scoring_main()
    scored = json.loads(next(line[5:] for line in score_output.getvalue().splitlines() if line.startswith("LOOP ")))["data"]
    assert scored["dataset_hash"] == "tiny-fixture" and scored["runtime"]["parameter_dtype"] == "torch.float32"
    assert list(scored["variants"]) == ["fp32"]
    assert scored["variants"]["fp32"][0]["input_ids"] == sample["input_ids"]
    assert {p.name: p.read_bytes() for p in base.iterdir()} == before_files
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
        frozen = {**dataset, "rows":[{"id":r["row_id"], "stream":r["stream"]} for r in dataset["samples"]]}
        accounting = completed_portfolio(portfolio(frozen, data["config"]), records[-1]["data"]["manifest"])
        assert accounting["status"] == "verified", accounting
        return records[-1]["data"], [r["data"] for r in records if r["type"] == "metric"]
    first, _ = train(base, "first")
    second, metrics = train(root/"first", "second")
    whole, _ = train(base, "whole", train_steps=4)
    automatic, _ = train(base, "automatic", train_steps=0, train_epochs=2, tokens_per_update=3)
    assert automatic["manifest"]["tokens"] == 8 and automatic["manifest"]["steps"] == 4
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
    assert len(answers[0]["generated_token_ids"]) == answers[0]["tokens"]
    assert len(answers[0]["prompt_token_ids"]) == answers[0]["prompt_tokens"]
    assert answers[0]["generation_audit"]["finite_logits"]
    assert answers[0]["generation_audit"]["mismatches"] == []
    # Force the first generated token in this tiny fixture. A secondary
    # checkpoint EOS must stop generation even though it differs from tokenizer EOS.
    for configured_eos, forced_token, expected_eos in [([2, 7], 7, [2, 7]), (2, 2, [2]), (None, 2, [2])]:
        model.generation_config.eos_token_id = configured_eos
        model.generation_config.forced_bos_token_id = forced_token
        model.generation_config.save_pretrained(base)
        generation.write_text(json.dumps({"config":{**config,"max_new_tokens":3},"checkpoint":str(base),"rows":[{"id":"stop","prompt":"a"}]}))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            main()
        answer = next(json.loads(line[5:])["data"] for line in output.getvalue().splitlines() if line.startswith('LOOP ') and json.loads(line[5:])["type"] == "answer")
        assert answer["generated_token_ids"] == [forced_token], answer
        assert answer["eos_token_ids"] == expected_eos
        assert answer["stop_reason"] == "eos"
        if forced_token == 2:
            assert answer["text"] == ""
            assert answer["raw_text"] == "<eos>"
        assert answer["generation_audit"]["tokens_checked"] == 1
        assert answer["generation_settings"]["forced_bos_token_id"] == forced_token
    # Same architecture family as the live student: changing prompt lengths and
    # repeating the first prompt must not carry another row's generation cache.
    llama_dir = root/"llama"
    llama = LlamaForCausalLM(LlamaConfig(vocab_size=8, hidden_size=16,
        intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, max_position_embeddings=64, bos_token_id=1,
        eos_token_id=2, pad_token_id=2))
    llama.save_pretrained(llama_dir)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(base)
    tokenizer.save_pretrained(llama_dir)
    generation.write_text(json.dumps({"config":{**config,"max_new_tokens":5},
        "checkpoint":str(llama_dir),"rows":[{"id":str(i),"prompt":p}
        for i,p in enumerate(["a b c", "d", "e d c b a", "a b c"])]}))
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        main()
    answers = [json.loads(line[5:])["data"] for line in output.getvalue().splitlines()
               if line.startswith('LOOP ') and json.loads(line[5:])["type"] == "answer"]
    assert answers[0]["generated_token_ids"] == answers[-1]["generated_token_ids"]
    for answer in answers:
        assert answer["generation_audit"]["tokens_checked"] == answer["tokens"]
        assert answer["generation_audit"]["mismatches"] == []
        assert answer["runtime"]["dtype"] == "torch.float32"
    # Verify the audit detects a token inconsistent with the model's raw logits.
    llama.eval()
    encoded = tokenizer("a b c", return_tensors="pt")
    with torch.inference_mode():
        predicted = llama(**encoded).logits[0,-1].argmax().item()
    incorrect = (predicted+1) % 8
    sequence = torch.cat([encoded.input_ids, torch.tensor([[incorrect]])], dim=1)
    audit = audit_generation(llama, encoded, sequence)
    assert audit["mismatches"][0]["generated_token_id"] == incorrect
    assert audit["mismatches"][0]["raw_argmax_token_id"] == predicted
    assert audit["mismatches"][0]["logit_gap"] > 0
    # Greedy argmax selects the first index on a tie; topk can order ties
    # differently. Matching tied predictions must not be reported as divergence.
    class TiedLogits:
        def __call__(self, input_ids, **kwargs):
            return SimpleNamespace(logits=torch.zeros((*input_ids.shape, 8)))
    for generated_id in [0, 7]:
        sequence = torch.cat([encoded.input_ids, torch.tensor([[generated_id]])], dim=1)
        audit = audit_generation(TiedLogits(), encoded, sequence)
        assert audit["raw_argmax_token_ids"] == [0]
        if generated_id == 0:
            assert audit["mismatches"] == []
        else:
            assert audit["mismatches"] == [{"position": 0,
                "generated_token_id": 7, "raw_argmax_token_id": 0, "logit_gap": 0.0}]
    # Exercise native formatting through the actual prepare/generate subprocess
    # entrypoint with tiny CPU fixture weights, distinct chat/tokenizer EOS, and a
    # prefill present only at generation time. This is not live student evidence.
    tokenizer.chat_template = "{{ bos_token }} a {{ messages[0]['content'] }} b {% if add_generation_prompt %}{% if enable_thinking is defined and not enable_thinking %}c {% endif %}{% else %}{{ messages[1]['content'] }} e\n{% endif %}"
    tokenizer.save_pretrained(base)
    model.generation_config.eos_token_id = [2, 7]
    model.generation_config.forced_bos_token_id = None
    model.generation_config.forced_eos_token_id = 7
    model.generation_config.save_pretrained(base)
    chat_config = {**config, "student_format": "chat_template", "max_new_tokens": 1,
                   "token_mix": {"teacher": 1, "corpus": 0, "replay": 0}}
    source = {"id": "chat", "stream": "sft", "text": "d", "training_tokenization": "chat_response",
              "training_prompt": "a", "training_response": "d"}
    generation.write_text(json.dumps({"config": chat_config, "checkpoint": str(base), "rows": [source]}))
    sys.argv = ["model.py", "prepare", str(generation)]
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        main()
    prepared = next(json.loads(line[5:])["data"] for line in output.getvalue().splitlines() if line.startswith("LOOP "))
    generation.write_text(json.dumps({"config": chat_config, "checkpoint": str(base),
                                     "rows": [{"id": "chat", "prompt": "a"}]}))
    sys.argv = ["model.py", "generate", str(generation)]
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        main()
    answer = next(json.loads(line[5:])["data"] for line in output.getvalue().splitlines()
                  if line.startswith("LOOP ") and json.loads(line[5:])["type"] == "answer")
    prefix = prepared["rows"][0]["serialization"]["prompt_token_ids"]
    assert answer["prompt_token_ids"] == prefix
    assert prefix.count(1) == 1 and prefix[-1] == 5
    assert prepared["samples"][0]["input_ids"] == prefix + [6, 7]
    assert answer["prompt_serialization"]["format"] == "chat_template"
    assert answer["prompt_serialization"]["prompt_text"] == prepared["rows"][0]["serialization"]["prompt_text"]
    assert answer["generated_token_ids"] == [7] and answer["stop_reason"] == "eos"
    print(json.dumps({"device":"cpu","global_steps":4,"max_parameter_difference":maximum,"weighted_loss_verified":True,"prompt_observations_verified":True,"generation_stopping_verified":True,"generation_audit_verified":True,"native_chat_verified":True}))


if __name__ == "__main__":
    probe(Path(sys.argv[1]))
