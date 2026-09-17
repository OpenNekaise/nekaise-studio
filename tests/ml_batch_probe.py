"""Tiny CPU-only execution checks, never live provider or learning measurements."""
import contextlib
import io
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast, LogitsProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from nekaise_loop.workers.generation import main
from nekaise_loop.workers.model import main as legacy_main


def probe(root):
    torch.manual_seed(81)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=8, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=128, bos_token_id=1, eos_token_id=2, pad_token_id=2))
    base = root/"tiny"
    model.save_pretrained(base)
    tok = Tokenizer(WordLevel({"<unk>": 0, "<bos>": 1, "<eos>": 2, "a": 3, "b": 4, "c": 5, "d": 6, "e": 7}, unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", bos_token="<bos>", eos_token="<eos>", pad_token="<eos>")
    tokenizer.model_input_names = ["input_ids", "attention_mask"]
    tokenizer.chat_template = "{{ bos_token }} a {{ messages[0]['content'] }} b {% if add_generation_prompt %}c {% else %}{{ messages[1]['content'] }} e{% endif %}"
    tokenizer.save_pretrained(base)
    config = {"seed": 81, "max_seq_len": 32, "max_new_tokens": 5,
              "generation_batch_size": 4, "generation_batch_tokens": 256}
    rows = [{"id": str(i), "prompt": prompt} for i, prompt in enumerate(["a b c", "d", "e d c b a", "a b c"])]

    def run(rows=rows, *, groups=None, entrypoint=main, **updates):
        payload = {"checkpoint": str(base), "rows": rows, "config": {**config, **updates}}
        if groups is not None:
            payload["batch_groups"] = groups
        path = root/"generate.json"
        path.write_text(json.dumps(payload))
        sys.argv = ["generation.py", "generate", str(path)]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            entrypoint()
        events = [json.loads(line[5:]) for line in output.getvalue().splitlines() if line.startswith("LOOP ")]
        return next(e["data"] for e in events if e["type"] == "result"), events

    batched, events = run()
    serial, _ = run(generation_batch_size=1)
    legacy, _ = run(entrypoint=legacy_main)
    assert any(e["type"] == "generation_plan" for e in events)
    assert len({a["generation_batch"]["index"] for a in batched}) == 1
    for a, b, c in zip(batched, serial, legacy):
        assert a["prompt_token_ids"] == b["prompt_token_ids"] == c["prompt_token_ids"]
        assert a["generated_token_ids"] == b["generated_token_ids"] == c["generated_token_ids"]
        assert a["generation_audit"]["finite_logits"]
        assert a["generation_audit"]["mismatches"] == []
        assert a["tokens"] == a["generation_audit"]["tokens_checked"]
        assert a["runtime"]["device"] == "cpu" and a["runtime"]["dtype"] == "torch.float32"
    grouped, _ = run(groups=[["3", "1"], ["0", "2"]])
    assert [a["id"] for a in grouped] == [r["id"] for r in rows]
    assert grouped[3]["generation_batch"]["row_ids"] == ["3", "1"]
    assert grouped[0]["generation_batch"]["row_ids"] == ["0", "2"]
    chat, _ = run(student_format="chat_template")
    chat_serial, _ = run(student_format="chat_template", generation_batch_size=1)
    for a, b in zip(chat, chat_serial):
        assert a["generated_token_ids"] == b["generated_token_ids"]
        assert a["prompt_token_ids"] == b["prompt_token_ids"]
        assert a["prompt_serialization"]["format"] == "chat_template"
        assert not a["generation_audit"]["mismatches"]

    # A real generate loop with a fixture logits processor makes four rows end
    # at distinct steps, including one that reaches its completion budget.
    controlled_rows = [{"id": str(i), "prompt": p} for i, p in enumerate(["a", "a b", "d", "a b c"])]
    original_generate = model.generate
    class EndAt(LogitsProcessor):
        def __init__(self, width, endings, terminator):
            self.width, self.endings, self.terminator = width, endings, terminator
        def __call__(self, input_ids, scores):
            scores.fill_(-float("inf"))
            step = input_ids.shape[1] - self.width + 1
            for index, end in enumerate(self.endings):
                scores[index, self.terminator if step == end else 3] = 0
            return scores
    for terminator in (2, 7):
        model.generation_config.eos_token_id = [2, 7]
        def controlled(**kwargs):
            last = kwargs["input_ids"][:, -1].tolist()
            endings = [{3: 1, 4: 2, 5: 3, 6: 99}[token] for token in last]
            kwargs["logits_processor"] = [EndAt(kwargs["input_ids"].shape[1], endings, terminator)]
            return original_generate(**kwargs)
        with patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model), patch.object(model, "generate", side_effect=controlled):
            answers, _ = run(controlled_rows)
        assert [a["tokens"] for a in answers] == [1, 2, 5, 3]
        assert [a["stop_reason"] for a in answers] == ["eos", "eos", "max_new_tokens", "eos"]
        for answer, size in zip(answers, [1, 2, 5, 3]):
            expected = [3]*(size-1)+[terminator] if size != 5 else [3]*5
            assert answer["generated_token_ids"] == expected
            assert answer["generation_audit"]["tokens_checked"] == size
    with patch("transformers.AutoModelForCausalLM.from_pretrained", side_effect=AssertionError("Weights must not load during invalid preflight")):
        for kwargs in ({"generation_batch_tokens": 6}, {"groups": [["0", "0"], ["1", "2", "3"]]}):
            try:
                run(**kwargs)
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid preflight was accepted")
        try:
            run([rows[0], rows[0]])
        except ValueError:
            pass
        else:
            raise AssertionError("Duplicate IDs accepted")
    print(json.dumps({"device": "cpu", "mixed_lengths": True, "native_chat": True,
                      "early_eos": True, "preflight": True, "result_order": True}))


if __name__ == "__main__":
    probe(Path(sys.argv[1]))
