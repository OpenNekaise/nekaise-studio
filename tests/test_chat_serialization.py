"""Serialization fixtures are not student learning results."""
import hashlib

import pytest

from nekaise_loop.config import CampaignConfig
from nekaise_loop.prompt_evidence import prompt_training_prefixes
from nekaise_loop.serialization import chat_training, student_prompt
from nekaise_loop.stages import _student_prompts, _tokenization
from nekaise_loop.training import prepare_dataset


class ChatTokenizer:
    """Distinct control tokens; generation-only prefill like MiniCPM's template."""
    eos_token_id = 1
    specials = {"<s>": 0, "</s>": 1, "<user>": 2, "<assistant>": 3,
                "<end>": 4, "<think>": 5, "</think>": 6}
    template = "fixture native template v1"
    add_bos = False
    rewrite = False
    bad_ending = False

    def get_chat_template(self):
        return self.template

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        assert tokenize is False
        rendered = "<s><user>" + messages[0]["content"] + "<end>\n"
        if add_generation_prompt:
            return rendered + "<assistant>" + ("<think>\n\n</think>\n\n" if kwargs.get("enable_thinking") is False else "")
        content = messages[1]["content"]
        if self.rewrite:
            content = content.strip()
        return rendered + "<assistant>" + content + ("BAD" if self.bad_ending else "<end>\n")

    def encode(self, text, add_special_tokens=True):
        ids = [0] if add_special_tokens and self.add_bos else []
        while text:
            for token, value in self.specials.items():
                if text.startswith(token):
                    ids.append(value)
                    text = text[len(token):]
                    break
            else:
                ids.append(ord(text[0]) + 100)
                text = text[1:]
        return ids

    def decode(self, ids, skip_special_tokens=False):
        inverse = {v: k for k, v in self.specials.items()}
        return ''.join(inverse[i] if i in inverse else chr(i - 100) for i in ids)


def row(response="  Correct answer.\n"):
    return {"id": "lesson", "stream": "sft", "text": response,
            "training_tokenization": "chat_response", "training_prompt": "Question?",
            "training_response": response}


def config(**updates):
    return CampaignConfig(max_seq_len=512, token_mix={"teacher": 1, "corpus": 0, "replay": 0}, **updates).model_dump()


@pytest.mark.parametrize("response", ["  Correct answer.\n", "", "Literal </think> text."])
def test_chat_prefix_response_and_native_end_are_exact(response):
    tokenizer = ChatTokenizer()
    source = row(response)
    prompt, special, evidence = student_prompt(tokenizer, source["training_prompt"], "chat_template")
    ids, frozen = chat_training(source, tokenizer, [1, 4])
    prefix = tokenizer.encode(prompt, add_special_tokens=special)
    assert ids == prefix + tokenizer.encode(response, add_special_tokens=False) + [4]
    assert ids.count(0) == 1 and ids[-1] == 4 and 1 not in ids
    assert frozen["text"] == prompt + response + "<end>"
    assert frozen["serialization"]["prompt_token_ids"] == prefix
    assert evidence["enable_thinking"] is False
    assert evidence["template_sha256"] == hashlib.sha256(tokenizer.template.encode()).hexdigest()
    assert "<think>\n\n</think>\n\n" in prompt


def test_freeze_keeps_serialization_and_actual_prefix_evidence():
    tokenizer = ChatTokenizer()
    dataset = prepare_dataset([row()], tokenizer, config(), [1, 4])
    frozen = dataset["rows"][0]
    lesson = {"id": "lesson", "student_prompt": "Question?",
              "effective_prompt": frozen["serialization"]["prompt_text"],
              "prompt_token_ids": frozen["serialization"]["prompt_token_ids"], "prompt_truncated": False}
    evidence = prompt_training_prefixes([lesson], dataset)["lessons"][0]
    assert evidence["text_prefix_matches"]
    assert evidence["sample_comparisons"][0]["prompt_is_sample_prefix"]
    assert dataset["samples"][0]["input_ids"][-1] == 4
    assert dataset["ledger"]["total_tokens"] == len(dataset["samples"][0]["input_ids"]) - 1


def test_raw_modes_keep_current_tokenizer_bos_policy_and_legacy_eos():
    tokenizer = ChatTokenizer()
    source = {"id": "old", "stream": "replay", "text": "raw example"}
    for bos in [False, True]:
        tokenizer.add_bos = bos
        dataset = prepare_dataset([source], tokenizer, CampaignConfig(token_mix={"teacher": 0, "corpus": 0, "replay": 1}).model_dump())
        assert dataset["rows"] == [source]
        assert dataset["samples"][0]["input_ids"] == tokenizer.encode(source["text"]) + [1]
        assert student_prompt(tokenizer, "raw", "raw_text", max_tokens=1)[:2] == ("raw", True)


def test_chat_prefix_text_uses_saved_rendering_when_decode_normalizes_whitespace():
    tokenizer = ChatTokenizer()
    dataset = prepare_dataset([row()], tokenizer, config(), [1, 4])
    serialization = dataset["rows"][0]["serialization"]
    lesson = {"id": "lesson", "student_prompt": "Question?",
              "effective_prompt": serialization["prompt_text"].replace("\n", " "),
              "prompt_serialization": {"prompt_text": serialization["prompt_text"]},
              "prompt_token_ids": serialization["prompt_token_ids"], "prompt_truncated": False}
    evidence = prompt_training_prefixes([lesson], dataset)["lessons"][0]
    assert evidence["text_prefix_matches"] is True
    assert evidence["sample_comparisons"][0]["prompt_is_sample_prefix"] is True


@pytest.mark.parametrize("bad_format", ["chat_template", "unsupported"])
def test_generation_entrypoint_preflights_later_rows_before_loading_weights(tmp_path, monkeypatch, capsys, bad_format):
    import json
    import sys
    from types import SimpleNamespace
    from nekaise_loop.workers.model import main

    class Tokenizer(ChatTokenizer):
        pad_token_id = 1

        def __call__(self, text, **kwargs):
            # Preflight retains CPU encodings; it must not move them or generate
            # the valid first answer before discovering the invalid second row.
            calls.append((text, kwargs))
            return object()

    calls = []
    loads = []

    def load_model(*args, **kwargs):
        loads.append(args)
        raise AssertionError("Weights loaded before all generation rows validated")

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        manual_seed=lambda seed: None, set_num_threads=lambda n: None,
        cuda=SimpleNamespace(is_available=lambda: False), float32="float32"))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: Tokenizer()),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model)))
    path = tmp_path / "generation.json"
    path.write_text(json.dumps({"checkpoint": "fixture", "config": config(student_format="chat_template"),
        "rows": [{"id": "valid", "prompt": "Question?"},
                 {"id": "bad-second-row", "prompt": "x" * 600, "format": bad_format}]}))
    monkeypatch.setattr(sys, "argv", ["model.py", "generate", str(path)])
    detail = "exceeding max_seq_len=512" if bad_format == "chat_template" else "Unknown student format"
    with pytest.raises(ValueError, match=f"Generation prompt row 'bad-second-row': .*{detail}"):
        main()
    assert len(calls) == 1 and calls[0][1]["truncation"] is False
    assert loads == []
    assert capsys.readouterr().out == ""


def test_replay_template_change_is_not_silently_reformatted():
    tokenizer = ChatTokenizer()
    source = {**row(), "training_template_sha256": "old-template-hash"}
    with pytest.raises(ValueError, match="Historical chat template changed"):
        chat_training(source, tokenizer, [1, 4])


@pytest.mark.parametrize("defect,message", [("rewrite", "rewrites"), ("bad_ending", "no effective generation EOS")])
def test_unsupported_template_fails_explicitly(defect, message):
    tokenizer = ChatTokenizer()
    setattr(tokenizer, defect, True)
    with pytest.raises(ValueError, match=message):
        chat_training(row(), tokenizer, [1, 4])


def test_chat_header_cannot_be_truncated():
    with pytest.raises(ValueError, match="exceeding max_seq_len"):
        student_prompt(ChatTokenizer(), "too long", "chat_template", max_tokens=2)


def test_only_exact_student_content_and_explicit_override_reach_provider():
    lessons = [{"id": "a", "student_prompt": "question", "reference": "HIDDEN", "rubric": ["SECRET"]},
               {"id": "b", "student_prompt": "raw diagnostic", "student_format": "raw_text"}]
    assert _student_prompts(lessons) == [{"id": "a", "prompt": "question"},
        {"id": "b", "prompt": "raw diagnostic", "format": "raw_text"}]
    lesson = {**row(), "student_prompt": "Question?", "prompt_serialization": {"format": "chat_template", "template_sha256": "hash"}}
    assert _tokenization(lesson)["training_template_sha256"] == "hash"
    assert CampaignConfig().student_format == "raw_text"


def test_chat_replay_uses_training_template_even_after_raw_diagnostic(setup_loop):
    from conftest import FakeModel, FakeTeacher
    from nekaise_loop.engine import Engine
    from nekaise_loop.teacher_tools import replay_lesson
    settings, service, campaign, _ = setup_loop

    class Teacher(FakeTeacher):
        def revise(self, lessons):
            return [{**r, "training_tokenization": "chat_response", "training_text": "",
                     "training_response": "Fixture assistant text"} for r in super().revise(lessons)]

        def evaluate(self, curriculum, lessons):
            return []

        def reflect(self, observations):
            return {"student_notes": "fixture", "next_round_instructions": "fixture", "action": "complete", "reason": "fixture"}

    class Model(FakeModel):
        def prepare(self, checkpoint, rows):
            return prepare_dataset(rows, ChatTokenizer(), self.config.model_dump(), [1, 4])

    Engine(settings, Teacher, Model).run(campaign["id"])
    assert service.store.campaign(campaign["id"])["status"] == "complete"
    rnd = service.store.one("SELECT id FROM rounds WHERE campaign_id=?", (campaign["id"],))
    replay = replay_lesson(settings.workspace, rnd["id"], "l1")
    assert replay.get("prompt_serialization", {}).get("format") != "chat_template"
    expected = hashlib.sha256(ChatTokenizer.template.encode()).hexdigest()
    assert replay["training_template_sha256"] == expected
    tokenization = _tokenization(replay)
    assert tokenization["origin_freeze_artifact"] == replay["origin_freeze_artifact"]
    changed = ChatTokenizer()
    changed.template = "different template"
    with pytest.raises(ValueError, match="Historical chat template changed"):
        chat_training({**row(), **tokenization}, changed, [1, 4])
