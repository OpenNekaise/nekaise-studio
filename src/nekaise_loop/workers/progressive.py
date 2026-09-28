"""CPU tokenizer subprocess for whole-passage progressive dataset preparation."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nekaise_loop.progressive_preparation import prepare_progressive
from nekaise_loop.serialization import checkpoint_eos_ids


def main():
    task, path = sys.argv[1:]
    if task != "progressive_prepare":
        raise ValueError("Unsupported progressive worker operation")
    from transformers import AutoTokenizer
    data = json.loads(Path(path).read_text())
    tokenizer = AutoTokenizer.from_pretrained(data["checkpoint"], local_files_only=True, trust_remote_code=False)
    if tokenizer.eos_token_id is None:
        raise ValueError("The student tokenizer must have an EOS token")
    result = prepare_progressive(data["rows"], tokenizer, data["config"], checkpoint_eos_ids(data["checkpoint"], tokenizer))
    print("LOOP " + json.dumps({"type": "result", "data": result}, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
