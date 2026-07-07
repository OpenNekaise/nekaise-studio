"""campaign runner — the spec-to-command mapping and metric extraction it relies on."""
from conftest import REPO, load_module

cp = load_module(REPO / "tools" / "campaign.py")


def test_eval_run_command_flags():
    cmd = cp.run_command({"kind": "eval", "checkpoint": "m", "split": "test",
                          "limit": 8, "batch_size": 4, "milestone": True, "exp": "e"})
    s = " ".join(cmd)
    assert "--checkpoint m" in s and "--split test" in s
    assert "--limit 8" in s and "--batch-size 4" in s
    assert "--milestone" in s and "--exp e" in s


def test_train_run_command_uses_recipe_path():
    cmd = cp.run_command({"kind": "train", "recipe": "experiments/x/train.py"})
    assert cmd[1].endswith("experiments/x/train.py")


def test_metric_regex_matches_both_line_styles():
    assert cp.METRIC_RE.search("METRIC nekaise_bench_dev=0.1453 stage=x").group(1) == "0.1453"
    assert cp.METRIC_RE.search("BENCH[m@dev] nekaise_bench=0.1606 n=523").group(1) == "0.1606"
    assert cp.METRIC_RE.search("no metric here") is None
