"""Explicit numerical compatibility for execution-only training changes; no ML imports."""
import hashlib
from pathlib import Path

from .training import recipe_hash, training_code_hash

# This exact serial implementation is the reviewed predecessor. Future edits do
# not become compatible merely because the optimizer tensors have matching shapes.
SERIAL_PREDECESSOR = "bec6927258b084448ca51167aaab6b7bd9fb61858e16810243576ba5acdf9cb2"


def runtime_hash(config):
    if config.get("training_execution", "serial_v1") == "serial_v1":
        return training_code_hash()
    root = Path(__file__).parent
    paths = [root / "training_runtime.py", root / "workers/batched_training.py"]
    return hashlib.sha256(training_code_hash().encode() + b"".join(p.read_bytes() for p in paths)).hexdigest()


def optimizer_transition(previous, config):
    """Return an auditable bridge or reject; never reset or rewrite saved state."""
    if previous.get("recipe_hash") != recipe_hash(config):
        raise ValueError("Optimizer recipe differs; execution-only migration cannot change the learning recipe")
    source, target = previous.get("training_code"), runtime_hash(config)
    if source == target:
        policy = "identical_runtime"
    elif (config.get("training_execution") == "batched_v1"
          and source == SERIAL_PREDECESSOR and training_code_hash() == SERIAL_PREDECESSOR):
        policy = "serial_to_batched_v1"
    else:
        raise ValueError("Unreviewed training runtime; compatible Adam migration is unavailable")
    return {"policy": policy, "from_training_code": source, "to_training_code": target,
            "recipe_hash": previous["recipe_hash"], "preserved": ["parameters", "Adam moments", "Adam steps", "global_step", "global_tokens"],
            "numerical_contract": "Same token-weighted causal objective and update boundaries; floating-point regrouping is not bitwise identity"}
