"""Explicit numerical compatibility for execution-only training changes; no ML imports."""
import hashlib
import ast
from pathlib import Path

from .training import recipe_hash, training_code_hash

# This exact serial implementation is the reviewed predecessor. Future edits do
# not become compatible merely because the optimizer tensors have matching shapes.
SERIAL_PREDECESSOR = "bec6927258b084448ca51167aaab6b7bd9fb61858e16810243576ba5acdf9cb2"
BATCHED_PREDECESSOR = "0bea6517cd6252d6d5b9eb42f74117dba35bb480053f4c690a58f654ffb1d868"
BATCHED_NUMERICAL_CORE = "dcb39da8dc255e517d8d1f9954ef372b0a93d8ad44b71e2dea1afe52d4ea7f77"
REVIEWED_CONTINUOUS_IMPLEMENTATION = "efb828f5121f4f50bd61b558a426f818cf614cd60e313839234ced15791df112"


def continuous_implementation_hash():
    root = Path(__file__).with_name('workers')
    return hashlib.sha256(b''.join((root/name).read_bytes() for name in
                          ('batched_training.py', 'resident.py', 'generation.py'))).hexdigest()


def numerical_core_hash():
    names = {'microbatches', 'collate', 'backward_update', 'step_update'}
    source = Path(__file__).with_name('workers').joinpath('batched_training.py').read_text()
    tree = ast.parse(source)
    # Source bytes are stable across the app's Python 3.13 and ML Python 3.12.
    # ast.dump defaults differ across those versions, despite identical code.
    value = '\n'.join(ast.get_source_segment(source, n) for n in tree.body
                      if isinstance(n, ast.FunctionDef) and n.name in names)
    return hashlib.sha256(value.encode()).hexdigest()


def runtime_hash(config):
    if config.get("training_execution", "serial_v1") == "serial_v1":
        return training_code_hash()
    root = Path(__file__).parent
    paths = [root / "training_runtime.py", root / "workers/batched_training.py"]
    if config.get("training_execution") == "resident_v1":
        paths.extend([root / "workers/resident.py", root / "workers/generation.py"])
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
    elif (config.get("training_execution") in {"batched_v1", "resident_v1"}
          and source == BATCHED_PREDECESSOR and training_code_hash() == SERIAL_PREDECESSOR
          and numerical_core_hash() == BATCHED_NUMERICAL_CORE
          and continuous_implementation_hash() == REVIEWED_CONTINUOUS_IMPLEMENTATION):
        policy = "batched_to_continuous_v1"
    else:
        raise ValueError("Unreviewed training runtime; compatible Adam migration is unavailable")
    return {"policy": policy, "from_training_code": source, "to_training_code": target,
            "recipe_hash": previous["recipe_hash"], "preserved": ["parameters", "Adam moments", "Adam steps", "global_step", "global_tokens"],
            "numerical_contract": "Same token-weighted causal objective and update boundaries; floating-point regrouping is not bitwise identity"}
