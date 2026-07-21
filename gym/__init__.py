"""gym — tasks + verifiers + runner. The project's single definition of correct.

Hard boundary (R1): gym never imports studio/lib/packs/tools/experiments — dependency is
one-way (studio → gym). Training rewards, SFT filters, and evaluation all import the SAME
verifier functions from here; TRL wrappers and eval CLIs are thin adapters on the studio
side. Enforced by tests/test_gym_isolation.py.

Terminology: the batch generation/eval orchestration here is the *runner* (gym/runner/).
The word "harness" is reserved for the agent runtime (opennekaise / nekaise-edge).
"""
