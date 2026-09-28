# Physical training batches

`training_execution=batched_v1` is an opt-in execution path for CoAPT Mid-training.
Historical campaigns default to `serial_v1`. The Teacher still owns material, passes,
learning rate and tokens per optimizer update. Physical batching does not increase
the effective update batch or change the curriculum.

`training_microbatch_size` limits independent sequences in one forward/backward pass.
Consecutive rows from each existing token-exact update are grouped, right padded and
given separate attention masks. There is no cross-document concatenation. Only added
padding labels are ignored; real EOS tokens remain targets, including when EOS is also
the padding ID. The existing full-sequence CPT/SFT objective remains unchanged.
FP32 summed causal cross-entropy is divided by the complete update's actual target
count, including short final updates. Counters exclude padding.

`training_activation_checkpointing` controls non-reentrant activation recomputation.
Disabling it can trade additional memory for speed. Batch size and checkpointing are
frozen in configuration; there is no automatic OOM fallback. Failure goes through the
ordinary orchestrator. FP32 parameters and Adam moments, BF16 autocast, SDPA, clipping,
learning-rate schedule, token traversal and checkpoint frequency remain explicit.
Non-capturable Adam step counters load on CPU, avoiding one device-to-host scalar
synchronization per parameter inside each optimizer update. Their values are retained;
moments still load onto the model's device. The serial diagnostic preserves the old
CUDA placement so that its timing represents the prior implementation.

## Compatible optimizer continuation

The legacy trainer files remain byte-identical. The new runtime hashes its implementation
and the existing tokenization/update code. An explicit `serial_to_batched_v1` bridge
accepts only the reviewed legacy fingerprint and an unchanged optimizer recipe.
Every inherited load requires the predecessor's validated torch/transformers versions,
matching checkpoint/state metadata, parameter grouping and FP32 moment shapes. It checks exact
equality of every loaded Adam step and moment and records the named parameter-layout
hash. Subsequent saves carry the new runtime fingerprint and transition evidence; later
batched loads require the saved parameter-name ordering to match as well. A switch back
to the serial runtime cannot silently reset an inherited batched optimizer.

Numerical regrouping is not bitwise equivalence. CPU objective/gradient tests and a
real-model BF16 comparison establish the intended numerical contract. Unknown runtime
or recipe changes cannot silently reset Adam when entering the batched path. Source
changes require a fresh continuation after validation; no saved checkpoint is rewritten.

## Reproducible profiling

Stop automatic execution through the command queue, then run:

```bash
.venv/bin/nekaise-loop profile-training ROUND_ID --mixed --equivalence-only
.venv/bin/nekaise-loop profile-training ROUND_ID --warmup 8 --steps 32
.venv/bin/nekaise-loop profile-training ROUND_ID --mixed --warmup 8 --steps 32
```

The standalone diagnostic worker shares the source lock and takes the same exclusive
workspace worker lock as campaign execution. It records subprocess PID/start identity,
honors termination and pending operator commands, and verifies a completed checkpoint
and its frozen dataset. Each case reloads pristine weights and Adam, updates only its
in-memory copy, and never saves those weights. Original checkpoint bytes verify again
after profiling. Cases and failures remain in `workspace/profiles/` with an isolated
SQLite journal; they do not enter teaching records, coverage or learning metrics.

The sweep compares the original serial loop with physical batches 1, 2 and 4, each
with activation checkpointing on and off. It reports actual valid targets per second,
warmup/measurement counts, peak allocated and reserved memory, library/runtime provenance
and optimizer continuity. The raw-only case selects existing corpus rows; `--mixed`
uses all frozen rows. Samples may repeat during a timing experiment and receive zero
official exposure credit. Equivalence checks use uneven real sequences, comparing
loss and all parameter gradients against a reimplementation of the preserved serial
BF16 loop, including repeated serial controls and a complete optimizer update. This
reference is a diagnostic implementation, not a saved official training result.

ML numerical tests can run separately in an environment containing torch, transformers
and pytest. Normal application tests never require torch in the API environment.

Only one GPU is currently installed. The trainer does not implement DDP; no multi-GPU
speedup is claimed. A future distributed implementation must preserve update normalization,
data accounting, optimizer compatibility and owned-process cancellation before activation.
