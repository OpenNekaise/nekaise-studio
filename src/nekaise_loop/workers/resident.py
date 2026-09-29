"""One worker-owned model/Adam instance; one complete durable window per request.

A crash discards the unsaved window. Only a verified checkpoint can be a parent;
there is no mid-window resume or second resident model.
"""
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nekaise_loop.artifacts import atomic_write, canonical
from nekaise_loop.training import recipe_hash
from nekaise_loop.training_runtime import runtime_hash
from nekaise_loop.workers.batched_training import load_training, train
from nekaise_loop.workers.generation import generate
from nekaise_loop.workers.model import emit


def compatible(previous, data):
    if data['checkpoint'] != previous['checkpoint']:
        raise ValueError('Resident parent differs from the last durable checkpoint')
    if recipe_hash(data['config']) != previous['recipe_hash']:
        raise ValueError('Resident Adam recipe changed')
    for key in ('training_microbatch_size', 'training_activation_checkpointing', 'training_execution'):
        if data['config'][key] != previous['config'][key]:
            raise ValueError('Resident execution configuration changed: '+key)
    if not data['config']['inherit_optimizer']:
        raise ValueError('Resident continuation cannot implicitly reset Adam')


def generate_from_state(data, state):
    import torch
    if state[3] == 'cuda':
        torch.cuda.empty_cache()
    padding_side = state[1].padding_side
    try:
        return generate(data, resident=(state[0], state[1], state[3]))
    finally:
        state[1].padding_side = padding_side


def main():
    task, path = sys.argv[1:]
    if task != 'resident':
        raise ValueError('Resident worker only accepts resident')
    directory = Path(path).parent
    command = directory/'command.json'
    seen, state, previous = None, None, None
    from nekaise_loop.workers import batched_training, generation
    def tagged(kind, data):
        emit('resident_event', {'request_id':seen, 'event':{'type':kind, 'data':data}})
    batched_training.emit = generation.emit = tagged
    # The coordinator owns cancellation. Parent death prevents an orphan GPU holder.
    import ctypes
    parent = os.getppid()
    ctypes.CDLL(None).prctl(1, 15)
    if os.getppid() != parent:
        return
    while not (directory/'close').exists():
        if not command.exists():
            time.sleep(.05)
            continue
        request = json.loads(command.read_text())
        if request['id'] == seen:
            time.sleep(.05)
            continue
        seen = request['id']
        data = json.loads(Path(request['input']).read_text())
        if request['task'] == 'train':
            load_started = time.monotonic()
            reused = state is not None
            if state is None:
                state = load_training(data)
            else:
                compatible(previous, data)
                state[4].update(optimizer_origin='inherited_resident', optimizer_transition={
                    'policy': 'same_resident_adam', 'from_training_code': runtime_hash(data['config']),
                    'to_training_code': runtime_hash(data['config']), 'recipe_hash': recipe_hash(data['config']),
                    'loaded_state_equal': True, 'basis': 'Same live parameter and Adam tensor objects, no reload or reset'})
            load_seconds = time.monotonic()-load_started
            result = train(data, state=state)
            result['execution_timing'].update(load_seconds=load_seconds, reused_resident=reused)
            previous = {**result, 'config': data['config'], 'recipe_hash': recipe_hash(data['config'])}
        elif request['task'] == 'generate':
            if state is None or data['checkpoint'] != previous['checkpoint']:
                raise ValueError('Resident evaluation requires its exact last saved checkpoint')
            result = generate_from_state(data, state)
        else:
            raise ValueError('Unsupported resident request')
        atomic_write(directory/(seen+'.result.json'), canonical({'request_id': seen, 'result': result}))
        # The FIFO acknowledgement is after all metrics/answers and the durable
        # result file. It prevents late events from leaking into the next stage.
        emit('resident_complete', {'request_id':seen})
        # Remove large per-window token data before idling; weights and Adam remain.
        del data, result


if __name__ == '__main__':
    main()
