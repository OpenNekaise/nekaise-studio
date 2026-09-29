"""Application-side lifetime and request protocol. No ML imports in this module."""
import json
import os
import threading
import time

from .artifacts import atomic_write, canonical
from .config import ROOT
from .processes import Cancelled, ProcessRunner
from .storage import new_id, now


class ResidentTrainer:
    def __init__(self, engine, campaign_id):
        self.engine, self.campaign_id = engine, campaign_id
        self.thread = None
        self.stop = threading.Event()
        self.error = None
        self.receiver = None
        self.latest_checkpoint = None
        self.request_lock = threading.Lock()

    def start(self):
        if self.thread:
            return
        self.stop.clear()
        self.error = None
        self.id = new_id('trainer')
        self.directory = self.engine.settings.workspace/'trainers'/self.id
        self.directory.mkdir(parents=True)
        self.engine.store.execute("INSERT INTO trainer_sessions(id,campaign_id,status,created_at) VALUES(?,?,'running',?)",
                                  (self.id, self.campaign_id, now()))
        runner = ProcessRunner(self.engine.store, self.id, self.stop.is_set, table='trainer_sessions')
        path = self.directory/'resident.input.json'
        atomic_write(path, canonical({'session_id': self.id}))
        def receive(message):
            current = self.receiver
            if not current:
                return
            if message['type'] == 'resident_event' and message['data']['request_id'] == current[0]:
                current[1](message['data']['event'])
            elif message['type'] == 'resident_complete' and message['data']['request_id'] == current[0]:
                current[2].set()
        def serve():
            try:
                env = {**os.environ, 'HF_HUB_OFFLINE':'1', 'TRANSFORMERS_OFFLINE':'1',
                       'TOKENIZERS_PARALLELISM':'false', 'PYTHONDONTWRITEBYTECODE':'1'}
                runner.run([self.engine.settings.model_python, '-u', str(ROOT/'src/nekaise_loop/workers/resident.py'),
                    'resident', str(path)], cwd=self.directory, log=self.directory/'resident.log',
                    timeout=None, on_message=receive, env=env)  # Each request has its own deadline.
                if not self.stop.is_set():
                    self.error = RuntimeError('Resident trainer exited before session shutdown')
            except BaseException as exc:
                if not self.stop.is_set():
                    self.error = exc
            finally:
                self.engine.store.execute('UPDATE trainer_sessions SET status=?,finished_at=? WHERE id=?',
                                          ('failed' if self.error else 'closed', now(), self.id))
        self.thread = threading.Thread(target=serve, name='resident-trainer')
        self.thread.start()

    def request(self, local, task, payload, on_message):
        with self.request_lock:
            self.start()
            request_id = new_id('request')
            path = local.directory/(request_id+'.input.json')
            atomic_write(path, canonical(payload))
            completed = threading.Event()
            self.receiver = (request_id, on_message, completed)
            result_path = self.directory/(request_id+'.result.json')
            atomic_write(self.directory/'command.json', canonical({'id':request_id, 'task':task, 'input':str(path)}))
            timeout = (local.config.teaching_cycle.train_timeout_seconds
                       if task == 'train' and local.config.teaching_cycle else local.config.max_stage_seconds)
            deadline = time.monotonic()+timeout
            try:
                while not result_path.exists() or not completed.is_set():
                    if local.runner.cancelled():
                        raise Cancelled('Resident training cancelled by operator')
                    if self.error:
                        raise self.error
                    if time.monotonic() > deadline:
                        raise TimeoutError('Resident request exceeded the window execution budget')
                    time.sleep(.05)
                result = json.loads(result_path.read_text())
                if result['request_id'] != request_id:
                    raise ValueError('Resident response identity mismatch')
                if task == 'train':
                    self.latest_checkpoint = result['result']['checkpoint']
                return result['result']
            except BaseException:
                self.close()
                raise
            finally:
                self.receiver = None

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=15)
            if self.thread.is_alive():
                raise RuntimeError('Owned resident trainer did not stop within its cancellation bound')
            self.thread = None
        self.latest_checkpoint = None
