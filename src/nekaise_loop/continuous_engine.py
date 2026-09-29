"""Bounded cross-cycle preparation, resident GPU consumer, asynchronous Teacher review.

Every dataset is still an immutable saved window. Coverage commits only after its
verified checkpoint. At most two teaching cycles await review and at most two future
windows prepare. A failed producer is drained before the exact failure is handed off.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import queue
import threading

from . import cycle_stages
from .config import CampaignConfig, STAGES
from .cycle_engine import execute, PREPARE, PauseCycle
from .cycle_store import open_cycle, cycle_for, bind_block
from .engine import Context
from .failures import TeacherUnavailable
from .processes import Cancelled
from .resident_training import ResidentTrainer
from .storage import now


class StoragePressure(RuntimeError):
    pass


def run(engine, campaign_id, controls, pause):
    store = engine.store
    campaign = store.campaign(campaign_id)
    config = CampaignConfig.model_validate(campaign['config'])
    stop, producer_stop, drain, review_failed = (threading.Event() for _ in range(4))
    ready = queue.Queue()
    capacity = threading.Semaphore(config.teaching_cycle.prefetch_blocks)
    cycles_pending = threading.Semaphore(2)
    failures, reviews = [], []
    engine.resident_trainer = ResidentTrainer(engine, campaign_id) if config.training_execution == 'resident_v1' else None
    store.set_status(campaign_id, 'running')

    def check():
        if controls():
            stop.set()
            raise Cancelled('Stopped by operator')
        if pause():
            stop.set()
            raise PauseCycle('Paused by operator')
        return False

    def acquire(semaphore):
        while not semaphore.acquire(timeout=.1):
            if stop.is_set() or producer_stop.is_set():
                return False
        return not stop.is_set() and not producer_stop.is_set()

    def cancelled_prepare():
        return stop.is_set() or producer_stop.is_set()

    def produce():
        previous = None
        try:
            while not cancelled_prepare():
                if previous and drain.is_set():
                    break
                if not acquire(cycles_pending):
                    break
                if previous and drain.is_set():
                    cycles_pending.release()
                    break
                cycle = open_cycle(engine, campaign, after_cycle=previous)
                if cycle is None:
                    cycles_pending.release()
                    break
                first = store.one('SELECT round_id FROM teaching_blocks WHERE cycle_id=? AND position=0', (cycle['id'],))['round_id']
                execute(engine, campaign, first, 'cycle_research', cancelled_prepare)
                execute(engine, campaign, first, 'cycle_plan', cancelled_prepare)
                blocks = store.query('SELECT r.* FROM rounds r JOIN teaching_blocks b ON b.round_id=r.id WHERE b.cycle_id=? ORDER BY b.position', (cycle['id'],))
                for block in blocks:
                    if block['status'] == 'complete':
                        continue
                    if not acquire(capacity):
                        return
                    for stage in PREPARE:
                        execute(engine, campaign, block['id'], stage, cancelled_prepare)
                    if not store.one("SELECT id FROM stage_runs WHERE round_id=? AND stage='train' AND status='complete'", (block['id'],)):
                        store.execute("UPDATE rounds SET status='prepared',updated_at=? WHERE id=?", (now(), block['id']))
                    ready.put(('block', block['id']))
                ready.put(('cycle', (cycle, blocks[-1]['id'])))
                previous = cycle
                if len(blocks) == 1:
                    planned = engine.artifacts.get(store.one('SELECT plan_artifact FROM teaching_cycles WHERE id=?', (cycle['id'],))['plan_artifact'])
                    if planned['blocks'][0]['curriculum']['train_epochs'] == 0:
                        # Diagnostic windows need their actual review before planning
                        # against a forward frontier they intentionally did not advance.
                        while not cancelled_prepare():
                            saved = store.one('SELECT status,review_artifact FROM teaching_cycles WHERE id=?', (cycle['id'],))
                            if saved['status'] == 'complete':
                                break
                            stop.wait(.1)
                        previous = None  # Reopen at the unchanged committed frontier.
        except BaseException as exc:
            if not isinstance(exc, Cancelled) or not cancelled_prepare():
                failures.append(exc)
        finally:
            ready.put(('end', None))

    def finish_review(cycle, round_id):
        try:
            if review_failed.is_set():
                return None  # Saved answers remain owed after the earlier review is repaired.
            result = None
            for stage in ('grade', 'adapt'):
                result = execute(engine, campaign, round_id, stage, stop.is_set)
            if result.get('review_pending'):
                raise ValueError('Continuous cycle requires an actual Teacher review')
            store.execute("UPDATE rounds SET status='complete',updated_at=? WHERE id=?", (now(), round_id))
            store.execute("UPDATE teaching_cycles SET status='complete',review_artifact=?,updated_at=? WHERE id=?",
                          (engine.artifacts.put(result), now(), cycle['id']))
            store.event(campaign_id, round_id, 'round_complete', 'Training window saved and Teacher cycle reviewed')
            if result['action'] != 'continue':
                producer_stop.set()
            return result
        except BaseException as exc:
            review_failed.set()
            failures.append(exc)
            producer_stop.set()
            raise
        finally:
            cycles_pending.release()

    producer = threading.Thread(target=produce, name='continuous-preparation')
    teacher = ThreadPoolExecutor(max_workers=1, thread_name_prefix='online-review')
    producer.start()
    decision = None
    try:
        while True:
            check()
            for review in reviews:
                if review.done() and not review.cancelled() and review.exception() is None:
                    result = review.result()
                    if result and result['action'] != 'continue':
                        decision = result
                        break
            if decision:
                break
            try:
                kind, value = ready.get(timeout=.1)
            except queue.Empty:
                continue
            if kind == 'end':
                break
            if kind == 'cycle':
                cycle, last_id = value
                reviews.append(teacher.submit(finish_review, cycle, last_id))
                if config.auto_recover and config.manage_history:
                    from .history import review_due
                    if review_due(store):
                        # Finish already authorized production and saves before a
                        # quiescent orchestrator review; never cancel paid work for
                        # routine maintenance or run a repair beside this source reader.
                        drain.set()
                continue
            round_id = value
            cycle = cycle_for(store, round_id)
            if not store.one("SELECT id FROM stage_runs WHERE round_id=? AND stage='train' AND status='complete'", (round_id,)):
                row = store.one('SELECT * FROM rounds WHERE id=?', (round_id,))
                row['stage'] = 'train'
                ctx = Context(engine, campaign, row, None, 0, check)
                bind_block(ctx, cycle, ctx.output('freeze'))
            capacity.release()
            from .checkpoint_retention import storage_status
            parent = store.one('SELECT model_before FROM rounds WHERE id=?', (round_id,))['model_before']
            storage = storage_status(engine.settings.workspace, parent)
            if storage['pressure']:
                raise StoragePressure(f"Insufficient checkpoint headroom: {storage['free_bytes']} bytes free; orchestrator must inspect storage")
            for stage in ('train', 'evaluate', 'answer'):
                execute(engine, campaign, round_id, stage, check)
            count = len(engine.artifacts.get(cycle['plan_artifact'])['blocks'])
            if cycle['position'] != count-1:
                for stage in ('grade', 'adapt'):
                    execute(engine, campaign, round_id, stage, check)
                store.execute("UPDATE rounds SET status='complete',updated_at=? WHERE id=?", (now(), round_id))
                store.event(campaign_id, round_id, 'round_complete', 'Training window saved; cycle review pending')
        # A producer failure never discards an already saved window. Wait for
        # reviews of actual answers; then report the original exact failed stage.
        for review in reviews:
            while not review.done():
                check()
                stop.wait(.1)
            if review.exception() is None and review.result() and review.result()['action'] != 'continue':
                decision = review.result()
        if failures:
            raise failures[0]
        if decision and decision['action'] == 'pause':
            from .service import Service
            Service(engine.settings).action(campaign_id, 'pause', spawn=False, actor='teacher', reason=decision['reason'])
        elif drain.is_set() and not decision:
            store.recover(campaign_id, 'history_review', 'Quiescent review after continuous preparation drained and verified windows saved')
        else:
            store.set_status(campaign_id, 'complete')
    except (Cancelled, PauseCycle) as exc:
        store.set_status(campaign_id, 'paused' if isinstance(exc, PauseCycle) else 'stopped')
    except BaseException as exc:
        if controls() or pause():
            store.set_status(campaign_id, 'paused' if pause() else 'stopped')
        elif isinstance(exc, Exception):
            unavailable = isinstance(exc, TeacherUnavailable)
            if config.auto_recover:
                retry_at = None
                if unavailable and exc.kind not in {'budget', 'material_budget'}:
                    delay = exc.retry_seconds or (60 if exc.kind == 'rate_limit' else config.teacher_retry_seconds)
                    retry_at = (datetime.now(timezone.utc)+timedelta(seconds=delay)).isoformat(timespec='milliseconds')
                store.recover(campaign_id, exc.kind if unavailable else 'storage_pressure' if isinstance(exc, StoragePressure) else 'failure', exc,
                    retry_at=retry_at, failed_stage_id=getattr(exc, 'cycle_stage_id', None))
            else:
                store.set_status(campaign_id, 'waiting' if unavailable else 'failed', str(exc)[-3000:])
            store.event(campaign_id, None, 'error', str(exc)[-1500:])
        else:
            raise
    finally:
        stop.set()
        producer_stop.set()
        producer.join()
        teacher.shutdown(wait=True, cancel_futures=True)
        if engine.resident_trainer:
            engine.resident_trainer.close()
        engine.resident_trainer = None
