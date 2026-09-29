"""Small derived preparation views; immutable full datasets remain the execution input."""


def save(store, artifacts, round_id, key, frozen):
    from .learning_work import prepared_coverage
    from .material_accounting import author_yield
    summary = {k:v for k,v in frozen.items() if k not in {'rows','samples'}}
    summary['prepared_target_coverage'] = prepared_coverage(frozen)
    summary['author_yield'] = author_yield(store, artifacts, round_id, frozen)
    summary['full_dataset_artifact'] = key
    artifact = artifacts.put(summary)
    store.execute('INSERT OR REPLACE INTO preparation_summaries VALUES(?,?,?)', (round_id,key,artifact))


def read(store, artifacts, key):
    row = store.one('SELECT artifact FROM preparation_summaries WHERE freeze_artifact=?', (key,))
    return artifacts.get(row['artifact'] if row else key)
