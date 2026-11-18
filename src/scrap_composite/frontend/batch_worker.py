"""Keep a web worker's launch visible before the generation pipeline takes its lock."""
import json
import os
from pathlib import Path
import runpy
import sys
import time

# Works with both the packaged web command and the historical script entry point.
if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scrap_composite.data.files import FileLease, atomic_json
from scrap_composite.data.store import Store


def record_start_failure(store, out, ident, failure):
    run = store.last_run()
    if not run or run['id'] != ident or run['state'] != 'running':
        return
    detail = ' '.join(str(failure).splitlines())[:300]
    run.update(state='interrupted', error=f'{type(failure).__name__}: {detail}',
               remaining=store.remaining() if store.counts() else run.get('remaining'))
    store.record_run(run)
    path = out / 'summary.json'
    summary = json.loads(path.read_text(encoding='utf8')) if path.is_file() else {}
    # No work has been dispatched in this attempt; retain prior batch totals.
    summary.update(run=run, counts=store.counts(), usage_current_run=None)
    atomic_json(path, summary)
    atomic_json(out / f'run-{ident}.json', run)


def main():
    out, script, config = map(Path, sys.argv[1:4])
    ident = sys.argv[4]
    guard = FileLease(out / '.web-worker.lock')
    # A status query can briefly hold this lease while checking for an orphan.
    for _ in range(10):
        if guard.acquire():
            break
        time.sleep(.02)
    else:
        return 1
    try:
        # stdout and stdin are only the startup handshake. EOF means the parent
        # disappeared before dispatch; the worker must never start in that case.
        print('ready', flush=True)
        if sys.stdin.buffer.readline() != b'run\n':
            return 1
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())  # Native tools and Python share the batch log.
        os.environ['SCRAP_BATCH_RUN_ID'] = ident
        sys.path.insert(0, str(script.parent))
        sys.argv = [str(script), 'run', '--config', str(config)]
        try:
            runpy.run_path(str(script), run_name='__main__')
        except BaseException as exc:
            if not isinstance(exc, SystemExit) or exc.code not in (None, 0):
                store = Store(out / 'jobs.sqlite')
                try:
                    record_start_failure(store, out, ident, exc)
                finally:
                    store.close()
            raise
    finally:
        guard.release()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
