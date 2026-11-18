"""Offline integration tests for the batch worker, its ledger and the restart API."""
import asyncio
from contextlib import closing
import json
from pathlib import Path
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

from PIL import Image
from starlette.applications import Starlette
from starlette.testclient import TestClient

from scrap_composite.data.files import atomic_json, file_hash
from scrap_composite.data.snapshot import save_snapshot, snapshot
from scrap_composite.data.store import Store
from scrap_composite.frontend import batches
from scrap_composite.generation import cli
from scrap_composite.generation.pipeline import Pipeline, make_jobs
from scrap_composite.generation.schema import BatchConfig

BATCH = '1234567890abcdef'
ROUTE = '/api/batches/' + BATCH
FAULT = 'test fixture: interrupted while reviewing item 3'


def read_json(path):
    return json.loads(path.read_text(encoding='utf8'))


async def offline_job(pipe, job):
    """Replace only the expensive single-item work; exercise real queue and accounting."""
    number = job['ordinal'] + 1
    with (pipe.out / 'attempts.jsonl').open('a', encoding='utf8') as stream:
        stream.write(json.dumps({'number': number, 'checkpoint': job['checkpoint']}) + '\n')
    # One simulated, already-metered request per item, persisted before the fault.
    if not job['checkpoint'].get('metered'):
        pipe.vision.calls.append({'seconds': 0, 'usage': {'total_tokens': 11, 'prompt_tokens': 8, 'completion_tokens': 3}})
        atomic_json(pipe.vision.usage_path, pipe.vision.usage())
        pipe.store.checkpoint(job['id'], 'review', {'metered': True})
    fault = pipe.out / 'fault.txt'
    mode = fault.read_text() if fault.exists() else None
    if mode == 'concurrent' and number == 2:
        pipe.third_item_started = asyncio.Event()
        await pipe.third_item_started.wait()
    if number == 3 and mode:
        if mode == 'concurrent':
            pipe.third_item_started.set()
        if mode == 'wait':
            (pipe.out / 'ready').touch()
            await asyncio.Event().wait()  # Parent kills this exact test worker.
        raise RuntimeError(FAULT)
    result = {'job_id': job['id'], 'asset_id': job['payload']['asset_id'], 'seconds': 0, 'reason': 'fixture'}
    if number == 2:
        return {**result, 'status': 'failed'}
    image = pipe.out / f'{job["id"]}.png'
    Image.new('RGB', (4, 4), (number, 0, 0)).save(image)
    status = 'accepted_display' if pipe.config.count == 1 else 'review'
    return {**result, 'status': status, 'image': str(image)}


def fixture_catalog(root):
    mesh = root / 'fixture.glb'
    return {'3': {'mesh': str(mesh), 'mesh_hash': file_hash(mesh), 'available': True, 'human_review': 'pending'}}


def fixture_cli(config_path):
    """Invoked inside the child interpreter in place of the production worker script."""
    gate = config_path.parent / 'hold-before-pipeline'
    if gate.exists():  # The web worker has acknowledged launch, but no pipeline lease exists yet.
        (config_path.parent / 'entered-worker').touch()
        deadline = time.monotonic() + 45
        while gate.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError('Timed out waiting to initialize the test pipeline')
            time.sleep(.05)
    catalog = fixture_catalog(config_path.parent)
    with patch('scrap_composite.generation.pipeline.load_catalog', return_value=catalog), patch.object(Pipeline, 'job', offline_job):
        cli.main()


def web_request(root, action):
    with patch.object(batches, 'DATA', root), patch.object(batches, 'HERE', root), TestClient(Starlette(routes=batches.routes)) as client:
        response = client.get(ROUTE) if action == 'status' else client.post(ROUTE + '/resume')
        if action == 'resume' and response.status_code == 202:
            code = batches.PROCESSES[BATCH].wait(timeout=60)
            if code:
                raise AssertionError((root / (BATCH + '.log')).read_text(encoding='utf8'))
        return {'code': response.status_code, 'body': response.json()}


class BatchRecoveryTests(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.out = self.root / 'output'
        self.out.mkdir()
        self.config = self.root / (BATCH + '.json')
        # A minimal GLB JSON chunk is enough for the real catalog preflight.
        gltf = b'{"asset":{"version":"2.0"}} '
        mesh = self.root / 'fixture.glb'
        mesh.write_bytes(struct.pack('<4sIIII', b'glTF', 2, 20 + len(gltf), len(gltf), 0x4E4F534A) + gltf)
        Image.new('RGB', (8, 8)).save(self.root / 'frame.jpg')
        provider = self.root / 'provider.json'
        atomic_json(provider, {'vision': {'url': 'http://127.0.0.1/chat/completions', 'model': 'offline-fixture'}})
        config = BatchConfig(backgrounds=str(self.root), output=str(self.out), cache=str(self.root / 'cache'),
                             provider_config=str(provider), assets=['3'], count=4, offline=True, cpu_concurrency=1)
        atomic_json(self.config, config.model_dump())
        self.worker = self.root / 'batch_display.py'
        self.worker.write_text(
            f'import sys\nsys.path.insert(0, {str(Path(__file__).resolve().parent)!r})\n'
            f'sys.path.insert(0, {str(Path(__file__).resolve().parents[1] / "src")!r})\n'
            'from pathlib import Path\nfrom test_batch_recovery import fixture_cli\n'
            'fixture_cli(Path(sys.argv[sys.argv.index("--config") + 1]))\n', encoding='utf8')

    def fresh_web(self, action):
        # Each request has a new interpreter, ASGI application and empty Popen registry.
        result = subprocess.run([sys.executable, __file__, 'web', str(self.root), action],
                                capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def rows(self):
        with closing(Store(self.out / 'jobs.sqlite')) as store:
            return store.rows()

    def wait_for_marker(self, path, process):
        deadline = time.monotonic() + 45
        while not path.exists():
            if process.poll() is not None:
                self.fail(process.stderr.read())
            if time.monotonic() >= deadline:
                self.fail('Test worker did not reach ' + path.name)
            time.sleep(.05)

    def run_worker(self, mode=None):
        if mode:
            (self.out / 'fault.txt').write_text(mode)
        return subprocess.Popen([sys.executable, str(self.worker), 'run', '--config', str(self.config)],
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)

    def test_legacy_complete_receipt_is_corrected_only_when_jobs_are_unfinished(self):
        config = read_json(self.config)
        for remaining in (1, 0):
            with self.subTest(remaining=remaining):
                out = self.out / f'legacy-{remaining}'
                atomic_json(self.config, {**config, 'output': str(out)})
                legacy_run = {'id': 'legacy-run', 'state': 'complete', 'error': None}
                atomic_json(out / 'summary.json', {'run': legacy_run, 'counts': {'failed': 2}})
                summary_hash = file_hash(out / 'summary.json')
                with closing(Store(out / 'jobs.sqlite')) as store:
                    store.initialize('legacy', [{'asset_id': '3', 'seed': n} for n in range(2)])
                    for _ in range(2 - remaining):
                        job = store.claim()
                        store.finish(job['id'], 'failed', {'job_id':job['id'], 'asset_id':'3', 'status':'failed'})
                    store.db.execute('DROP TABLE runs')  # Historical schema, no per-run ledger.

                info = self.fresh_web('status')['body']
                self.assertFalse(info['running'])
                self.assertTrue(info['legacy_runtime_required'])
                self.assertFalse(info['resumable'])
                self.assertEqual(info['counts'].get('queued', 0), remaining)
                if remaining:
                    self.assertEqual((info['run']['state'], info['run']['remaining']), ('interrupted', 1))
                    self.assertIn('原运行未记录具体原因', info['run']['error'])
                    self.assertEqual(read_json(out / 'run-legacy-run.json'), info['run'])
                    self.assertEqual(read_json(out / 'summary.json')['run'], info['run'])
                else:
                    self.assertEqual(info['run'], legacy_run)
                    self.assertEqual(file_hash(out / 'summary.json'), summary_hash)
                    self.assertFalse((out / 'run-legacy-run.json').exists())

    def test_one_item_batch_finishes_without_interruption_and_refuses_resume(self):
        atomic_json(self.config, {**read_json(self.config), 'count': 1})
        with self.run_worker() as worker:
            _, stderr = worker.communicate(timeout=60)
            self.assertEqual(worker.returncode, 0, stderr)
        info = self.fresh_web('status')['body']
        self.assertEqual(info['counts'], {'accepted_display': 1})
        self.assertEqual((info['run']['state'], info['run']['remaining']), ('complete', 0))
        self.assertIsNone(info['run']['error'])
        self.assertIsNone(info.get('previous_interruption'))
        self.assertFalse(info['resumable'])
        self.assertIn(self.fresh_web('resume')['code'], (400, 409))

    def test_exception_survives_web_restart_and_only_remaining_jobs_run(self):
        with self.run_worker('raise') as worker:
            _, stderr = worker.communicate(timeout=60)
            self.assertNotEqual(worker.returncode, 0)
            self.assertIn(FAULT, stderr)

        rows = self.rows()
        self.assertEqual([r['state'] for r in rows[:2]], ['review', 'failed'])
        before = read_json(self.out / 'summary.json')
        info = self.fresh_web('status')
        self.assertEqual(info['code'], 200)
        info = info['body']
        self.assertTrue(info['resumable'])
        self.assertFalse(info['running'])
        run = info['run']
        self.assertEqual(run['id'], before['run']['id'])
        self.assertEqual((run['state'], run['job_id'], run['stage'], run['remaining']),
                         ('interrupted', rows[2]['id'], 'review', 2))
        self.assertIn(FAULT, run['error'])
        self.assertEqual(before['run']['error'], run['error'])
        self.assertEqual(before['cost_total']['actual_requests'], 3)
        self.assert_resume(rows, run)

    def assert_resume(self, rows, previous_run):
        first_image = Path(rows[0]['result']['image'])
        digest = file_hash(first_image)
        (self.out / 'fault.txt').unlink()
        resumed = self.fresh_web('resume')
        self.assertEqual(resumed['code'], 202, resumed)
        finished = self.fresh_web('status')['body']
        self.assertEqual(finished['run']['state'], 'complete')
        self.assertNotEqual(finished['run']['id'], previous_run['id'])
        self.assertEqual(finished['previous_interruption']['id'], previous_run['id'])
        self.assertEqual(finished['previous_interruption']['error'], previous_run['error'])
        self.assertEqual(read_json(self.out / ('run-' + previous_run['id'] + '.json')), previous_run)
        self.assertEqual(finished['run']['remaining'], 0)
        self.assertEqual(finished['counts'], {'review': 3, 'failed': 1})
        self.assertFalse(finished['resumable'])
        self.assertFalse(finished['running'])
        self.assertEqual(finished['generated_pngs'], 3)
        self.assertEqual(finished['cost_total']['actual_requests'], 4)
        self.assertEqual(finished['cost_total']['reported_tokens'], 44)
        self.assertEqual(read_json(self.out / 'summary.json')['run'], finished['run'])
        self.assertEqual(file_hash(first_image), digest)
        self.assertEqual(self.rows()[:2], rows[:2])
        attempts = [json.loads(line) for line in (self.out / 'attempts.jsonl').read_text().splitlines()]
        self.assertEqual([a['number'] for a in attempts], [1, 2, 3, 3, 4])
        self.assertEqual(attempts[3]['checkpoint'], {'metered': True})
        self.assertIn(self.fresh_web('resume')['code'], (400, 409))

    def test_abrupt_worker_death_keeps_last_checkpoint_and_active_poll_does_not_interrupt(self):
        with self.run_worker('wait') as worker:
            try:
                self.wait_for_marker(self.out / 'ready', worker)
                active = web_request(self.root, 'status')['body']
                self.assertTrue(active['running'])
                self.assertFalse(active['resumable'])
                self.assertEqual(self.rows()[2]['state'], 'running')
            finally:
                if worker.poll() is None:
                    worker.kill()
                worker.communicate(timeout=15)

        info = self.fresh_web('status')['body']
        row = self.rows()[2]
        self.assertFalse(info['running'])
        self.assertTrue(info['resumable'])
        self.assertEqual((info['run']['state'], info['run']['job_id'], info['run']['stage'], info['run']['remaining']),
                         ('interrupted', row['id'], 'review', 2))
        self.assertTrue(info['run']['error'])
        stable = self.fresh_web('status')['body']['run']
        self.assertEqual(stable, info['run'])
        self.assertEqual(row['checkpoint'], {'metered': True})
        self.assertEqual(read_json(self.out / 'summary.json')['run'], info['run'])
        self.assert_resume(self.rows(), info['run'])

    def test_second_web_process_cannot_launch_during_worker_handoff(self):
        with self.run_worker('raise') as interrupted:
            _, stderr = interrupted.communicate(timeout=45)
            self.assertIn(FAULT, stderr)
        completed = self.rows()[:2]
        (self.out / 'fault.txt').unlink()
        gate = self.root / 'hold-before-pipeline'
        gate.touch()
        command = [sys.executable, __file__, 'web', str(self.root), 'resume']
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as first:
            try:
                self.wait_for_marker(self.root / 'entered-worker', first)
                active = self.fresh_web('status')['body']
                self.assertTrue(active['running'])
                self.assertFalse(active['resumable'])
                self.assertEqual((active['run']['state'], active['run']['stage']), ('running', 'launch'))
                competing = self.fresh_web('try_resume')
                self.assertIn(competing['code'], (400, 409), competing)
                self.assertIn('正在运行', competing['body']['error'])
                self.assertEqual(self.fresh_web('status')['body']['run'], active['run'])
            finally:
                gate.unlink()
                output, stderr = first.communicate(timeout=60)
        self.assertEqual(first.returncode, 0, stderr)
        self.assertEqual(json.loads(output)['code'], 202)
        final = self.fresh_web('status')['body']
        self.assertEqual((final['run']['id'], final['run']['state']), (active['run']['id'], 'complete'))
        self.assertEqual(self.rows()[:2], completed)
        self.assertEqual(final['cost_total']['actual_requests'], 4)
        attempts = [json.loads(line)['number'] for line in (self.out / 'attempts.jsonl').read_text().splitlines()]
        self.assertEqual(attempts, [1, 2, 3, 3, 4])

    def test_early_web_worker_failure_can_resume_from_snapshot_before_first_claim(self):
        config = BatchConfig.model_validate(read_json(self.config))
        catalog = fixture_catalog(self.root)
        save_snapshot(self.out, snapshot(config, make_jobs(config, catalog), catalog))
        normal_worker = self.worker.read_text(encoding='utf8')
        message = 'fixture import failed before constructing Pipeline'
        self.worker.write_text(f'raise RuntimeError({message!r})', encoding='utf8')
        with patch.object(batches, 'DATA', self.root), patch.object(batches, 'HERE', self.root), patch.dict(batches.PROCESSES, clear=True):
            batches.launch(BATCH)
            self.assertNotEqual(batches.PROCESSES[BATCH].wait(timeout=30), 0)

        info = self.fresh_web('status')['body']
        self.assertFalse(info['running'])
        self.assertTrue(info['resumable'])
        self.assertEqual((info['run']['state'], info['run']['stage']), ('interrupted', 'launch'))
        self.assertIn(message, info['run']['error'])
        self.assertFalse(self.rows())
        self.worker.write_text(normal_worker, encoding='utf8')
        self.assertEqual(self.fresh_web('resume')['code'], 202)
        finished = self.fresh_web('status')['body']
        self.assertEqual(finished['run']['state'], 'complete')
        self.assertEqual(finished['counts'], {'review': 3, 'failed': 1})
        self.assertIn(message, finished['previous_interruption']['error'])

    def test_concurrent_worker_finishes_its_claim_but_never_starts_the_next(self):
        config = read_json(self.config)
        atomic_json(self.config, {**config, 'cpu_concurrency': 2})
        with self.run_worker('concurrent') as worker:
            _, stderr = worker.communicate(timeout=60)
            self.assertNotEqual(worker.returncode, 0)
            self.assertIn(FAULT, stderr)
        rows = self.rows()
        self.assertEqual([r['state'] for r in rows], ['review', 'failed', 'queued', 'queued'])
        run = read_json(self.out / 'summary.json')['run']
        self.assertEqual((run['job_id'], run['stage'], run['remaining']), (rows[2]['id'], 'review', 2))
        self.assert_resume(rows, run)


if __name__ == '__main__':
    if sys.argv[1:2] == ['web']:
        print(json.dumps(web_request(Path(sys.argv[2]), sys.argv[3])))
    else:
        unittest.main()
