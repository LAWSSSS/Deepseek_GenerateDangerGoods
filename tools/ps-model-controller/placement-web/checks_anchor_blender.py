"""Verify browser anchor fixtures against real Blender mesh projections."""
import argparse
import asyncio
import json
import re
from pathlib import Path

import server


async def verify(browser_run, output):
    payload = json.loads(browser_run.read_text(encoding='utf-8-sig'))
    run = json.loads(payload['result']) if isinstance(payload.get('result'), str) else payload
    if run.get('status') != 'passed' or not run.get('states'):
        raise ValueError('Browser checks did not pass')
    if not re.fullmatch(r'[0-9a-f]{32}', run.get('id', '')):
        raise ValueError('Invalid browser session ID')
    directory = server.DATA / run['id']
    if not directory.is_dir() or len(run['states']) > 24:
        raise ValueError('Invalid browser session or too many fixtures')
    results = []
    for state in run['states']:
        row = server.row_for(state['asset'])
        if not row or server.digest(Path(row['mesh'])) != row['mesh_sha256']:
            raise ValueError('Mesh provenance mismatch')
        request = {'mesh': row['mesh'], 'azimuth': state['azimuth'], 'elevation': state['elevation'],
                   'rotation_deg': [state['rot_x'], state['rot_y'], state['rot_z']]}
        ground, hit = await server.contact_preflight(request, directory, row['mesh_sha256'])
        evidence = server.anchor_evidence(state, ground)
        results.append({'state': state, 'contact_anchor': evidence, 'cache_hit': hit})
        print(f"case {len(results)}: drift {evidence['drift_px']} px", flush=True)
    report = {**run, 'browser_run_sha256': server.digest(browser_run),
              'actual_blender_checks': results,
              'max_drift_px': max(x['contact_anchor']['drift_px'] for x in results),
              'physical_support': 'unknown',
              'scope': 'Browser pose/size changes checked against actual Blender projections; no physical acceptance'}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--browser-run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(verify(args.browser_run, args.out))
