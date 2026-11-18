"""Execute the Photoshop lane using verified layers and registered absolute operations."""
import asyncio
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from time import monotonic
from uuid import uuid4

from PIL import Image

from scrap_composite.data.files import FileLease, atomic_json, file_hash, signature
from scrap_composite.photography_agent import run_photography
from scrap_composite.photography_controls import photoshop_changes
from scrap_composite.photography_protocol import MIN_ACTION_CONFIDENCE, PhotoReport
from scrap_composite.photoshop import LIMITS, UNITS, PhotoshopAdapter, availability, review_binding
from scrap_composite.photoshop_workflow import CAPABILITIES, load_composition
from scrap_composite.runtime import ROOT
from scrap_composite.workflow import plan_execution
from .ai import InvalidStructuredReply

MCP = ROOT / 'work/photoshop-mcp-env/Scripts/photoshop-mcp-server.exe'


class PhotoshopUnavailable(RuntimeError):
    """The requested Photoshop lane cannot be executed by this host."""


def needs_photoshop(mode, *, observation=None, remaining=0):
    if mode == 'always' and observation is None:
        return True
    return mode != 'off' and remaining > 0 and bool(photo_findings(observation))


def photo_findings(observation):
    if observation is None:
        return []
    report = observation.legacy() if isinstance(observation, PhotoReport) else observation
    return [finding for finding in report.findings
            if finding.scope == 'image' and finding.code in ('lighting', 'color')
            and finding.visibility == 'visible' and finding.direction != 'unknown'
            and finding.confidence >= MIN_ACTION_CONFIDENCE]


async def photoshop_plan(client, observation, comp, current, seen):
    """Validate edits against the PNG under review and the actual supported values."""
    findings = photo_findings(observation)
    if not findings or not comp.get('composition_manifest'):
        return None, None
    if file_hash(comp['image']) != observation.image_sha256:
        raise ValueError('photoshop_plan_image_changed')
    source, _ = load_composition(comp['composition_manifest'])
    try:
        if isinstance(observation, PhotoReport):
            changes = [
                change.model_copy(update={'part_id': source.action_parts.get(change.control, 'whole'),
                                          'unit': UNITS[change.control]})
                for change in photoshop_changes(current, LIMITS)
                if signature({**current, change.control: change.value}) not in seen
            ]
            decision = await run_photography(client, observation.context, observation.image_sha256,
                target=source.object_id, changes=changes, plan=True, report=observation, parameters=current)
            plan, updates = decision.plan, decision.updates
        else:
            capabilities = {
                name: {**capability, 'causes': sorted({finding.code for finding in findings
                            if finding.code in capability['causes']}),
                       'evidence_ids': [finding.id for finding in findings if finding.code in capability['causes']]}
                for name, capability in CAPABILITIES.items()
                if any(finding.code in capability['causes'] for finding in findings)
            }
            plan, updates = await plan_execution(client, observation, current, capabilities,
                {'backend': 'photoshop', 'parameters': current,
                 'measurements': comp.get('execution_state', {}).get('measurement_summary', {})},
                target=source.object_id)
        values = {**current, **updates}
        if not updates or signature(values) in seen:
            return None, plan.model_dump(mode='json')
        if file_hash(comp['image']) != plan.image_sha256:
            raise ValueError('photoshop_plan_image_changed')
        return values, plan.model_dump(mode='json')
    except (InvalidStructuredReply, ValueError) as error:
        return None, {'status': 'invalid', 'reason': str(error)}


def pixel_hash(path):
    with Image.open(path) as image:
        rgb = image.convert('RGB')
        return hashlib.sha256(repr(rgb.size).encode('ascii') + rgb.tobytes()).hexdigest()


class PhotoshopExecutor:
    def __init__(self, *, adapter=None, lease=None):
        self.adapter = adapter or PhotoshopAdapter(mcp=MCP)
        self.lease = lease if lease is not None else ROOT / 'work/photoshop-batch.lock'
        self.check_local = adapter is None

    async def preflight(self):
        if self.check_local:
            status = await asyncio.to_thread(availability, MCP)
            if not status['available']:
                raise PhotoshopUnavailable(status['reason'])

    async def execute(self, comp, directory, *, actions=None, prior=None, reviewed=None):
        proof = comp.get('composition_manifest')
        if not proof:
            raise ValueError('photoshop_requires_verified_composition')
        source, _ = load_composition(proof)
        image = Path(comp['image'])
        original_hash = source.source_hashes['expected_png']
        if file_hash(image) != original_hash:
            raise ValueError('photoshop_source_frame_changed')
        if reviewed and file_hash(reviewed['image']) != reviewed['image_hash']:
            raise ValueError('photoshop_reviewed_frame_changed')
        if reviewed and not prior and reviewed['image_hash'] != original_hash:
            raise ValueError('photoshop_reviewed_frame_is_not_the_source_composite')
        binding = review_binding(prior) if prior else {}
        if prior and (not reviewed or binding['image_sha256'] != reviewed['image_hash']):
            raise ValueError('photoshop_readback_is_not_the_reviewed_frame')
        proof_hash = file_hash(proof)
        root = Path(directory) / signature(proof_hash, actions or {}, binding)[:24]
        index = root / 'export.json'
        if index.is_file():
            exported = json.loads(index.read_text(encoding='utf8'))
            records = [*exported['outputs'].values(), exported['readback']]
            inputs = {name: getattr(source, name) for name in source.source_hashes}
            if (all(Path(item['path']).is_file() and file_hash(item['path']) == item['sha256'] for item in records)
                    and all(path and file_hash(path) == source.source_hashes[name] for name, path in inputs.items())):
                return self._frame(comp, exported, 0., True)
        await self.preflight()
        lease = FileLease(self.lease)
        while not lease.acquire():
            await asyncio.sleep(.25)
        start = monotonic()
        try:
            request = replace(source, actions=actions or {}, review_binding=binding)
            exported = await self.adapter.export(request, root / uuid4().hex)
        except RuntimeError as error:
            if str(error) in {'photoshop_unavailable', 'photoshop_mcp_execution_failed'}:
                raise PhotoshopUnavailable(str(error)) from error
            raise
        finally:
            lease.release()
        atomic_json(index, exported)
        return self._frame(comp, exported, round(monotonic() - start, 3), False)

    @staticmethod
    def _frame(comp, exported, seconds, cache_hit):
        image = Path(exported['outputs']['png']['path'])
        with Image.open(image) as opened:
            final = opened.convert('RGB')
        crop = image.with_suffix('.crop.jpg')
        preview = image.with_suffix('.full.jpg')
        area = final.crop(comp['edit_bounds'])
        area.thumbnail((1000, 800))
        area.save(crop, quality=92)
        final.thumbnail((1536, 1024))
        final.save(preview, quality=88)
        current = {**dict.fromkeys(LIMITS, 0), **exported['validation'].get('actual_parameters', {})}
        return {**comp, 'image': str(image), 'crop': str(crop), 'preview': str(preview),
                'photoshop': exported, 'photoshop_seconds': seconds, 'photoshop_cache_hit': cache_hit,
                'execution_state': {'backend': 'photoshop', 'parameters': current,
                                    'image_sha256': exported['outputs']['png']['sha256'],
                                    'state_sha256': exported['readback']['sha256'],
                                    'measurement_summary': exported.get('local_measurements', {})}}
