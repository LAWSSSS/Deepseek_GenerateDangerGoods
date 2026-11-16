"""Compare the generated PBR finish on one hydraulic trial and six frozen frames."""
import argparse
import asyncio
from dataclasses import dataclass, replace
import html
import os
from pathlib import Path
import shutil
import time
from typing import Annotated, Literal

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import cv2
import numpy as np
from PIL import Image
from pydantic import Field

from display_batch.ai import InvalidStructuredReply, ProviderUnavailable
from display_batch.common import ROOT, atomic_json, file_hash, implementation_manifest, resolve, signature
from display_batch.compose import compose, object_window, warp
from display_batch.costs import cumulative, label
from display_batch.generated_material import VERSION as FIRST_PACK_VERSION, finish_regions, load_pack
from display_batch.pipeline import Pipeline
from display_batch.rendering import CORROSION_PASS, FINISH_PASS, HYDRAULIC_PASS, render_dependencies, render_request
from display_batch.schema import BatchConfig
from display_batch.strict_contact import visual_veto
from display_batch.structured import _review, frame_inputs
from scrap_composite.photography_agent import role_prompt
from scrap_composite.photography_protocol import PhotoBrief, PhotoReport
from scrap_composite.contracts import Strict
from validate_local_fusion_six import (check_invariants, make_overview, model_hashes,
                                     raw_alpha_audit, read, relative)
from verify_geometry_rebuild import saved_render


SOURCE = 'outputs/局部薄雾与金属做旧-六图-20261113-v2'
OUTPUT = 'outputs/生成贴图-固定摆位-20261113'
PACK = 'work/render-materials/generated-pbr-v1/pack.json'
PILOT = '36_6_6-31'
VERSION = 'fixed-generated-pbr-v1'
REVIEW_CONTRACT = 'paired-brief-v3'
RENDER_SOURCE_FILES = (
    'blender_display_worker.py', 'blender_worker.py', 'display_batch/geometry.py',
    'display_batch/framing.py', 'display_batch/materials.py', 'display_batch/rendering.py',
    'display_batch/contact_geometry.py', 'display_batch/scanned_shader.py', 'display_batch/local_light.py',
)
PILOT_LEGACY_DEPS = 'cb25c083092008cd5cd451c7d1eaddc3d2ef5a4a4d2bc8f804ee8e2258d39523'
METAL_COMPARISONS = {
    '3': 'outputs/实物细节复测-20261116',
    '22': 'outputs/实物细节复测-20261116',
    '31': 'outputs/液压缸三倍划痕锈迹-20261116',
    '33': 'outputs/圆柱划痕与锈蚀-20261116',
}


@dataclass(frozen=True)
class Case:
    key: str
    group: str
    asset_id: str
    asset_record: Path
    scene_key: str
    background: str
    baseline: str
    baseline_hash: str
    candidate: dict
    params: dict
    baseline_entry: dict
    folder: Path
    meta: dict
    shadow_reference: dict
    resolution: int
    seed: int
    previous_image: str | None = None
    previous_hash: str | None = None

    @property
    def comparison_image(self):
        return self.previous_image or self.baseline

    @property
    def comparison_hash(self):
        return self.previous_hash or self.baseline_hash

    def binding(self):
        result = {'key': self.key, 'asset_id': self.asset_id, 'group': self.group,
                'scene_key': self.scene_key, 'background': self.background,
                'background_sha256': file_hash(self.background), 'baseline': self.baseline,
                'baseline_sha256': self.baseline_hash, 'asset_sha256': file_hash(self.asset_record),
                'candidate_signature': signature(self.candidate), 'params_signature': signature(self.params),
                'resolution': self.resolution, 'seed': self.seed, 'shadow_reference': self.shadow_reference}
        if self.previous_image:
            result.update(previous_image=self.previous_image, previous_hash=self.previous_hash)
        return result


def frozen_cases(source, cache):
    frozen = read(source / 'frozen.json')
    pairs = {(i['group'], i['asset_id']) for i in frozen['items']}
    if len(frozen['items']) != 6 or pairs != {(g, a) for g in ('36_6_24', '53_7_35') for a in ('3', '22', '31')}:
        raise ValueError('expected the same three items in both archived scenes')
    original = read(source / 'implementation.json')['implementation']['sources']
    original = {k.replace('\\', '/'): v for k, v in original.items()}
    deps = signature('blender-4.5.3', {name: original[name] for name in RENDER_SOURCE_FILES})
    cases = []
    for item in frozen['items']:
        key = item['key']
        path = source / 'cases' / key
        result, render = read(path / 'result.json'), read(path / 'render.json')
        baseline = path / 'after.png'
        digest = file_hash(baseline)
        if digest != result['image_hash'] or file_hash(item['background']) != item['background_hash']:
            raise ValueError(f'archived image changed: {key}')
        candidate = read(path / 'candidate.json')
        if signature(candidate) != item['candidate_signature']:
            raise ValueError(f'archived geometry changed: {key}')
        shadow = render['metadata']['render_cache']['without_shadow']['signature']
        cases.append(Case(key, item['group'], item['asset_id'],
            cache / 'assets' / item['asset_signature'] / 'result-strict.json',
            item['scene_signature'], item['background'], str(baseline), digest, candidate,
            result['params'], result['composition'], Path(render['folder']), render['metadata'],
            {'signature': shadow, 'dependencies': deps, 'standard_profile_without_detail_key': True},
            render['resolution'], item['seed']))
    return cases, frozen['source_models']


def hydraulic_case(cache):
    source = ROOT / 'work/test-all-two-scenes-20261113/run/jobs/0033-8e9fb14b14'
    result, iteration = read(source / 'result.json'), read(source / 'iteration-0.json')
    trial = ROOT / 'work/test-hydraulic-detail-20261113/run'
    record = read(trial / 'hydraulic-fine-detail.json')
    baseline = record['output_image']
    digest = file_hash(baseline)
    if file_hash(trial / 'hydraulic-fine-detail.png') != digest:
        raise ValueError('delivered hydraulic baseline differs from its recorded render')
    candidate = next(c for c in read(source / 'candidates.json')['candidates'] if c['id'] == record['candidate_id'])
    params = {**iteration['params'], 'render_detail': 'fine', 'rust_spots': '0.95',
              'surface_dust': '0.5', 'rust_pattern': 'photo', 'rust_relief': True,
              'rust_relief_strength': 3., 'contrast_matching': 'local'}
    folder, meta = saved_render(cache, {'render_cache': record['render_cache'], 'params': params})
    shadow = record['render_cache']['without_shadow']['signature']
    return Case(PILOT, '36_6_6', '31', cache / 'assets' / result['asset_signature'] / 'result.json',
        result['scene_signature'], result['background'], baseline, digest, candidate, params,
        {'photo': record['photography'], 'checks': record['checks']}, folder, meta,
        {'signature': shadow, 'dependencies': PILOT_LEGACY_DEPS}, iteration['render_resolution'], 20261146)


def generated_params(case, pack):
    return {**case.params, 'surface_finish': 'generated_pbr', 'surface_finish_manifest': str(pack),
            'rust_spots': '0', 'rust_pattern': 'procedural', 'rust_relief': False, 'rust_relief_strength': 1.}


def bind_comparison(case, previous):
    row = read(previous / 'cases' / case.key / 'result.json')
    if (row['invariants']['fixed_geometry_signature'] != signature(case.candidate)
            or row['invariants']['status'] != 'pass' or file_hash(row['image']) != row['image_hash']):
        raise ValueError(f'comparison image or geometry mismatch: {case.key}')
    return replace(case, previous_image=row['image'], previous_hash=row['image_hash'])


def bind_inputs(path, payload):
    if path.is_file() and read(path) != payload:
        raise ValueError(f'frozen inputs changed: {path.name}; choose a new output directory')
    atomic_json(path, payload)


def finish_image(folder, filename=FINISH_PASS):
    exr = read_exr(folder / filename)
    if exr is None or exr.ndim != 3 or exr.shape[2] < 3:
        raise ValueError('missing material diagnostic pass')
    return np.ascontiguousarray(exr[..., :3][..., ::-1])


def read_exr(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)


def material_diagnostic(folder, path):
    values = finish_image(folder)
    with Image.open(folder / 'object.png') as image:
        core = np.asarray(image.getchannel('A')) > 245
    if not core.any() or values.shape[:2] != core.shape:
        raise ValueError('material diagnostic and coverage mismatch')
    Image.fromarray(np.uint8(np.clip(values * [1., 2., 2.], 0, 1) * 255)).save(path)
    result = {'path': str(path), 'red': 'oxidation and paint chips', 'green': 'scratches', 'blue': 'powder',
            'mean': values[core].mean(axis=0).tolist(),
            'p95': np.percentile(values[core], 95, axis=0).tolist()}
    if (folder / CORROSION_PASS).is_file():
        corrosion = finish_image(folder, CORROSION_PASS)
        image = path.with_name('corrosion.png')
        Image.fromarray(np.uint8(np.clip(corrosion * [1.5, 1.5, .35], 0, 1) * 255)).save(image)
        result['corrosion'] = {'path': str(image), 'channels': ['oxide', 'exposed_steel', 'ferrous_region']}
    return result


def frame_detail(case, comp, meta, folder, visible_alpha):
    """Measure the actual projection and exported texture away from the outline."""
    bounds = comp['edit_bounds']
    full = case.candidate['full_px']
    offset = np.asarray(case.candidate['anchor_px']) - np.asarray(meta['anchor_uv']) * full
    coverage = cv2.erode(np.uint8(visible_alpha > .95), np.ones((11, 11), np.uint8)) > 0
    if not coverage.any():
        return {'status': 'unknown', 'reason': 'insufficient_visible_core'}
    window = object_window(meta)
    projected = warp(finish_image(folder), full, offset, bounds, window=window)

    def measure(path):
        with Image.open(path) as image:
            y = np.float32(image.crop(bounds).convert('RGB')) @ np.float32([.2126, .7152, .0722])
        detail = cv2.GaussianBlur(y, (0, 0), 1.) - cv2.GaussianBlur(y, (0, 0), 5.)
        return {'median_luma': float(np.median(y[coverage])),
                'detail_p75': float(np.percentile(np.abs(detail[coverage]), 75))}

    result = {'status': 'measured', 'core_pixels': int(coverage.sum()),
              'projected_fraction_gt_0_03': (projected[coverage] > .03).mean(axis=0).tolist(),
              'previous': measure(case.comparison_image), 'current': measure(comp['image'])}
    def share_between(first, second):
        if first.sum() < 20 or second.sum() < 20:
            return None
        gloss, diffuse = (warp(np.maximum(0., read_exr(folder / pass_file)[..., :3].mean(axis=2)),
            full, offset, bounds, window=window) for pass_file in ('specular0001.exr', 'diffuse0001.exr'))
        share = gloss / np.maximum(gloss + diffuse, 1e-5)
        return float(np.median(share[first])), float(np.median(share[second]))

    if (folder / CORROSION_PASS).is_file():
        oxide, steel, ferrous = warp(finish_image(folder, CORROSION_PASS), full, offset, bounds,
                                    window=window).transpose(2, 0, 1)
        affected = coverage & (ferrous > .5)
        result['corrosion'] = {
            'ferrous_pixels': int(affected.sum()),
            'rust_fraction': float(np.mean(oxide[coverage] > .03)),
            'rust_fraction_gt_0_20': float(np.mean(oxide[coverage] > .20)),
            'exposed_steel_fraction': float(np.mean(steel[coverage] > .03)),
        }
        shares = share_between(affected & (steel > .22) & (oxide < .16), affected & (oxide > .45))
        if shares is not None:
            result['corrosion']['specular_fraction_median'] = dict(zip(('exposed_steel', 'oxide'), shares))
    elif (folder / HYDRAULIC_PASS).is_file():
        wear = warp(finish_image(folder, HYDRAULIC_PASS), full, offset, bounds, window=window)
        result['hydraulic_channels'] = dict(zip(('rust', 'exposed_steel', 'scrapes'),
            (wear[coverage] > .03).mean(axis=0).tolist()))
        steel = coverage & (wear[..., 1] > .20)
        nearby = cv2.dilate(np.uint8(steel), np.ones((19, 19), np.uint8)) > 0
        paint = coverage & nearby & (wear.max(axis=2) < .03)
        shares = share_between(steel, paint)
        if shares is not None:
            result['specular_fraction_median'] = dict(zip(('exposed_steel', 'nearby_intact_paint'), shares))
    return result


async def preview(pipe, case, asset, pack, out):
    """Independently vary the light and the model's yaw in an isolated render."""
    candidate = case.candidate
    params = generated_params(case, pack['manifest'])
    region = finish_regions(asset, pack, pipe.cache)
    environment = {'world_color': [.44, .44, .44], 'light_color': [1., 1., 1.], 'dust_color': [.19, .19, .19],
                   'light_level': 1., 'light_spread': .38, 'integration_mode': 'coherent',
                   'local_irradiance': {'direction': {'direction': [-.8, -.25], 'confidence': .9},
                                        'level': 1., 'fill_ratio': .28}}
    first = render_request(asset, candidate, params, environment)
    first.update(appearance='local', mesh_hash=asset['mesh_hash'], dust_control=1.,
                 material_classes=asset['semantic']['material_classes'],
                 generated_finish={**pack, 'regions': region},
                 windows={part: {'frame': 448, 'rect': [0, 0, 448, 448]} for part in ('object', 'shadow')},
                 render_detail='standard')
    second = {**first, 'environment': {**environment, 'local_irradiance': {**environment['local_irradiance'],
              'direction': {'direction': [.8, -.25], 'confidence': .9}}}}
    rotated = {**first, 'yaw': candidate['yaw'] + 55}
    renders = []
    out.mkdir(parents=True, exist_ok=True)
    timing = {}
    for tag, request in (('left_light', first), ('right_light', second), ('turned', rotated)):
        folder = out / tag
        folder.mkdir(exist_ok=True)
        metadata = await pipe.blender.ask('render_parts', {**request, 'parts': {'object': str(folder)}})
        timing[tag] = metadata['timings']
        material_diagnostic(folder, folder / 'material.png')
        renders.append(folder)
    coverage = []
    for folder in renders:
        with Image.open(folder / 'object.png') as image:
            coverage.append(np.asarray(image.getchannel('A')) > 245)
    core = cv2.erode((coverage[0] & coverage[1]).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    maps = [finish_image(folder) for folder in renders]
    specs = [read_exr(folder / 'specular0001.exr')[..., :3] for folder in renders[:2]]
    stable_maps = float(np.abs(maps[0][core] - maps[1][core]).mean())
    reflection_change = float(np.abs(specs[0][core] - specs[1][core]).mean())
    turned = coverage[0] & coverage[2]
    turn_change = float(np.abs(maps[0][turned] - maps[2][turned]).mean()) if turned.any() else 0.
    report = {'status': 'pass' if stable_maps < .003 and reflection_change > .001 and turn_change > .005 else 'fail',
              'material_map_mae_under_changed_light': stable_maps,
              'specular_linear_mae_under_changed_light': reflection_change, 'turn_material_map_mae': turn_change,
              'turn_degrees': 55, 'preview_only_renders': 3, 'timings': timing}
    atomic_json(out / 'result.json', report)
    if report['status'] == 'fail':
        raise ValueError('material preview failed light/rotation separation')
    return report


class BlindBrief(PhotoBrief):
    findings_order: object = Field(default=None, exclude=True)


class BlindPair(Strict):
    a: BlindBrief
    b: BlindBrief
    preferred: Literal['a', 'b', 'tie']
    reason: Annotated[str, Field(min_length=1, max_length=80)]
    type: str | None = Field(default=None, exclude=True)

async def compare(pipe, case, result, asset, out):
    comp = result['composition']
    width, height = comp['checks']['size']['pixels']
    left, top, right, bottom = comp['edit_bounds']
    region = (left / width, top / height, right / width, bottom / height)
    # Both versions receive the same scene context and identically sized crops.
    images = [result['baseline'], result['image']]
    new_index = int(signature(case.key, 'surface-material-blind-order')[-1], 16) % 2
    if new_index == 0:
        images.reverse()
    crops = []
    for index, image in enumerate(images):
        path = out / f'pair-local-{index}.png'
        with Image.open(image) as frame:
            frame.crop(comp['edit_bounds']).save(path)
        crops.append(path)
    prompt = role_prompt('observer').replace(
        '首图是唯一评审画面，其余用于局部细节或物品识别。', '两张全景分别评审，每张全景有一张对应的局部图。')
    prompt = prompt.replace('只输出含 trace、scores、findings 的 JSON。', '')
    prompt = prompt.replace('最多五条观察。', '每张最多三条观察。')
    prompt += ('\n图1是 A 全景，图2是 A 局部，图3是 B 全景，图4是 B 局部。a 和 b 分别写各版评分、trace 和 findings。'
               'region 始终对应各自全景。给出更像现场的材质与磨损版本 preferred，微小或不可辨差异用 tie。'
               'reason 简述可见锈斑、粉尘、划痕、反光及识别细节的差异。保持正常材料差异，勿以变化多寡推定质量。'
               '根字段仅 a、b、preferred、reason。a 与 b 仅 trace、scores、findings。')
    pair = await pipe.vision.ask('photographer_material_blind_pair', BlindPair, prompt,
        {'item': asset['name'], 'materials': asset['semantic']['material_classes'],
         'target_region': region, 'camera_and_placement': 'same for A and B'},
        (images[0], crops[0], images[1], crops[1]), compact=True, plain_json=True,
        image_limits=((1280, 720), (1000, 800), (1280, 720), (1000, 800)), max_output_tokens=2400)
    keys = ('a', 'b')
    old_key, new_key = keys[1 - new_index], keys[new_index]
    before = getattr(pair, old_key)
    brief = getattr(pair, new_key)
    context, _ = frame_inputs(asset, comp)
    new_report = PhotoReport.bind(brief, result['image_hash'], 'object', context)
    before_report = PhotoReport.bind(before, case.comparison_hash, 'object', context)
    checked = _review(new_report.legacy(), comp['checks'])
    direction = 'tie' if pair.preferred == 'tie' else 'new' if pair.preferred == new_key else 'old'
    vetoes = visual_veto(checked)
    serious = [f.factor for f in new_report.findings if f.visibility == 'visible'
               and f.severity in ('major', 'fatal') and f.score_key in ('material', 'texture', 'edge', 'structure')]
    scores = new_report.scores
    passes = direction == 'new' and min(scores.material, scores.texture) >= 3 and not serious and not vetoes
    return {'blind_order': {old_key: 'existing', new_key: 'generated_pbr'}, 'preferred': direction,
            'photography_pass': passes, 'serious_factors': serious, 'visual_vetoes': vetoes,
            'old_scores': before_report.scores.model_dump(), 'new_scores': scores.model_dump(),
            'reason': pair.reason, 'response': pair.model_dump(mode='json'),
            'before_report': before_report.model_dump(mode='json'), 'after_report': new_report.model_dump(mode='json')}


def material_target_met(review):
    scores = review.get('new_scores', {})
    return review.get('preferred') == 'new' and min(scores.get('material', 0), scores.get('texture', 0)) >= 3


def write_report(out, cases, hashes, pack, preview_report, *, metal_review=False):
    rows = []
    for case in cases:
        file = out / 'cases' / case.key / 'result.json'
        rows.append(read(file) if file.exists() else {'key': case.key, 'asset_id': case.asset_id,
            'group': case.group, 'baseline': case.comparison_image, 'status': 'not_run', 'image': None})
    sample = [row for row in rows if row['key'] != PILOT]
    passed = [row for row in sample if row.get('review', {}).get('photography_pass') and row['status'] == 'review']
    unchanged = model_hashes() == hashes
    usage = cumulative(out)
    runs = [read(path) for path in sorted((out / 'runs').glob('*.json'))]
    report = {'version': VERSION, 'generated': sum(bool(r.get('image')) for r in rows),
              'review': sum(r['status'] == 'review' for r in rows),
              'failed': sum(r['status'] == 'failed' for r in rows),
              'material_targets': sum(material_target_met(r.get('review', {})) for r in sample),
              'reviewed': sum(bool(r.get('review')) for r in rows), 'sample_photography_pass': len(passed),
              'sample_count': len(sample),
              'sample_gate': not metal_review and len(passed) >= 5 and {row['asset_id'] for row in passed} == {'3', '22', '31'}
                  and unchanged and all(row.get('invariants', {}).get('status') == 'pass' for row in rows),
              'source_models_unchanged': unchanged, 'source_model_count': len(hashes),
              'cost_total': usage, 'runs': runs, 'pilot_preview': preview_report,
              'pack': pack['manifest'], 'pack_signature': pack['signature'],
              'imagegen': {'builtin_calls': len(pack['maps']) if pack['version'] == FIRST_PACK_VERSION else 0,
                           'estimated_cost': None}, 'jobs': rows}
    atomic_json(out / 'validation.json', report)
    names = {'3': '03 轴承座', '11': '11 千斤顶', '22': '22 气缸', '31': '31 液压缸', '33': '33 灭火器'}
    cards = []
    display_rows = sorted(rows, key=lambda row: (names[row['asset_id']], row['group'])) if metal_review else rows
    for row in display_rows:
        frames = []
        new_title = '新材质 · 待审' if row['status'] == 'review' else '新材质 · 诊断'
        for title, path in (('对照版', row['baseline']), (new_title, row.get('image'))):
            if path:
                url = html.escape(relative(path, out), quote=True)
                frames.append(f'<figure><figcaption>{title}</figcaption><a href="{url}">'
                              f'<img loading="lazy" src="{url}"></a></figure>')
            else:
                frames.append('<p>未生成</p>')
        key = html.escape(f"{names[row['asset_id']]} · {row['group']}" if metal_review else row['key'])
        review = html.escape(row.get('review', {}).get('reason', row.get('stop_reason', '')))
        diagnosis = row.get('material_diagnostic', {}).get('path')
        link = f'<a href="{html.escape(relative(diagnosis, out), quote=True)}">材质分布</a>' if diagnosis else ''
        material = row.get('material_diagnostic', {})
        corrosion = material.get('corrosion', {}).get('path')
        if corrosion:
            link += f' · <a href="{html.escape(relative(corrosion, out), quote=True)}">锈蚀与露钢分布</a>'
        corrosion_stats = material.get('final_frame', {}).get('corrosion')
        coverage = (f'<p>可见物品内部锈蚀覆盖 {corrosion_stats["rust_fraction"]:.1%}，'
                    f'露钢覆盖 {corrosion_stats["exposed_steel_fraction"]:.1%}。</p>') if corrosion_stats else ''
        scores = row.get('review', {}).get('new_scores', {})
        rating = f"材质 {scores['material']}/4，纹理 {scores['texture']}/4。 " if scores else ''
        cards.append(f'<section><h2>{key}</h2><div>{"".join(frames)}</div><p>{rating}{review}</p>{coverage}{link}</section>')
    summary = (f'本轮生成 {report["generated"]}/{len(cases)} 张完整画面，供人工点评。' if metal_review
               else f'摄影门槛 {report["sample_photography_pass"]}/{report["sample_count"]}。')
    (out / 'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8">'
        '<title>废钢材质同摆位对照</title><style>body{font:16px sans-serif;margin:24px;color:#192323}'
        'section{margin:32px 0}section div{display:grid;grid-template-columns:repeat(auto-fit,minmax(400px,1fr))}'
        'figure{margin:6px}img{width:100%;height:auto}</style><h1>废钢材质同摆位对照</h1>'
        '<p>点击完整画面可查看原尺寸 PNG。现场尺度和接触证据为估计时，结果保留待审。</p>'
        f'<p>{summary}{html.escape(label(usage))}</p>'
        + ''.join(cards), encoding='utf8')
    if any(r.get('image') for r in rows):
        make_overview(out, rows)
    return report


async def pilot_entry(case, pipe, scene, asset, out):
    """Recompose the archived channels once to recover its original geometry mask hashes."""
    async def archived_render(*args, **kwargs):
        return case.folder, case.meta, True
    original = pipe.renders.get
    pipe.renders.get = archived_render
    try:
        folder, meta, environment, samples, _ = await pipe.render(
            asset, scene, case.candidate, case.params, case.resolution)
    finally:
        pipe.renders.get = original
    comp = await asyncio.to_thread(compose, scene, case.candidate, folder, meta, environment,
                                   samples, case.params, case.seed, out)
    if file_hash(comp['image']) != case.baseline_hash:
        with Image.open(comp['image']) as actual, Image.open(case.baseline) as expected:
            if actual.size != expected.size or not np.array_equal(np.asarray(actual), np.asarray(expected)):
                raise ValueError('could not reproduce the delivered hydraulic baseline pixels')
    return comp


async def run_case(pipe, case, pack, hashes, out):
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'result.json'
    if path.is_file():
        previous = read(path)
        if (previous.get('image') and file_hash(previous['image']) == previous['image_hash']
                and previous.get('invariants', {}).get('status') == 'pass'):
            return previous
    result = {'key': case.key, 'asset_id': case.asset_id, 'group': case.group, 'baseline': case.comparison_image,
              'status': 'failed', 'image': None}
    stage = 'load_frozen_scene'
    try:
        asset = read(case.asset_record)
        if file_hash(asset['mesh']) != hashes[case.asset_id]:
            raise ValueError('source mesh SHA-256 mismatch')
        scene = await pipe.scenes.prepare(case.background)
        if scene['key'] != case.scene_key:
            raise ValueError('frozen scene signature changed')
        candidate = dict(case.candidate)
        baseline = case.baseline_entry
        if case.key == PILOT:
            baseline = await pilot_entry(case, pipe, scene, asset, out / 'baseline-replay.png')
        item = {'candidate': candidate, 'candidate_signature': signature(candidate),
                'params': case.params, 'params_signature': signature(case.params),
                'baseline': case.baseline, 'baseline_entry': baseline}
        params = generated_params(case, pack['manifest'])
        stage = 'render'
        folder, meta, env, samples, _ = await pipe.render(asset, scene, candidate, params, case.resolution,
                                                         shadow_reference=case.shadow_reference)
        atomic_json(out / 'render.json', {'folder': str(folder), 'metadata': meta})
        raw = raw_alpha_audit(item, folder, meta, case.folder, case.meta)
        fixed = {**candidate, 'fixed_alpha_dir': str(case.folder), 'fixed_alpha_window': object_window(case.meta)}
        stage = 'compose'
        comp = await asyncio.to_thread(compose, scene, fixed, folder, meta, env, samples, params, case.seed,
                                       out / 'after.png', capture_local=True)
        layers = comp.pop('local_layers')
        invariants, metrics = check_invariants(item, comp, meta, case.meta, scene, fixed, layers,
                                               estimated_occlusion=pack.get('corrosion', False))
        invariants['raw_render_alpha'] = raw
        checks = invariants['checks']
        checks['camera_blur_unchanged'] = abs(comp['photo']['blur_sigma'] - baseline['photo']['blur_sigma']) < 1e-6
        checks['air_amount_unchanged'] = abs(comp['photo']['haze'] - baseline['photo']['haze']) < 1e-6
        checks['receiver_baseline_reused'] = meta['render_cache']['without_shadow']['hit']
        checks['material_actually_changed'] = metrics.get('visible_changed_pixels', 0) > 0
        invariants['status'] = 'pass' if all(checks.values()) and raw['status'] == 'pass' else 'fail'
        failures = [k for k, v in comp['checks'].items() if v['status'] == 'fail']
        diagnostic = material_diagnostic(folder, out / 'material.png')
        diagnostic['final_frame'] = frame_detail(case, comp, meta, folder, layers['alpha'])
        checks['surface_diagnostic_nonzero'] = max(diagnostic['p95']) > .01
        if not checks['surface_diagnostic_nonzero']:
            invariants['status'] = 'fail'
        result.update(image=comp['image'], image_hash=file_hash(comp['image']), composition=comp, metrics=metrics,
                      invariants=invariants, program_failures=failures, material_diagnostic=diagnostic,
                      generated_finish=meta['generated_finish'], render_cache=meta['render_cache'],
                      render_timings=meta['render_timings'])
        if invariants['status'] == 'pass' and not failures:
            result.update(status='review', stop_reason='same_geometry_scene_evidence_estimated')
        else:
            result['stop_reason'] = 'program_checks_failed'
    except (RuntimeError, ValueError, OSError, KeyError) as error:
        result.update(stop_reason=stage + '_failed', error=str(error)[:300])
    finally:
        result['seconds'] = time.monotonic() - started
        atomic_json(path, result)
        print(case.key, result['status'], result['stop_reason'], f"{result['seconds']:.1f}s", flush=True)
    return result


async def review_case(pipe, case, out):
    path = out / 'result.json'
    if not path.exists():
        return
    result = read(path)
    if (result.get('review') or result['status'] != 'review'
            or (result.get('review_error') and result.get('review_contract') == REVIEW_CONTRACT)):
        return
    if file_hash(result['image']) != result['image_hash']:
        raise ValueError('generated image changed before review')
    asyncio.current_task().set_name(case.key)
    before = len(pipe.vision.calls)
    started = time.monotonic()
    result['review_contract'] = REVIEW_CONTRACT
    result.pop('review_error', None)
    try:
        result['review'] = await compare(pipe, case, result, read(case.asset_record), out)
        if result['review']['visual_vetoes']:
            result.update(status='failed', stop_reason='visible_physical_defect')
    except (ProviderUnavailable, InvalidStructuredReply) as error:
        if isinstance(error, ProviderUnavailable):
            raise
        result['review_error'] = str(error)[:250]
    finally:
        result['review_seconds'] = time.monotonic() - started
        result['review_requests'] = len(pipe.vision.calls) - before
        atomic_json(path, result)
        print(case.key, 'blind review', result.get('review', {}).get('preferred', 'unavailable'), flush=True)


async def run(args):
    out = resolve(args.out)
    source = resolve(args.source)
    pack = load_pack(resolve(args.pack))
    config = BatchConfig.model_validate({**read(source / 'config.json'), 'output': str(out), 'offline': True,
        'max_iterations': 0, 'review_profile': 'photographer'})
    cache = resolve(config.cache)
    sample, hashes = frozen_cases(source, cache)
    pilot = hydraulic_case(cache)
    if args.hydraulic or args.cylinders:
        sample = [case for case in sample if case.asset_id == '31']
    if args.cylinders or args.metal_review:
        from cylinder_wear_cases import extinguisher_cases, jack_cases
        sample += extinguisher_cases(cache)
        if args.metal_review:
            sample += jack_cases(cache)
    cases = [pilot, *sample]
    previous_cases = []
    for case in cases:
        comparison = args.compare_to
        if args.cylinders and case.asset_id == '33':
            comparison = None
        elif args.metal_review and not comparison:
            comparison = METAL_COMPARISONS.get(case.asset_id)
        previous_cases.append(bind_comparison(case, resolve(comparison)) if comparison else case)
    cases = previous_cases
    pilot, *sample = cases
    if model_hashes() != hashes:
        raise ValueError('44-model source baseline differs')
    binding = {'version': VERSION, 'pack': pack['signature'], 'models': hashes,
               'cases': [case.binding() for case in cases]}
    preview_path = out / 'preview' / 'result.json'
    preview_report = read(preview_path) if preview_path.is_file() else None
    if args.stage in ('report', 'review'):
        recorded = read(out / 'frozen.json')
        if any(recorded.get(key) != value for key, value in binding.items()):
            raise ValueError('archived comparison inputs changed')
        for case in cases:
            path = out / 'cases' / case.key / 'result.json'
            if path.is_file() and (row := read(path)).get('image') and file_hash(row['image']) != row['image_hash']:
                raise ValueError(f'archived image changed: {case.key}')
        if args.stage == 'report':
            write_report(out, cases, hashes, pack, preview_report, metal_review=args.metal_review)
            return
    else:
        bind_inputs(out / 'frozen.json', {**binding, 'render_dependencies': render_dependencies(),
            'implementation': signature(implementation_manifest())})
    pipe = Pipeline(config)
    start = time.monotonic()
    try:
        if args.stage in ('preview', 'all') and not preview_report:
            preview_report = await preview(pipe, pilot, read(pilot.asset_record), pack, preview_path.parent)
        if args.stage in ('pilot', 'all'):
            result = await run_case(pipe, pilot, pack, hashes, out / 'cases' / pilot.key)
            if result['status'] != 'review':
                raise ValueError(f"pilot stopped: {result['stop_reason']}")
        if args.stage in ('six', 'all'):
            if not preview_report or preview_report['status'] != 'pass':
                raise ValueError('complete the material preview before the six-frame comparison')
            if read(out / 'cases' / pilot.key / 'result.json')['invariants']['status'] != 'pass':
                raise ValueError('pilot program checks failed')
            for case in sample:
                await run_case(pipe, case, pack, hashes, out / 'cases' / case.key)
        if args.stage == 'review' or (args.stage == 'all' and not args.offline and not args.metal_review):
            pipe.vision.offline = args.offline
            for case in cases if args.compare_to else sample:
                await review_case(pipe, case, out / 'cases' / case.key)
    finally:
        await pipe.vision.close()
        await pipe.blender.close()
        pipe.store.close()
        atomic_json(out / 'runs' / f'{pipe.run_id}.json', {'stage': args.stage,
            'seconds': time.monotonic() - start, 'blender_seconds': pipe.blender.seconds,
            'render_counts': dict(pipe.renders.totals), 'render_timings': dict(pipe.renders.timings),
            'api_requests': len(pipe.vision.calls)})
        report = write_report(out, cases, hashes, pack, preview_report, metal_review=args.metal_review)
        print('report', out / 'index.html', report['generated'], report['reviewed'], report['sample_gate'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('preview', 'pilot', 'six', 'review', 'report', 'all'))
    parser.add_argument('--source', default=SOURCE)
    parser.add_argument('--out', default=OUTPUT)
    parser.add_argument('--pack', default=PACK)
    parser.add_argument('--compare-to', help='prior fixed-geometry trial used as the visual baseline')
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument('--cylinders', action='store_true', help='replay the 31 hydraulic cylinder and 33 extinguisher')
    scope.add_argument('--hydraulic', action='store_true', help='replay only the three fixed-view 31 hydraulic cylinder samples')
    scope.add_argument('--metal-review', action='store_true',
                       help='prepare fixed-frame comparisons for five metal objects for manual review')
    parser.add_argument('--offline', action='store_true')
    asyncio.run(run(parser.parse_args()))


if __name__ == '__main__':
    main()
