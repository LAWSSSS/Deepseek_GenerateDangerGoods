"""Background-independent, metric material evidence for physical surface layers.

The inherited atlas describes known source regions, never a model-specific look.
RGB-derived maps are estimates of wear masks, not measured PBR parameters.
"""
import math
import copy
from pathlib import Path
import re

from .common import file_hash, signature
from .generated_material import finish_regions, load_pack
from .surface_maps import prepare_uv_masks


VERSION = 'physical-layers-v1'
RECIPE = {
    'oxide_period_m': .12, 'scratch_period_m': .08, 'dust_period_m': .09,
    'relief_depth_m': .00008, 'dust_depth_m': .00004,
    'oxide_max': .30, 'abrasion_max': .22,
    'dust_albedo_linear': [.20, .185, .165],
    'max_dust': .5, 'paint_dust_multiplier': .5,
}
PROTECTED = r'\b(?:label|mark|decal|print|chrome|machined|bearing|polished|zerk|rubber|plastic|glass)\b'


def metric_size(value):
    if isinstance(value, bool):
        raise ValueError('physical_layers requires a positive physical size in metres')
    try:
        size = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError('physical_layers requires a positive physical size in metres') from exc
    if not math.isfinite(size) or size <= 0:
        raise ValueError('physical_layers requires a positive physical size in metres')
    return size


def metal_family(name, classes, authored):
    """Do not infer iron corrosion from metalness alone."""
    if authored:
        return 'partitioned'
    if re.search(r'\b(?:alumin\w*|brass|bronze|copper|stainless)\b', name, re.I):
        return 'nonferrous_or_stainless'
    if re.search(r'\b(?:steel|iron|cast.iron)\b', name, re.I):
        return 'ferrous'
    # Whole-asset semantics cannot distinguish an unknown insert from its body.
    return 'unknown'


def prepare_finish(asset_or_mesh, cache, pack, size_m, profile=None):
    """Return the worker payload; changing the background cannot change its hash.

    ``asset_or_mesh`` should be the prepared asset dict (mesh, mesh_hash,
    semantic.material_classes). A path is supported with conservative semantics.
    ``size_m`` is the physical length represented by a canonical mesh long axis 1.
    """
    size = metric_size(size_m)
    asset = dict(asset_or_mesh) if isinstance(asset_or_mesh, dict) else {'mesh': str(asset_or_mesh)}
    asset['mesh'] = str(Path(asset['mesh']).resolve())
    actual_hash = file_hash(asset['mesh'])
    if asset.get('mesh_hash') and asset['mesh_hash'] != actual_hash:
        raise ValueError('physical material source mesh hash mismatch')
    asset['mesh_hash'] = actual_hash
    if profile and profile.get('source_mesh_hash') != actual_hash:
        raise ValueError('asset profile is bound to a different source mesh')
    asset.setdefault('semantic', {'material_classes': []})
    classes = asset['semantic'].get('material_classes', [])
    source = load_pack(pack) if not isinstance(pack, dict) else pack
    maps = {name: dict(source['maps'][name]) for name in ('oxide', 'paint', 'dust')}
    for entry in maps.values():
        if file_hash(entry['path']) != entry['sha256']:
            raise ValueError('physical material wear map hash mismatch')
    raw = finish_regions(asset, source, cache)
    if actual_hash in source.get('atlases', {}):
        # A documented atlas can cover one body material while the asset also
        # has separate valve/label/polymer materials. Audit and preserve those
        # through the same conservative rules instead of dropping their records.
        inferred, _ = prepare_uv_masks(asset, Path(cache) / 'physical-source-uv')
        raw = {**raw, 'materials': {**inferred['materials'], **raw['materials']}}
    materials = {}
    for name, evidence in raw['materials'].items():
        entry = dict(evidence)
        authored = bool(entry.get('authored_atlas'))
        protected = not authored and bool(re.search(PROTECTED, name, re.I))
        # The source atlas may contain protected inserts: the audited zero-mask
        # cells take precedence over its mixed-material name.
        entry.update(metal_family=metal_family(name, classes, authored), protected=protected)
        if profile:
            entry.update(profile.get('material_evidence', {}).get(name, {}))
        entry['eligible'] = not protected and any(entry.get('coverage', [])[:3])
        entry['skip_reason'] = ('protected_source_material' if protected else
                                'unknown_source_region' if not entry['eligible'] else None)
        materials[name] = entry
    if profile:
        from .source_detail import prepare_source_detail
        materials=prepare_source_detail(asset,materials,Path(cache)/'source-detail-variants',profile)
    regions = {**raw, 'materials': materials, 'signature': signature(VERSION, materials)}
    audit = [{'material': name, 'eligible': row['eligible'], 'coverage': row.get('coverage'),
              'metal_family': row['metal_family'], 'basis': row.get('basis'),
              'skip_reason': row['skip_reason']} for name, row in materials.items()]
    recipe=dict(RECIPE)
    if profile:
        recipe.update(detail_mode='evidence_layers_v2',oxide_threshold=.13,
                      scratch_threshold=.025,oxide_max=.45,abrasion_max=.48,
                      oxide_period_m=.065,scratch_period_m=.045,dust_period_m=.055)
    identity = signature(VERSION, actual_hash, size, recipe, regions, profile,
                         {name: entry['sha256'] for name, entry in maps.items()},
                         file_hash(__file__), file_hash(Path(__file__).with_name('physical_layers.py')))
    return {'version': VERSION, 'signature': identity, 'source_mesh_hash': actual_hash,
            'manifest': source.get('manifest'), 'maps': maps, 'regions': regions,
            'size_m': size, 'recipe': recipe, 'layer_stage': 'full',
            'asset_profile': profile,
            'material_audit': audit, 'geometry_modified': False,
            'basis': 'source_material_and_metric_estimated_wear; no_background_material_fit'}


CONTROL_LIMITS={'oxide_gain':(.5,1.5),'scratch_gain':(.5,1.5),
                'roughness_shift':(-.08,.08),'relief_gain':(.5,1.5)}


def refine_finish(finish, controls):
    """Absolute, bounded controls; repeat application never compounds a recipe."""
    unknown=set(controls)-set(CONTROL_LIMITS)
    if unknown:
        raise ValueError('unknown physical material controls: '+','.join(sorted(unknown)))
    actual={key:float(controls.get(key,0. if key=='roughness_shift' else 1.)) for key in CONTROL_LIMITS}
    for key,value in actual.items():
        lo,hi=CONTROL_LIMITS[key]
        if isinstance(controls.get(key),bool) or not math.isfinite(value) or not lo<=value<=hi:
            raise ValueError('physical material control out of range: '+key)
    result=copy.deepcopy(finish)
    base=result.setdefault('base_signature',finish['signature'])
    result['controls']=actual
    result['signature']=signature(base,'absolute-material-controls-v1',actual)
    return result
