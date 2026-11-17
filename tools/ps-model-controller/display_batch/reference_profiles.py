"""Exact-mesh source-authoring evidence for reference material eligibility.

This is a material-domain audit, not a look recipe. The source GLB hash and
authoring request hash must match. Names from an AI reference, its rust colours,
and whole-object semantics never turn an unknown material into iron.

Atlas indexing follows the original authoring code: item 01 uses a top-first
2x2 image grid; item 25/27 use bottom-first 4x4 UV cells. Machined surfaces,
labels, polymers and ambiguous alloys deliberately remain source material.
"""
import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .common import file_hash, signature
from .surface_maps import save_image


VERSION = 'source-reference-regions-v1'
# Portable provenance: files are resolved beside the exact source mesh. These
# are hashes of source-authoring records, not copied private absolute paths.
PROFILES = {
    'ab8f217913365186f65eaf04980d93b03dcee077f06cba42883b043b6c158ae5': {
        'asset_id': '1', 'request_sha256': '81dc3f9040e6938c7663871f0ad1f09348472b59bd0610bf8dc669d69c46b3ba',
        'selectors': ['config.material_profile.cast', 'config.material_profile.machined', 'config.material_profile.blue_paint', 'config.material_profile.rust'],
        'builder': 'mesh_rebuild_item_worker.py:assign; mesh_rebuild_item.py:make_atlas',
        'atlas': {'material': 'Observed and inferred PBR atlas', 'grid': 2, 'origin': 'top',
                  'channels': {0: 0, 3: 0}, 'authored_atlas': True, 'metal_family': 'partitioned'},
        'reason': 'Author separates casting and rust cells; machined and blue-paint cells are preserved. The blue cell is also used by raised inscriptions.',
        'limits': ['Casting treated as the original iron/oxidation modelling assumption; exact alloy not measured.',
                   'Entire blue-paint cell is protected because handwheel and inscription share UVs.']},
    '9e3cfe9b1b0ddb6487a419bdff5ecc5c574ae51e6b7d799fd14b5a79c69515e0': {
        'asset_id': '25', 'request_sha256': '8d7b77992918bd5e792dda47d2ad5d6fbefe2a0018e53630914d073459d3658a',
        'selectors': ['config.texture.tiles', 'config.material_profile'],
        'builder': 'mesh_rebuild_25_worker.py:tile_bounds,assign; mesh_rebuild_item_25.py:tile_rect',
        'atlas': {'material': 'Observed blue paint, chrome, steel and bronze PBR atlas', 'grid': 4, 'origin': 'bottom',
                  'channels': {0: 1, 1: 1, 5: 0, 7: 1, 8: 0, 9: 1},
                  'authored_atlas': True, 'metal_family': 'partitioned'},
        'reason': 'Original 4x4 UV atlas identifies blue paint, dark steel, weld coating and rust. Chrome, bronze-colour alloy and rubber are protected.',
        'limits': ['Cell3 steel mixes machined bushings and small fittings; preserved conservatively.',
                   'Painted-iron substrate is an authored engineering estimate, not a measured alloy.',
                   'Blue hydraulic cylinder is asset25; this profile must not be applied to yellow asset31.']},
    'a141f20dc36f89e5ee0dd5b7fb0931a2d9b72e21854798001558ec24c71a1e67': {
        'asset_id': '27', 'request_sha256': '87e115cb2672f25be6cbfc521a3749624cf7e388519e2d2351dd0c5efd9b84fe',
        'selectors': ['config.texture.tiles', 'config.material_profile.blue_spring', 'config.material_profile.black_ano', 'config.material_profile.braid'],
        'builder': 'mesh_rebuild_27_worker.py:tile_bounds,assign; mesh_rebuild_item_27.py:tile_rect',
        'atlas': {'material': 'Observed black anodized, blue paint, chrome, steel and reservoi',
                  'grid': 4, 'origin': 'bottom', 'channels': {1: 1},
                  'authored_atlas': False, 'metal_family': 'unknown'},
        'protect': ['YU-AN cyan/white label', 'Silver band YU-AN etching'],
        'reason': 'Only the known blue coating cell receives surface wear. Original author says black anodized aluminium/steel ambiguously; no iron oxidation is authorized there.',
        'limits': ['Spring substrate is not composition-confirmed; coating wear retains its source response, with no exposed-iron or rust conversion.',
                   'Anodized reservoir, machined rod, bronze guide, stainless braid and all remaining cells stay source.']},
    '45f83ce3c1424820e1d6c43fe884dbcd97e22c3505157ff3125efb706ada0554': {
        'asset_id': '16', 'request_sha256': 'cf7d5cbcb9e7c80b20f850d3ddffde82dd4f4067b1c01119db766cf317aa5a47',
        'selectors': ['config.material_profile'], 'protect_all': True,
        'reason': 'Source author identifies body as dielectric printed label, cap as moulded plastic and base as aluminium; red rust in generated six views does not override that evidence.',
        'limits': ['No new rust material is applied. A future stain-only printed-label treatment would need its own safe domains.']},
    '5a51a1483b667e1c9024bdcab18018c21187a6ad26c0dee5ecad914af2f67478': {
        'asset_id': '36', 'request_sha256': 'f7505a8ab6e6233443cf0727b3bf4d864b5b5e4f16cc99b61818097983e51891',
        'selectors': ['config.material_profile.label', 'config.material_profile.cap', 'config.material_profile.base_metal', 'config.material_profile.valve'],
        'protect_all': True,
        'reason': 'Body is authored printed dielectric, grey cap and hidden valve are plastic, and exposed base is authored aluminium; protect all rather than turn the silver cap or base into rusty steel.',
        'limits': ['Visible material change may be zero. Report this honestly instead of claiming a successful retexture.']},
    'c45081cc0047abc2ef3601f15b150d88d374906680e9a7d671736435874f9ee8': {
        'asset_id': '17', 'request_sha256': 'aec8892be973d867bea933f0953432de24700660425d149bb74ea28047f5109f',
        'selectors': ['config.material_profile.logo_paint'], 'protect_all': True,
        'reason': 'Body is authored anodized with embedded dielectric AERO print; other generic material names do not distinguish alloys safely.',
        'limits': ['No blanket iron corrosion or source decal replacement.']},
    '013340ec08c49409482f39c982ee8fa427235f0d62f4f0c5d98fea820e490262': {
        'asset_id': '18', 'request_sha256': '098ef3d7e81930a73e3717b1acbaee09f07d8cbe886038e229ca85dc4f1ebad1',
        'selectors': ['config.material_profile'], 'protect_all': True,
        'reason': 'Anodized body/adjusters, integrated prints and rubber O-ring are distinct from steel bushings; bushings are protected machining surfaces.',
        'limits': ['Brown AI-reference marks are not evidence that all anodized parts are iron.']},
    '080b30140d78ad8a223a547a286de25ade1ed49e2d7deb3015b294b5bc6e477f': {
        'asset_id': '10', 'request_sha256': None,
        'selectors': ['exact_GLTF_material_names_after_approved_operating_bar_removal'],
        'ferrous_materials': ['10 weathered red enamel', '10 rusted red enamel', '10 worn red enamel', '10 clean red enamel'],
        'protect': ['steel', 'dark', 'chrome', 'saddle', 'bore', 'label'],
        'reason': 'Exact current authored red enamel domains belong to the jack body; separately authored ram, bore, saddle and label stay untouched.',
        'limits': ['Painted-steel substrate follows the original modelling assumption; it is not a material assay.']},
    'b90bf464cfc3b2b97e358380124735f811c0ec080b483392f360801703117285': {
        'asset_id': '38', 'request_sha256': '4dce0bfff92b2d7b659cbfc4960ee358fc794aacf53be2f7541c6ac31b59d9c3',
        'selectors': ['config.material_profile.red', 'config.material_profile.brass', 'config.material_profile.label'],
        'ferrous_materials': ['38 weathered red enamel', '38 clean red enamel', '38 rusted red enamel', '38 stained red enamel'],
        'protect': ['steel', 'brass', 'black', 'bore', 'label'],
        'reason': 'Painted vessel body is separate from valve fittings, black polymer, bores and label in source authoring.',
        'limits': ['Exact substrate alloy is not measured; no corrosion of the brass valve or label is introduced.']},
    '5e779052a25e4ec035857ab66031c9feddc62daf0d49873bb4eed2cf2d4b8e47': {
        'asset_id': '13', 'request_sha256': None, 'protect_all': True,
        'selectors': ['exact_GLTF_material_names_after_approved_single_bucket_revision'],
        'reason': 'Plastic protection control: white polymer and moulded mark retain the exact source material.',
        'limits': ['Brown reference dirt is not plastic oxidation.']},
}


def atlas_mask(grid, origin, channels, size=512):
    if origin not in ('top', 'bottom') or size % grid:
        raise ValueError('invalid audited atlas layout')
    image = np.zeros((size, size, 4), dtype=np.uint8)
    tile = size // grid
    for cell, channel in channels.items():
        cell, channel = int(cell), int(channel)
        if not 0 <= cell < grid*grid or channel not in (0, 1, 2):
            raise ValueError('invalid audited atlas cell')
        column, row = cell % grid, cell // grid
        if origin == 'bottom':
            row = grid-1-row
        image[row*tile:(row+1)*tile, column*tile:(column+1)*tile, channel] = 255
    return image


def _protect(entry, reason):
    result = dict(entry)
    for key in ('mask', 'mask_hash', 'constant', 'albedo_variant'):
        result.pop(key, None)
    result.update(coverage=[0., 0., 0., 0.], constant=[0., 0., 0., 0.],
                  eligible=False, protected=True, authored_atlas=False,
                  metal_family='unknown', skip_reason='source_authoring_protection', basis=reason)
    return result


def apply_reference_regions(asset, materials, cache):
    """Return independent material entries and an auditable source-evidence summary."""
    source = Path(asset['mesh']).resolve()
    actual = file_hash(source)
    if asset.get('mesh_hash') != actual:
        raise ValueError('reference material evidence mesh hash mismatch')
    rows = copy.deepcopy(materials)
    profile = PROFILES.get(actual)
    if not profile:
        return rows, {'version': VERSION, 'status': 'no_exact_source_profile',
                      'source_mesh_hash': actual, 'changes': [], 'unknown_regions_preserved': True}
    provenance = [{'file': source.name, 'sha256': actual, 'role': 'exact_source_mesh'}]
    request = source.with_name('request.json')
    if profile.get('request_sha256'):
        if not request.is_file() or file_hash(request) != profile['request_sha256']:
            raise ValueError('source authoring request missing or changed for asset ' + profile['asset_id'])
        provenance.append({'file': 'request.json', 'sha256': profile['request_sha256'],
                           'selectors': profile['selectors'], 'role': 'existing_source_authoring_not_new_reference'})
    changed = []
    if profile.get('protect_all'):
        for name in rows:
            rows[name] = _protect(rows[name], profile['reason'])
            changed.append({'material': name, 'action': 'preserve_source'})
    for name in profile.get('protect', []):
        if name in rows:
            rows[name] = _protect(rows[name], profile['reason'])
            changed.append({'material': name, 'action': 'preserve_source'})
    for name in profile.get('ferrous_materials', []):
        if name in rows:
            entry = rows[name]
            entry.update(metal_family='ferrous', protected=False,
                         eligible=any(entry.get('coverage', [])[:3]),
                         basis='source_authoring_confirmed_body_coating',
                         reference_evidence=profile['reason'])
            changed.append({'material': name, 'action': 'confirmed_body_coating_only'})
    atlas = profile.get('atlas')
    if atlas:
        name = atlas['material']
        if name not in rows:
            raise ValueError('audited atlas name missing from exact source mesh')
        key = signature(VERSION, actual, profile, file_hash(__file__))
        folder = Path(cache) / 'source-reference-regions' / key
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / 'known-regions.png'
        values = atlas_mask(atlas['grid'], atlas['origin'], atlas['channels'])
        save_image(Image.fromarray(values), path)
        entry = rows[name]
        for field in ('constant', 'albedo_variant'):
            entry.pop(field, None)
        entry.update(mask=str(path.resolve()), mask_hash=file_hash(path),
                     coverage=(values > 0).mean(axis=(0, 1)).tolist(), size=[512, 512],
                     authored_atlas=atlas['authored_atlas'], metal_family=atlas['metal_family'],
                     eligible=True, protected=False, skip_reason=None,
                     basis='exact_source_authoring_uv_cells', reference_evidence=profile['reason'])
        changed.append({'material': name, 'action': 'audited_uv_domains',
                        'atlas': atlas, 'coverage': entry['coverage'], 'mask_hash': entry['mask_hash']})
    summary = {'version': VERSION, 'status': 'applied_source_evidence',
               'asset_id': profile['asset_id'], 'source_mesh_hash': actual,
               'provenance': provenance, 'reason': profile['reason'],
               'limitations': profile.get('limits', []), 'changes': changed,
               'evidence_role': 'authoring_material_assignment_not_measured_alloy',
               'unknown_regions_preserved': True}
    return rows, summary
