"""Validate component PBR packages and bind them to an exact, unchanged GLB."""
import copy
import math
from pathlib import Path
from urllib.parse import urlsplit

from .common import file_hash, signature


VERSION = 'component-pbr-v1'
ROLES = {'steel', 'paint', 'oxide', 'rubber', 'plastic', 'nonferrous'}
AOV = 'component_domains'
FILES = {'albedo': 'sRGB', 'surface': 'Non-Color'}


def number(value, low, high, name):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not low <= value <= high):
        raise ValueError(f'{name} outside {low:g}–{high:g}')
    return float(value)


def aligned_maps(maps, base, name):
    from PIL import Image
    if set(maps) != set(FILES):
        raise ValueError(f'Incomplete or unexpected component channels: {name}')
    shape = None
    for kind, entry in maps.items():
        path = Path(entry['path'])
        if not path.is_absolute():
            if base is None:
                raise ValueError('relative component texture has no manifest base')
            path = Path(base) / path
        path = path.resolve()
        if file_hash(path) != entry['sha256']:
            raise ValueError('component texture hash changed: ' + kind)
        with Image.open(path) as image:
            if (image.mode != 'RGB' or min(image.size) < 16 or max(image.size) > 2048
                    or (shape is not None and shape != image.size)):
                raise ValueError(f'Misaligned or invalid component channels: {name}')
            shape = image.size
        entry.update(path=str(path), color_space=FILES[kind])


def observed_projection(record, base):
    """Validate a registered local texture frame and the two baked maps."""
    mode = record.get('mode')
    if mode not in ('cylinder', 'plane'):
        raise ValueError('Expected a cylindrical or planar reference projection')
    vectors = [record['axes'][key] for key in ('u', 'v', 'n')]
    for vector in vectors:
        if len(vector) != 3 or not math.isclose(math.hypot(*(number(x,-1.,1.,'projection axis') for x in vector)),
                                                 1., abs_tol=.001):
            raise ValueError('Reference projection needs unit vectors')
    pairs = ((vectors[0], vectors[1]), (vectors[0], vectors[2]), (vectors[1], vectors[2]))
    if any(abs(sum(a*b for a,b in zip(left, right))) > .001 for left,right in pairs):
        raise ValueError('Reference projection axes must be perpendicular')
    if len(record['origin']) != 3:
        raise ValueError('Reference projection needs a local origin')
    for value in record['origin']:
        number(value, -1., 1., 'projection origin')
    for axis in ('v',) if mode == 'cylinder' else ('u', 'v'):
        low, high = (number(value, -2., 2., f'projection limits {axis}') for value in record['limits'][axis])
        if high-low < .002:
            raise ValueError('Reference projection extent is too small')
    sha = record.get('source_sha256', '')
    if len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
        raise ValueError('Reference projection needs pinned source image bytes')
    aligned_maps(record['maps'], base, 'registered view')


def validate(reference, base=None):
    """Only aligned, content-addressed material channels enter the renderer."""
    result = copy.deepcopy(reference)
    if result.get('status') != 'review':
        raise ValueError('component material must be an appearance candidate')
    packs = result.get('packages', {})
    if not isinstance(packs, dict) or not packs:
        raise ValueError('component material needs a package dictionary')
    for name, pack in packs.items():
        role = pack.get('role')
        provenance = pack.get('source', {})
        if (role not in ROLES or provenance.get('license') != 'CC0-1.0'
                or urlsplit(provenance.get('url', '')).scheme != 'https'):
            raise ValueError(f'Missing role or public material provenance: {name}')
        number(pack['tile_m'], .002, 2., name + ' texture period')
        number(pack['relief_um'], 0., 80., name + ' relief')
        colour = pack['anchor_linear']
        if not isinstance(colour, list) or len(colour) != 3:
            raise ValueError(f'Invalid linear albedo anchor: {name}')
        for value in colour:
            number(value, .001, 1., name + ' linear albedo')
        number(pack['albedo_variation'], 0., 1., name + ' albedo variation')
        number(pack['albedo_median_luma'], .001, 1., name + ' albedo normalization')
        aligned_maps(pack['maps'], base, name)
    regions = result.get('component_regions')
    if not isinstance(regions, list) or not regions or len(regions) > 64:
        raise ValueError('component material needs 1–64 regions')
    used_parts = set()
    for index, region in enumerate(regions, 1):
        parts = region.get('parts')
        if (not isinstance(parts, list) or not parts or
                not all(isinstance(part, str) and part for part in parts) or
                len(set(parts)) != len(parts) or used_parts.intersection(parts)):
            raise ValueError('component part names must be nonempty and assigned once')
        used_parts.update(parts)
        region['id'] = index
        if not isinstance(region.get('material'), str) or not region['material']:
            raise ValueError('exact source material name required')
        if region.get('package') not in packs:
            raise ValueError('Unknown component material package')
        main = packs[region['package']]['role']
        steel = region.get('substrate')
        oxidized = region.get('oxide')
        if steel and (main != 'paint' or packs.get(steel, {}).get('role') != 'steel'):
            raise ValueError('exposed steel requires paint on an audited ferrous substrate')
        if oxidized and (packs.get(oxidized, {}).get('role') != 'oxide' or not (steel or main == 'steel')):
            raise ValueError('ferrous oxidation cannot be assigned to a nonferrous or polymer surface')
        number(region.get('wear', 1.), 0., 1., 'wear')
        for field, lo, hi in (('parent_roughness', .08, .95), ('relief_gain', 0., 1.),
                              ('paint_loss_um', 0., 40.), ('scuff_strength', 0., 1.),
                              ('source_roughness_mix', 0., .75), ('oxide_spread', 0., 1.),
                              ('oxide_grain_mix', 0., 1.), ('steel_patina', 0., .48)):
            if field in region:
                number(region[field], lo, hi, field)
        if (region.get('oxide_spread', 0.) or region.get('oxide_grain_mix', 0.)) and not oxidized:
            raise ValueError('Oxide detail requires an audited ferrous oxide layer')
        if region.get('steel_patina', 0.) and main != 'steel':
            raise ValueError('Steel patina requires a ferrous material')
        if 'view_projection' in region:
            projection = region['view_projection']
            role = projection.get('finish')
            if not ((role == 'oxide' and oxidized) or (role == 'dirt' and main == 'plastic')):
                raise ValueError('Reference appearance is incompatible with the part material')
            observed_projection(projection, base)
        if region.get('paint_loss_um', 0.) and not steel:
            raise ValueError('Paint film relief requires a separate underlying steel layer')
        if region.get('scuff_strength', 0.) and not steel:
            raise ValueError('Coating scuffs require an audited paint stack')
        if region.get('source_roughness_mix', 0.) and main not in ('rubber', 'plastic'):
            raise ValueError('Source roughness blending is limited to opaque polymers')
        if 'machining' in region:
            profile = region['machining']
            if main not in ('steel', 'nonferrous') or not isinstance(profile, dict):
                raise ValueError('Directional machining requires a metal part and an audited axis')
            axis = profile.get('axis')
            if not isinstance(axis, list) or len(axis) != 3:
                raise ValueError('Machining axis must contain three object-space components')
            length = math.hypot(*(number(v, -1., 1., 'machining axis') for v in axis))
            if not math.isclose(length, 1., abs_tol=.001):
                raise ValueError('Machining axis must be a unit vector')
            number(profile.get('anisotropy'), 0., .45, 'machining anisotropy')
        if 'preserve_print' in region and region['preserve_print'] not in ('white_on_red', 'white_on_black'):
            raise ValueError('unrecognized authored print mask')
    return result


def texture_records(reference):
    packs = [(entry['path'], entry['sha256'], f'{name}.{kind}')
             for name, pack in reference['packages'].items() for kind, entry in pack['maps'].items()]
    return packs + [(entry['path'], entry['sha256'], f'registered.{region["id"]}.{kind}')
                    for region in reference['component_regions'] if 'view_projection' in region
                    for kind, entry in region['view_projection']['maps'].items()]


def slot_name(region):
    return f'Component PBR {region["id"]:02d} {region["package"]}'


def print_guard(image, kind):
    """An enlarged soft mask for authored white ink. Coordinates stay in source UV."""
    import cv2
    import numpy as np
    r, g, b = (np.float32(image[..., index]) / 255. for index in range(3))
    neutral = np.minimum(np.minimum(r, g), b)
    if kind == 'white_on_red':
        ink = (neutral > .29) & (np.minimum(b, g) >= .49 * r)
    else:
        ink = (neutral > .19) & (np.maximum(np.maximum(r, g), b) - neutral < .22)
    # Dilate by the larger of two texels or 0.2% of the long side so the
    # original antialiased borders remain on both 1K and 2K atlases.
    radius = max(2, round(max(image.shape[:2]) * .002))
    mask = cv2.dilate(ink.astype(np.float32), cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)))
    soft = cv2.GaussianBlur(mask, (0, 0), max(.8, radius / 2))
    return np.uint8(np.rint(255 * np.maximum(mask, soft)))


def prepare(asset, reference, cache):
    from PIL import Image
    from .surface_maps import glb_images, save_image
    gltf, load_image = glb_images(asset['mesh'])
    source = {mat.get('name') or f'Material_{i}': mat for i, mat in enumerate(gltf)}
    materials = {name: {'eligible': False, 'components': [], 'skip_reason': 'no_component_assignment'}
                 for name in source}
    masks = {}
    for original in reference['component_regions']:
        region = copy.deepcopy(original)
        name = region['material']
        material = source.get(name)
        if (material is None or material.get('alphaMode', 'OPAQUE') != 'OPAQUE'
                or 'KHR_materials_transmission' in material.get('extensions', {})):
            raise ValueError(f'Component uses missing or non-opaque source material: {name}')
        if kind := region.get('preserve_print'):
            key = (name, kind)
            if key not in masks:
                pixels = load_image(material['pbrMetallicRoughness']['baseColorTexture'])
                mask = print_guard(pixels, kind)
                fraction = float((mask > 127).mean())
                if not 0 < fraction < .35:
                    raise ValueError(f'Unreliable printed-region mask on {name}: {fraction:g}')
                stamp = signature(VERSION, file_hash(__file__), asset['mesh_hash'], key)
                path = Path(cache).resolve() / 'component-pbr-regions' / stamp / 'source-print.png'
                path.parent.mkdir(parents=True, exist_ok=True)
                save_image(Image.fromarray(mask), path)
                masks[key] = {'path': str(path), 'sha256': file_hash(path), 'uv_fraction': fraction}
            region['print_mask'] = masks[key]
        region['slot'] = slot_name(region)
        materials[name]['components'].append(region)
        materials[name].update(eligible=True, skip_reason=None)
    return {'materials': materials, 'signature': signature(VERSION, materials)}
