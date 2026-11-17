"""Source-backed dimensions. Recognition is not itself a measurement."""
import json
import math
from pathlib import Path
from typing import Literal
from pydantic import Field
from .common import atomic_json, file_hash, signature, resolve
from .schema import Strict

VERSION = 'dimension-evidence-v1'
CATALOG = Path(__file__).with_name('specification_catalog.json')


class Identification(Strict):
    brand: str = Field(max_length=80)
    model: str = Field(max_length=80)
    visible_markings: list[str] = Field(max_length=8)
    confidence: Literal['clear', 'partial', 'unknown']
    feature: Literal['housing_length', 'body_length', 'extended_total_length', 'overall_length', 'unknown']
    extension: Literal['retracted', 'extended', 'unknown', 'not_applicable']
    evidence: str = Field(max_length=260)


class SceneReferenceObservation(Strict):
    entity_id: str = Field(max_length=32)
    identity: str = Field(max_length=100)
    visible_markings: str = Field(max_length=100)
    feature: str = Field(max_length=100)
    endpoints: list[tuple[float,float]] = Field(min_length=2,max_length=2)
    status: Literal['identifiable_but_dimension_unverified','category_estimate','no_dimension_evidence']
    evidence: str = Field(max_length=180)


class SceneReferenceAudit(Strict):
    entities: list[SceneReferenceObservation] = Field(max_length=8)
    evidence: str = Field(max_length=260)


class FeatureConfirmation(Strict):
    feature_visible: bool
    longest_extent_matches_feature: bool
    evidence: str = Field(max_length=220)


async def scene_observations(scene,vision,cache):
    path=Path(cache)/'reference-observations'/(signature(VERSION,scene['key'],vision.endpoint.model_dump())+'.json')
    if path.is_file():return json.loads(path.read_text(encoding='utf8'))
    answer=await vision.ask('scene-reference-audit',SceneReferenceAudit,
        'Audit possible REAL SIZE references in this scrap scene. Only identify visible entities and dimensioned '
        'features; do not invent dimensions, manufacturer, nominal steel sizes or tube diameter from appearance. '
        'Arbitrary cut edges and guessed hopper width are category estimates. Read any actual markings. '
        'Use normalized image endpoints of the specific feature. These observations alone NEVER certify size. '
        'No usable reference is a valid result. No metric depth may be inferred from a neural relative-depth map.',
        {'existing_estimated_references':[{'id':s['id'],'feature':s.get('label'),'endpoints':s.get('reference')}
                                        for s in scene['supports'] if s['id'].startswith('S')]},[scene['path']],
        image_limits=[(1920,1080)],plain_json=True)
    result=answer.model_dump();atomic_json(path,result);return result


PROMPT = '''Read only visible labels/cast lettering in the ORIGINAL object photograph.
Return brand/model only when legible; never complete a partially visible identifier from memory.
Do not infer extinguisher capacity, cylinder stroke or product dimensions from typical appearance.
For cylinders distinguish barrel/body from the total at its current extension; if not known say unknown.
For a pillow-block bearing read the HOUSING marking (e.g. Pxxx), not an imagined bearing insert number.
No dimensions are requested: official specifications are matched separately by code.'''


def profile(asset, identification, registry=None):
    catalog = registry if registry is not None else json.loads(CATALOG.read_text(encoding='utf8'))
    identity = identification.model_dump() if hasattr(identification, 'model_dump') else identification
    lo, hi = map(float, asset['semantic']['major_mm'])
    result = {'version': VERSION, 'status': 'unknown', 'basis': 'category_estimate_not_specification',
              'identification': identity, 'nominal_mm': math.sqrt(lo*hi), 'interval_mm': [lo, hi],
              'source_interval_mm': [lo, hi], 'source_nominal_mm': math.sqrt(lo*hi),
              'user_factor': 1., 'legacy_display_factor_applied': False, 'unit': 'mm',
              'feature': 'overall_length_at_observed_extension', 'source': None,
              'reasons': ['exact_model_or_corresponding_dimension_not_confirmed']}
    # Exact brand AND model AND dimensioned feature. Similar product rows are never eligible.
    for row in catalog.get('specifications', []):
        exact = (identity.get('confidence') == 'clear' and
                 identity.get('brand', '').strip().casefold() == row['brand'].casefold() and
                 identity.get('model', '').strip().casefold() == row['model'].casefold() and
                 identity.get('feature') == row['feature'])
        source = row.get('source', {})
        if not exact or not source.get('sha256') or not source.get('url') or not source.get('page'):
            continue
        if row.get('extension') and identity.get('extension') != row['extension']:
            continue
        value = float(row['length_mm'])
        result.update(status='pass', basis='manufacturer_exact_marking_and_feature', nominal_mm=value,
                      source_nominal_mm=value, interval_mm=[value,value], source_interval_mm=[value,value],
                      feature=row['feature'], source=source, reasons=[], dimensions=row.get('dimensions', {}))
        break
    result['signature'] = signature(VERSION, asset['mesh_hash'], identity, catalog, result)
    return result


async def identify(asset, vision, cache, registry_path=None):
    original = asset.get('original')
    key = signature(VERSION, file_hash(original) if original else asset['mesh_hash'], PROMPT,
                    vision.endpoint.model_dump())
    path = Path(cache)/'dimensions'/(key+'.json')
    if path.is_file():
        recognition = Identification.model_validate_json(path.read_text(encoding='utf8'))
    else:
        recognition = await vision.ask('dimension-identification', Identification, PROMPT,
            {'revision': asset['name']}, [original] if original else [asset['board']])
        atomic_json(path, recognition.model_dump())
    registry = json.loads(resolve(registry_path).read_text(encoding='utf8')) if registry_path else None
    catalog=registry if registry is not None else json.loads(CATALOG.read_text(encoding='utf8'))
    confirmation=None
    matches=[r for r in catalog.get('specifications',[]) if recognition.confidence=='clear' and
             recognition.brand.casefold()==r['brand'].casefold() and recognition.model.casefold()==r['model'].casefold()]
    if len(matches)==1:
        row=matches[0]
        confirmation=await vision.ask('dimension-feature-match',FeatureConfirmation,
            'The ORIGINAL marking exactly matches the supplied official catalog entry. Verify only the dimensioned '
            'FEATURE in the photo and current model: does that feature span the model longest extent? '
            'Housing foot-to-foot length is distinct from shaft bore, axis height, mounting centres or cylinder stroke. '
            'Reject if model has extra parts extending farther, is incomplete, or feature is not identifiable. '
            'Do not estimate any dimension from appearance.',{'specification':row,'read_markings':recognition.visible_markings},
            [original,asset['board']],image_limits=[(1280,1000),(1000,1000)],plain_json=True)
        if confirmation.feature_visible and confirmation.longest_extent_matches_feature:
            recognition=recognition.model_copy(update={'feature':row['feature']})
        else:
            recognition=recognition.model_copy(update={'feature':'unknown'})
    result = profile(asset, recognition, registry)
    if confirmation:result['feature_confirmation']=confirmation.model_dump()
    result['recognition_cache'] = str(path)
    atomic_json(Path(cache)/'dimensions'/(result['signature']+'-profile.json'), result)
    return result
