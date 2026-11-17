"""Bind audited source assets to two reusable flat surface-reference packages.

Six-view sheets establish appearance identity and requested wear style. The
separate flat ImageGen tiles supply estimated texture channels. Neither is a
measurement of material composition, geometry, dimensions, or a BRDF.
"""
import argparse
import json
from pathlib import Path

from display_batch.common import atomic_json, file_hash, load_catalog, resolve, signature
from display_batch.reference_material import validate_reference


VERSION = 'reference-material-library-v1'
EXCLUDED = {'5', '6', '7'}
STEEL_STYLE = {'1', '2', '9'}
PROTECTED_CONTROLS = {'13', '16', '17', '18', '36'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def select_references(audit, catalog):
    """Only exact identity candidates are bound; style targets never become IDs."""
    selected = {}
    for row in audit.get('mappings', []):
        if row.get('mapping_status') != 'identity_style_reference':
            continue
        candidates = row.get('candidate_asset_ids', [])
        if len(candidates) != 1:
            raise ValueError('identity binding needs exactly one audited asset')
        aid = str(candidates[0])
        if aid in EXCLUDED:
            raise ValueError('excluded tank asset cannot be enabled by reference images')
        if aid in selected:
            raise ValueError('multiple identity references need explicit selection: ' + aid)
        if aid not in catalog:
            raise ValueError('reference asset missing from catalog: ' + aid)
        expected = next((entry for entry in row.get('asset_references', [])
                         if str(entry.get('asset_id')) == aid), None)
        if not expected or expected.get('mesh_sha256') != catalog[aid]['mesh_hash']:
            raise ValueError('identity audit belongs to a different source mesh: ' + aid)
        selected[aid] = row
    if not selected:
        raise ValueError('audit contains no active exact identity correspondence')
    return selected


def build(audit_path, steel_path, paint_path, output):
    audit_path, steel_path, paint_path, output = map(Path, (audit_path, steel_path, paint_path, output))
    audit = read(audit_path)
    if audit.get('version') != 'rust-reference-audit-v1':
        raise ValueError('unsupported identity audit')
    catalog = load_catalog()
    selected = select_references(audit, catalog)
    packages = {'steel': steel_path.resolve(), 'paint': paint_path.resolve()}
    package_evidence = {}
    for name, path in packages.items():
        reference = validate_reference(read(path), path.parent)
        package_evidence[name] = {'path': str(path), 'sha256': file_hash(path),
                                  'status': reference['status'], 'estimated': True,
                                  'binding_signature': reference['binding_signature'],
                                  'role': 'flat_generated_surface_motif_not_identity_or_measured_PBR'}
    assets = {}
    for aid in sorted(selected, key=lambda value: (int(value.split('-')[0]), value)):
        row = selected[aid]
        if file_hash(row['image_path']) != row['sha256']:
            raise ValueError('audited six-view input changed: ' + aid)
        style = 'steel' if aid in STEEL_STYLE else 'paint'
        source_hash = catalog[aid]['mesh_hash']
        own_source = next(entry for entry in row['asset_references'] if str(entry['asset_id']) == aid)
        assets[source_hash] = {'asset_id': aid, 'reference_path': str(packages[style]),
                              'reference_sha256': package_evidence[style]['sha256'],
                              'surface_package': style,
                              'surface_selection_basis': 'dominant_source_authored_body_class; exact_region_masks_decide_where_application_is_allowed',
                              'control': 'source_material_protection' if aid in PROTECTED_CONTROLS else 'eligible_domains_only',
                              'evidence': {'mapping_status': 'identity_style_reference',
                                  'candidate_asset_ids': [aid],
                                  'identity_confidence': row['identity_confidence'],
                                  'identity_evidence': row['identity_evidence'],
                                  'six_view_path': row['image_path'], 'six_view_sha256': row['sha256'],
                                  'six_view_role': 'identity_and_requested_wear_style_not_texture_or_geometry_truth',
                                  'source_asset': own_source,
                                  'material_roles': row['material_roles'],
                                  'allowed_transfer': row['allowed_transfer'],
                                  'cautions': row['cautions'],
                                  'flat_texture_role': 'independent_generated_surface_reference_with_estimated_channels'}}
    result = {'version': VERSION, 'algorithm': 'audited-identity-flat-style-binding-v1',
              'identity_audit': {'path': str(audit_path.resolve()), 'sha256': file_hash(audit_path)},
              'packages': package_evidence, 'assets': assets,
              'excluded_asset_ids': sorted(EXCLUDED), 'protected_control_asset_ids': sorted(PROTECTED_CONTROLS),
              'unbound_style_only_references': [row['image_path'] for row in audit['mappings'] if row['mapping_status'] == 'class_style_only'],
              'production_default_changed': False, 'source_assets_modified': False,
              'limits': ['Two reusable motif packages are not per-object material measurements.',
                         'The selected style package never overrides source-material protection or unknown-region rules.',
                         'Plastic, labels, machined parts and ambiguous alloys may show no change; count these as protection controls.',
                         'Blue hydraulic cylinder reference belongs to25, not31;31 is not bound.',
                         'Six-view layout, text, white background, object geometry and photographic illumination are not pasted into source UVs.']}
    result['signature'] = signature(result)
    if output.exists() and read(output).get('signature') != result['signature']:
        raise FileExistsError('reference library already exists with other inputs; use a new path')
    atomic_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', required=True)
    parser.add_argument('--steel', required=True, help='validated steel reference-pbr entry.json')
    parser.add_argument('--paint', required=True, help='validated paint reference-pbr entry.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = build(resolve(args.audit), resolve(args.steel), resolve(args.paint), resolve(args.output))
    print(json.dumps({'library': str(resolve(args.output)), 'assets': len(result['assets']),
                      'asset_ids': [entry['asset_id'] for entry in result['assets'].values()],
                      'signature': result['signature']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
