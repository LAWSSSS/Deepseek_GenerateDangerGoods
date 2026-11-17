"""Item-15 integration checks for the review-only aerosol-can candidate.

Run from the repository root with the project Python:

    & work/triposr-env/Scripts/python.exe -X utf8 -B tools/ps-model-controller/checks_rebuild_15.py

The checks read work/direct-deepseek/15-progress.json to find the final output
directory, then verify manifests, hashes, mesh topology, real openings, PBR
texture statistics, render boards and the independent review page.  They do
not turn the candidate into an accepted asset: the record must stay review /
manual_only with physical and visual review pending.
"""
import json
import re
import unittest
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from scene_v2 import digest

ROOT = Path(__file__).resolve().parents[2]
PROGRESS = ROOT / 'work/direct-deepseek/15-progress.json'
LIBRARY = ROOT / 'outputs/三维模型库-圆润复验/results.jsonl'
VIEWS = ('front', 'back', 'left', 'right', 'top', 'bottom')


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


class RebuildChecks15(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not PROGRESS.is_file():
            raise unittest.SkipTest('item 15 progress record is missing')
        cls.progress = load_json(PROGRESS)
        cls.out = Path(cls.progress['out'])
        if not (cls.out / 'result.json').is_file():
            raise unittest.SkipTest('item 15 candidate has not been generated')
        cls.record = load_json(cls.out / 'result.json')
        cls.worker = load_json(cls.out / 'worker-result.json')
        cls.page = load_json(cls.out / 'browser-check.json')
        cls.hashes = load_json(cls.out / 'hashes.json')
        cls.row = next(r for r in (json.loads(line) for line in
                                   LIBRARY.read_text(encoding='utf8').splitlines()
                                   if line.strip()) if r['number'] == 15)

    # ------------------------------------------------------------------
    # Manifest / review state
    # ------------------------------------------------------------------
    def test_manifest_hashes_and_manual_only_state(self):
        self.assertEqual(self.record['number'], 15)
        self.assertEqual(self.record['version'], 'spray15-parametric-v1')
        self.assertEqual(self.record['status'], 'review')
        self.assertEqual(self.record['model_review'], 'pending')
        self.assertEqual(self.record['scene_validation'], 'pending')
        self.assertTrue(self.record['manual_only'])
        self.assertEqual(Path(self.record['out']).resolve(), self.out.resolve())
        self.assertEqual(Path(self.record['mesh']).resolve(),
                         (self.out / 'mesh.glb').resolve())
        self.assertEqual(Path(self.record['blend']).resolve(),
                         (self.out / 'model.blend').resolve())
        self.assertEqual(self.record['mesh_sha256'], digest(self.out / 'mesh.glb'))
        self.assertEqual(self.record['blend_sha256'], digest(self.out / 'model.blend'))
        self.assertEqual(self.record['original_mesh_sha256'], self.row['mesh_sha256'])
        self.assertEqual(self.record['original_clean_sha256'], self.row['clean_sha256'])
        self.assertEqual(self.record['original_rust_sha256'], self.row['rust_sha256'])
        self.assertEqual(self.record['reference_sha256'], digest(Path(self.record['reference'])))
        self.assertIn(str(ROOT / 'references/parallel-six-view'), self.record['reference'])
        self.assertIn(str(ROOT / 'outputs/三维模型库-圆润复验'), self.record['original_mesh'])
        self.assertNotIn('scrap-composite', self.record['reference'])
        self.assertNotIn('scrap-composite', self.record['original_mesh'])

    # ------------------------------------------------------------------
    # Mesh and generated openings
    # ------------------------------------------------------------------
    def test_mesh_topology_and_scale(self):
        audit = self.record['mesh_audit']
        self.assertEqual(audit['parts'], 16)
        self.assertEqual(audit['closed_after_welding_seams'], audit['parts'])
        self.assertEqual(audit['open_parts'], [])
        self.assertEqual(audit['nonfinite_parts'], [])
        self.assertEqual(self.worker['parts_count'], 16)
        self.assertEqual(self.worker['self_intersecting_components'], {})
        self.assertEqual(self.record['checks']['mesh_integrity'], True)
        bounds = np.asarray(self.worker['bounds'])
        self.assertTrue(np.isfinite(bounds).all())
        size = bounds[1] - bounds[0]
        self.assertGreater(size[2], 0.25)
        self.assertLess(size[2], 0.28)
        self.assertGreater(size[0], 0.065)
        self.assertLess(size[0], 0.075)
        scene = trimesh.load(self.out / 'mesh.glb', force='scene')
        self.assertGreater(len(scene.geometry), 0)

    def test_real_window_and_nozzle_openings(self):
        opening = self.record['opening_checks']
        self.assertTrue(opening['window_opening']['passed'])
        self.assertTrue(opening['window_opening']['control_passed'])
        self.assertTrue(all(r['clearance_ok'] for r in
                            opening['window_opening']['rays']))
        self.assertTrue(opening['nozzle_opening']['passed'])
        self.assertTrue(opening['nozzle_opening']['control_passed'])
        self.assertTrue(all(r['through_ok'] for r in
                            opening['nozzle_opening']['rays']))
        self.assertEqual(opening['nozzle_face_holes'], 2)
        self.assertEqual(self.record['checks']['window_opening'], True)
        self.assertEqual(self.record['checks']['nozzle_opening'], True)

    # ------------------------------------------------------------------
    # Materials / textures
    # ------------------------------------------------------------------
    def test_pbr_textures_and_metal_response(self):
        for name, size in (('label_base', (2048, 2048)),
                           ('label_mr', (2048, 2048)),
                           ('label_normal', (2048, 2048)),
                           ('cap_base', (1024, 1024)),
                           ('cap_mr', (1024, 1024)),
                           ('metal_base', (1024, 1024)),
                           ('metal_mr', (1024, 1024)),
                           ('rust_base', (1024, 1024)),
                           ('dust_base', (1024, 1024))):
            path = self.out / f'{name}.png'
            self.assertTrue(path.is_file(), name)
            with Image.open(path) as image:
                self.assertEqual(image.size, size, name)
        self.assertTrue(self.record['label_texture_checks']['passed'])
        self.assertTrue(self.record['cap_texture_checks']['passed'])
        self.assertTrue(self.record['metal_texture_checks']['passed'])
        self.assertTrue(self.record['rust_texture_checks']['passed'])
        self.assertEqual(self.record['label_texture_checks']['mean_metallic'], 0.0)
        self.assertEqual(self.record['cap_texture_checks']['mean_metallic'], 0.0)
        self.assertGreater(self.record['metal_texture_checks']['mean_metallic'], 0.85)
        reflection = self.record['metal_reflection_check']
        self.assertEqual(reflection['status'], 'checked_bottom_metal')
        self.assertTrue(reflection['changed'])
        self.assertGreater(reflection['mean_abs_luma_difference'], 2.0)
        self.assertEqual(self.record['checks']['metal_reflection'], True)
        self.assertTrue(self.record['checks']['material_tiles'])

    # ------------------------------------------------------------------
    # Diagnostics / evidence boundaries
    # ------------------------------------------------------------------
    def test_source_and_reference_are_diagnostics_only(self):
        fidelity = self.record['source_fidelity_diagnostic']
        self.assertFalse(fidelity['acceptance_gate'])
        self.assertEqual(fidelity['holes'], 0)
        self.assertEqual(fidelity['filled_holes'], 0)
        self.assertGreaterEqual(fidelity['narrow_recall'], 0.80)
        self.assertEqual(self.record['source_view_hole_count'], 0)
        self.assertEqual(self.record['new_front_hole_count'], 0)
        self.assertEqual(len(self.record['reference_views']), 6)
        self.assertTrue(self.record['checks']['source_fidelity_is_diagnostic'])
        self.assertTrue(self.record['checks']['reference_diagnostic_only'])
        self.assertEqual(set(self.record['old_new_silhouette_iou']), set(VIEWS))
        self.assertEqual(self.record['reference_role'], 'visual_match_unverified')
        self.assertEqual(self.record['shape_basis'],
                         'source_photo_with_uncalibrated_six_view_reference')

    # ------------------------------------------------------------------
    # Renders, boards, page
    # ------------------------------------------------------------------
    def test_render_sets_and_boards_exist(self):
        renders = self.out / 'renders'
        for view in VIEWS:
            self.assertTrue((renders / f'new-{view}.png').is_file(), view)
            self.assertTrue((renders / f'old-{view}.png').is_file(), view)
        for i in range(24):
            self.assertTrue((renders / f'turn-{i:02d}.png').is_file())
        for name in ('detail-cap-front', 'detail-cap-top', 'detail-window',
                     'detail-nozzle', 'detail-cap-seam', 'detail-bottom',
                     'detail-bottom-rim', 'detail-label-front',
                     'detail-label-back', 'detail-label-left',
                     'detail-label-bottom', 'detail-surface',
                     'cap-only', 'cap-top', 'nozzle-only', 'nozzle-face',
                     'bottom-only', 'window-look', 'body-only',
                     'new-material-close', 'new-light-alternate',
                     'new-surface-alternate', 'bottom-light-alternate',
                     'old-material-close'):
            self.assertTrue((renders / f'{name}.png').is_file(), name)
        for board in ('六视图-新旧对照.jpg', '局部检查.jpg',
                      '结构开口-隔离检查.jpg', '材质-同光照对照.jpg',
                      '材质-变光检查.jpg', '表面反射-变光对照.jpg',
                      '转台-24方向.jpg'):
            path = self.out / board
            self.assertTrue(path.is_file(), board)
            self.assertGreater(path.stat().st_size, 10000, board)
        with Image.open(self.out / 'renders/new-front.png') as image:
            self.assertEqual(image.size, (768, 768))
        with Image.open(self.out / 'renders/nozzle-face.png') as image:
            alpha = np.asarray(image.getchannel('A'))
        self.assertGreater((alpha > 128).sum(), 1000)

    def test_independent_page_and_browser_record(self):
        page_path = self.out / 'review-15.html'
        self.assertTrue(page_path.is_file())
        html = page_path.read_text(encoding='utf8')
        self.assertIn('manual_only', html)
        self.assertIn('data-page-ready', html)
        self.assertNotIn('http://', html)
        self.assertNotIn('https://', html)
        self.assertNotIn('/api/', html)
        for match in re.finditer(r'src="([^"]+)"', html):
            source = match.group(1)
            if source.startswith('#'):
                continue
            self.assertTrue((self.out / source).is_file(), source)
        self.assertEqual(self.page['status'], 'passed')
        self.assertTrue(self.page['page_ready_marker'])
        self.assertTrue(self.page['screenshot_exists'])
        self.assertEqual(self.record['checks']['browser_page'], 'passed')
        self.assertEqual(self.record['page_check']['status'], 'passed')

    # ------------------------------------------------------------------
    # Hash manifest / progress
    # ------------------------------------------------------------------
    def test_hashes_and_progress_record(self):
        files = self.hashes['files']
        self.assertEqual(files['mesh.glb'], self.record['mesh_sha256'])
        self.assertEqual(files['model.blend'], self.record['blend_sha256'])
        self.assertEqual(files['label_base.png'],
                         digest(self.out / 'label_base.png'))
        self.assertEqual(self.hashes['inputs']['reference'],
                         self.record['reference_sha256'])
        self.assertEqual(self.hashes['inputs']['mesh'],
                         self.record['original_mesh_sha256'])
        self.assertEqual(self.progress['number'], 15)
        self.assertEqual(self.progress['status'],
                         'candidate_generated_pending_review')
        self.assertTrue(self.progress['manual_only'])
        self.assertEqual(self.progress['next_number'], 16)
        for key in ('out', 'mesh', 'blend'):
            path = Path(self.progress[key])
            self.assertTrue(path.is_absolute(), key)
            self.assertTrue(str(path).startswith(str(ROOT)), key)
            self.assertTrue(path.exists(), key)
        self.assertEqual(Path(self.progress['out']).resolve(), self.out.resolve())
        self.assertIn('pending_review', self.progress)
        self.assertGreaterEqual(len(self.progress['pending_review']), 5)
        self.assertEqual(self.progress['checks']['manual_only'], True)
        self.assertEqual(self.progress['checks']['model_review'], 'pending')
        self.assertEqual(self.progress['checks']['scene_validation'], 'pending')


if __name__ == '__main__':
    unittest.main(verbosity=2)
