"""Build and publish one item from immutable, locally registered sources."""
import json
from pathlib import Path
from time import monotonic

from PIL import Image, ImageOps

from scrap_composite.data.catalog import load_catalog
from scrap_composite.data.files import FileLease, atomic_json, file_hash, signature
from scrap_composite.data.registry import Registry, validate_artifacts
from scrap_composite.data.inventory import is_storage_tank
from scrap_composite.runtime import BLENDER, ROOT, enable_legacy_tools, resolve
from .contracts import BuildOptions
from .sources import VIEWS, check_source, glb_document, stage, stage_path

WORKER = Path(__file__).with_name('blender_worker.py')


def file_record(path):
    return {'path': str(path), 'sha256': file_hash(path)}


class ModelingService:
    def __init__(self, root=None):
        self.registry = Registry() if root is None else Registry(root)
        self.sources = self.registry.root / 'sources'

    def _stage(self, source, *, photo_only=False):
        if isinstance(source, tuple):
            name, stream = source
            return stage(self.sources, name, stream, photo_only=photo_only)
        return stage_path(self.sources, source, photo_only=photo_only)

    def items(self):
        catalog = load_catalog(workspace=self.registry.root)
        existing = self.registry.assets()
        return [
            {'id': key, 'name': current.get('name', row.get('name')), 'available': row.get('available', False),
             'published': current.get('published'), 'archived': row.get('archived', False)
             or is_storage_tank(row.get('number'))}
            for key in dict.fromkeys([*catalog, *existing])
            for row, current in [(catalog.get(key, {}), existing.get(key, {}))]
        ]

    def submit(self, *, name=None, asset_id=None, source=None, views=None, options=None):
        options = options or BuildOptions()
        views = views or {}
        if set(views) - set(VIEWS):
            raise ValueError('六视图方位无效')
        if asset_id is None and (not name or not name.strip() or len(name.strip()) > 100):
            raise ValueError('新物品需要 1–100 字的名称')
        parent = None
        stored = self.registry.assets().get(asset_id)
        asset = stored
        if asset_id is not None:
            if stored is None:
                legacy = load_catalog(workspace=self.registry.root).get(asset_id)
                if not legacy:
                    raise ValueError('物品编号不存在')
                asset = {key: legacy[key] for key in ('id', 'number', 'name')}
            builds = self.registry.builds(asset_id)
            parent = next((item for item in builds if item['state'] == 'ready'), builds[0] if builds else None)
        prior = parent['request'] if parent else {}
        legacy_axes = prior.get('legacy_axes', False)
        if source is not None:
            incoming = self._stage(source)
            legacy_axes = False
        elif prior.get('source'):
            incoming = prior['source']
        elif asset_id is not None and stored is None and legacy.get('available'):
            incoming = self._stage(Path(legacy['mesh']))
            legacy_axes = incoming['kind'] == 'glb'
        else:
            raise ValueError('请上传主照片、GLB 或 Blender 文件')
        references = {**prior.get('views', {}), **{key: self._stage(path, photo_only=True) for key, path in views.items()}}
        identity = incoming if incoming['kind'] == 'photo' else references.get('front')
        if not identity and asset_id is not None and stored is None and legacy.get('original'):
            identity = self._stage(Path(legacy['original']), photo_only=True)
        dimension = prior.get('dimension')
        if options.major_mm is not None:
            dimension = {'major_mm': options.major_mm, 'source': options.dimension_source or None}
        same_source = incoming['sha256'] == (prior.get('source') or {}).get('sha256')
        library = prior.get('material_library') if same_source else None
        if 'material_library' in options.model_fields_set:
            library = None
            if options.material_library:
                path = resolve(options.material_library)
                if not path.is_relative_to(ROOT) or path.suffix.lower() != '.json':
                    raise ValueError('材质库应为仓库内已校验的 JSON 清单')
                library = file_record(path)
        if options.operation == 'retexture' and not library:
            raise ValueError('重贴图需要绑定到源模型散列的部件材质库')
        verification = None
        if options.operation == 'verify' and parent and parent['state'] == 'ready' and source is None:
            validate_artifacts(parent, self.registry.directory(parent['id']))
            verification = parent['artifacts']['render']
        request = {
            'source': incoming, 'views': references, 'operation': options.operation,
            'dimension': dimension, 'material_library': library,
            'subject': options.subject or (prior.get('subject') if source is None else None),
            'legacy_axes': legacy_axes, 'verify_model': verification,
            'identity_photo': identity['path'] if identity else prior.get('identity_photo') if source is None else None,
            'identity_sha256': identity['sha256'] if identity else prior.get('identity_sha256') if source is None else None,
        }
        return self.registry.create(name.strip() if name else None, request, asset=asset)

    def run(self, ident):
        folder = self.registry.directory(ident)
        lease = FileLease(folder / '.build.lock')
        if not lease.acquire():
            raise ValueError('建模任务正在执行')
        start = monotonic()
        try:
            record = self.registry.get(ident)
            if record['state'] == 'ready':
                return record
            record.update(state='running', error=None)
            self.registry.save(record)
            try:
                self._run(record, folder)
            except Exception as error:
                record.update(state='failed', error=str(error)[:2000])
            record['seconds'] = round(record.get('seconds', 0) + monotonic()-start, 3)
            self.registry.save(record)
            return record
        finally:
            lease.release()

    def _run(self, record, folder):
        request = record['request']
        record['stage'] = 'reconstruction' if request['source']['kind'] == 'photo' else 'import'
        self.registry.save(record)
        original = check_source(request['source'])
        for image in request['views'].values():
            check_source(image)
        if request.get('identity_photo'):
            check_source({'path': request['identity_photo'], 'sha256': request['identity_sha256']})
        mesh = {**request['source'], 'legacy_axes': request['legacy_axes']}
        if request.get('verify_model'):
            check_source(request['verify_model'])
            mesh = {**request['verify_model'], 'legacy_axes': False}
        elif request['source']['kind'] == 'photo':
            from .reconstruction import photo_to_mesh
            mesh = photo_to_mesh(original, folder, self.registry.root / 'cache')
        if 'source_fidelity' in mesh:
            record['checks']['source_fidelity'] = mesh['source_fidelity']
        if request['views']:
            record['checks']['six_views'] = {'status': 'review', 'directions': sorted(request['views']),
                                            'reason': '保存对应视角作结构对照，遮挡和背面由人工确认'}
        if request.get('dimension'):
            record['checks']['dimension'] = {'status': 'review', **request['dimension']}
        else:
            record['checks']['dimension'] = {'status': 'unknown', 'reason': '缺少实物尺寸与来源'}
        finish = None
        if request.get('material_library') and not request.get('verify_model'):
            record['stage'] = 'material'
            self.registry.save(record)
            finish = self._material(check_source(request['material_library']), mesh, request['dimension'])
        payload = {'mesh': mesh['path'], 'legacy_axes': mesh['legacy_axes'], 'finish': finish,
                   'subject': 'object:PublishedSubject' if request.get('verify_model') else request.get('subject'),
                   'output': str(folder)}
        record['stage'] = 'blender_validation'
        self.registry.save(record)
        report = self._blender(folder, payload, mesh['sha256'])
        record['checks']['blender'] = report
        if finish:
            record['checks']['material'] = {'status': 'applied', 'signature': finish['signature']}
        artifacts = {'render': 'model.blend', 'preview': 'preview.glb',
                     'board': 'preview.jpg', 'verification': 'blender.json'}
        glb_document(folder / artifacts['preview'])
        self._board(folder, request)
        record.update(artifacts={kind: file_record(folder/name) for kind, name in artifacts.items()},
                      packed_subject=report['subject'], state='ready', stage='ready')
        validate_artifacts(record, folder)

    def _blender(self, folder, payload, mesh_hash):
        enable_legacy_tools()
        from mesh_runtime import run_bounded, safe_tail
        output = folder / 'blender.json'
        checkpoint = folder / 'blender-checkpoint.json'
        manifest = folder / 'blender-input.json'
        stamp = signature(payload, mesh_hash, file_hash(WORKER))
        if checkpoint.is_file():
            saved = json.loads(checkpoint.read_text(encoding='utf8'))
            if saved['signature'] == stamp and all(Path(path).is_file() and file_hash(path) == expected
                                                    for path, expected in saved['files'].items()):
                return json.loads(output.read_text(encoding='utf8'))
        if not BLENDER.is_file():
            raise ValueError('Blender 运行时未安装')
        atomic_json(manifest, payload)
        result = run_bounded([BLENDER, '-b', '--factory-startup', '--disable-autoexec', '--python-exit-code', '1',
                              '--python', WORKER, '--', manifest], log=folder/'blender.log', seconds=240)
        if result['state'] != 'ok' or not output.is_file():
            raise ValueError('Blender 导入或检查失败。' + safe_tail(folder/'blender.log', 1400))
        files = [folder/name for name in ('model.blend','preview.glb','front.png','quarter.png','blender.json')]
        atomic_json(checkpoint, {'signature': stamp, 'files': {str(path): file_hash(path) for path in files}})
        return json.loads(output.read_text(encoding='utf8'))

    def _material(self, library, mesh, dimension):
        from scrap_composite.generation.component_pbr import prepare
        from scrap_composite.generation.reference_material import lookup_reference

        reference = lookup_reference(library, mesh['sha256'])
        if 'component_regions' not in reference or Path(mesh['path']).suffix != '.glb':
            raise ValueError('该材质库需匹配具有同名部件的源 GLB')
        if not dimension:
            raise ValueError('物理尺度贴图需要物品长轴毫米数')
        regions = prepare({'mesh': mesh['path'], 'mesh_hash': mesh['sha256']}, reference, self.registry.root/'cache')
        return {'reference': reference, 'regions': regions, 'source_mesh_hash': mesh['sha256'],
                'size_m': dimension['major_mm']/1000, 'signature': signature(reference, regions, dimension),
                'layer_stage': 'full'}

    @staticmethod
    def _board(folder, request):
        paths = [folder/'front.png', folder/'quarter.png']
        paths += [Path(request['views'][view]['path']) for view in VIEWS if view in request['views']]
        columns = 2
        side = 512
        board = Image.new('RGB', (columns*side, ((len(paths)+columns-1)//columns)*side), '#e9eaeb')
        for index, path in enumerate(paths):
            with Image.open(path) as image:
                panel = ImageOps.contain(ImageOps.exif_transpose(image).convert('RGBA'), (side, side))
                x = (index % columns) * side + (side - panel.width)//2
                y = (index // columns) * side + (side - panel.height)//2
                board.paste(panel, (x, y), panel)
        board.save(folder/'preview.jpg', quality=91)

    def publish(self, asset_id, version):
        return self.registry.publish(asset_id, version)
