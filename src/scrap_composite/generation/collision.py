"""Component-preserving collision proxies; never rewrite the render geometry."""
import importlib.metadata
import json
import time
import numpy as np
import trimesh
from .common import atomic_json, file_hash, signature
from pathlib import Path
from functools import lru_cache

VERSION = 'component-collision-v4-certified-contact'
# Normalized long axis is one. Two Bullet margins consume .001 of the .002
# contact band for the trial's physical sizes; reserve another .00025 for solve error.
OUTWARD_BUDGET = .00075


@lru_cache(maxsize=1)
def cache_signature():
    return signature(VERSION,file_hash(__file__),{
        name:importlib.metadata.version(name) for name in ('numpy','trimesh','coacd','rtree')})


def outward_error(source, parts):
    """Surface witnesses include face interiors, so a hull spanning a hole fails."""
    witnesses = []
    for hull in parts:
        witnesses.extend((hull.vertices, hull.triangles_center))
    points = np.unique(np.round(np.concatenate(witnesses), 9), axis=0)
    # Source vertices are exact boundary witnesses; the harder witnesses are new faces.
    _, distance, _ = trimesh.proximity.closest_point(source, points)
    select = distance > 1e-7
    outside = np.zeros(len(points), bool)
    outside[select] = ~source.contains(points[select])
    return float(distance[outside].max(initial=0)), int(len(points))


def build(geometry, path, refine=False):
    import coacd
    coacd.set_log_level('error')
    started = time.monotonic()
    data = np.load(geometry)
    mesh = trimesh.Trimesh(data['vertices'], data['faces'], process=True)
    pieces = mesh.split(only_watertight=False)
    closed = [p for p in pieces if p.is_watertight and abs(p.volume) > 1e-10]
    fraction = sum(p.area for p in closed) / max(mesh.area, 1e-9)
    if not closed or fraction < .8:
        raise ValueError('insufficient_closed_volume_for_mass')
    volume = sum(abs(p.volume) for p in closed)
    centroid = sum(p.center_mass * abs(p.volume) for p in closed) / volume
    parts = []; audits = []
    cache=Path(path).parent/'component-proxies';cache.mkdir(exist_ok=True)
    for index, source in enumerate(closed):
        if source.volume < 0: source.invert()
        origin=source.bounds.mean(axis=0)
        source.apply_translation(-origin)
        exact = bool(source.is_convex)
        attempts = []
        key=signature(cache_signature(),{'refine':refine},np.round(source.vertices,7).tolist(),source.faces.tolist())
        cached=cache/(key+'.json')
        print('component',index,'faces',len(source.faces),'convex',exact,'cached',cached.exists(),flush=True)
        if cached.exists():
            saved=json.loads(cached.read_text(encoding='utf8'))
            parts.extend({**p,'vertices':(np.asarray(p['vertices'])+origin).tolist(),'source_component':index} for p in saved['parts'])
            audits.append({**saved['audit'],'component':index,'cache_hit':True})
            continue
        for attempt in range(1 if exact or not refine else 2):
            mode = 'off' if source.is_volume else 'on'
            if exact or not refine:
                hulls = [source.convex_hull]
            else:
                raw = coacd.run_coacd(coacd.Mesh(source.vertices, source.faces),
                    threshold=.01 / (2 ** attempt),
                    preprocess_mode=mode, preprocess_resolution=100,
                    resolution=1000 if attempt == 0 else 2000,
                    mcts_nodes=4, mcts_iterations=1 if attempt == 0 else 3,
                    mcts_max_depth=2, merge=False, decimate=False, extrude=False, seed=17)
                hulls = [trimesh.Trimesh(v, f, process=False) for v, f in raw]
            surface_error, count = outward_error(source, hulls)
            # A component hull may fill an internal concavity, but cannot extend
            # its outer support envelope: all hull vertices come from that component.
            # Such a proxy is only usable behind the mandatory true-mesh audit.
            if not refine:
                cloud=np.concatenate([h.vertices for h in hulls])
                error=float(np.maximum(cloud.max(0)-source.vertices.max(0),source.vertices.min(0)-cloud.min(0)).max(initial=0))
            else:error=surface_error
            attempts.append({'outward_error':error, 'witnesses':count,
                'internal_surface_fill_error':surface_error,
                'hulls':len(hulls), 'preprocess': 'exact_convex' if exact else ('conservative_component_envelope' if not refine else mode)})
            if error <= OUTWARD_BUDGET: break
        audit = {'component': index, 'exact_convex':exact, 'attempts':attempts,
                 'concavities_require_true_mesh_contact_audit':not exact and not refine,
                 'status':'pass' if error <= OUTWARD_BUDGET else 'fail'}
        audits.append(audit)
        print('component result',index,attempts,flush=True)
        if audit['status'] == 'fail' or len(parts) + len(hulls) > 1024:
            atomic_json(str(path)+'.audit.json', {'version':VERSION, 'components':audits,
                'reason':'proxy_surface_error' if audit['status']=='fail' else 'proxy_complexity_limit'})
            raise ValueError('component_collision_failed: inspect proxy audit')
        local=[{'vertices':hull.vertices.tolist(), 'faces':hull.faces.tolist(),
                'volume':float(abs(hull.volume))} for hull in hulls]
        atomic_json(cached,{'parts':local,'audit':audit})
        parts.extend({**p,'vertices':(np.asarray(p['vertices'])+origin).tolist(),'source_component':index} for p in local)
    result = {'version':VERSION, 'parts':parts, 'centroid':centroid.tolist(),
        'volume':float(volume), 'closed_surface_fraction':float(fraction), 'components':audits,
        'outward_budget_normalized':OUTWARD_BUDGET,
        'max_outward_error':max(a['attempts'][-1]['outward_error'] for a in audits),
        'mass_basis':'uniform_density_closed_components; overlap volume remains estimated',
        'coacd_version':importlib.metadata.version('coacd'), 'geometry_hash':file_hash(geometry),
        'seconds':time.monotonic()-started, 'render_geometry_unchanged':True}
    result.update(collision_kind='component_convex',mandatory_true_mesh_contact=True,
        approximation='per_component_outer_envelopes_with_explicit_internal_fill' if not refine else 'component_coacd',
        concavity_filled_components=sum(a.get('concavities_require_true_mesh_contact_audit',False) for a in audits))
    atomic_json(path,result)
    return result
