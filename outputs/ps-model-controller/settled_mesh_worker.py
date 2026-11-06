"""Blender worker that retains triangle provenance through rigid-body settling."""
import json
from pathlib import Path
import sys

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blender_worker as worker


def triangles(obj, *, require_ids=False):
    mesh = obj.data
    mesh.calc_loop_triangles()
    attribute = mesh.attributes.get("source_face_id")
    if require_ids and attribute is None:
        raise ValueError("Face provenance was lost while separating components")
    matrix = obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).matrix_world
    vertices = np.asarray([tuple(matrix @ v.co) for v in mesh.vertices], dtype=np.float32)
    faces = np.asarray([list(t.vertices) for t in mesh.loop_triangles], dtype=np.int32)
    ids = np.asarray([attribute.data[t.polygon_index].value if attribute else t.polygon_index
                      for t in mesh.loop_triangles], dtype=np.int32)
    return vertices[faces], ids


def main():
    argv = sys.argv[sys.argv.index("--")+1:]
    request = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    output = Path(request["out"])
    output.mkdir(parents=True, exist_ok=True)
    render_device = request.get("render_device", "CPU")
    scene = worker.setup_scene(render_device)
    obj = worker.load_mesh(request["mesh"])
    attribute = obj.data.attributes.new("source_face_id", "INT", "FACE")
    attribute.data.foreach_set("value", np.arange(len(obj.data.polygons),dtype=np.int32))
    original, source_ids = triangles(obj, require_ids=True)
    result = worker.settle_view({**request,"skip_unstable_renders":True}, scene, obj, output)
    result["render_device"] = render_device
    if not result["physics_stable"]:
        (output/"blender-result.json").write_text(json.dumps(result),encoding="utf-8")
        return
    posed = []
    ids = []
    for part in [o for o in scene.objects if o.type == "MESH" and o.name != "support"]:
        coords, face_ids = triangles(part, require_ids=True)
        posed.append(coords)
        ids.append(face_ids)
    if not posed:
        raise ValueError("No settled mesh components")
    posed = np.concatenate(posed)
    ids = np.concatenate(ids)
    if set(source_ids.tolist()) != set(ids.tolist()):
        raise ValueError("Settled geometry does not match reconstructed source faces")
    np.savez_compressed(output/"triangles.npz", source=original, source_ids=source_ids,
                        posed=posed, posed_ids=ids)
    result["geometry"]="triangles.npz"
    (output/"blender-result.json").write_text(json.dumps(result),encoding="utf-8")
    print("SETTLED_MESH_OK", flush=True)


if __name__=="__main__":
    main()
