"""Blender-only check of the pinned add-on, imported maps and optional surface layer."""
import json
from pathlib import Path
import sys

import addon_utils
import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from display_batch.common import atomic_json
from display_batch.scanned_shader import scanned_surface


def verify(manifest):
    root = manifest.parent
    library = json.loads(manifest.read_text(encoding="utf8"))
    addon_utils.enable("asset_importer", default_set=False)
    import asset_importer
    assert tuple(asset_importer.bl_info["version"]) == (2, 0, 0)
    assert addon_utils.check("asset_importer")[1]
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    report = {"blender": bpy.app.version_string, "addon": library["addon"], "maps": {}}
    names = {"base": "base_color", "roughness": "roughness", "normal": "normal", "metallic": "metallic"}
    for index, (name, maps) in enumerate(library["materials"].items()):
        expected = {category: root / maps[key]["path"] for key, category in names.items()}
        matches = asset_importer._classify_textures(list(expected.values()))
        assert {k: v["path"] for k, v in matches.items()} == expected
        material = asset_importer._create_node_wrangler_style_material(name, matches, use_ao_cavity=False)
        shader = next(n for n in material.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
        loaded = {}
        for key, socket in zip(names, ("Base Color", "Roughness", "Normal", "Metallic"), strict=True):
            assert shader.inputs[socket].is_linked
            path = expected[names[key]]
            image = next(n.image for n in material.node_tree.nodes if n.type == "TEX_IMAGE"
                         and Path(bpy.path.abspath(n.image.filepath)) == path)
            assert tuple(image.size) == (2048, 2048)
            assert image.colorspace_settings.is_data == (key != "base")
            loaded[key] = {"size": list(image.size), "color_space": image.colorspace_settings.name}
        report["maps"][name] = loaded
        profile = {"partitions": {"materials": {name: {"coverage": [0, 1, 0, 0],
                   "constant": [0, 1, 0, 0]}}},
                   "maps": {key: {k: str(root / v["path"]) for k, v in value.items()}
                            for key, value in library["materials"].items()}}
        test_material = material.copy()
        test_material.name = name + " trial"
        test_shader = next(n for n in test_material.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
        assert not scanned_surface(test_material, test_shader, profile, (.2, .2, .2))
        test_material.name = name + ".001"
        assert scanned_surface(test_material, test_shader, profile, (.2, .2, .2))
        assert scanned_surface(test_material, test_shader, profile, (.21, .21, .21))
        dust = test_material.node_tree.nodes.get("Scanned surface dust")
        assert dust.inputs["Metallic"].default_value == 0 and dust.inputs["Emission Strength"].default_value == 0
        assert sum(n.type == "MIX_SHADER" for n in test_material.node_tree.nodes) == 1
        assert test_shader.inputs["Alpha"].default_value == 1 and not test_shader.inputs["Alpha"].is_linked
        assert not next(n for n in test_material.node_tree.nodes
                        if n.type == "OUTPUT_MATERIAL").inputs["Displacement"].is_linked
        bpy.ops.mesh.primitive_uv_sphere_add(segments=40, ring_count=24, radius=.82,
                                            location=(-1 + 2 * index, 0, 0))
        obj = bpy.context.object
        obj.data.materials.append(material)
        for polygon in obj.data.polygons:
            polygon.use_smooth = True

    env = bpy.data.images.load(str(root / library["environment"]["path"]), check_existing=True)
    assert tuple(env.size) == (1024, 512) and env.is_float
    scene = bpy.context.scene
    scene.world.use_nodes = True
    nodes = scene.world.node_tree.nodes
    tex = nodes.new("ShaderNodeTexEnvironment")
    tex.image = env
    scene.world.node_tree.links.new(tex.outputs["Color"], nodes.get("Background").inputs["Color"])
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 12
    scene.cycles.use_denoising = True
    bpy.ops.object.camera_add(location=(0, -6, 2))
    scene.camera = bpy.context.object
    scene.camera.rotation_euler = (-Vector(scene.camera.location)).to_track_quat("-Z", "Y").to_euler()
    scene.camera.data.type = "ORTHO"
    scene.camera.data.ortho_scale = 4.4
    scene.render.resolution_x, scene.render.resolution_y = 384, 224
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(root / "material-preview.png")
    blend = root / "import-check.blend"
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    assert all(bpy.data.materials.get(name) for name in library["materials"])
    bpy.ops.render.render(write_still=True)
    assert (root / "material-preview.png").stat().st_size > 0
    report["roundtrip"] = True
    report["surface_layers_checked"] = len(library["materials"])
    report["preview"] = str(root / "material-preview.png")
    atomic_json(root / "blender-check.json", report)


if __name__ == "__main__":
    verify(Path(sys.argv[sys.argv.index("--") + 1]))
