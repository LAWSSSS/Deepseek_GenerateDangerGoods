"""Blender shading for the optional scanned material and overcast light trial."""
import re

import bpy
import numpy as np


def original(socket):
    if socket.is_linked:
        return socket.links[0].from_socket
    value = socket.default_value
    return float(value) if isinstance(value, (int, float)) else tuple(value)


def assign(tree, value, socket):
    if isinstance(value, bpy.types.NodeSocket):
        tree.links.new(value, socket)
    else:
        socket.default_value = value


def math_node(tree, operation, a, b, *, clamp=False):
    node = tree.nodes.new("ShaderNodeMath")
    node.operation = operation
    node.use_clamp = clamp
    assign(tree, a, node.inputs[0])
    assign(tree, b, node.inputs[1])
    return node.outputs[0]


def mix(tree, weight, a, b, *, blend="MIX"):
    node = tree.nodes.new("ShaderNodeMixRGB")
    node.blend_type = blend
    assign(tree, weight, node.inputs[0])
    for index, value in enumerate((a, b), start=1):
        if isinstance(value, (int, float)):
            value = (value, value, value, 1.)
        assign(tree, value, node.inputs[index])
    return node.outputs[0]


def texture(tree, path, uv, name):
    node = tree.nodes.new("ShaderNodeTexImage")
    node.label = name
    node.image = bpy.data.images.load(path, check_existing=True)
    node.image.colorspace_settings.name = "Non-Color"
    tree.links.new(uv, node.inputs["Vector"])
    return node


def scanned_surface(material, shader, profile, dust_color):
    """Retain source metalness, alpha and geometry. Mask all supplemental surface work."""
    tree = material.node_tree
    name = re.sub(r"\.\d{3}$", "", material.name)
    entry = profile["partitions"]["materials"].get(name)
    if not entry or not any(entry["coverage"][:3]):
        return False
    dust = tree.nodes.get("Scanned surface dust")
    if dust:
        dust.inputs["Base Color"].default_value = (*dust_color, 1.)
        return True
    output = next((n for n in tree.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output), None)
    if not output or not output.inputs["Surface"].is_linked or output.inputs["Surface"].links[0].from_node != shader:
        return False
    alpha = shader.inputs["Alpha"]
    transmission = shader.inputs["Transmission Weight"]
    if alpha.is_linked or alpha.default_value < .999 or transmission.is_linked or transmission.default_value > .02:
        return False
    uv = tree.nodes.new("ShaderNodeTexCoord").outputs["UV"]
    scaled = tree.nodes.new("ShaderNodeVectorMath")
    scaled.operation = "SCALE"
    assign(tree, uv, scaled.inputs[0])
    scaled.inputs["Scale"].default_value = 2.4
    tiled = scaled.outputs["Vector"]
    if entry.get("mask"):
        part = texture(tree, entry["mask"], uv, "Source UV material regions")
        separate = tree.nodes.new("ShaderNodeSeparateColor")
        tree.links.new(part.outputs["Color"], separate.inputs["Color"])
        weights = [separate.outputs[channel] for channel in ("Red", "Green", "Blue")]
    else:
        weights = entry["constant"][:3]
    normal_input = shader.inputs["Normal"]
    if not normal_input.is_linked:
        normal_node = tree.nodes.new("ShaderNodeNormalMap")
        normal_node.inputs["Color"].default_value = (.5, .5, 1., 1.)
        tree.links.new(normal_node.outputs["Normal"], normal_input)
    else:
        normal_node = normal_input.links[0].from_node
    normal_color = original(normal_node.inputs["Color"]) if normal_node.type == "NORMAL_MAP" else None
    base = original(shader.inputs["Base Color"])
    roughness = original(shader.inputs["Roughness"])
    coverage = 0.
    for channel, scan_id in enumerate(("rusty_metal_03", "rusty_painted_metal", "rusty_metal_03")):
        if not entry["coverage"][channel]:
            continue
        weight = weights[channel]
        coverage = math_node(tree, "ADD", coverage, weight, clamp=True)
        scan = profile["maps"][scan_id]
        detail = texture(tree, scan["detail"], tiled, "Scanned albedo detail").outputs["Color"]
        gain = math_node(tree, "ADD", math_node(tree, "MULTIPLY", detail, .26), .87)
        base = mix(tree, weight, base, gain, blend="MULTIPLY")
        scan_roughness = texture(tree, scan["roughness"], tiled, "Scanned roughness").outputs["Color"]
        mapped = math_node(tree, "ADD", math_node(tree, "MULTIPLY", scan_roughness, .46),
                           (.18, .30, .38)[channel], clamp=True)
        blend = math_node(tree, "MULTIPLY", weight, (.18, .34, .28)[channel])
        roughness = mix(tree, blend, roughness, mapped)
        if normal_color is not None:
            micro = texture(tree, scan["micro_normal"], tiled, "Scanned fine normal").outputs["Color"]
            normal_color = mix(tree, math_node(tree, "MULTIPLY", weight, .32), normal_color, micro)
    assign(tree, base, shader.inputs["Base Color"])
    assign(tree, roughness, shader.inputs["Roughness"])
    if normal_color is not None:
        assign(tree, normal_color, normal_node.inputs["Color"])

    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    up = tree.nodes.new("ShaderNodeVectorMath")
    up.operation = "DOT_PRODUCT"
    up.inputs[1].default_value = (0., 0., 1.)
    tree.links.new(geometry.outputs["True Normal"], up.inputs[0])
    top = math_node(tree, "MAXIMUM", up.outputs["Value"], 0.)
    amount = math_node(tree, "ADD", .045, math_node(tree, "MULTIPLY", top, .18))
    amount = math_node(tree, "MULTIPLY", coverage, amount, clamp=True)
    dust = tree.nodes.new("ShaderNodeBsdfPrincipled")
    dust.name = "Scanned surface dust"
    dust["display_deposit"] = True
    dust.inputs["Base Color"].default_value = (*dust_color, 1.)
    dust.inputs["Roughness"].default_value = .88
    dust.inputs["Metallic"].default_value = 0.
    dust.inputs["Emission Strength"].default_value = 0.
    dust.inputs["Specular IOR Level"].default_value = .20
    tree.links.new(normal_input.links[0].from_socket, dust.inputs["Normal"])
    layer = tree.nodes.new("ShaderNodeMixShader")
    assign(tree, amount, layer.inputs[0])
    tree.links.new(shader.outputs[0], layer.inputs[1])
    tree.links.new(dust.outputs[0], layer.inputs[2])
    tree.links.new(layer.outputs[0], output.inputs["Surface"])
    return True


def overcast_pixels(state, hdri, environment):
    """Use a broad sky/ground gradient measured from the HDRI and locally tint its energy."""
    cached = state.get("scanned_hdri")
    if cached is None or cached[0] != hdri["sha256"]:
        image = bpy.data.images.load(hdri["path"], check_existing=False)
        try:
            w, h = image.size
            if (w, h) != (1024, 512):
                raise ValueError("Expected the checked 1K equirectangular HDRI")
            raw = np.empty(w * h * 4, np.float32)
            image.pixels.foreach_get(raw)
        finally:
            bpy.data.images.remove(image)
        rgb = raw.reshape(128, 4, 256, 4, 4)[..., :3].mean(axis=(1, 3))
        if not np.isfinite(rgb).all() or float(rgb.max()) <= 0:
            raise ValueError("Invalid linear HDR environment")
        # No local direction measurement. Average azimuth to avoid imposing the source horizon.
        axis = np.clip(rgb, 0., np.percentile(rgb, 98)).mean(axis=1, keepdims=True)
        luminance = axis @ np.float32([.2126, .7152, .0722])
        normalized = (.15 * axis + .85 * luminance[..., None]) / max(.001, float(np.mean(luminance[64:])))
        state["scanned_hdri"] = (hdri["sha256"], normalized)
    else:
        normalized = cached[1]
    local = np.asarray(environment["world_color"], np.float32)
    return np.clip(normalized * local * 1.65, 0, 4.)
