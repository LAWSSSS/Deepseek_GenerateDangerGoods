"""Conservative classification for render-time material adjustments."""
import re


def material_kind(name, metallic, metallic_linked, classes):
    name = name.lower()
    if metallic_linked and 'atlas' in name:
        return 'mixed'
    if re.search(r'gasket|seal|rubber|plastic|polymer', name):
        return 'polymer'
    if re.search(r'label|mark|glass', name):
        return 'unknown'
    if re.search(r'paint|enamel', name):
        return 'paint'
    if 'rust' in name:
        return 'rust'
    if re.search(r'steel|stainless|chrome|machined|brass|bronze|zerk', name) or (not metallic_linked and metallic >= .65):
        return 'metal'
    if classes and set(classes) <= {'rubber', 'plastic'}:
        return 'polymer'
    return 'unknown'


SURFACE_DUST = {'paint': .13, 'rust': .08, 'metal': .065, 'polymer': .035}
