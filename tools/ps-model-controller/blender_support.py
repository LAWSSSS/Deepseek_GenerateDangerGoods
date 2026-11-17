"""Compute a conservative support graph for settled Blender mesh components."""
from mathutils.bvhtree import BVHTree


def support_graph(parts,vertices,span):
    trees=[BVHTree.FromPolygons(points,[list(poly.vertices) for poly in part.data.polygons])
           for points,part in zip(vertices,parts)]
    collisions=[]
    for i in range(len(trees)):
        for j in range(i+1,len(trees)):
            if trees[i].overlap(trees[j]):
                collisions.append([i,j])
    minimum=[min(point.z for point in points) for points in vertices]
    grounded=[i for i,z in enumerate(minimum) if -.005*span<=z<=.025*span]
    edges=[]
    for upper,points in enumerate(vertices):
        if upper in grounded:
            continue
        low=minimum[upper]
        candidates=[point for point in points if point.z<=low+.025*span]
        stride=max(1,len(candidates)//2000)
        for lower,tree in enumerate(trees):
            if lower==upper or minimum[lower]>=low-.005*span:
                continue
            supported=False
            for point in candidates[::stride]:
                hit=tree.find_nearest(point,.015*span)
                if hit[0] is not None:
                    location,normal,_,distance=hit
                    if (distance<=.015*span and location.z<=point.z+.003*span
                            and normal.z>.1):
                        supported=True
                        break
            if supported:
                edges.append([upper,lower])
    anchored=set(grounded)
    changed=True
    while changed:
        changed=False
        for upper,lower in edges:
            if lower in anchored and upper not in anchored:
                anchored.add(upper)
                changed=True
    return {"grounded":grounded,"support_edges":edges,"anchored":sorted(anchored),
            "all_anchored":len(anchored)==len(parts),"component_collisions":collisions}
