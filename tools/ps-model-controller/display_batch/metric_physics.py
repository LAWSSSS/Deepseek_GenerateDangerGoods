"""Blender-only metre simulation and true render-mesh contact audit."""
import math
import json
from pathlib import Path
import numpy as np

VERSION='metre-contact-v3-upward-receivers'


def tolerances(length_m):
    if not .005<=length_m<=20:raise ValueError('invalid_physical_length_m')
    return {'margin_m':float(np.clip(length_m*.0005,.00002,.0005)),
            'contact_m':float(np.clip(length_m*.002,.0001,.0015)),
            'penetration_m':float(np.clip(length_m*.001,.00005,.001)),
            'motion_m':float(np.clip(length_m*.0002,.00001,.0002))}


def run(request, reset, rigid, make_receiver):
    import bpy
    from mathutils import Matrix, Vector
    from mathutils.bvhtree import BVHTree
    from mathutils.geometry import convex_hull_2d
    from .geometry import transformed, horizontal_angle
    unit=float(request['length_m']);tol=tolerances(unit)
    data=np.load(request['geometry']);v=data['vertices']*unit;faces=data['faces']
    proxy=json.loads(Path(request['proxy']).read_text(encoding='utf8'))
    if proxy.get('max_outward_error',0)*unit+2*tol['margin_m']>tol['contact_m']:
        raise ValueError('proxy_and_margin_exceed_contact_budget')
    com=np.asarray(proxy['centroid'])*unit
    receiver={**request.get('receiver',{})}
    receiver['patches']=[{**p,'vertices':(np.asarray(p['vertices'])*unit).tolist()}
                         for p in receiver.get('patches',[])]
    normal=np.asarray(request.get('plane_normal',[0,0,1]),float);normal/=np.linalg.norm(normal)
    high=bool(request.get('high_precision'));results=[]
    for index,initial in enumerate(request['initial_matrices'][:6]):
        scene=reset(request['device']);scene.unit_settings.system='METRIC';scene.unit_settings.scale_length=1
        mesh=bpy.data.meshes.new('metric_render_mesh');mesh.from_pydata((v-com).tolist(),[],faces.tolist());mesh.update()
        obj=bpy.data.objects.new('metric_compound',mesh);scene.collection.objects.link(obj)
        body=rigid(obj,'COMPOUND');body.mass=max(1e-6,float(proxy['volume'])*unit**3*7800)
        body.collision_margin=tol['margin_m']
        matrix=np.asarray(initial,float).copy();matrix[:3,3]*=unit
        obj.matrix_world=Matrix(matrix)@Matrix.Translation(Vector(com))
        points=transformed(v,matrix)
        # Lift only the initial state, never the evaluated result. Gravity decides contact.
        obj.location.z+=max(0,unit*.015-float((points@normal).min()))/normal[2]
        for part in proxy['parts']:
            mesh=bpy.data.meshes.new('metric_convex_part')
            mesh.from_pydata((np.asarray(part['vertices'])*unit-com).tolist(),[],part['faces']);mesh.update()
            child=bpy.data.objects.new('metric_part',mesh);scene.collection.objects.link(child)
            child.parent=obj;child.matrix_parent_inverse=Matrix.Identity(4)
            rb=rigid(child,'CONVEX_HULL');rb.collision_margin=tol['margin_m']
            rb.mass=max(1e-7,float(part['volume'])*unit**3*7800)
        ground=make_receiver({**request,'receiver':receiver},normal,True)
        ground.rigid_body.collision_margin=tol['margin_m']
        scene.gravity=(0,0,-9.81);world=scene.rigidbody_world
        world.substeps_per_frame=40 if high else 15;world.solver_iterations=120 if high else 60
        end=800 if high else 400;world.point_cache.frame_start=1;world.point_cache.frame_end=end
        last=None;quiet=0;move=999.;angle=999.
        for frame in range(1,end+1):
            scene.frame_set(frame)
            if frame%20:continue
            m=obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).matrix_world.copy()
            if last is not None:
                move=(m.translation-last.translation).length
                angle=math.degrees(m.to_quaternion().rotation_difference(last.to_quaternion()).angle)%360
                angle=min(angle,360-angle)
                quiet=quiet+1 if frame>=80 and move<=tol['motion_m'] and angle<=.15 else 0
                if quiet>=3:break
            last=m
        m=obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).matrix_world.copy()@Matrix.Translation(Vector(-com))
        matrix=np.asarray(m).copy();points=transformed(v,matrix);center=transformed(com[None],matrix)[0]
        floor_v=np.asarray([ground.matrix_world@p.co for p in ground.data.vertices])
        ground.data.calc_loop_triangles()
        floor_f=np.asarray([list(f.vertices) for f in ground.data.loop_triangles])
        tree=BVHTree.FromPolygons(floor_v.tolist(),floor_f.tolist(),all_triangles=True)
        from .contact_audit import contact_witnesses, support
        witnesses=contact_witnesses(points,faces,floor_v,floor_f,tol['contact_m'])
        contacts=[];gaps=[];penetrations=[];nearest_points=[]
        for point in witnesses:
            hit=tree.find_nearest(Vector(point))
            if hit[0] is None:continue
            delta=point-np.asarray(hit[0]);n=np.asarray(hit[1]);signed=float(delta@n)
            tangent=float(np.linalg.norm(delta-n*signed))
            if hit[3]<=tol['contact_m']:
                contacts.append(point);gaps.append(max(0,signed));nearest_points.append(list(hit[0]))
            if tangent<=tol['contact_m'] and signed < -tol['penetration_m']:
                penetrations.append(-signed)
        # Reverse witnesses catch a narrow receiver crossing a large render face
        # whose vertices/centroid all lie outside the finite receiving patch.
        object_tree=BVHTree.FromPolygons(points.tolist(),faces.tolist(),all_triangles=True)
        reverse=np.concatenate([floor_v,floor_v[floor_f].mean(1),
                                (floor_v[floor_f[:,0]]+floor_v[floor_f[:,1]])*.5])
        for q in reverse:
            hit=object_tree.find_nearest(Vector(q))
            if hit[0] is None or hit[3]>tol['contact_m']:continue
            p=np.asarray(hit[0]);ground_hit=tree.find_nearest(Vector(p))
            if ground_hit[0] is None or ground_hit[1].z<=.15:continue
            contacts.append(p);nearest_points.append(q.tolist());gaps.append(float(hit[3]))
        contacts=np.asarray(contacts).reshape(-1,3)
        support_info=support(contacts,center,points,tol['contact_m'],tol['motion_m'],quiet>=3)
        supported=support_info['supported']
        normalized=matrix.copy();normalized[:3,3]/=unit
        tilt=horizontal_angle(request['axis'],normalized)
        failures=[]
        if quiet<3:failures.append('not_converged')
        if penetrations:failures.append('ground_penetration')
        if not len(contacts):failures.append('floating')
        if not supported:failures.append('center_of_mass_unsupported')
        if tilt>15:failures.append('upright')
        results.append({'id':f'P{index}','matrix':normalized.tolist(),'stable':not failures,
            'failure_codes':failures,'axis_angle_deg':tilt,'height':float(np.ptp(points[:,2])/unit),
            'min_z':float((points@normal).min()/unit),'center_supported':supported,'frames':frame,
            'motion_m':move,'rotation_motion_deg':angle,'length_m':unit,'units':'metres',
            'tolerances':tol,'max_penetration_m':max(penetrations,default=0.),
            'contact_count':len(contacts),'max_contact_gap_m':max(gaps,default=None),
            'support_detail':support_info,'contact_method':'triangle_edge_and_bidirectional_receiver_witnesses',
            'contact_points':(contacts[::max(1,len(contacts)//1024)]/unit).tolist(),
            'receiver_points':(np.asarray(nearest_points).reshape(-1,3)[::max(1,len(contacts)//1024)]/unit).tolist(),
            'high_precision':high,'result_translation_adjusted':False,
            'collision_kind':'component_convex'})
    return {'version':VERSION,'poses':results,'mass_basis':'uniform_density_closed_volume_7800kg_m3_estimate',
            'length_m':unit,'assembly':request['assembly'],'tolerances':tol}
