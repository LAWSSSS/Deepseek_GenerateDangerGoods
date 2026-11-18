"""Generate and validate geometry candidates, with scene evidence retained."""
import asyncio
import json
import math
from pathlib import Path

import numpy as np

from .common import atomic_json, signature
from .geometry import contact_support
from .pile import support_union
from .policy import resize_candidate
from .scene import choose_candidates, fitted_masks, fit_checks, projection_geometry


class Placement:
    def __init__(self, config, output, cache, vision, blender, prepare_lock):
        self.config, self.out, self.cache = config, output, cache
        self.vision, self.blender, self.prepare_lock = vision, blender, prepare_lock

    async def choose_placements(self,asset,scene,seed):
        preview='estimated_preview_scale' in scene
        if self.config.geometry_preset=='rebuild' and not preview:
            from .vehicle_scale import prepare as prepare_vehicle, projection_preflight
            observation_mode='audited' if self.config.integration_mode=='coherent' else 'legacy'
            if ('vehicle_scale' not in scene or
                    scene['vehicle_scale'].get('observation_mode','legacy')!=observation_mode):
                scene['vehicle_scale']=await prepare_vehicle(scene,self.config.vehicle_reference,self.vision,self.cache,
                    observation_mode=observation_mode)
            scene['vehicle_scale']['enforce_projection_preflight']=self.config.integration_mode=='coherent'
            gate=scene['vehicle_scale'].get('projection_preflight') or projection_preflight(scene['vehicle_scale']['calibration'])
            if self.config.integration_mode=='coherent' and not gate['allow_expensive_steps']:
                atomic_json(self.out/'scale-scenes'/(scene['key']+'.json'),{'vehicle_scale':scene['vehicle_scale'],
                    'projection_preflight':gate})
                return [],[{'reason':'high_risk_vehicle_projection','evidence':gate}]
        if self.config.scale_mode=='reference_projection' or self.config.contact_mode=='strict':
            from .specifications import identify, scene_observations
            from .reference_projection import audit
            from .strict_contact import simulate
            from .assets import initial_poses
            async with self.prepare_lock:
                if 'physical_size' not in asset:
                    asset['physical_size']=await identify(asset,self.vision,self.cache,self.config.dimension_registry)
                if self.config.contact_mode=='strict' and not asset.get('metric_prepared'):
                    data=np.load(asset['geometry']);_,initial=initial_poses(data['vertices'])
                    physics=await simulate(asset,self.blender,initial)
                    poses=[p for p in physics['poses'] if p['stable']]
                    unique=[]
                    for p in sorted(poses,key=lambda q:q['height']):
                        if not any(np.linalg.norm(np.asarray(p['matrix'])[:3,:3]-np.asarray(q['matrix'])[:3,:3])<.05 for q in unique):unique.append(p)
                    poses=unique
                    ext=sorted(asset['inspection']['extents'])
                    if poses and ext[-1]/max(ext[-2],1e-6)<1.5:
                        low=min(p['height'] for p in poses);poses=[p for p in poses if p['height']<=low*1.15]
                    asset.update(poses=poses,metric_prepared=True,metric_physics=physics)
                    atomic_json(self.out/'physics-models'/(str(asset['id'])+'.json'),physics)
            scene['reference_audit']=audit(scene,self.config.scene_reference_registry)
            if preview:
                scene['reference_audit']['preview_scale']=scene['estimated_preview_scale']
            else:
                scene['reference_audit']['visual_observations']=await scene_observations(scene,self.vision,self.cache)
            if self.config.geometry_preset=='rebuild' and not preview:
                from .vehicle_scale import prepare as prepare_vehicle
                if 'vehicle_scale' not in scene:
                    scene['vehicle_scale']=await prepare_vehicle(scene,self.config.vehicle_reference,self.vision,self.cache)
                scene['reference_audit']['vehicle_scale']=scene['vehicle_scale']
            atomic_json(self.out/'scale-models'/(str(asset['id'])+'.json'),asset['physical_size'])
            atomic_json(self.out/'scale-scenes'/(scene['key']+'.json'),scene['reference_audit'])
            if not asset['poses']:
                return [],[{'reason':'no_valid_metric_lying_pose','evidence':asset.get('metric_physics')}]
            return await asyncio.to_thread(choose_candidates,asset,scene,seed,self.config)
        if self.config.scale_mode == 'physical_consistent':
            from .consistent_scale import scaled_asset, scene_calibration
            asset = scaled_asset(asset,self.config.asset_scale_factors.get(str(asset.get('id','')),1.))
            atomic_json(self.out/'scale-models'/(str(asset['id'])+'.json'),asset['physical_size'])
            atomic_json(self.out/'scale-scenes'/(scene['key']+'.json'),scene_calibration(scene))
            return await asyncio.to_thread(choose_candidates,asset,scene,seed,self.config)
        candidates,rejected=await asyncio.to_thread(choose_candidates,asset,scene,seed,self.config)
        factor=self.config.asset_scale_factors.get(str(asset.get('id','')),1.)
        if factor!=1:
            candidates=[await asyncio.to_thread(resize_candidate,asset,scene,c,factor) for c in candidates]
        return candidates,rejected

    async def validate_local_pose(self,asset,scene,candidate):
        if candidate.get('placement'):
            return await self.validate_pile_pose(asset,scene,candidate)
        normal=candidate['plane_normal']
        key=signature('terrain-physics-v1',asset['physics_signature'],candidate['matrix'],normal)
        path=Path(asset['folder'])/'terrain'/(key+'.json')
        hit=path.is_file()
        if hit:physics=json.loads(path.read_text(encoding='utf8'))
        else:
            physics=await self.blender.ask('settle',{'geometry':asset['geometry'],
                'proxy':str(Path(asset['folder'])/'collision.json'),'axis':asset['axis'],
                'initial_matrices':[candidate['matrix']],'assembly':asset['semantic']['assembly'],
                'plane_normal':normal},timeout=90)
            atomic_json(path,physics)
        pose=physics['poses'][0]
        if not pose['stable']:raise ValueError('local_plane_'+','.join(pose['failure_codes']))
        data=np.load(asset['geometry']);candidate=dict(candidate);candidate['matrix']=pose['matrix']
        site=next(s for s in scene['supports'] if s['id']==candidate['site_id'])
        minimum=1.;middle=candidate['elevation'];world_ppm=candidate['scale']['world_pixels_per_unit']
        for elevation in sorted(set([*candidate['elevation_interval'],middle])):
            geom=projection_geometry(data['vertices'],data['faces'],pose,0,elevation,normal)
            distance=np.linalg.norm(np.array(geom['camera']['eye'])-geom['camera']['target'])
            full_px=world_ppm*2*distance*math.tan(math.radians(geom['camera']['fov']/2))
            fit=fitted_masks(geom,candidate['anchor_px'],full_px,scene)
            if fit is None:raise ValueError('local_plane_out_of_frame')
            covers,protected,inside,occluded=fit_checks(fit,site,scene)
            if not covers or min(covers)<.9 or not protected or inside<.98 or occluded>.3:
                raise ValueError('local_plane_contact_or_clearance_failed')
            minimum=min(minimum,min(covers))
            if elevation==middle:
                x,y,r,b=fit['bounds']
                candidate.update(full_px=full_px,target_px=max(r-x,b-y),offset=fit['offset'],
                                 bounds=fit['bounds'],anchor_uv=geom['anchor'].tolist(),
                                 expected_major_px=float(np.ptp(geom['uv'],axis=0).max()*full_px))
        candidate.update(contact_min=minimum,terrain_physics={**pose,'signature':key,'cache_hit':hit,
                         'plane_normal':normal,'basis':candidate['plane_basis']})
        return candidate

    async def validate_pile_pose(self,asset,scene,candidate):
        from .geometry import transformed,horizontal_angle
        candidate=dict(candidate);candidate['allow_uncertain']=self.config.uncertain_policy=='render_review'
        strict=self.config.contact_mode=='strict'
        receiver=candidate['receiver'];data=np.load(asset['geometry'])
        original=transformed(data['vertices'],candidate['matrix'])
        if not strict and original[:,2].min()<-.004:raise ValueError('confirmed_initial_ground_penetration')
        key=signature('finite-support-physics-v1',asset['physics_signature'],candidate['matrix'],receiver['signature'])
        path=Path(asset['folder'])/'terrain'/(key+'.json');hit=path.is_file()
        if strict:
            from .strict_contact import simulate, validate_receiver, projected_gap
            from .metric_physics import tolerances
            validate_receiver(candidate)
            physics=await simulate(asset,self.blender,[candidate['matrix']],receiver,candidate['plane_normal'])
            simulated=physics['poses'][0];key=physics['signature'];hit=physics['cache_hit']
            if not simulated['stable']:
                raise ValueError('strict_contact_'+','.join(simulated['failure_codes']))
            gap=projected_gap(simulated,candidate)
            candidate['contact_gap']=gap
            if gap.get('contact_evidence_version'):
                atomic_json(self.out/'contact-evidence'/(signature(key,candidate.get('id'))+'.json'),
                    {'candidate_id':candidate.get('id'),'physics_signature':key,'contact_gap':gap})
            if gap['status']=='fail':
                raise ValueError('visible_contact_support_below_90pct' if
                    gap.get('visible_support',{}).get('status')=='fail' else 'visible_contact_gap_over_2px')
            unit=asset['physical_size']['nominal_mm']/1000
            candidate['contact_band']=tolerances(unit)['contact_m']/unit
        elif receiver['patches']:
            if hit:physics=json.loads(path.read_text(encoding='utf8'))
            else:
                physics=await self.blender.ask('settle',{'geometry':asset['geometry'],
                    'proxy':str(Path(asset['folder'])/'collision.json'),'axis':asset['axis'],
                    'initial_matrices':[candidate['matrix']],'assembly':asset['semantic']['assembly'],
                    'plane_normal':[0,0,1],'receiver':receiver},timeout=90)
                atomic_json(path,physics)
            simulated=physics['poses'][0]
        else:simulated={'stable':False,'failure_codes':['receiver_geometry_insufficient']}
        pose={'matrix':candidate['matrix'],'stable':False,'status':'unknown','axis_angle_deg':candidate['axis_angle_deg']}
        if simulated['stable']:
            pose=simulated
        elif rejected := set(simulated['failure_codes']) & {
                'ground_penetration', 'floating', 'center_of_mass_unsupported', 'upright'}:
            raise ValueError('confirmed_local_pose_failure:' + ','.join(sorted(rejected)))
        elif not candidate['allow_uncertain']:
            raise ValueError('estimated_support_simulation_'+','.join(simulated['failure_codes']))
        else:
            candidate['uncertainties']=list(candidate['uncertainties'])+['hidden_support_or_height_unknown']
        # Keep the scene camera fixed when a simulated object moves; update the contact pixel.
        middle=candidate['elevation'];camera=candidate['camera'];full=candidate['full_px']
        geom=projection_geometry(data['vertices'],data['faces'],pose,0,middle,
                                 candidate['plane_normal'] if strict else (0,0,1),camera=camera,
                                 contact_band=candidate.get('contact_band',.008),receiver=receiver if strict else None)
        if geom['anchor'] is None:raise ValueError('no_visible_contact_geometry')
        anchor=(np.asarray(candidate['offset'])+geom['anchor']*full).tolist()
        fit=fitted_masks(geom,anchor,full,scene)
        if fit is None:raise ValueError('local_pose_outside_image')
        x,y,r,b=fit['bounds'];mask=fit['hull']
        if np.any(mask&scene['protected'][y:b,x:r]):raise ValueError('protected_region_conflict')
        if (mask&scene['allowed'][y:b,x:r]).sum()/max(1,mask.sum())<.98:raise ValueError('local_pose_outside_hopper')
        cover=contact_support(
            fit['contact'],support_union(scene,candidate['support_ids'],fit['bounds']))
        if not cover or min(cover)<.9:
            if strict or not candidate['allow_uncertain']:raise ValueError('support_coverage_insufficient')
            candidate['uncertainties']=list(candidate['uncertainties'])+['support_coverage_insufficient']
        tilt=horizontal_angle(asset['axis'],pose['matrix'])
        if tilt>15:raise ValueError('confirmed_non_horizontal_pose')
        candidate.update(matrix=pose['matrix'],anchor_px=anchor,offset=fit['offset'],bounds=fit['bounds'],
            target_px=max(r-x,b-y),expected_major_px=float(np.ptp(geom['uv'],axis=0).max()*full),
            anchor_uv=geom['anchor'].tolist(),contact_polygons=[p.tolist() for p in geom['contacts']],
            contact_min=min(cover) if cover else 0,axis_angle_deg=tilt)
        endpoint_risks=[]
        for elevation in ([] if candidate['scale'].get('calibration',{}).get('projectable') else candidate['elevation_interval']):
            endpoint=projection_geometry(data['vertices'],data['faces'],pose,0,elevation,
                                         candidate['plane_normal'] if strict else (0,0,1),
                                         contact_band=candidate.get('contact_band',.008),receiver=receiver if strict else None)
            other=fitted_masks(endpoint,anchor,full,scene)
            if other is None:endpoint_risks.append({'elevation':elevation,'reason':'outside_image'});continue
            ox,oy,rr,bb=other['bounds'];m=other['hull']
            coverage=contact_support(
                other['contact'],support_union(scene,candidate['support_ids'],other['bounds']))
            if np.any(m&scene['protected'][oy:bb,ox:rr]) or not coverage or min(coverage)<.9:
                endpoint_risks.append({'elevation':elevation,'reason':'support_or_clearance_uncertain'})
        candidate['camera_risks']=endpoint_risks
        if candidate['scale'].get('version','').startswith('vehicle-width-projection'):
            from .vehicle_scale import check_projection_interval
            interval=check_projection_interval(scene,candidate,pose,data['vertices'],data['faces'])
            candidate['camera_interval_check']=interval
            candidate['camera_risks']=[{'reason':interval['reason']}]+[
                r for r in interval['checks'] if r['status']!='pass']
        if endpoint_risks and not candidate['allow_uncertain']:raise ValueError('camera_interval_uncertain')
        if receiver['height_confidence']=='unknown':
            candidate['uncertainties']=list(candidate['uncertainties'])+['support_heights_uncertain']
        candidate['terrain_physics']={**pose,'signature':key,'cache_hit':hit,'simulation':simulated,
            'status':'pass' if pose.get('stable') and (receiver['height_confidence']=='measured' if strict else receiver['height_confidence']!='unknown') else 'unknown',
            'basis':receiver['basis'],'hidden_heights':receiver['hidden_heights']}
        return candidate
