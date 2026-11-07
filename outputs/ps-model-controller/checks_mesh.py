"""Offline checks; --blender also exercises the installed CPU renderer."""
import argparse
import asyncio
import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import httpx
import numpy as np
from PIL import Image

from mesh_checks import face_id_raster, placement_gate, source_fidelity, visible_source_fraction
from mesh_camera import fit_source_camera
from mesh_carve import adaptive_resolution, carve
from mesh_ai import RenderAnswer, StructureAnswer, check_structure
from mesh_placement import compose_spec, open_fit
from mesh_optimize import optimize
from mesh_runtime import run_bounded
from mesh_scene import candidate_scale
from mesh_visibility import coverage, fragments
from batch_v2 import Budget, Runner
from metric_scale import SceneReference
from run import Config, Endpoint
from scene_v2 import NeedsReview


class GeometryChecks(unittest.TestCase):
    def test_subpixel_occluder_is_counted_at_exact_source_position(self):
        from mesh_visibility import SourceProjection
        camera={'eye':[0,0,0],'target':[0,0,1],'up':[0,1,0],'orthographic_scale':2}
        triangles=np.array([[[-.8,-.8,2],[.8,-.8,2],[0,.8,2]],
                            [[-.001,-.001,1],[.001,-.001,1],[0,.001,1]]])
        labels,_,_=fragments(triangles,camera,64)
        self.assertFalse(np.any(labels==1))
        mask=np.full((64,64),255,dtype=np.uint8)
        source=SourceProjection(triangles,camera,mask,mask)
        self.assertEqual(source.observed(np.array([[0,0,2],[.1,0,2]])).tolist(),[False,True])

    def test_zbuffer_resolves_crossing_depths_per_pixel(self):
        triangles=np.array([[[-.8,-.8,1],[.8,-.8,3],[0,.8,2]],
                            [[-.8,-.8,2],[.8,-.8,2],[0,.8,2]]],dtype=float)
        camera={'eye':[0,0,0],'target':[0,0,1],'up':[0,1,0],'orthographic_scale':2}
        labels,_,_=fragments(triangles,camera,128)
        self.assertEqual(labels[90,35],1)
        self.assertEqual(labels[90,93],0)

    def test_partial_triangle_visibility_is_not_whole_face_evidence(self):
        from mesh_visibility import _coverage
        source=np.array([[[-.8,-.8,2],[.8,-.8,2],[0,.8,2]],
                         [[-.8,-.8,1.5],[0,-.8,1.5],[0,.8,1.5]]],dtype=float)
        posed=source.copy(); posed[1,:,0]+=4
        camera={'eye':[0,0,0],'target':[0,0,1],'up':[0,1,0],'orthographic_scale':2}
        src,_,_=fragments(source,camera,256); dst,_,_=fragments(posed,camera,256)
        fraction,_=_coverage(source,np.array([0,1]),posed,np.array([0,1]),camera,camera,
                             (src>=0).astype('uint8')*255,(dst>=0).astype('uint8')*255,
                             (src>=0).astype('uint8')*255,256)
        self.assertIsNotNone(fraction)
        self.assertLess(fraction,.9)
        self.assertGreater(fraction,.1)

    def test_same_camera_view_retains_visible_surface(self):
        from mesh_visibility import _coverage
        triangles=np.array([[[-.8,-.8,2],[.8,-.8,2],[0,.8,2]]],dtype=float)
        camera={'eye':[0,0,0],'target':[0,0,1],'up':[0,1,0],'orthographic_scale':2}
        labels,_,_=fragments(triangles,camera,256)
        alpha=(labels>=0).astype('uint8')*255
        fraction,_=_coverage(triangles,np.array([0]),triangles,np.array([0]),camera,camera,
                             alpha,alpha,alpha,256)
        self.assertGreater(fraction,.99)

    def test_structure_review_rejects_contradictory_stage(self):
        with self.assertRaises(ValueError):
            StructureAnswer.model_validate_json('{"preserved":true,"missing":[],"stages":'
                 '{"source_to_rust":"pass","rust_to_mesh":"fail"}}')

    def test_hole_cannot_be_hidden_by_good_iou(self):
        mask = np.zeros((160, 160), np.uint8)
        cv2.rectangle(mask, (15, 20), (145, 140), 255, -1)
        cv2.circle(mask, (80, 80), 12, 0, -1)
        result = source_fidelity(mask, mask)
        self.assertTrue(result["passed"])
        filled = mask.copy()
        cv2.circle(filled, (80, 80), 12, 255, -1)
        result = source_fidelity(mask, filled)
        self.assertGreater(result["iou"], .9)
        self.assertFalse(result["passed"])
        self.assertEqual(result["filled_holes"], 1)

    def test_missing_narrow_hose_fails(self):
        mask = np.zeros((160, 160), np.uint8)
        cv2.rectangle(mask, (30, 20), (120, 135), 255, -1)
        cv2.line(mask, (120, 45), (155, 45), 255, 2)
        damaged = mask.copy()
        damaged[40:51, 124:] = 0
        self.assertFalse(source_fidelity(mask, damaged)["passed"])

    def test_geometry_requires_independent_evidence(self):
        good = dict(source_visible_fraction=.8, physics_stable=True,
                    component_support=[.90, 1.], collision=False,
                    occlusion_order="clear", scene_scale="calibrated")
        self.assertEqual(placement_gate(good), [])
        for field, value in [("source_visible_fraction",.79), ("physics_stable",False),
                             ("component_support",[.89, 1.]), ("collision",None),
                             ("occlusion_order","unknown"), ("scene_scale","unknown")]:
            self.assertTrue(placement_gate({**good,field:value}), field)

    def test_small_openings_get_finer_bounded_voxels(self):
        holes=np.zeros((50,50),np.uint8)
        holes[10:30,10:14]=1
        self.assertEqual(adaptive_resolution(holes),384)
        holes[:]=0
        holes[10:30,10:20]=1
        self.assertEqual(adaptive_resolution(holes),192)

    def test_approved_size_requires_same_local_depth(self):
        scene={"size":(800,600),"depth":np.full((150,200),.5,dtype=np.float32)}
        row={"physical_size":None,"approved_size_reference":{"bounds_px":[350,250,450,350],"major_px":100.}}
        candidate={"center_px":[420,310],"target_px":100.}
        result=candidate_scale(row,candidate,np.ones((30,100),np.uint8)*255,scene)
        self.assertEqual(result["method"],"approved_pixels_at_comparable_depth")
        changed={"size":(800,600),"depth":np.full((150,200),.5,dtype=np.float32)}
        changed["depth"][76:81,103:109]=.8
        with self.assertRaises(NeedsReview):
            candidate_scale(row,candidate,np.ones((30,100),np.uint8)*255,changed)

    def test_saved_dimensions_can_be_checked_against_a_measured_reference(self):
        ref=SceneReference(id="G1",label="measured steel edge",span_mm=(315,325),
                           points=((400,500),(600,500)),basis="measured",source="test fixture")
        scene={"size":(800,600),"depth":np.full((150,200),.5,dtype=np.float32),
               "references":{"G1":ref}}
        row=json.loads('{"physical_size":{"major_mm":[200,200],"minor_mm":[60,60],'
                       '"basis":"measured","source":"test fixture"}}')
        alpha=np.full((30,100),255,np.uint8)
        candidate={"center_px":[400,300],"target_px":100.}
        self.assertEqual(candidate_scale(row,candidate,alpha,scene)["status"],"calibrated")
        with self.assertRaisesRegex(NeedsReview,"超出局部参照范围"):
            candidate_scale(row,{**candidate,"target_px":105.},alpha,scene)

    def test_surface_coverage_uses_new_visible_pixels(self):
        source = np.array([[0,1,1],[0,2,2]], dtype=np.int32)
        new = np.array([[0,1,3],[0,2,3]], dtype=np.int32)
        self.assertEqual(visible_source_fraction(source,new),.5)

    def test_deadline_terminates_worker(self):
        with tempfile.TemporaryDirectory() as folder:
            result = run_bounded([sys.executable,"-c","import time; time.sleep(20)"],
                                 log=Path(folder)/"worker.log",seconds=.5)
        self.assertEqual(result["state"],"timeout")

    @unittest.skipUnless(sys.platform=="win32","Windows job-object limit")
    def test_hard_memory_limit_rejects_oversized_allocation(self):
        with tempfile.TemporaryDirectory() as folder:
            result=run_bounded([sys.executable,"-c","x=bytearray(128*1024*1024)"],
                               log=Path(folder)/"worker.log",seconds=10,memory_bytes=64*1024*1024)
        self.assertNotEqual(result["state"],"ok")
        self.assertEqual(result["memory_limit_mechanism"],"windows_job_commit")

    def test_visual_review_cannot_override_serious_defects(self):
        answer=RenderAnswer.model_validate_json('{"pick":"A","scores":[4,4,4,4,4],"defects":["floating"]}')
        self.assertFalse(answer.passes())
        with self.assertRaises(ValueError):
            StructureAnswer.model_validate_json('{"preserved":true,"missing":["hole"]}')
        with self.assertRaises(ValueError):
            RenderAnswer.model_validate_json('{"pick":null,"scores":[4,4,4,4,4],"defects":[]}')

    def test_structure_tool_cache_and_actual_token_accounting(self):
        async def check(root):
            calls=[]
            def handler(request):
                data=json.loads(request.content)
                calls.append(data)
                return httpx.Response(200,json={"choices":[{"message":{"tool_calls":[{"type":"function",
                    "function":{"name":"check_structure","arguments":'{"preserved":false,"missing":["hole"]}'}}]}}],
                    "usage":{"prompt_tokens":300,"completion_tokens":20,"total_tokens":320}})
            for name in ("original.png","rust.png","view.png"):
                Image.new("RGBA",(24,24),(30,40,50,255)).save(root/name)
            (root/"mesh.glb").write_bytes(b"fixture")
            row={"job_id":"fixture","signature":"version","calls":[],"original":str(root/"original.png"),
                 "rust":str(root/"rust.png"),"source_view":str(root/"view.png"),"mesh":str(root/"mesh.glb")}
            endpoint=Endpoint(url="http://localhost/v1/chat/completions",model="deepseek-test")
            budget=Budget(root/"usage.json",2000000)
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                runner=Runner(None,client,Config(vision=endpoint),None,root,budget)
                first,hit=await check_structure(runner,row,root)
                self.assertFalse(hit)
                self.assertEqual(first.missing,["hole"])
                _,hit=await check_structure(runner,row,root)
                self.assertTrue(hit)
            self.assertEqual(len(calls),1)
            self.assertEqual(calls[0]["tool_choice"]["function"]["name"],"check_structure")
            self.assertEqual(budget.image_usage(row)["total_tokens"],320)
        with tempfile.TemporaryDirectory() as root:
            asyncio.run(check(Path(root)))

    def test_open_placement_uses_actual_component_contacts(self):
        class Atlas:
            def clearance_crop(self, support_id, crop):
                return np.full((300,400),100,dtype=np.float32)
            def native_position_context(self,*args,**kwargs):
                return {"cost":.2,"relief":{"kind":"flat"}}
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/"A").mkdir()
            support=np.zeros((300,400),np.uint8)
            support[5:-5,5:-5]=255
            Image.fromarray(support).save(root/"A"/"support.png")
            Image.new("L",(400,300)).save(root/"A"/"foreground.png")
            pixels=np.zeros((60,120,4),np.uint8)
            pixels[4:52,4:116]=(110,100,90,255)
            shadow=np.zeros((60,120),np.float32)
            shadow[52:57,6:114]=.2
            left=np.zeros((60,120),bool); left[48:55,10:22]=True
            right=np.zeros_like(left); right[48:55,96:108]=True
            view=(pixels,shadow,[left,right])
            frame=Image.new("RGB",(800,600),(93,89,82))
            candidate={"id":"A","placement_mode":"open","target_px":120,"crop_px":[200,120,600,420],
                       "center_px":[400,270],"support_id":"S1","scene_scale_status":"calibrated"}
            fit=open_fit(candidate,root,Atlas(),frame,view,source_visible_fraction=.9,physics_stable=True)
            self.assertGreaterEqual(min(fit["evidence"]["component_support"]),.9)
            spec,_=compose_spec(fit,frame,root/"output","A")
            self.assertTrue(Path(spec["shade"]["path"]).is_file())
            self.assertEqual(spec["tone"]["blur_px"],0)
            with self.assertRaises(NeedsReview):
                open_fit(candidate,root,Atlas(),frame,view,source_visible_fraction=.79,physics_stable=True)
            with self.assertRaises(NeedsReview):
                open_fit({**candidate,"scene_scale_status":"unknown"},root,Atlas(),frame,view,
                         source_visible_fraction=.9,physics_stable=True)
            unknown={**candidate,"scene_scale_status":"unknown"}
            preview=open_fit(unknown,root,Atlas(),frame,view,source_visible_fraction=.9,
                             physics_stable=True,pixel_size_review=True)
            self.assertEqual(placement_gate(preview["evidence"]),["scene_scale_uncertain"])
            with self.assertRaises(NeedsReview):
                open_fit(unknown,root,Atlas(),frame,view,source_visible_fraction=.79,
                         physics_stable=True,pixel_size_review=True)


def blender_smoke(blender):
    import trimesh
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        mesh = root/"cube.glb"
        trimesh.creation.box(extents=[.8, .6, .4]).export(mesh)
        script = Path(__file__).with_name("blender_worker.py")
        for operation in ("source", "settle"):
            folder = root/operation
            folder.mkdir()
            request = root/f"{operation}.json"
            request.write_text(json.dumps({"operation":operation,"mesh":str(mesh),"out":str(folder),
                                           "camera_elevation_deg":45}),encoding="utf-8")
            result = run_bounded([blender,"--background","--factory-startup","--python",script,
                                  "--",request],log=folder/"worker.log",seconds=180)
            if result["state"] != "ok" or not (folder/"blender-result.json").exists():
                raise RuntimeError(f"Blender {operation} failed: " + (folder/"worker.log").read_text(errors="replace")[-1500:])
            output = folder/("source-view.png" if operation=="source" else "object.png")
            with Image.open(output) as image:
                assert image.getchannel("A").getextrema() == (0,255)
            if operation == "settle":
                data = json.loads((folder/"blender-result.json").read_text())
                assert data["physics_stable"] and data["component_count"] == 1
                assert data["camera"]["elevation_deg"] == 45
                assert data["contacts_px"][0]
                assert (folder/"with-shadow.png").is_file() and (folder/"without-shadow.png").is_file()
            else:
                data = json.loads((folder/"blender-result.json").read_text())
                spans=[hi-lo for lo,hi in data["mesh_axis_bounds"]]
                assert np.allclose(spans,[.8,.6,.4],atol=.0001), spans
                with Image.open(output) as image:
                    original=np.asarray(image.getchannel("A"))
                transform=cv2.getRotationMatrix2D((256,256),0,1.15)
                transform[:,2]+=[7,-5]
                target=cv2.warpAffine(original,transform,(512,512))
                camera=fit_source_camera(target,original)
                request.write_text(json.dumps({"operation":"source","mesh":str(mesh),"out":str(folder),
                                               "camera":camera}),encoding="utf-8")
                result=run_bounded([blender,"--background","--factory-startup","--python",script,
                                    "--",request],log=folder/"calibrated.log",seconds=180)
                if result["state"]!="ok":
                    raise RuntimeError((folder/"calibrated.log").read_text(errors="replace")[-1000:])
                with Image.open(output) as image:
                    actual=np.asarray(image.getchannel("A"))
                measured=source_fidelity(target,actual)["iou"]
                assert measured>.90 and camera["predicted_iou"]-measured<.04,(camera,measured)
        # A drilled synthetic body should retain its visible hole in 3-D.
        cube_source=root/"source"/"source-view.png"
        first=root/"carve-source"
        first.mkdir()
        request=root/"carve-source.json"
        request.write_text(json.dumps({"operation":"source","mesh":str(mesh),"out":str(first)}),encoding="utf8")
        command=[blender,"--background","--factory-startup","--python",script,"--",request]
        result=run_bounded(command,log=first/"worker.log",seconds=180)
        assert result["state"]=="ok"
        with Image.open(first/"source-view.png") as image:
            alpha=np.array(image.getchannel("A"))
            texture=image.convert("RGB")
        cv2.circle(alpha,(256,256),20,0,-1)
        # An off-center narrow opening exposes projection and voxel footprint
        # errors that the much wider central hole can tolerate.
        cv2.rectangle(alpha,(308,233),(312,265),0,-1)
        Image.fromarray(alpha).save(first/"alpha.png")
        texture.save(first/"texture.png")
        camera={"eye":[1.9,0,0],"target":[0,0,0],"fov_deg":40}
        carved=root/"carved"
        carve(mesh,first/"alpha.png",first/"texture.png",camera,carved,resolution=96)
        rendered=root/"carve-render"
        rendered.mkdir()
        request.write_text(json.dumps({"operation":"source","mesh":str(carved/"mesh.glb"),"out":str(rendered)}),encoding="utf8")
        result=run_bounded(command,log=rendered/"worker.log",seconds=180)
        assert result["state"]=="ok"
        with Image.open(rendered/"source-view.png") as image:
            qa=source_fidelity(alpha,np.asarray(image.getchannel("A")))
        assert qa["holes"]==2 and qa["passed"],qa
        optimized=root/"optimized"
        optimize(carved/"mesh.glb",first/"alpha.png",rendered/"source-view.png",camera,optimized)
        after=root/"optimized-render"
        after.mkdir()
        request.write_text(json.dumps({"operation":"source","mesh":str(optimized/"mesh.glb"),"out":str(after)}),encoding="utf8")
        result=run_bounded(command,log=after/"worker.log",seconds=180)
        assert result["state"]=="ok"
        with Image.open(after/"source-view.png") as image:
            qa=source_fidelity(alpha,np.asarray(image.getchannel("A")))
        assert qa["passed"],qa
        # The visibility check must refer to the same faces before and after physics.
        provenance=root/"provenance"
        provenance.mkdir()
        request=root/"provenance.json"
        request.write_text(json.dumps({"mesh":str(mesh),"out":str(provenance),
                                       "rotation_deg":[0,0,0]}),encoding="utf8")
        result=run_bounded([blender,"--background","--factory-startup","--python",
                            Path(__file__).with_name("settled_mesh_worker.py"),"--",request],
                           log=provenance/"worker.log",seconds=180)
        if result["state"]!="ok":
            raise RuntimeError((provenance/"worker.log").read_text(errors="replace")[-1500:])
        with np.load(provenance/"triangles.npz") as geometry:
            assert set(geometry["source_ids"])==set(geometry["posed_ids"])
        source_meta=json.loads((first/"blender-result.json").read_text())
        posed_meta=json.loads((provenance/"blender-result.json").read_text())
        with Image.open(first/"source-view.png") as source_image,Image.open(provenance/"object.png") as posed_image:
            source_alpha=np.asarray(source_image.getchannel("A"))
            check=coverage(provenance/"triangles.npz",source_meta["camera"],posed_meta["camera"],
                           source_alpha,np.asarray(posed_image.getchannel("A")),source_alpha)
        assert check["known"],check
        assert 0<=check["source_visible_fraction"]<=1
        rotated=root/"rotated"
        rotated.mkdir()
        request.write_text(json.dumps({"mesh":str(mesh),"out":str(rotated),
                                       "rotation_deg":[0,90,0],"camera_elevation_deg":45}),encoding="utf8")
        result=run_bounded([blender,"--background","--factory-startup","--python",
                            Path(__file__).with_name("settled_mesh_worker.py"),"--",request],
                           log=rotated/"worker.log",seconds=180)
        if result["state"]!="ok":
            raise RuntimeError((rotated/"worker.log").read_text(errors="replace")[-1500:])
        rotated_data=json.loads((rotated/"blender-result.json").read_text())
        assert np.allclose(rotated_data["initial_extents"],[.4,.6,.8],atol=.001),rotated_data["initial_extents"]
        assert rotated_data["physics_stable"],rotated_data
        separate=root/"separate.glb"
        first=trimesh.creation.box(extents=[.35,.3,.2]); first.apply_translation([-.4,0,0])
        second=trimesh.creation.box(extents=[.35,.3,.2]); second.apply_translation([.4,0,0])
        trimesh.util.concatenate((first,second)).export(separate)
        pairs=root/"separate"
        pairs.mkdir()
        request.write_text(json.dumps({"mesh":str(separate),"out":str(pairs),
                                       "rotation_deg":[0,0,0],"camera_elevation_deg":45}),encoding="utf8")
        result=run_bounded([blender,"--background","--factory-startup","--python",
                            Path(__file__).with_name("settled_mesh_worker.py"),"--",request],
                           log=pairs/"worker.log",seconds=180)
        if result["state"]!="ok":
            raise RuntimeError((pairs/"worker.log").read_text(errors="replace")[-1500:])
        data=json.loads((pairs/"blender-result.json").read_text())
        assert data["component_count"]==2 and not data["component_collisions"] and data["physics_stable"],{
            k:data[k] for k in ("component_count","component_collisions","physics_stable","motion",
                                 "rotation_motion_deg","component_min_z","simulated_frames","stable_windows")}
        stacked=root/"stacked.glb"
        lower=trimesh.creation.box(extents=[.8,.6,.3])
        upper=trimesh.creation.box(extents=[.3,.2,.2]); upper.apply_translation([0,0,.25])
        trimesh.util.concatenate((lower,upper)).export(stacked)
        supported=root/"supported"
        supported.mkdir()
        request.write_text(json.dumps({"mesh":str(stacked),"out":str(supported),
                                       "rotation_deg":[0,0,0],"camera_elevation_deg":45}),encoding="utf8")
        result=run_bounded([blender,"--background","--factory-startup","--python",
                            Path(__file__).with_name("settled_mesh_worker.py"),"--",request],
                           log=supported/"worker.log",seconds=180)
        if result["state"]!="ok":
            raise RuntimeError((supported/"worker.log").read_text(errors="replace")[-1500:])
        data=json.loads((supported/"blender-result.json").read_text())
        assert data["physics_stable"] and data["support_graph"]["support_edges"],data["support_graph"]
    print("Blender CPU source and rigid-body smoke passed")


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--blender",type=Path)
    args=parser.parse_args()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(GeometryChecks)
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
        raise SystemExit(1)
    if args.blender:
        blender_smoke(args.blender)
