"""Offline regression checks for gates and batch bookkeeping."""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import numpy as np
from PIL import Image
from pydantic import ValidationError

from asset_catalog import Catalog
from auto_v3 import CandidateReview, SceneAnswer, checked_scene
from batch_v2 import (Budget, BudgetReached, Fix, GapPlan, Job, MetricPlan, Quality, Runner,
                      correct, process, render_geometry, save_mask, verify_background)
from metric_scale import (ObjectSize, ScaleReview, SceneAnchor, SceneEstimate, SceneReference, SceneSpan,
                          fit_consensus, fit_scale, reference_geometry, reference_region, verify_scale)
from open_pose import bearing_pixels
from pose_retest import choose_recess, generation_request, register_view
from relief import ReliefField
from run import Config, Endpoint
from scene_v2 import (MIN_EDGE_AGREEMENT, NeedsReview, boundary_agreement, digest, frame_point,
                      gap_planes, measure_geometry, original_edge_neighborhood, posed_alpha)


class Checks(unittest.TestCase):
    def test_relief_separates_tilted_floor_recess_and_raised_support(self):
        y,x=np.indices((150,160),dtype=np.float32)
        plane=.25+.0010*x+.0007*y
        valid=np.ones(plane.shape,bool)
        bounds=(52,48,106,102)
        for mode,sign in (("recess",1),("raised",-1),("flat",0)):
            depth=plane.copy()
            depth[48:102,38:52]+=sign*.07
            depth[48:102,106:120]+=sign*.07
            self.assertEqual(ReliefField(depth,plane,valid,.007).measure(bounds,60)["kind"],mode)
        one_sided=plane.copy()
        one_sided[48:102,106:120]+=.08
        self.assertEqual(ReliefField(one_sided,plane,valid,.007).measure(bounds,60)["kind"],"uncertain")

    def test_one_compact_tool_call_only_for_tie_and_cache_reuse(self):
        async def check(folder):
            hits=[]
            async def handler(request):
                body=json.loads(request.content)
                self.assertEqual(body["max_tokens"],64)
                self.assertEqual(body["thinking"],{"type":"disabled"})
                self.assertEqual(body["tool_choice"]["function"]["name"],"choose_slot")
                self.assertEqual(len(body["tools"]),1)
                parts=body["messages"][1]["content"]
                self.assertEqual(len(parts),2)
                self.assertEqual(parts[1]["image_url"]["detail"],"low")
                hits.append(body)
                return httpx.Response(200,json={
                    "choices":[{"finish_reason":"tool_calls","message":{"tool_calls":[
                        {"type":"function","function":{"name":"choose_slot","arguments":'{"slot":"B"}'}}]}}],
                    "usage":{"prompt_tokens":210,"completion_tokens":10,"total_tokens":220,"prompt_cache_hit_tokens":32},
                })
            endpoint=Endpoint(url="http://localhost/v1/chat/completions",model="deepseek-flash")
            ledger=Budget(folder/"ledger.json",4096)
            frame=Image.new("RGB",(360,260),(72,79,83))
            rust=folder/"object.png"
            frame.save(rust)
            pixels=np.full((28,64,4),255,np.uint8)
            relief={"kind":"recess","raised_rim":.57,"lower_sides":.06}
            def candidate(ident,x,score):
                return ((score,0,x,84,pixels,[],1.,1.,12.,0.,{"relief":relief,"glare":.04}),
                        {"id":ident,"support_id":"S1","placement_mode":"open","target_px":64,
                         "crop_px":[0,0,360,260]},
                        {"status":"eligible"})
            base={"job_id":"bearing","asset_id":"bearing","rust":str(rust),"background_sha256":"fixture"}
            a,b=candidate("M",40,.72),candidate("N",224,.69)
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
                runner=Runner(None,http,Config(planner=endpoint),None,folder,ledger)
                clear={**base,"signature":"clear","calls":[]}
                await choose_recess(runner,clear,frame,folder,[a,candidate("O",224,.55)])
                self.assertEqual(ledger.image_usage(clear)["total_tokens"],0)
                tied={**base,"signature":"tied","calls":[]}
                selected=await choose_recess(runner,tied,frame,folder,[a,b])
                self.assertEqual(selected[0][1]["id"],"N")
                self.assertEqual(ledger.image_usage(tied)["total_tokens"],220)
                self.assertEqual(ledger.image_usage(tied)["prompt_cache_hit_tokens"],32)
                cached={**base,"signature":"next","calls":[]}
                await choose_recess(runner,cached,frame,folder,[a,b])
                self.assertEqual(cached["slot_decision"]["source"],"cache")
                self.assertEqual(ledger.image_usage(cached)["total_tokens"],0)
                self.assertEqual(len(hits),1)
        with tempfile.TemporaryDirectory() as tmp:
            asyncio.run(check(Path(tmp)))

    def test_unreported_request_is_not_counted_as_zero_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger=Budget(Path(tmp)/"ledger.json",5000)
            current={"job_id":"part","signature":"new","calls":[]}
            earlier=ledger.reserve("part","choose",signature="old",reservation=2048)
            ledger.settle(earlier,{"usage":{"prompt_tokens":190,"completion_tokens":11}})
            timed_out=ledger.reserve("part","choose",signature="new",reservation=2048)
            ledger.settle(timed_out,{"role":"choose"})
            usage=ledger.image_usage(current)
            self.assertIsNone(usage["total_tokens"])
            self.assertEqual((usage["reported_tokens"],usage["unreported_calls"],usage["budget_tokens"]),(0,1,2048))

    def test_pose_generation_is_limited_to_isolated_objects(self):
        row={
            "job_id":"cylinder","number":30,"kind":"ram","status":"awaiting_generation",
            "selection":{"pick":"A","keep":"K1"},"features":{"K1":"rod eye"},
            "candidates":[{"id":"A","axis_px":[[1,4],[7,4]],"placement_mode":"scrap"}],
            "generations":[],"original":"object.jpg","rust":"object-aged.png",
        }
        request=generation_request(row)
        self.assertEqual(request["references"],["object.jpg","object-aged.png"])
        self.assertEqual(request["mode"],"view")
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/"opaque-scene.png"
            Image.new("RGBA",(1024,1024),(60,62,59,255)).save(path)
            with self.assertRaisesRegex(ValueError,"独立物品"):
                register_view(Path(root),row,path)
        self.assertEqual(row["generations"],[])

    def test_remote_reservoir_keeps_its_own_support_band(self):
        solid=np.zeros((126,150),dtype=bool)
        solid[74:94,28:126]=True
        solid[38:49,40:73]=True
        solid[46:77,68:70]=True
        body,accessory=bearing_pixels(solid,np.array([[48,90],[99,90],[55,48]]),100,"shock")
        self.assertTrue(body[93,55])
        self.assertTrue(accessory[48,55])
        self.assertFalse(accessory[93,55])
        self.assertFalse(body[48,55])

    def test_absolute_gate_and_fatal(self):
        raw=dict(support=4,occlusion=3,lighting=3,edge=3,texture=3,fatal=[],fix=None,note="")
        self.assertTrue(Quality(**raw).passes())
        self.assertFalse(Quality(**{**raw,"edge":2}).passes())
        self.assertFalse(Quality(**{**raw,"fatal":["floating"]}).passes())
        self.assertIsNone(correct({"id":"A"},Fix(brightness=7),{},["floating"]))
        with self.assertRaises(ValidationError) as error:
            GapPlan.model_validate_json(json.dumps({"sites":[{"view":1,"pose":[500,1001,70,0],
                                             "seat":[530,550],"front":[[560,530]],"keep":[510,580]}]}))
        self.assertEqual(error.exception.errors()[0]["loc"],("sites",0,"pose",1))

    def test_frame_and_asset_coordinates(self):
        self.assertEqual(frame_point((0,1000),(80,60)),(0,59))
        self.assertEqual(frame_point((1000,0),(80,60)),(79,0))
        alpha=np.full((40,80),255,dtype=np.uint8)
        rotated,keep=posed_alpha(alpha,40,90,(10,10))
        self.assertEqual(rotated.shape,(80,40))
        self.assertAlmostEqual(keep[0]*40,29.5)
        self.assertAlmostEqual(keep[1]*80,10.5)
        self.assertEqual(GapPlan.model_validate_json('{"sites":[]}').sites,[])

    def test_measured_occlusion_crossing_and_feature(self):
        alpha=np.zeros((64,64),dtype=np.uint8)
        alpha[15:45,18:54]=255
        alpha[20:26,21:25]=0
        foreground=np.zeros_like(alpha)
        foreground[7:55,:36]=255
        support=np.zeros_like(alpha,dtype=bool)
        support[43:47,36:56]=True
        expected=(30*18-24)/(30*36-24)
        good=measure_geometry(alpha,foreground,support,(46,28),36)
        self.assertAlmostEqual(good["occlusion_ratio"],expected)
        with self.assertRaisesRegex(NeedsReview,"遮挡率"):
            measure_geometry(alpha,np.zeros_like(alpha),support,(46,28),36)
        with self.assertRaisesRegex(NeedsReview,"遮挡率"):
            measure_geometry(alpha,np.full_like(alpha,255),support,(46,28),36)
        with self.assertRaisesRegex(NeedsReview,"识别特征"):
            measure_geometry(alpha,foreground,support,(25,35),36)
        with self.assertRaisesRegex(NeedsReview,"承托"):
            measure_geometry(alpha,foreground,np.zeros_like(support),(46,28),36)
        interior=np.zeros_like(alpha)
        interior[16:44,19:37]=255
        with self.assertRaisesRegex(NeedsReview,"轮廓"):
            measure_geometry(alpha,interior,support,(46,28),36)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            crop=[16,16,80,80]
            foreground_entry=save_mask(foreground,root/"foreground.png",crop,(255,255,255))
            rgba=Image.new("RGBA",(64,64))
            rgba.putalpha(Image.fromarray(alpha))
            rgba.save(root/"alpha.png")
            Image.fromarray(support).save(root/"support.png")
            spec={"crop":crop,"canvas":[100,100],"object_alpha":root/"alpha.png","occlusion":foreground_entry,
                  "support_mask":root/"support.png","width":36,"keep":[28.5/36,13.5/30]}
            result=render_geometry(spec,{"bounds":[34,31,70,61]})
            self.assertAlmostEqual(result["occlusion_ratio"],expected)
            with self.assertRaisesRegex(NeedsReview,"保护带"):
                render_geometry({**spec,"canvas":[100,60]},{"bounds":[34,31,70,61]})

    def test_local_depth_order(self):
        front=np.zeros((32,32),dtype=bool)
        front[:, :15]=True
        support=~front
        depth=np.where(front,.65,.43)
        points=[(20,20),(10,10)]
        self.assertEqual(len(list(gap_planes(depth,[[support],[front]],points,24))),1)
        self.assertEqual(list(gap_planes(1-depth,[[support],[front]],points,24)),[])

    def test_uncovered_render_does_not_reach_vision(self):
        class Uncovered(Runner):
            async def asset(self,source,scratch):
                return source

            async def guided_specs(self,job,report,background,assets,folder,scratch):
                alpha=np.zeros((64,64),dtype=np.uint8)
                alpha[15:45,18:54]=255
                rgba=Image.new("RGBA",(64,64))
                rgba.putalpha(Image.fromarray(alpha))
                rgba.save(scratch/"alpha.png")
                Image.fromarray(alpha).save(scratch/"support.png")
                return [{"id":"A","site":1,"crop":[16,16,80,80],"canvas":[100,100],
                         "object_alpha":scratch/"alpha.png","support_mask":scratch/"support.png",
                         "occlusion":None,"width":36,"keep":[.75,.5],"preview":scratch/"candidate.jpg"}]

            async def ps(self,payload):
                Image.new("RGB",(32,32)).save(payload["candidates"][0]["preview"])
                return [{"bounds":[34,31,70,61]}]

            async def audit(self,*args,**kwargs):
                raise AssertionError("Zero occlusion reached the visual model")

        async def check(root):
            frame=root/"frame.png"
            Image.new("RGB",(100,100)).save(frame)
            endpoint=Endpoint(url="http://localhost/v1/chat/completions",model="unused")
            runner=Uncovered(None,None,Config(planner=endpoint,vision=endpoint),None,root,None)
            report={"calls":[]}
            with self.assertRaisesRegex(NeedsReview,"实际渲染几何"):
                await runner.item(Job(job_id="uncovered",background="frame.png",objects=["frame.png"],
                                      mode="preview"),frame,[frame],root,report)
            self.assertIn("0.0%",report["rejected"][0]["reason"])
            self.assertEqual(report["calls"],[])

        with tempfile.TemporaryDirectory() as tmp:
            asyncio.run(check(Path(tmp)))

    def test_metric_scale_changes_with_local_reference(self):
        alpha=np.full((40,120),255,dtype=np.uint8)
        depth=np.full((600,800),.65,dtype=np.float32)
        depth[:300]=.30
        size=ObjectSize(major_mm=(500,500),minor_mm=(160,160),basis="measured",source="fixture")
        near=SceneReference(id="NEAR",label="200 px ruler",span_mm=(1000,1000),
                            points=((250,700),(500,700)),basis="measured",source="fixture")
        far=SceneReference(id="FAR",label="100 px ruler",span_mm=(1000,1000),
                           points=((250,300),(375,300)),basis="measured",source="fixture")
        near_fit=fit_scale(alpha,(800,600),depth,(380,700),size,near)
        far_fit=fit_scale(alpha,(800,600),depth,(320,300),size,far)
        self.assertEqual(near_fit.record["projected_major_px"],100)
        self.assertEqual(far_fit.record["projected_major_px"],50)
        self.assertEqual(near_fit.record["status"],"calibrated")
        rendered,_=posed_alpha(alpha,None,12,(80,20),pixel_scale=near_fit.factor)
        self.assertAlmostEqual(verify_scale(rendered,near_fit.record)["rendered_major_px"],100,delta=3)
        oversized,_=posed_alpha(alpha,None,12,(80,20),pixel_scale=near_fit.factor*1.5)
        with self.assertRaisesRegex(ScaleReview,"实际尺寸"):
            verify_scale(oversized,near_fit.record)
        with self.assertRaises(ScaleReview) as error:
            fit_scale(alpha,(800,600),depth,(380,470),size,near)
        self.assertEqual(error.exception.code,"reference_depth_mismatch")
        eligible=reference_region(depth,(800,600),reference_geometry(near,(800,600),depth))
        self.assertTrue(eligible[420,300])
        self.assertFalse(eligible[150,300])

    def test_estimates_are_not_calibrated(self):
        alpha=np.full((40,120),255,dtype=np.uint8)
        depth=np.full((600,800),.5,dtype=np.float32)
        size=ObjectSize(major_mm=(450,550),minor_mm=(145,175),basis="similar_catalog",source="catalog")
        reference=SceneSpan(label="beam width",span_mm=(800,1000),points=((250,700),(500,700)))
        fit=fit_scale(alpha,(800,600),depth,(380,700),size,reference)
        self.assertEqual(fit.record["status"],"estimated")
        low,high=fit.record["projected_major_px_range"]
        self.assertLess(low,fit.record["projected_major_px"])
        self.assertGreater(high,fit.record["projected_major_px"])
        with self.assertRaises(ScaleReview) as error:
            fit_scale(alpha,(800,600),depth,(380,700),size,reference.model_copy(update={"span_mm":(10,1000)}))
        self.assertEqual(error.exception.code,"scale_too_uncertain")
        with self.assertRaises(ValidationError):
            MetricPlan.model_validate_json('{"sites":[{"view":1,"edge":"B","pose":[500,500,70,0],"keep":"1","ref":"R1"}]}')

    def test_missing_dimensions_stop_before_api(self):
        runner=Runner(None,None,Config(),None,None,None)
        job=Job(job_id="missing",background="frame.png",objects=["object.png"])
        report={"calls":[]}
        with self.assertRaises(ScaleReview) as error:
            asyncio.run(runner.item(job,None,[],None,report))
        self.assertEqual(error.exception.code,"missing_object_size")
        self.assertEqual(report["calls"],[])

    def test_scene_estimate_cached_by_background(self):
        class EstimateRunner(Runner):
            count=0

            async def ask(self,job,report,endpoint,role,schema,prompt,payload,picture):
                assert role=="estimate_scene_scale"
                self.count+=1
                return SceneEstimate.model_validate_json(
                    '{"references":[{"id":"R1","label":"beam edge","span_mm":[500,800],'
                    '"points":[[300,500],[500,500]]}]}')

        async def check(root):
            endpoint=Endpoint(url="http://localhost/v1/chat/completions",model="test")
            runner=EstimateRunner(None,None,Config(planner=endpoint),None,root,None)
            source=root/"frame.png"
            frame=Image.new("RGB",(800,600))
            frame.save(source)
            job=Job(job_id="fixture",background="frame.png",objects=["object.png"])
            first,second={},{}
            self.assertEqual(await runner.scene_references(job,first,source,frame,root),
                             await runner.scene_references(job,second,source,frame,root))
            self.assertEqual(runner.count,1)
            self.assertFalse(first["scene_scale"]["cache_hit"])
            self.assertTrue(second["scene_scale"]["cache_hit"])

        with tempfile.TemporaryDirectory() as tmp:
            asyncio.run(check(Path(tmp)))

    def test_budget_survives_interruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"budget.json"
            Budget(path,4500).reserve("item","choose")
            ledger=Budget(path,4500)
            with self.assertRaises(BudgetReached):
                ledger.reserve("other","audit")
            ledger.settle(ledger.entries[0],{"usage":{"total_tokens":370}})
            self.assertEqual(Budget(path,4500).accounted,370)
            ledger.settle(ledger.reserve("no-request","audit"),{})
            self.assertEqual(Budget(path,4500).accounted,370)

    def test_invalid_model_reply_is_charged_without_edit(self):
        async def check(tmp):
            async def handler(request):
                content=json.loads(request.content)
                self.assertEqual(len(content["messages"][1]["content"]),2)
                return httpx.Response(200,json={"choices":[{"message":{"content":'{"candidate":"Z","note":""}'}}],
                                                "usage":{"total_tokens":401}})
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
                budget=Budget(tmp/"ledger.json",4096)
                runner=Runner(None,http,Config(),None,tmp,budget)
                picture=tmp/"preview.jpg"
                Image.new("RGB",(40,40)).save(picture)
                endpoint=Endpoint(url="http://localhost/v1/chat/completions",model="test")
                report={"calls":[]}
                with self.assertRaises(ValueError):
                    await runner.ask(SimpleNamespace(job_id="test"),report,endpoint,"find_gaps",GapPlan,
                                     "Return JSON",{"views":1},picture)
                self.assertEqual(budget.accounted,401)
                self.assertEqual(report["calls"][0]["schema_errors"][0]["path"],["sites"])
        with tempfile.TemporaryDirectory() as tmp:
            asyncio.run(check(Path(tmp)))

    def test_failed_image_and_resume(self):
        class Fixture:
            calls=0
            async def item(self,job,background,objects,folder,report):
                self.calls+=1
                with Image.open(background) as image:
                    image.verify()
                (folder/"final.png").write_bytes(b"fixture")
                (folder/"final.psd").write_bytes(b"fixture")
                report.update(status="passed",outputs={"png":"final.png","psd":"final.psd"})
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/"bad.jpg").write_bytes(b"not an image")
            Image.new("RGB",(32,32)).save(root/"ok.png")
            args=SimpleNamespace(out=root/"out",manifest=root/"task.jsonl",retry=False)
            fixture=Fixture()
            config,engine=Config(),SimpleNamespace(versions=["fixture"])
            bad=Job(job_id="bad",background="bad.jpg",objects=["ok.png"])
            good=Job(job_id="good",background="ok.png",objects=["ok.png"])
            self.assertEqual(asyncio.run(process(bad,args,config,engine,fixture))["status"],"failed")
            self.assertEqual(asyncio.run(process(good,args,config,engine,fixture))["status"],"passed")
            self.assertTrue(asyncio.run(process(good,args,config,engine,fixture))["resumed"])
            self.assertEqual(fixture.calls,2)

    def test_pixel_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            pixels=np.zeros((100,100,3),np.uint8)
            Image.fromarray(pixels).save(root/"original.png")
            pixels[48:54,45:51]=120
            Image.fromarray(pixels).save(root/"composite.png")
            spec={"shade":None,"occlusion":None}
            verify_background(root/"original.png",root/"composite.png",[44,47,52,55],spec)
            pixels[3,50]=255
            Image.fromarray(pixels).save(root/"composite.png")
            with self.assertRaises(NeedsReview):
                verify_background(root/"original.png",root/"composite.png",[44,47,52,55],spec)

    def test_pixel_guard_uses_actual_alpha(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            Image.new("RGB",(100,100)).save(root/"original.png")
            image=np.zeros((100,100,3),np.uint8)
            image[45:50,45:50]=100
            Image.fromarray(image).save(root/"composite.png")
            alpha=np.zeros((20,20),np.uint8)
            alpha[5:10,5:10]=255
            rgba=Image.new("RGBA",(20,20))
            rgba.putalpha(Image.fromarray(alpha))
            rgba.save(root/"alpha.png")
            spec={"crop":[40,40,60,60],"object_alpha":root/"alpha.png","shade":None}
            verify_background(root/"original.png",root/"composite.png",[44,44,51,51],spec)
            image[44,44]=100
            Image.fromarray(image).save(root/"composite.png")
            with self.assertRaisesRegex(RuntimeError,"超出物品"):
                verify_background(root/"original.png",root/"composite.png",[44,44,51,51],spec)

    def test_independent_local_scale_consensus(self):
        alpha=np.full((40,120),255,dtype=np.uint8)
        depth=np.full((600,800),.53,dtype=np.float32)
        size=ObjectSize(major_mm=(500,500),minor_mm=(160,160),basis="measured",source="fixture")
        left=SceneAnchor(id="R1",label="plate",span_mm=(780,850),points=((230,675),(480,675)))
        right=SceneAnchor(id="R2",label="beam",span_mm=(590,640),points=((430,650),(618,650)))
        local=(alpha,(800,600),depth,(460,700),size)
        fit=fit_consensus(*local,[left,right])
        self.assertEqual(fit.record["status"],"estimated")
        self.assertAlmostEqual(fit.record["projected_major_px"],123,delta=5)
        self.assertEqual(len(fit.record["references"]),2)
        for claims,code in (
            ([left],"insufficient_local_references"),
            ([left,left.model_copy(update={"id":"R2"})],"insufficient_local_references"),
            ([left,right.model_copy(update={"span_mm":(1800,2100)})],"reference_conflict"),
            ([left.model_copy(update={"span_mm":(650,1250)}),
              right.model_copy(update={"span_mm":(487,937)})],"scale_too_uncertain"),
        ):
            with self.subTest(code=code), self.assertRaises(ScaleReview) as error:
                fit_consensus(*local,claims)
            self.assertEqual(error.exception.code,code)
        measured=SceneReference(**left.model_dump(),basis="measured",source="fixture ruler")
        self.assertEqual(fit_consensus(*local,[measured]).record["status"],"calibrated")

    def test_original_pixel_edges_reject_mask_cut_through_a_panel(self):
        scene=np.full((100,100,3),120,dtype=np.uint8)
        scene[:,:50]=(75,50,43)
        subject=np.zeros((100,100),dtype=np.uint8)
        subject[30:70,28:76]=255
        foreground=np.zeros_like(subject)
        foreground[:,:50]=255
        neighborhood=original_edge_neighborhood(scene)
        self.assertGreaterEqual(boundary_agreement(subject,foreground,neighborhood),MIN_EDGE_AGREEMENT)
        foreground[:,:60]=255
        self.assertLess(boundary_agreement(subject,foreground,neighborhood),MIN_EDGE_AGREEMENT)

    def test_compact_response_and_repair_keep_measured_curves(self):
        result=CandidateReview.model_validate_json('{"pick":"B","scores":[4,3,3,3,3],"defects":[]}')
        self.assertTrue(result.passes())
        with self.assertRaises(ValidationError):
            CandidateReview.model_validate_json('{"pick":"A","scores":[4,3,3,3,3],"defects":[],"fix":{"blur_px":2}}')
        scene=SceneAnswer.model_validate_json('{"zones":["Z1"],"refs":[["R1","plate",650,500]],"protect":[]}')
        self.assertEqual(checked_scene(scene,{"Z1":{}},{"R1":{"points":[[300,300],[400,400]]}})
                         ["R1"].span_mm,(500,650))
        with self.assertRaisesRegex(ValueError,"unknown"):
            checked_scene(scene,{"Z1":{}},{})
        spec={"id":"A","tone":{"brightness":0,"warmth":0,"saturation":-7,"blur_px":.2,
                                 "luma_curve":[[0,0],[128,140],[255,255]],"rgb_cast":[-2,1,1]}}
        fixed=correct(spec,Fix(brightness=4,blur_px=.6),{},False)
        self.assertEqual(fixed["tone"]["luma_curve"],spec["tone"]["luma_curve"])
        self.assertEqual(fixed["tone"]["brightness"],4)
        self.assertEqual(spec["tone"]["brightness"],0)

    def test_catalog_hash_preflight_and_estimated_resume(self):
        class Fixture:
            count=0

            async def item(self,job,background,objects,folder,report):
                self.count+=1
                self.size=job.object_size
                for suffix in ("png","psd"):
                    (folder/f"estimated-preview.{suffix}").write_bytes(b"fixture")
                report.update(status="review",delivery="estimated_preview",preview_outputs={
                    "png":"estimated-preview.png","psd":"estimated-preview.psd"})

        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            object_path=root/"object.png"
            Image.new("RGB",(48,32)).save(object_path)
            Image.new("RGB",(48,32),"gray").save(root/"frame.png")
            sha=digest(object_path)
            ident=f"asset-{sha[:12]}"
            catalog=root/"catalog.json"
            catalog.write_text(json.dumps(dict(schema_version=1,source_root=str(root),assets=[dict(
                file=object_path.name,sha256=sha,asset_id=ident,category="fixture",object_size=dict(
                    major_mm=[140,155],minor_mm=[50,70],basis="similar_catalog",source="manufacturer"))])),
                encoding="utf-8")
            fixture=Fixture()
            fixture.catalog=Catalog(catalog)
            args=SimpleNamespace(out=root/"out",manifest=root/"tasks.jsonl",retry=False)
            engine=SimpleNamespace(versions=["test"])
            bad=Job(job_id="unknown",object="frame.png",background="frame.png")
            good=Job(job_id="known",object=ident,background="frame.png")
            self.assertEqual(asyncio.run(process(bad,args,Config(),engine,fixture))["reason_code"],"unknown_asset")
            self.assertEqual(fixture.count,0)
            first=asyncio.run(process(good,args,Config(),engine,fixture))
            self.assertEqual(first["delivery"],"estimated_preview")
            self.assertEqual(fixture.size.major_mm,(140,155))
            self.assertTrue(asyncio.run(process(good,args,Config(),engine,fixture))["resumed"])
            self.assertEqual(fixture.count,1)
            Image.new("RGB",(48,32),"yellow").save(object_path)
            result=asyncio.run(process(good,args,Config(),engine,fixture))
            self.assertEqual(result["reason_code"],"catalog_asset_changed")

    def test_prepared_generated_keeps_original_alpha(self):
        class Fixture(Runner):
            async def asset(self,source,scratch):
                raise AssertionError("prepared image was processed again")

            def offline_specs(self,job,report,background,assets,scratch):
                self.inputs=assets
                raise NeedsReview("fixture stopped before rendering")

        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            image=root/"generated.png"
            mask=np.zeros((96,128),dtype=np.uint8)
            mask[9:72,13:109]=254
            cutout=Image.new("RGBA",(128,96),(180,123,92,0))
            cutout.putalpha(Image.fromarray(mask))
            cutout.save(image)
            runner=Fixture(None,None,Config(),None,root,None)
            job=Job(job_id="prepared",object=image.name,background=image.name,mode="preview")
            with self.assertRaisesRegex(NeedsReview,"fixture stopped"):
                asyncio.run(runner.item(job,image,[image],root,{"asset":{"prepared_generated":True}}))
            self.assertEqual(runner.inputs,[image])
            self.assertEqual(digest(image),digest(runner.inputs[0]))

    def test_scene_content_cache_and_malformed_coalescing(self):
        class Fixture(Runner):
            count=0
            answer='{"zones":["Z1"],"refs":[["R2","plate",500,650]]}'

            async def ask(self,job,report,endpoint,role,schema,prompt,payload,picture):
                self.count+=1
                self.assertion=endpoint.max_output_tokens
                return schema.model_validate_json(self.answer)

        async def check(root):
            frame=root/"frame.png"
            renamed=root/"copy.png"
            changed=root/"changed.png"
            Image.new("RGB",(160,90),"gray").save(frame)
            renamed.write_bytes(frame.read_bytes())
            Image.new("RGB",(160,90),"yellow").save(changed)
            def perceive(path):
                return np.full((90,160,3),128,np.uint8),np.full((90,160),.5,np.float32),[],(160,90)
            engine=SimpleNamespace(versions=["test"],scene=perceive)
            endpoint=Endpoint(url="http://localhost/v1/chat/completions",model="test",max_output_tokens=256)
            runner=Fixture(None,None,Config(planner=endpoint),engine,root,None)
            job=Job(job_id="scene",object="fixture.png",background="frame.png")
            zones={"Z1":{"anchors":[[500,500],[500,490],[500,510]],"refs":["R1"]}}
            refs={"R1":{"points":[[300,300],[400,400]],"piece":0}}
            with patch("batch_v2.atlas_geometry",return_value=(zones,refs)):
                for path in (frame,renamed):
                    with self.assertRaises(ScaleReview) as error:
                        await runner.auto_scene(job,{},path,root)
                    self.assertEqual(error.exception.code,"invalid_model_response")
                self.assertEqual(runner.count,1)
                runner.answer='{"zones":["Z1"],"refs":[["R1","plate",500,650]]}'
                await runner.auto_scene(job,{},changed,root)
                report={}
                await runner.auto_scene(job,report,changed,root)
                self.assertTrue(report["scene_analysis"]["cache_hit"])
                self.assertEqual(runner.count,2)
                self.assertEqual(runner.assertion,192)
        with tempfile.TemporaryDirectory() as tmp:
            asyncio.run(check(Path(tmp)))


if __name__=="__main__":
    unittest.main()
