import copy
import json
import tempfile
import unittest
from pathlib import Path

from scrap_composite.knowledge import (FEATURE_VERSION, KnowledgeBase, camera_group,
    context_from_result, features_from_environment, recommend, versions_from_params)
from scrap_composite.knowledge_cli import discover_history, main


VERSIONS={"photography":"fixture-v1","camera_pipeline":"premultiplied","fog_mode":"local_auto",
          "weathering":"preserve_source","integration_mode":"coherent","workflow":"legacy","feature_schema":FEATURE_VERSION}


def case_pair(camera, asset, background, *, good=True, material="painted_metal"):
    cases=[]
    for index,dust in enumerate(("0.5","0.25")):
        params={"surface_dust":dust,"pool_kernel":"3"}
        features={"material_group":material,"luma_median":90.,"contrast_span":60.,"camera_sigma_px":.9,
            "noise_sigma":.004,"veil_estimate":.06,"target_major_px":200.,
            "local_sample_confidence":"estimated","current_params":params.copy()}
        cases.append({"study_id":f"study-{camera}-{asset}-{background}","sequence":index,
            "asset_id":str(asset),"background_id":f"{camera}-{background}","background_group":f"{camera}-{background}","camera_group":camera,
            "image":f"fixture-{camera}-{asset}-{background}-{index}.png",
            "geometry_key":f"geometry-{camera}-{asset}-{background}","params":params,
            "features":features,"versions":dict(VERSIONS),
            "assessment":{"source":"vision","image_bound":True,
                "scores":{"lighting":3,"material":3 if index and good else 2,"texture":3,"structure":3},"defects":[]},
            "checks":{k:{"status":"pass"} for k in ("protected","outside_edit","holes_alpha")},
            "legacy_status":"review"})
    return cases


def populate(knowledge, *, heldout=True):
    cases=[]
    for asset in ("a","b"):
        for background in ("one","two","three"):
            cases.extend(case_pair("53",asset,background))
    if heldout:
        for asset in ("a","b"):
            for background in ("holdout-one","holdout-two"):cases.extend(case_pair("36",asset,background))
    for case in cases:knowledge.record_case(case)
    knowledge.derive_trials();knowledge.induce_rules()
    return cases


class KnowledgeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.knowledge=KnowledgeBase(self.root/"facts.sqlite")

    def tearDown(self):
        self.knowledge.close();self.temp.cleanup()

    def write(self,name,value):
        path=self.root/name;path.write_text(json.dumps(value),encoding="utf8");return path

    def test_missing_database_does_not_create_one_or_recommend(self):
        path=self.root/"missing/facts.sqlite"
        result=recommend({}, {}, db_path=path)
        self.assertEqual(result["overrides"],{});self.assertFalse(path.parent.exists())

    def test_versions_and_local_features_are_required(self):
        cases=populate(self.knowledge)
        features=cases[0]["features"]
        result=self.knowledge.recommend(features,VERSIONS)
        self.assertEqual(result["status"],"recommended")
        self.assertEqual(result["overrides"],{"surface_dust":"0.25"})
        self.assertIn("object_render",result["affected_stages"])
        self.assertFalse(result["geometry_status_changed"])
        self.assertEqual(self.knowledge.recommend(features,{**VERSIONS,"photography":"different"})["overrides"],{})
        self.assertEqual(self.knowledge.recommend(features,{**VERSIONS,"workflow":"different"})["overrides"],{})
        self.assertEqual(self.knowledge.recommend({**features,"noise_sigma":None},VERSIONS)["overrides"],{})
        self.assertEqual(self.knowledge.recommend({**features,"veil_estimate":.4},VERSIONS)["overrides"],{})
        self.assertEqual(self.knowledge.recommend({**features,"current_params":{"surface_dust":"0.25"}},VERSIONS)["overrides"],{})

    def test_development_does_not_approve_without_holdout(self):
        cases=populate(self.knowledge,heldout=False)
        self.assertEqual(self.knowledge.db.execute("SELECT state FROM rules").fetchone()[0],"trial")
        self.assertEqual(self.knowledge.recommend(cases[0]["features"],VERSIONS)["overrides"],{})

    def test_holdout_cannot_supply_rule_conditions_or_replace_failed_validation(self):
        for asset in ("a","b","c","d"):
            for case in case_pair("36",asset,"one"):self.knowledge.record_case(case)
        self.knowledge.derive_trials();self.assertEqual(self.knowledge.induce_rules(),{})
        self.assertEqual(self.knowledge.counts()["rules"],0)
        cases=populate(self.knowledge,heldout=False)
        for asset in ("e","f","g","h"):
            for case in case_pair("36",asset,"bad",good=False):self.knowledge.record_case(case)
        self.knowledge.derive_trials();self.knowledge.induce_rules()
        self.assertEqual(self.knowledge.recommend(cases[0]["features"],VERSIONS)["overrides"],{})

    def test_insufficient_samples_do_not_produce_rules(self):
        for case in case_pair("53","a","one"):self.knowledge.record_case(case)
        self.knowledge.derive_trials();self.knowledge.induce_rules()
        self.assertEqual(self.knowledge.counts()["rules"],0)

    def test_many_frames_of_one_holdout_group_do_not_replace_two_groups(self):
        cases=populate(self.knowledge,heldout=False)
        for asset in ("a","b"):
            for frame in ("frame-one","frame-two","frame-three"):
                for case in case_pair("36",asset,frame):
                    case["background_group"]="one-heldout-scene"
                    self.knowledge.record_case(case)
        self.knowledge.derive_trials();self.knowledge.induce_rules()
        row=json.loads(self.knowledge.db.execute("SELECT data FROM rules").fetchone()[0])
        self.assertEqual(row["state"],"trial")
        self.assertEqual(row["holdout"]["backgrounds"],1)
        self.assertEqual(row["holdout"]["background_images"],3)
        self.assertEqual(self.knowledge.recommend(cases[0]["features"],VERSIONS)["overrides"],{})

    def test_geometry_or_version_changes_cannot_teach_effect(self):
        a,b=case_pair("53","a","one");b["geometry_key"]="moved"
        for case in (a,b):self.knowledge.record_case(case)
        result=self.knowledge.derive_trials()
        self.assertIn("geometry_changed",result["rejected_comparisons"])
        self.assertEqual(self.knowledge.counts()["trials"],0)

    def test_unreviewed_bad_geometry_and_missing_pixel_checks_are_not_successes(self):
        for index,change in enumerate(("unreviewed","highrisk","floating","pixels")):
            a,b=case_pair("53",str(index),"one")
            if change=="unreviewed":b["assessment"]["source"]="unreviewed"
            elif change=="highrisk":b["high_risk_projection"]=True
            elif change=="floating":b["assessment"]["defects"]=[{"code":"floating","severity":"major"}]
            else:b["checks"]["protected"]={"status":"unknown"}
            for case in (a,b):self.knowledge.record_case(case)
        self.knowledge.derive_trials();self.assertEqual(self.knowledge.counts()["trials"],0)

    def test_duplicate_case_and_conflicting_review_do_not_inflate_success(self):
        cases=populate(self.knowledge)
        before=self.knowledge.counts()
        for case in cases:self.knowledge.record_case(case,source_id="copy")
        self.knowledge.derive_trials();self.knowledge.induce_rules()
        self.assertEqual(before["cases"],self.knowledge.counts()["cases"])
        self.assertEqual(before["trials"],self.knowledge.counts()["trials"])
        for original in (cases[1],cases[7]):
            conflicting=copy.deepcopy(original);conflicting["assessment"]["scores"]["material"]=1
            self.knowledge.record_case(conflicting,source_id="different-review")
        self.knowledge.derive_trials();self.knowledge.induce_rules()
        self.assertEqual(self.knowledge.recommend(cases[0]["features"],VERSIONS)["overrides"],{})
        self.assertEqual(self.knowledge.db.execute("SELECT state FROM rules").fetchone()[0],"superseded")

    def test_cumulative_usage_prefix_copy_and_distinct_actual_requests(self):
        call={"model":"fixture","request_signature":"same-request","attempt":0,"usage":{
            "prompt_tokens":100,"completion_tokens":20,"total_tokens":120}}
        data={"ledger_name":"run-a","calls":[call]}
        path=self.write("usage-a.json",data)
        self.knowledge.import_history([path,path])
        copy_path=self.write("copied.json",data);self.knowledge.import_history([copy_path])
        self.assertEqual(self.knowledge.cost_summary()["actual_requests"],1)
        data["calls"].append({**call,"attempt":1});self.write("usage-a.json",data)
        self.knowledge.import_history([path])
        other=self.write("usage-b.json",{"ledger_name":"run-b","calls":[call]})
        self.knowledge.import_history([other])
        summary=self.knowledge.cost_summary()
        self.assertEqual(summary["actual_requests"],3);self.assertEqual(summary["reported_tokens"],360)
        self.assertFalse(summary["complete"]);self.assertIsNone(summary["estimated_max"])

    def test_aggregate_cost_never_added_to_event_totals(self):
        aggregate={"task_totals":{"cost_total":{"actual_requests":100}},"prior_attempts":[{"actual_requests":100}]}
        self.knowledge.import_history([self.write("cost-ledger.json",aggregate)])
        self.assertEqual(self.knowledge.cost_summary()["actual_requests"],0)
        self.assertEqual(self.knowledge.counts()["facts"],1)

    def test_exact_provider_id_dedup_and_not_sent_excluded(self):
        call={"model":"fixture","request_id":"provider-1","usage":{"prompt_tokens":10,"completion_tokens":2}}
        paths=[self.write(f"usage-{i}.json",{"ledger_name":str(i),"calls":[call,{"state":"not_sent"}]}) for i in range(2)]
        self.knowledge.import_history(paths)
        self.assertEqual(self.knowledge.cost_summary()["actual_requests"],1)

    def test_changed_ledger_event_marks_cost_incomplete(self):
        a={"ledger_name":"run-a","calls":[{"usage":{"prompt_tokens":2,"completion_tokens":1}}]}
        path=self.write("usage-a.json",a);self.knowledge.import_history([path])
        a["calls"][0]["usage"]["prompt_tokens"]=50;self.write("usage-a.json",a);self.knowledge.import_history([path])
        self.assertEqual(self.knowledge.cost_summary()["identity_conflicts"],1)
        self.assertEqual(self.knowledge.cost_summary()["actual_requests"],1)

    def test_child_iteration_never_inherits_final_image_review(self):
        record={"asset_id":"3","background":"53_2_14.jpg","image":"final.png","review":{"scores":{"material":4}},
            "iterations":[{"image":"first.png","iteration":0,"params":{}},{"image":"final.png","iteration":1,"params":{}}]}
        self.knowledge.import_history([self.write("results.json",[record])])
        for row in self.knowledge.db.execute("SELECT data FROM cases"):
            case=json.loads(row[0]);self.assertEqual(case["assessment"]["source"],"unreviewed")
        self.assertEqual(self.knowledge.counts()["cases"],2)

    def test_browse_report_escapes_evidence_and_keeps_full_images(self):
        cases=populate(self.knowledge)
        cases[0]["image"]="<script>alert(1)</script>";self.knowledge.record_case(cases[0])
        summary=self.knowledge.export_report(self.root/"report")
        page=(self.root/"report/index.html").read_text(encoding="utf8")
        self.assertIn("&lt;script&gt;",page);self.assertNotIn("<script>alert",page)
        self.assertEqual(summary["rule_states"],{"approved":1})

    def test_feature_context_does_not_use_adjusted_object_or_total_fog(self):
        record={"params":{"fog_mode":"local_auto"},"measured_major_px":200,"material_group":"metal",
            "photo":{"version":"v1","after_luma":250,"haze":.9,"measured_camera_sigma":.8,
                "sensor":{"noise_sigma":.002},"local_context":{"luma_range":[30,80,130],
                    "veil_estimate":.04,"confidence":"estimated"}}}
        result=context_from_result(record)
        self.assertEqual(result["features"]["luma_median"],80)
        self.assertEqual(result["features"]["veil_estimate"],.04)
        self.assertEqual(result["features"]["contrast_span"],100)
        self.assertEqual(result["versions"]["fog_mode"],"local_auto")

    def test_camera_split_is_evaluation_metadata_and_discovery_ignores_configs(self):
        self.assertEqual(camera_group(r"C:\images\20261015_parts_53_2_14.jpg"),"53")
        self.assertEqual(camera_group("36_6_6"),"36")
        self.assertIsNone(camera_group("guess53.jpg"))
        self.write("results.json",[]);self.write("config.json",{"secret":"do-not-read"})
        self.write(".env.json",{"secret":"do-not-read"})
        self.assertEqual([p.name for p in discover_history(self.root)],["results.json"])


if __name__=="__main__":unittest.main(verbosity=2)
