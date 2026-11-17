"""Prepare a human trial checklist and evaluate the production gate."""
import argparse
import csv
import html
import json
import random
from pathlib import Path
from statistics import median

from scene_v2 import atomic_json, contact_sheet


def rows(path):
    with path.open(encoding="utf-8-sig",newline="") as file:
        return list(csv.DictReader(file))


def write_rows(path,fields,records):
    with path.open("w",encoding="utf-8-sig",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--batch",required=True,type=Path)
    cli.add_argument("--old-tank",type=Path)
    cli.add_argument("--old-extinguisher",type=Path)
    cli.add_argument("--comparison-batch",type=Path)
    args=cli.parse_args()
    root=args.batch
    reports=rows(root/"index.csv")
    passed={r["job_id"]:r for r in reports if r["status"]=="passed" or r.get("delivery")=="estimated_preview"}
    filenames={job:("estimated-preview.png" if r.get("delivery")=="estimated_preview" else "final.png")
               for job,r in passed.items()}
    form=root/"人工验收.csv"
    fields=["job_id","image","accept","severe","notes"]
    old={r["job_id"]:r for r in rows(form)} if form.exists() else {}
    human={}
    for k,r in passed.items():
        image=r["folder"]+"/"+filenames[k]
        record=old.get(k)
        human[k]=record if record and record["image"]==image else dict(job_id=k,image=image,accept="",severe="",notes="")
    if human!=old or not form.exists():
        write_rows(form,fields,list(human.values()))
    missing=[k for k in passed if k not in human or human[k]["accept"] not in {"yes","no"}
             or human[k]["severe"] not in {"yes","no"}]
    accepted=sum(human.get(k,{}).get("accept")=="yes" for k in passed)
    severe=[k for k in passed if human.get(k,{}).get("severe")=="yes"]
    baseline={"36a-tank":args.old_tank,"53a-extinguisher":args.old_extinguisher}
    comparison={r["job_id"]:r for r in rows(args.comparison_batch/"index.csv")} if args.comparison_batch else {}
    rng=random.Random(23)
    pairs=[]
    answer={}
    for i,(job,previous) in enumerate(baseline.items(),1):
        current=None
        if job in passed:
            current=root/passed[job]["folder"]/filenames[job]
        elif job in comparison and comparison[job].get("folder"):
            candidate=args.comparison_batch/comparison[job]["folder"]
            current=candidate/("final.png" if comparison[job]["status"]=="passed" else "review-only.png")
        if not previous or not previous.is_file() or not current or not current.is_file():
            pairs.append(f"<p>对照 {i} 暂无两张可比成品。</p>")
            continue
        paths=[(previous,"旧版"),(current,"新版")]
        rng.shuffle(paths)
        name=f"盲比-{i}.jpg"
        contact_sheet([p[0] for p in paths],["A","B"],root/name,audit=True)
        answer[str(i)]={"job":job,**dict(zip(["A","B"],[p[1] for p in paths]))}
        pairs.append(f'<p>对照 {i}</p><a href="{html.escape(name)}"><img src="{html.escape(name)}" width="896"></a>')
    (root/"盲比.html").write_text('<!doctype html><meta charset="utf-8"><title>同背景盲比</title>'
                                  "<h1>同背景盲比</h1><p>按承托、遮挡、光照、边缘和画质比较 A/B。</p>"
                                  +"".join(pairs),encoding="utf8")
    atomic_json(root/"盲比答案.json",answer)
    auto_ok=len(reports)==24 and len(passed)>=16
    verdict=("not_ready" if not auto_ok or severe else "pending_human" if missing else
             "pass" if accepted/max(1,len(passed))>=.9 and len(answer)==2 else "not_ready")
    structured=root/"results.jsonl"
    entries=[json.loads(line) for line in structured.read_text(encoding="utf-8").splitlines()] if structured.is_file() else []
    warm=[r for r in entries if not r.get("resumed") and r.get("scene_analysis",{}).get("cache_hit")
          and [call["role"] for call in r["calls"]]==["review"]]
    timing={"count":len(warm),"seconds":median(r["elapsed_seconds"] for r in warm) if warm else None,
            "tokens":median(r["calls"][0].get("usage",{}).get("total_tokens",0) for r in warm) if warm else None}
    formal=sum(r["status"]=="passed" for r in passed.values())
    data={"status":verdict,"jobs":len(reports),"automatic_passed":len(passed),
          "formal":formal,"estimated_previews":len(passed)-formal,"warm_review_only":timing,"human_pending":missing,
          "human_accepted":accepted,"severe":severe,"blind_pairs_available":len(answer)}
    atomic_json(root/"试产验收.json",data)
    print(json.dumps(data,ensure_ascii=False))


if __name__=="__main__":
    main()
