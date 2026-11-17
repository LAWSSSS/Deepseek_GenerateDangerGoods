"""Offline audit of old vetoes. Never rewrites results, scores or acceptance."""
import argparse
from collections import Counter
import json
from pathlib import Path

import batch_display  # establish core package path, no network or render side effects
from display_batch.common import atomic_json, file_hash
from display_batch.schema import Defect
from display_batch.strict_contact import review_evidence_issue
from scrap_composite.photo_evidence import VERSION


def audit(source):
    records=[]
    for line in source.read_text(encoding='utf8').splitlines():
        row=json.loads(line)
        for item in row.get('iterations',[]):
            vetoes=item.get('physical_visual_veto',[])
            if not vetoes:continue
            findings=[]
            for raw in vetoes:
                defect=Defect.model_validate_json(json.dumps(raw))
                reason=review_evidence_issue(defect)
                findings.append({'original':raw,'evidence_audit':'unresolved' if reason else 'direct_cue_retained',
                    'reason':reason or 'explicit_current_frame_physical_cue',
                    'physical_verification':'not_performed'})
            records.append({'job_id':row['job_id'],'asset_id':row['asset_id'],'background':row['background'],
                'iteration':item.get('iteration'),'image':item.get('image'),'original_status':row['status'],
                'findings':findings,'resolved_status':'unresolved','automatic_pass':False,
                'requires_new_review':True,'original_program_checks':item.get('checks',{})})
    counts=Counter(f['evidence_audit'] for r in records for f in r['findings'])
    return {'version':VERSION,'source':str(source.resolve()),'source_sha256':file_hash(source),
        'mode':'offline_text_consistency_audit','api_requests':0,'render_count':0,
        'summary':{'veto_records':len(records),'unique_jobs':len({r['job_id'] for r in records}),
                   'findings':sum(counts.values()),**counts,'unresolved_jobs':len({r['job_id'] for r in records}),
                   'automatic_passes':0},
        'limitations':'Text consistency only. Retaining a direct cue is not independent confirmation; '
                      'removing an unsupported veto never establishes physical or visual acceptance.',
        'records':records}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();value=audit(args.source)
    atomic_json(args.output,value)
    print(json.dumps(value['summary'],ensure_ascii=False))


if __name__=='__main__':main()
