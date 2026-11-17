"""Recover the program-ranked best reviewed image after a legacy batch review exception."""
import argparse
import json
from pathlib import Path
import shutil

from display_batch.common import atomic_json,file_hash
from display_batch.pipeline import batch_lock,write_summary
from display_batch.schema import Review,version_rank
from display_batch.store import Store


def recover(folder):
    folder=Path(folder).resolve();recovered=[]
    with batch_lock(folder):
        summary=json.loads((folder/'summary.json').read_text(encoding='utf8'))
        if summary['run']['state']!='complete':raise ValueError('Only a finished batch can be repaired')
        store=Store(folder/'jobs.sqlite')
        try:
            for row in store.rows():
                if row['state']!='failed' or not row['result']:continue
                result=row['result']
                eligible=[r for r in result.get('iterations',[]) if r.get('review') and Path(r['image']).is_file()]
                if not eligible:continue
                best=max(eligible,key=lambda r:version_rank(r['checks'],Review.model_validate(r['review'])))
                if result.get('best_iteration')==best['iteration'] and result.get('checks')==best['checks']:continue
                job_folder=folder/'jobs'/row['id']
                if not job_folder.resolve().is_relative_to(folder):raise ValueError('invalid job location')
                original=job_folder/'recovery-original-result.json'
                if not original.exists():atomic_json(original,result)
                dest=folder/'review'/(row['id']+'.png');crop=dest.with_suffix('.crop.jpg');dest.parent.mkdir(exist_ok=True)
                previous_hash=file_hash(dest) if dest.is_file() else None
                shutil.copy2(best['image'],dest);shutil.copy2(best['crop'],crop)
                result.update(image=str(dest),crop=str(crop),best_iteration=best['iteration'],checks=best['checks'],
                              scores=best['review']['scores'],measured_major_px=best['measured_major_px'])
                evidence={'policy':'existing_version_rank_only','original_final_hash':previous_hash,
                          'selected_iteration':best['iteration'],'selected_hash':file_hash(dest),
                          'status_unchanged':'failed','new_api_calls':0,'new_render_calls':0}
                assert evidence['selected_hash']==best['image_hash']
                atomic_json(job_folder/'evidence-recovery.json',evidence)
                store.finish(row['id'],'failed',result);recovered.append(row['id'])
            if recovered:
                updated=write_summary(folder,store,summary['usage_current_run'],summary['run'])
                updated['evidence_recovery']={'jobs':recovered,'status_changes':0,'new_api_calls':0,'new_render_calls':0}
                atomic_json(folder/'summary.json',updated)
        finally:store.close()
    return {'batch':str(folder),'recovered':recovered,'status_changes':0,'new_api_calls':0,'new_render_calls':0}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--batches',nargs='+',required=True);args=parser.parse_args()
    print(json.dumps([recover(folder) for folder in args.batches],ensure_ascii=False,indent=2))
