"""Prepare estimated surface maps from six views or a generated flat master."""
import argparse
import json
from pathlib import Path
import time

from display_batch.common import atomic_json, file_hash
from display_batch.reference_textures import prepare_reference, prepare_flat_master


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True,help='One six-view PNG or a directory')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=173)
    parser.add_argument('--flat-master',action='store_true',help='Use one complete generated surface; no 3x2 split or quilting')
    parser.add_argument('--provenance',type=Path,help='JSON generation prompt and original six-view reference paths/hashes')
    parser.add_argument('--size',type=int,choices=(512,1024),default=512,help='Flat master output long edge, downsampling only')
    parser.add_argument('--material-kind',choices=('auto','metal','paint'),default='auto')
    args=parser.parse_args()
    if args.flat_master and (args.input.is_dir() or args.provenance is None):
        parser.error('--flat-master requires one image and --provenance JSON')
    files=sorted(args.input.glob('*.png')) if args.input.is_dir() else [args.input]
    if not files:parser.error('no PNG reference sheets')
    started=time.monotonic();records=[]
    for path in files:
        try:
            out=args.output/(path.stem+'-'+file_hash(path)[:10])
            if args.flat_master:
                provenance=json.loads(args.provenance.read_text(encoding='utf8'))
                if 'masters' in provenance:
                    matches=[entry for entry in provenance['masters'] if
                             Path(entry.get('path','')).name==path.name]
                    if len(matches)!=1:raise ValueError('generation manifest needs one matching master filename')
                    provenance={k:v for k,v in provenance.items() if k!='masters'}|matches[0]
                record=prepare_flat_master(path,out,provenance,size=args.size,material_kind=args.material_kind)
            else:
                if args.size!=512:raise ValueError('--size applies only to --flat-master')
                record=prepare_reference(path,out,seed=args.seed)
            records.append({'input':str(path),'pack':str(out/'reference.json'),'status':record['status'],
                            'reason':record['reason'],'map_count':len(record['maps']),'cache_hit':record['cache_hit']})
            print(path.name+': '+str(len(record['maps']))+' maps; '+record['reason'],flush=True)
        except (OSError,ValueError) as error:
            records.append({'input':str(path),'status':'review','reason':type(error).__name__+': '+str(error)})
    atomic_json(args.output/'manifest.json',{'version':'reference-pbr-v1','records':records,
        'seconds':round(time.monotonic()-started,3),'api_calls':0,'render_calls':0,
        'status':'review','note':'Estimated maps; extraction does not establish material realism.'})


if __name__=='__main__':main()
