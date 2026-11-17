"""Conservative directional light cues from narrow cylindrical scene objects."""
import numpy as np
from PIL import Image

from occlusion import thin_line_seeds


def estimate_light_from_rods(frame, center, size):
    """Infer a screen-plane direction only with multiple consistent rod cues.

    Assumes locally cylindrical rods with approximately uniform surface color.
    A still image cannot uniquely determine 3-D illumination. The returned
    direction is an uncertain appearance cue, not a measured light source.
    """
    gray=np.asarray(frame.convert('L'),dtype=np.float32)
    cues=[]
    for seed in thin_line_seeds(frame,center,size,limit=12):
        yy,xx=np.nonzero(seed)
        if len(xx)<50:continue
        coords=np.column_stack((xx,yy)).astype(float)
        _,vectors=np.linalg.eigh(np.cov(coords.T))
        axis=vectors[:,-1]
        if axis[0]<0 or (axis[0]==0 and axis[1]<0):axis=-axis
        normal=np.array([-axis[1],axis[0]])
        offset=(coords-coords.mean(axis=0))@normal
        lo,hi=np.percentile(offset,[30,70])
        first=gray[yy[offset<=lo],xx[offset<=lo]]
        last=gray[yy[offset>=hi],xx[offset>=hi]]
        if min(len(first),len(last))<12:continue
        contrast=float(np.median(last)-np.median(first))
        if not 5<=abs(contrast)<=65:continue
        cues.append((normal,contrast))
    if len(cues)<3:
        return {'status':'unknown','reason':'可用的独立钢件明暗线索不足','cues':len(cues)}
    matrix=np.array([c[0] for c in cues]);values=np.array([c[1] for c in cues])
    if np.linalg.cond(matrix)>3:
        return {'status':'unknown','reason':'钢件方向过于相近，无法估计光向','cues':len(cues)}
    vector,*_=np.linalg.lstsq(matrix,values,rcond=None)
    strength=float(np.linalg.norm(vector))
    error=float(np.median(np.abs(matrix@vector-values)))
    if strength<5 or error>max(4,strength*.35):
        return {'status':'unknown','reason':'钢件明暗线索不一致','cues':len(cues)}
    direction=vector/strength
    return {'status':'estimated','direction_screen':[round(float(v),4) for v in direction],
            'cues':len(cues),'residual':round(error,2),'contrast':round(strength,2),
            'assumptions':['细钢件近似圆柱且表面颜色一致','光向只在屏幕平面估计，不是三维实测'],
            'confidence':'low'}
