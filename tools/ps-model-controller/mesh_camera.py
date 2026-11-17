"""Calibrate a single focal length and principal point from source silhouettes."""
import math

import cv2
import numpy as np
from scipy.optimize import differential_evolution


def fit_source_camera(source_alpha, rendered_alpha):
    """Fit a perspective camera's field of view and optical center, not mesh shape."""
    source=cv2.resize(np.asarray(source_alpha,dtype=np.uint8),(128,128),interpolation=cv2.INTER_AREA)>=128
    rendered=cv2.resize(np.asarray(rendered_alpha,dtype=np.uint8),(128,128),interpolation=cv2.INTER_AREA)>=128
    if not source.any() or not rendered.any():
        raise ValueError("Camera calibration needs two nonempty silhouettes")
    def objective(value):
        scale,dx,dy=value
        matrix=cv2.getRotationMatrix2D((64,64),0,float(scale))
        matrix[:,2]+=[dx,dy]
        trial=cv2.warpAffine(rendered.astype("uint8"),matrix,(128,128),flags=cv2.INTER_NEAREST)>0
        union=np.logical_or(source,trial).sum()
        return -float(np.logical_and(source,trial).sum()/union) if union else 0.
    fit=differential_evolution(objective,[(.65,1.5),(-24,24),(-24,24)],
                               seed=42,popsize=8,maxiter=80,tol=.01,polish=False)
    scale,dx,dy=map(float,fit.x)
    fov=math.degrees(2*math.atan(math.tan(math.radians(40)/2)/scale))
    return {"eye":[1.9,0,0],"target":[0,0,0],"fov_deg":round(fov,6),
            "shift_x":round(-dx/128,6),"shift_y":round(dy/128,6),
            "predicted_iou":round(-float(fit.fun),4),"scale":round(scale,6)}
