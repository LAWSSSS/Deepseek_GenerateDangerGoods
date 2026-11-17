"""Compare steel around a placement with the local bearing plane."""
import numpy as np
import cv2


RELIEF_PRIORITY={"raised":-1,"uncertain":0,"flat":1,"recess":2}


def plane_fit(depth, mask, *, limit=None):
    ys,xs=np.where(mask & np.isfinite(depth))
    if len(xs)<40:
        return None,float("inf")
    if limit and len(xs)>limit:
        step=(len(xs)+limit-1)//limit
        ys,xs=ys[::step],xs[::step]
    z=depth[ys,xs]
    points=np.c_[np.ones(len(xs)),xs,ys]
    fit,_,rank,_=np.linalg.lstsq(points,z,rcond=None)
    if rank!=3:
        return None,float("inf")
    errors=z-points@fit
    inliers=np.abs(errors-np.median(errors))<=max(.013,2.5*np.median(np.abs(errors)))
    if inliers.sum()>=40:
        fit=np.linalg.lstsq(points[inliers],z[inliers],rcond=None)[0]
    return fit,float(np.quantile(np.abs(z-points@fit),.72))


class ReliefField:
    def __init__(self, depth, plane, valid, plane_error):
        # The local depth model uses greater values for nearer surfaces.
        residual=depth-plane
        valid=valid & np.isfinite(residual)
        step=max(.018,2.3*plane_error)
        masks=np.stack((valid&(residual>step),valid&(residual< -step),valid),axis=2)
        self.counts=cv2.integral(masks.astype(np.uint8),sdepth=cv2.CV_32S)
        self.height,self.width=depth.shape

    def measure(self,bounds,major):
        l,t,r,b=bounds
        reach=max(5,round(.18*major))
        inset_x=max(1,round(.11*(r-l)))
        inset_y=max(1,round(.11*(b-t)))
        # The bands start outside the occupied rectangle. The corners also
        # capture troughs whose steel walls run diagonally across the camera.
        sectors=np.asarray([
            [r,t+inset_y,r+reach,b-inset_y],[r,b,r+reach,b+reach],
            [l+inset_x,b,r-inset_x,b+reach],[l-reach,b,l,b+reach],
            [l-reach,t+inset_y,l,b-inset_y],[l-reach,t-reach,l,t],
            [l+inset_x,t-reach,r-inset_x,t],[r,t-reach,r+reach,t],
        ])
        wanted=np.maximum(1,(sectors[:,2]-sectors[:,0])*(sectors[:,3]-sectors[:,1]))
        x,y,xx,yy=np.clip(sectors,0,[self.width,self.height,self.width,self.height]).T
        counts=self.counts[yy,xx]-self.counts[yy,x]-self.counts[y,xx]+self.counts[y,x]
        sampled=counts[:,2]>=np.maximum(12,.60*wanted)
        fraction=counts[:,:2]/np.maximum(1,counts[:,2,None])
        pairs=[np.minimum(fraction[i],fraction[j]) for i in range(8) for j in range(i+3,min(8,i+6))
               if sampled[i] and sampled[j]]
        paired=np.max(pairs,axis=0) if pairs else (0.,0.)
        rim,drop=map(float,paired)
        n=int(sampled.sum())
        terrain="uncertain"
        if n>=4 and pairs:
            if drop>=.30 and drop-rim>=.13:
                terrain="raised"
            elif rim>=.30 and rim-drop>=.13:
                terrain="recess"
            elif float(fraction[sampled].max())<.30:
                terrain="flat"
        return {"kind":terrain,"raised_rim":round(rim,4),"lower_sides":round(drop,4),
                "valid_sectors":n}


def relief_value(relief):
    return relief["raised_rim"]-relief["lower_sides"]
