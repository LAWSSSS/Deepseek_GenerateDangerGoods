"""Audit the delivered receiver boundary, including its inner transition bands."""
import cv2
import numpy as np


def audit_boundary(before,after,alpha,receiver):
    before=np.float32(before)/255;after=np.float32(after)/255
    h,w=alpha.shape
    near=cv2.dilate(np.uint8(alpha>.001),np.ones((9,9),np.uint8))>0
    distance=cv2.distanceTransform(np.pad(receiver.astype(np.uint8),1),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)[1:-1,1:-1]
    yy,xx=np.mgrid[:h,:w];crop=np.minimum.reduce([xx,yy,w-1-xx,h-1-yy])
    delta=after-before
    luma=delta@np.float32([.2126,.7152,.0722])
    gradient=cv2.magnitude(cv2.Sobel(luma,cv2.CV_32F,1,0)/8,cv2.Sobel(luma,cv2.CV_32F,0,1)/8)
    bands=[]
    for low,high in ((0,2),(2,8),(8,16),(16,32)):
        mask=((distance>low)&(distance<=high)|((crop>=low)&(crop<high)))&~near
        values=gradient[mask];color=np.max(abs(delta[mask]),axis=1)
        bands.append({'range_px':[low,high],'samples':int(mask.sum()),
            'gradient_p95':float(np.percentile(values,95)) if values.size else 0.,
            'rgb_delta_p95':float(np.percentile(color,95)) if color.size else 0.})
    # Large smooth shadows remain legitimate inside the receiver. Abrupt cutoffs do not.
    passed=all(b['gradient_p95']<=.035 for b in bands) and bands[0]['rgb_delta_p95']<=.04
    return {'pass':passed,'bands':bands,'gradient_limit':.035,'outer_rgb_limit':.04,
            'basis':'final_png_minus_original_background_multiband','object_edge_excluded':True}
