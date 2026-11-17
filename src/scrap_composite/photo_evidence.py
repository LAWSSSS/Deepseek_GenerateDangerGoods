"""Conservative text guards; uncertainty never proves a physical pass.

These checks detect explicit contradictions, not photographic quality. New
requests also require typed state and reference roles; old records remain
readable and are auditable without changing their scores or final statuses.
"""
import re

VERSION = 'photo-evidence-v2'
_GAP = re.compile(r'缝隙|间隙|空隙|空档|透空|透出背景|visible gap|air gap|clear air below|daylight', re.I)
_NO_GAP = re.compile(r'(?:无|没有|未见|看不到)(?:明显|可见|任何)?的?(?:空气)?(?:缝隙|间隙|空隙|空档)|no (?:visible )?(?:air )?gap', re.I)
_SOURCE = re.compile(r'参考图|识别参考|源图|原图|图[234]|reference|source|棚拍|棚灯', re.I)
_STUDIO = re.compile(r'棚拍|棚灯|studio', re.I)
_BLUR = re.compile(r'(?:轮廓|边界|边缘|细节).{0,12}(?:模糊|糊化|柔化)|blurred|blurry', re.I)
_SHARP = re.compile(r'(?:边缘|轮廓|细节).{0,12}(?:过锐|过硬|过清晰)|oversharp|too sharp', re.I)
_LYING = re.compile(r'横向|横躺|侧卧|平躺|lying|horizontal', re.I)
_DENIED_LYING = re.compile(r'(?:并非|不是|没有|未|不|应该|应当|必须|需要).{0,8}(?:横躺|侧卧|平躺)|(?:not|should be|must be).{0,12}(?:lying|flat)', re.I)
_NO_UPRIGHT = re.compile(r'未见.{0,8}立起|并非.{0,8}直立|不是.{0,8}直立|not upright', re.I)
_INTERSECTION = re.compile(r'穿透|穿入|相交|交叉穿过|surface intersection|intersects? (?:steel|solid|surface)|penetrat', re.I)
_REVERSED = re.compile(r'反向叠压|遮挡关系倒置|前后次序.{0,5}(?:错误|反转)|错误遮挡|reversed overlap|wrong (?:depth|overlap)|depth.order', re.I)
_STANDING = re.compile(r'端部.{0,8}(?:站立|支地)|竖立|直立|standing on end|upright', re.I)


def contradiction(factor, direction, cue, reference=''):
    """Return only definite semantic contradictions; absence is not approval."""
    if factor == 'sharpness' and direction == 'decrease' and _BLUR.search(cue) and not _SHARP.search(cue):
        return 'sharpness_direction_conflicts_with_blurred_observation'
    if factor == 'sharpness' and direction == 'increase' and _SHARP.search(cue) and not _BLUR.search(cue):
        return 'sharpness_direction_conflicts_with_sharp_observation'
    if factor == 'identity' and re.search(r'(?:与.{0,16})?(?:识别参考|参考图).{0,8}一致', cue):
        return 'identity_loss_claim_describes_preserved_identity'
    if factor == 'upright' and _LYING.search(cue) and _NO_UPRIGHT.search(cue):
        return 'upright_claim_describes_lying_object'
    if factor == 'haze_amount' and direction == 'increase' and re.search(r'灰雾.{0,16}(?:暗部抬升|反差低)|雾幕过重|雾量过大', cue):
        return 'haze_direction_conflicts_with_excess_haze_observation'
    return None


def physical_evidence_issue(code, cue, reference='', physical_cue=None, reference_role=None):
    """Veto needs actual geometry, never mere visibility loss or weak shadow."""
    expected = {'floating':'visible_air_gap', 'penetration':'surface_intersection',
                'occlusion':'reversed_overlap', 'upright':'standing_on_end'}.get(code)
    if expected is None:
        return None
    if physical_cue not in (None, expected):
        return 'physical_cue_mismatch'
    if code == 'floating':
        return None if _GAP.search(cue) and not _NO_GAP.search(cue) else 'no_explicit_visible_air_gap'
    if code == 'penetration':
        return None if _INTERSECTION.search(cue) else 'no_explicit_surface_intersection'
    if code == 'occlusion':
        if reference_role == 'identity' or _SOURCE.search(reference):
            return 'unobstructed_identity_reference_cannot_prove_wrong_overlap'
        return None if _REVERSED.search(cue) else 'ordinary_or_ambiguous_occlusion'
    if _NO_UPRIGHT.search(cue) or _LYING.search(cue) and not _DENIED_LYING.search(cue) and _SOURCE.search(reference):
        return 'reference_pose_does_not_establish_standing'
    return None if _STANDING.search(cue) else 'no_explicit_standing_on_end'


def studio_reference(text):
    return bool(_STUDIO.search(text or ''))


def identity_reference(text):
    # A localized scene reference can also be called a "reference image".
    # Reject explicit identity/studio provenance, not that generic noun alone.
    return bool(re.search(r'识别参考|源图|原图|商品图|图[234]|image\s*[234]|source|identity|棚拍|棚灯|studio',
                          text or '', re.I))
