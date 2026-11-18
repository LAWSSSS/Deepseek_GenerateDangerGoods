"""Choose the stage that owns a validated correction."""
from ..review import fusion_actions, repair
from ..schema import FusionReview

GEOMETRY = frozenset({'floating', 'penetration', 'occlusion', 'upright', 'scale', 'camera'})


def geometry_issue(review):
    return any(finding.visibility == 'visible' and finding.code in GEOMETRY for finding in review.defects)


async def next_frame(client, observation, review, comp, params, allowed, ids, seen, *,
                     structured, dust_in_shader, recommendation):
    if structured:
        from ..structured import plan_frame
        updated, actions, plan = await plan_frame(
            client, observation, review, comp, params, allowed, ids, seen,
            dust_in_shader=dust_in_shader, recommendation=recommendation)
        return updated, actions, plan
    suggestions = fusion_actions(review, comp, allowed) if isinstance(review, FusionReview) else review.actions
    if not suggestions:
        return None, [], None
    updated, actions = repair(params, review, allowed, ids, seen, dust_in_shader=dust_in_shader,
                              suggestions=suggestions)
    return updated, actions, None
