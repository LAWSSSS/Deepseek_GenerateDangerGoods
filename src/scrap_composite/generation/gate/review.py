"""Visual observation and diagnosis of the current image."""
from scrap_composite.data.files import file_hash

from ..review import inspect
from ..strict_contact import visual_veto


async def review_frame(client, asset, comp, image_hash, params, allowed, *, structured, profile,
                       prompt, render_dir, compact, geometry_board, choices):
    if file_hash(comp['image']) != image_hash:
        raise ValueError('review_image_changed')
    if structured:
        from ..structured import inspect_frame
        return await inspect_frame(client, asset, comp, image_hash, profile=profile, parameters=params)
    return None, await inspect(client, prompt, asset, comp, params, allowed, render_dir,
        compact=compact, geometry_board=geometry_board,
        candidate_evidence=[{'id': item['id'], 'placement': item.get('placement'),
                             'uncertainties': item.get('uncertainties', [])} for item in choices])


def review_record(observation, review, *, enforce_physics=False):
    entry = {'review': review.model_dump()}
    if observation is not None:
        if hasattr(observation, 'legacy'):
            entry['photography_diagnosis'] = observation.model_dump(mode='json')
            observation = observation.legacy()
        entry['visual_evidence'] = observation.model_dump(mode='json')
    if enforce_physics:
        veto = visual_veto(review)
        entry.update(deliverable=not veto, physical_visual_veto=veto)
    return entry
