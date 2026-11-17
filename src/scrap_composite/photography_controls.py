"""Translate supported photographic quantities into the renderer's actual controls."""
import math

from .photography_protocol import ControlChange


def _steps(factor, control, current, values, strength, stage, effect, **metadata):
    """Use the nearest effective absolute value in each direction."""
    origin = strength(current)
    if origin is None:
        return []
    nearest = {}
    for value in values:
        achieved = strength(value)
        if value == current or achieved is None or math.isclose(achieved, origin, abs_tol=1e-4):
            continue
        direction = 'increase' if achieved > origin else 'decrease'
        distance = abs(achieved - origin)
        if direction not in nearest or distance < nearest[direction][0]:
            nearest[direction] = distance, value
    return [ControlChange(factor=factor, direction=direction, control=control, current=current,
                          value=value, effect=effect[direction], stage=stage, **metadata)
            for direction, (_, value) in nearest.items()]


def composite_changes(params, allowed, photo, *, tone_profiles, light_profiles=None):
    """Map only effects available in the current composition and measured rendering path."""
    changes = []
    realism = params.get('realism_version') == 'realism-v1'
    if realism:
        allowed={key:values for key,values in allowed.items()
                 if key not in ('chroma','contrast','dust','haze','pool_kernel','material')}
        # The existing renderer changes direct key+fill energy relative to its
        # fixed broad environment. It does NOT rotate a light or widen highlights.
        if ('light' in allowed and light_profiles and
                (photo.get('lighting_basis') == 'bounded-broad-light-v1' or
                 (photo.get('calibration') or {}).get('version') == 'bounded-broad-light-v1')):
            changes.extend(_steps('light_ratio','light',params['light'],allowed['light'],
                lambda value:light_profiles[value],'render',
                {'increase':'增加主补光相对环境光的能量','decrease':'减小主补光相对环境光的能量'}))
    if 'tone' in allowed and not photo.get('exposure_limited'):
        # A tone preset controls both axes. A temperature preset must not reset
        # active exposure, and an exposure preset must not reset active warmth.
        current = params['tone']
        for axis, factor, effect in ((0, 'exposure', {'increase': '提亮中间调', 'decrease': '压低中间调'}),
                                     (1, 'warmth', {'increase': '轻微偏暖', 'decrease': '轻微偏冷'})):
            values = [v for v in allowed['tone'] if tone_profiles[v][1-axis] == tone_profiles[current][1-axis]]
            changes.extend(_steps(factor, 'tone', current, values, lambda v: tone_profiles[v][axis], 'composite', effect))

    def effective(control, value):
        return photo.get('effective_controls', {}).get(value, {}).get(control)

    for control, factor, effect in (
        ('contrast', 'tonal_contrast', {'increase': '增大中尺度明暗反差', 'decrease': '减小中尺度明暗反差'}),
        ('chroma', 'chroma', {'increase': '保留更多本色色度', 'decrease': '减少过量色度'}),
        ('sharpness', 'sharpness', {'increase': '减弱相机模糊', 'decrease': '增加相机模糊'}),
        ('haze', 'haze_amount', {'increase': '增大估计空气雾幕', 'decrease': '减小估计空气雾幕'}),
    ):
        if control in allowed:
            def strength(value):
                amount = effective(control, value)
                # The implementation records blur radius. Its direction is
                # opposite to apparent sharpness.
                return -amount if control == 'sharpness' and amount is not None else amount

            steps=_steps(factor, control, params[control], allowed[control], strength, 'composite', effect)
            if control == 'sharpness' and photo.get('blur_sigma') is not None and photo['blur_sigma'] <= .18:
                steps=[step for step in steps if step.direction != 'increase']
            changes.extend(steps)

    effects = photo.get('effects', {})
    if 'fog_gain' in allowed and photo.get('atmosphere', {}).get('confidence') == 'estimated':
        current = params['fog_gain']
        median = effects.get('fog_total_median', photo.get('haze'))
        if median is not None and 0 < median < .55 and float(current) > 0:
            def veil(value):
                return min(.50 if realism else .55, 1 - (1 - median) ** (float(value) / float(current)))

            changes.extend(_steps('haze_amount', 'fog_gain', current, allowed['fog_gain'], veil, 'composite',
                                  {'increase': '增大局部空气雾幕', 'decrease': '减小局部空气雾幕'}))
    if 'fog_extra' in allowed and effects.get('fog_base', 0) > 0:
        changes.extend(_steps('haze_amount', 'fog_extra', params['fog_extra'], allowed['fog_extra'], float, 'composite',
                              {'increase': '增加附加空气雾幕', 'decrease': '减弱附加空气雾幕'}))
    if 'pool_kernel' in allowed:
        changes.extend(_steps('sharpness', 'pool_kernel', params['pool_kernel'], allowed['pool_kernel'],
                              lambda v: -int(v), 'composite',
                              {'increase': '减小池化格', 'decrease': '增大池化格'}))
    if 'surface_dust' in allowed:
        changes.extend(_steps('dust_coverage', 'surface_dust', params['surface_dust'], allowed['surface_dust'],
                              float, 'render', {'increase': '增加表面浮灰覆盖', 'decrease': '减少表面浮灰覆盖'}))
    if params.get('asset_refinement') == 'component_v1':
        controls = (
            ('oxide_gain', 'wear_detail', 'ratio', '增加已确认铁基区域的不规则氧化', '减少已确认铁基区域的不规则氧化'),
            ('scratch_gain', 'wear_detail', 'ratio', '增加已确认表面的擦伤覆盖', '减少已确认表面的擦伤覆盖'),
            ('relief_gain', 'wear_detail', 'ratio', '增大物理限幅内的磨损微凹凸', '减小磨损微凹凸'),
            ('roughness_shift', 'specular_width', 'roughness', '增加已分类基底粗糙度以展宽反射', '降低已分类基底粗糙度以收窄反射'),
            ('light_width_scale', 'specular_width', 'ratio', '增大宽面光源尺寸以展宽反射', '减小宽面光源尺寸以收窄反射'),
            ('fill_gain', 'key_fill_ratio', 'ratio', '增加补光以减小主辅光比', '减少补光以增大主辅光比'),
        )
        causes = {'oxide_gain':'oxide_distribution', 'scratch_gain':'scratch_distribution',
                  'relief_gain':'surface_normals', 'roughness_shift':'surface_roughness',
                  'light_width_scale':'light_size', 'fill_gain':'light_energy'}
        for control, factor, unit, increase, decrease in controls:
            if control not in allowed or control not in params:
                continue
            # key_fill_ratio is key/fill, distinct from direct/environment light_ratio.
            invert = control == 'fill_gain'
            meaning = {'increase': decrease if invert else increase, 'decrease': increase if invert else decrease}
            changes.extend(_steps(factor, control, params[control], allowed[control],
                                  (lambda value: -float(value)) if invert else float, 'render', meaning,
                                  backend='blender', unit=unit, cause=causes[control]))
    return changes


def photoshop_changes(current, limits):
    changes = []
    for factor, control, step, effect in (
        ('exposure', 'brightness', 6, {'increase': '提亮物品曲线中点', 'decrease': '压低物品曲线中点'}),
        ('warmth', 'warmth', 4, {'increase': '红蓝通道微调偏暖', 'decrease': '红蓝通道微调偏冷'}),
        ('chroma', 'saturation', 6, {'increase': '增加物品饱和度', 'decrease': '减少物品饱和度'}),
        ('black_point', 'black_point', 4, {'increase': '抬升物品暗部曲线点', 'decrease': '降低物品暗部曲线点'}),
        ('highlight_clip', 'highlights', 4, {'increase': '提亮物品亮部曲线点', 'decrease': '降低物品亮部曲线点'}),
        ('tint', 'tint', 4, {'increase': '通道曲线偏洋红', 'decrease': '通道曲线偏绿'}),
    ):
        if control not in current or control not in limits:
            continue
        low, high = limits[control]
        value = current[control]
        candidates = [max(low, value-step), min(high, value+step)]
        if control == 'saturation':
            candidates = [int(v) for v in candidates]
        changes.extend(_steps(factor, control, value, candidates, float, 'photoshop', effect,
                              unit='percent_point' if control == 'saturation' else 'curve_code_value',
                              backend='photoshop'))
    return changes
