# Material, light and camera evidence

## Curved metal

Highlight width depends on roughness, emitter angular size, curvature, viewing direction and tone mapping. Rough metal is not universally identified by a narrow bright line. Diagnose with the same pose under two controlled broad sources and compare diffuse/specular passes. Rotate the asset while holding the light fixed: texture coordinates must remain attached, whereas specular intensity changes with orientation. A streak baked into base color moves with the texture; a reflection responds to illumination.

Preserve polished rods, bearing rings and exposed abraded metal separately from cast surfaces and coatings. Do not decrease metallicness merely to darken steel. Oxide and dust are dielectric coverage over the substrate. Roughness, corrosion coverage and surface deposit are different controls. Paint scratches may reveal metal only where the material identification supports it. Aluminum oxidation does not justify orange iron rust.

## Wear evidence

Review oxide patches, smaller pits, sparse scratches and deposits separately. Match their physical scale to asset dimensions and projected pixels. Repeated parallel marks or stretched cylindrical UVs need coordinate/texture repair, not more noise. AI texture luminance supplies a pattern; it is not a calibrated height map. Preserve source labels, holes, seals and protected machined regions. An unreadable reference yields an unknown region, not invented segmentation.

## Local scene and air

Use comparable, similar-depth neighborhoods and multiple surface orientations. Exclude text, obvious reflections, holes and deep cast shadows. Without known reflectance and normals, local brightness is an illumination interval, not lux. Treat material-independent atmosphere and camera estimates separately from the object appearance.

Broad area lighting, receiver-matched shadows and bounded exposure precede air attenuation. Haze attenuates both diffuse and specular energy and raises dark values toward local airlight; alpha remains geometric. Shadow cannot darken existing scattered air as though it were surface radiance. Do not use dust to cover an exposure or haze-color error.

## Photoshop and final pixels

Inspect profile, bit depth, clipping and actual adjustment values before editing. Measure object and reference regions separately; report sample count and whether reference depth/material is verified. Final-pixel edge sharpness and flat-region noise guide the camera budget. Pooling consumes that same budget. PSD reopening and pixel comparison establish execution, not photographic realism.

Explicit physical failure needs a located visible gap, impossible intersection or contradictory depth relation. Weak shadows or hidden contact alone do not prove floating. New saturation or fog adjustments cannot resolve geometric evidence.
