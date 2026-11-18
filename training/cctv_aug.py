"""Targeted pixel-level augmentations for roadside CCTV, shared by the YOLO26 and DEIMv2 recipes.

Pixel-level only (no geometry), so boxes stay exact. Probabilities are per image.
`level`: 'cctv' (full), 'light' (every probability halved), 'off'. `strength` scales every probability.
`colour_jitter` adds an HSV jitter for pipelines that have no colour augmentation of their own (DEIMv2 once its
RandomPhotometricDistort is dropped; YOLO has Ultralytics' hsv_* instead).
"""


def build(level='cctv', strength=1.0, colour_jitter=False):
    if level == 'off':
        return None
    import cv2
    import albumentations as A
    s = strength * (0.5 if level == 'light' else 1.0)
    p = lambda x: min(1.0, x * s)
    ops = [
        # optics and motion: 20 fps footage of fast vehicles, cheap lenses, night shutter
        A.MotionBlur(blur_limit=(3, 7), p=p(0.12)),
        A.Defocus(radius=(2, 4), alias_blur=(0.1, 0.3), p=p(0.04)),
        # sensor and stream: gain noise at night, sub-stream resolution (1600x900 cameras), H.264/.dav compression
        A.GaussNoise(std_range=(0.02, 0.08), per_channel=True, p=p(0.10)),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=p(0.08)),
        A.Downscale(scale_range=(0.5, 0.9), interpolation_pair={'downscale': cv2.INTER_AREA, 'upscale': cv2.INTER_LINEAR},
                    p=p(0.15)),
        A.ImageCompression(quality_range=(40, 90), p=p(0.15)),
        # exposure: dawn/dusk gamma, backlight, auto-exposure contrast, infrared night mode (monochrome)
        A.RandomGamma(gamma_limit=(70, 140), p=p(0.15)),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.25, p=p(0.10)),
        A.CLAHE(clip_limit=(1, 3), p=p(0.05)),
        A.ToGray(num_output_channels=3, p=p(0.08)),
        # scene: mist and drizzle on the lens, tree shadows across the road, headlight / sun flare
        A.RandomFog(fog_coef_range=(0.1, 0.3), alpha_coef=0.08, p=p(0.05)),
        A.RandomRain(slant_range=(-10, 10), drop_length=15, drop_width=1, blur_value=3, brightness_coefficient=0.9,
                     rain_type='drizzle', p=p(0.04)),
        A.RandomShadow(shadow_roi=(0, 0.3, 1, 1), num_shadows_limit=(1, 2), shadow_dimension=5,
                       shadow_intensity_range=(0.3, 0.6), p=p(0.08)),
        A.RandomSunFlare(flare_roi=(0, 0, 1, 0.6), src_radius=120, num_flare_circles_range=(3, 6), p=p(0.03)),
    ]
    if colour_jitter:
        # close to Ultralytics hsv_h 0.015 / hsv_s 0.6 / hsv_v 0.5; small hue only: colour separates taxis from
        # buses and trucks, and channel permutation (torchvision's RandomPhotometricDistort) would destroy it
        ops.insert(0, A.HueSaturationValue(hue_shift_limit=3, sat_shift_limit=40, val_shift_limit=30, p=0.5))
    return ops
