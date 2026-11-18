# Error analysis: deimv2-l-base.json at score >= 0.41

4554 images, 16021 GT boxes, matched class-agnostically at IoU >= 0.5.

| | mAP50-95 | mAP50 |
|---|---|---|
| class-aware | 0.499 | 0.754 |
| class-agnostic (any class counts) | 0.503 | 0.834 |

The gap between the two rows is what wrong class labels cost.

## Confusion (rows = true class, share of its GT boxes)

| true \ pred | Bus | Cyclist | LV | Long Trk | Medium Trk | Taxi | Motorcycle | Pedestrian | Short Trk | missed | GT |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Bus | **0.60** | · | 0.01 | 0.07 | 0.02 | 0.22 | · | · | 0.01 | 0.05 | 81 |
| Cyclist | · | **0.77** | · | · | · | · | 0.04 | 0.12 | · | 0.08 | 26 |
| LV | · | · | **0.78** | · | · | · | · | · | · | 0.21 | 10985 |
| Long Trk | · | · | · | **0.85** | 0.05 | · | · | · | 0.01 | 0.09 | 511 |
| Medium Trk | · | · | 0.01 | 0.06 | **0.56** | · | · | · | 0.23 | 0.14 | 248 |
| Taxi | · | · | 0.08 | · | · | **0.68** | · | · | · | 0.24 | 1696 |
| Motorcycle | · | · | 0.03 | · | · | · | **0.77** | · | · | 0.21 | 39 |
| Pedestrian | · | · | · | · | · | · | · | **0.42** | · | 0.58 | 1840 |
| Short Trk | · | · | 0.06 | 0.02 | 0.06 | 0.03 | · | · | **0.54** | 0.29 | 595 |
| background FP | 5 | 4 | 1577 | 81 | 90 | 236 | 2 | 367 | 153 | | |

## Night infrared (grey frame) vs colour (Bus, Cyclist, LV, Long Trk, Medium Trk, Taxi, Motorcycle, Short Trk)

| | images | GT boxes | found (any class) | recall (right class) | precision |
|---|---|---|---|---|---|
| colour | 1623 | 10873 | 0.736 | 0.716 | 0.794 |
| IR / grey | 2931 | 3308 | 0.966 | 0.905 | 0.844 |

## Camera resolution (Bus, Cyclist, LV, Long Trk, Medium Trk, Taxi, Motorcycle, Short Trk)

| | images | GT boxes | found (any class) | recall (right class) | precision |
|---|---|---|---|---|---|
| 1600x900 | 713 | 5551 | 0.784 | 0.774 | 0.851 |
| 1920x1080 | 809 | 5189 | 0.679 | 0.650 | 0.735 |
| 2560x1440 | 3032 | 3441 | 0.967 | 0.902 | 0.836 |

## Box size at the model input (sqrt area, letterbox scale) (Bus, Cyclist, LV, Long Trk, Medium Trk, Taxi, Motorcycle, Short Trk)

| | images | GT boxes | found (any class) | recall (right class) | precision |
|---|---|---|---|---|---|
| <8 px |  | 1499 | 0.155 | 0.148 | – |
| 8-16 px |  | 2817 | 0.644 | 0.625 | – |
| 16-32 px |  | 3008 | 0.821 | 0.798 | – |
| 32-64 px |  | 3448 | 0.963 | 0.930 | – |
| 64-128 px |  | 2913 | 0.987 | 0.940 | – |
| >=128 px |  | 496 | 0.992 | 0.903 | – |

## Per test camera (Bus, Cyclist, LV, Long Trk, Medium Trk, Taxi, Motorcycle, Short Trk)

| | images | GT boxes | found (any class) | recall (right class) | precision |
|---|---|---|---|---|---|
| scene_079 | 749 | 5459 | 0.795 | 0.783 | 0.849 |
| scene_103 | 456 | 1776 | 0.642 | 0.597 | 0.780 |
| scene_025 | 105 | 1559 | 0.641 | 0.621 | 0.702 |
| scene_166 | 1049 | 1232 | 0.972 | 0.896 | 0.814 |
| scene_193 | 973 | 1060 | 0.991 | 0.943 | 0.884 |
| scene_064 | 48 | 812 | 0.770 | 0.757 | 0.710 |
| scene_014 | 70 | 768 | 0.648 | 0.626 | 0.707 |
| scene_208 | 681 | 767 | 0.928 | 0.879 | 0.845 |
| scene_113 | 131 | 277 | 0.942 | 0.903 | 0.814 |
| scene_059 | 85 | 247 | 0.676 | 0.623 | 0.770 |
| scene_169 | 207 | 224 | 0.942 | 0.862 | 0.811 |
