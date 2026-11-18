# Error analysis: deimv2-l-cctv.json at score >= 0.41

4554 images, 16021 GT boxes, matched class-agnostically at IoU >= 0.5.

| | mAP50-95 | mAP50 |
|---|---|---|
| class-aware | 0.503 | 0.763 |
| class-agnostic (any class counts) | 0.511 | 0.839 |

The gap between the two rows is what wrong class labels cost.

## Confusion (rows = true class, share of its GT boxes)

| true \ pred | Bus | Cyclist | LV | Long Trk | Medium Trk | Taxi | Motorcycle | Pedestrian | Short Trk | missed | GT |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Bus | **0.63** | · | · | 0.05 | 0.01 | 0.15 | · | · | 0.04 | 0.12 | 81 |
| Cyclist | · | **0.73** | · | · | · | · | · | 0.19 | · | 0.08 | 26 |
| LV | · | · | **0.80** | · | · | · | · | · | · | 0.20 | 10985 |
| Long Trk | · | · | · | **0.85** | 0.05 | · | · | · | 0.01 | 0.08 | 511 |
| Medium Trk | · | · | 0.01 | 0.04 | **0.63** | · | · | · | 0.19 | 0.12 | 248 |
| Taxi | · | · | 0.07 | · | · | **0.72** | · | · | · | 0.20 | 1696 |
| Motorcycle | · | 0.03 | 0.05 | · | · | · | **0.82** | · | · | 0.10 | 39 |
| Pedestrian | · | · | · | · | · | · | · | **0.42** | · | 0.58 | 1840 |
| Short Trk | · | · | 0.05 | 0.02 | 0.07 | 0.02 | · | · | **0.57** | 0.27 | 595 |
| background FP | 4 | 5 | 1847 | 75 | 51 | 354 | 9 | 361 | 122 | | |

## Night infrared (grey frame) vs colour (Bus, Cyclist, LV, Long Trk, Medium Trk, Taxi, Motorcycle, Short Trk)

| | images | GT boxes | found (any class) | recall (right class) | precision |
|---|---|---|---|---|---|
| colour | 1623 | 10873 | 0.757 | 0.737 | 0.771 |
| IR / grey | 2931 | 3308 | 0.966 | 0.910 | 0.857 |

## Camera resolution (Bus, Cyclist, LV, Long Trk, Medium Trk, Taxi, Motorcycle, Short Trk)

| | images | GT boxes | found (any class) | recall (right class) | precision |
|---|---|---|---|---|---|
| 1600x900 | 713 | 5551 | 0.798 | 0.787 | 0.832 |
| 1920x1080 | 809 | 5189 | 0.708 | 0.681 | 0.708 |
| 2560x1440 | 3032 | 3441 | 0.966 | 0.906 | 0.853 |

## Box size at the model input (sqrt area, letterbox scale) (Bus, Cyclist, LV, Long Trk, Medium Trk, Taxi, Motorcycle, Short Trk)

| | images | GT boxes | found (any class) | recall (right class) | precision |
|---|---|---|---|---|---|
| <8 px |  | 1499 | 0.214 | 0.204 | – |
| 8-16 px |  | 2817 | 0.673 | 0.654 | – |
| 16-32 px |  | 3008 | 0.841 | 0.817 | – |
| 32-64 px |  | 3448 | 0.963 | 0.932 | – |
| 64-128 px |  | 2913 | 0.985 | 0.944 | – |
| >=128 px |  | 496 | 0.994 | 0.915 | – |

## Per test camera (Bus, Cyclist, LV, Long Trk, Medium Trk, Taxi, Motorcycle, Short Trk)

| | images | GT boxes | found (any class) | recall (right class) | precision |
|---|---|---|---|---|---|
| scene_079 | 749 | 5459 | 0.807 | 0.794 | 0.834 |
| scene_103 | 456 | 1776 | 0.680 | 0.635 | 0.691 |
| scene_025 | 105 | 1559 | 0.651 | 0.635 | 0.735 |
| scene_166 | 1049 | 1232 | 0.970 | 0.897 | 0.825 |
| scene_193 | 973 | 1060 | 0.992 | 0.951 | 0.890 |
| scene_064 | 48 | 812 | 0.802 | 0.791 | 0.691 |
| scene_014 | 70 | 768 | 0.708 | 0.689 | 0.670 |
| scene_208 | 681 | 767 | 0.931 | 0.884 | 0.871 |
| scene_113 | 131 | 277 | 0.935 | 0.895 | 0.832 |
| scene_059 | 85 | 247 | 0.717 | 0.664 | 0.729 |
| scene_169 | 207 | 224 | 0.942 | 0.875 | 0.834 |
