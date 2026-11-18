# Error analysis: yolo26l-o2m-base.json at score >= 0.34

4554 images, 16021 GT boxes, matched class-agnostically at IoU >= 0.5.

| | mAP50-95 | mAP50 |
|---|---|---|
| class-aware | 0.457 | 0.678 |
| class-agnostic (any class counts) | 0.498 | 0.810 |

The gap between the two rows is what wrong class labels cost.

## Confusion (rows = true class, share of its GT boxes)

| true \ pred | Bus | Cyclist | LV | Long Trk | Medium Trk | Taxi | Motorcycle | Pedestrian | Short Trk | missed | GT |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Bus | **0.58** | · | · | 0.06 | 0.02 | 0.22 | · | · | · | 0.11 | 81 |
| Cyclist | · | **0.69** | · | · | · | · | · | 0.19 | · | 0.12 | 26 |
| LV | · | · | **0.74** | · | · | · | · | · | · | 0.25 | 10985 |
| Long Trk | · | · | · | **0.84** | 0.04 | · | · | · | 0.02 | 0.09 | 511 |
| Medium Trk | · | · | 0.01 | 0.04 | **0.60** | · | · | · | 0.19 | 0.15 | 248 |
| Taxi | · | · | 0.07 | · | · | **0.66** | · | · | · | 0.27 | 1696 |
| Motorcycle | · | 0.03 | 0.03 | · | · | · | **0.77** | · | · | 0.18 | 39 |
| Pedestrian | · | · | · | · | · | · | · | **0.32** | · | 0.68 | 1840 |
| Short Trk | · | · | 0.06 | 0.01 | 0.07 | 0.02 | · | · | **0.56** | 0.28 | 595 |
| background FP | 4 | 6 | 1473 | 44 | 21 | 225 | 5 | 209 | 73 | | |

## Night infrared (grey frame) vs colour (Bus, Cyclist, LV, Long Trk, Medium Trk, Taxi, Motorcycle, Short Trk)

| | images | GT boxes | found (any class) | recall (right class) | precision |
|---|---|---|---|---|---|
| colour | 1623 | 10873 | 0.689 | 0.672 | 0.798 |
| IR / grey | 2931 | 3308 | 0.962 | 0.902 | 0.882 |

## Camera resolution (Bus, Cyclist, LV, Long Trk, Medium Trk, Taxi, Motorcycle, Short Trk)

| | images | GT boxes | found (any class) | recall (right class) | precision |
|---|---|---|---|---|---|
| 1600x900 | 713 | 5551 | 0.741 | 0.732 | 0.846 |
| 1920x1080 | 809 | 5189 | 0.629 | 0.604 | 0.742 |
| 2560x1440 | 3032 | 3441 | 0.960 | 0.898 | 0.880 |

## Box size at the model input (sqrt area, letterbox scale) (Bus, Cyclist, LV, Long Trk, Medium Trk, Taxi, Motorcycle, Short Trk)

| | images | GT boxes | found (any class) | recall (right class) | precision |
|---|---|---|---|---|---|
| <8 px |  | 1499 | 0.121 | 0.117 | – |
| 8-16 px |  | 2817 | 0.567 | 0.553 | – |
| 16-32 px |  | 3008 | 0.772 | 0.751 | – |
| 32-64 px |  | 3448 | 0.934 | 0.902 | – |
| 64-128 px |  | 2913 | 0.983 | 0.938 | – |
| >=128 px |  | 496 | 0.992 | 0.911 | – |

## Per test camera (Bus, Cyclist, LV, Long Trk, Medium Trk, Taxi, Motorcycle, Short Trk)

| | images | GT boxes | found (any class) | recall (right class) | precision |
|---|---|---|---|---|---|
| scene_079 | 749 | 5459 | 0.753 | 0.742 | 0.847 |
| scene_103 | 456 | 1776 | 0.641 | 0.602 | 0.730 |
| scene_025 | 105 | 1559 | 0.531 | 0.515 | 0.760 |
| scene_166 | 1049 | 1232 | 0.959 | 0.883 | 0.861 |
| scene_193 | 973 | 1060 | 0.993 | 0.945 | 0.899 |
| scene_064 | 48 | 812 | 0.717 | 0.706 | 0.717 |
| scene_014 | 70 | 768 | 0.608 | 0.589 | 0.731 |
| scene_208 | 681 | 767 | 0.926 | 0.877 | 0.891 |
| scene_113 | 131 | 277 | 0.899 | 0.852 | 0.837 |
| scene_059 | 85 | 247 | 0.599 | 0.563 | 0.818 |
| scene_169 | 207 | 224 | 0.946 | 0.879 | 0.883 |
