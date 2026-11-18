# Error analysis: yolo26l-o2m-80e.json at score >= 0.34

4554 images, 16021 GT boxes, matched class-agnostically at IoU >= 0.5.

| | mAP50-95 | mAP50 |
|---|---|---|
| class-aware | 0.455 | 0.674 |
| class-agnostic (any class counts) | 0.498 | 0.809 |

The gap between the two rows is what wrong class labels cost.

## Confusion (rows = true class, share of its GT boxes)

| true \ pred | Bus | Cyclist | LV | Long Trk | Medium Trk | Taxi | Motorcycle | Pedestrian | Short Trk | missed | GT |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Bus | **0.53** | · | 0.01 | 0.06 | 0.02 | 0.22 | · | · | 0.01 | 0.14 | 81 |
| Cyclist | · | **0.69** | · | · | · | · | · | 0.23 | · | 0.08 | 26 |
| LV | · | · | **0.75** | · | · | · | · | · | · | 0.24 | 10985 |
| Long Trk | · | · | · | **0.84** | 0.04 | · | · | · | 0.02 | 0.09 | 511 |
| Medium Trk | · | · | 0.01 | 0.05 | **0.58** | · | · | · | 0.21 | 0.13 | 248 |
| Taxi | · | · | 0.08 | · | · | **0.66** | · | · | · | 0.26 | 1696 |
| Motorcycle | · | 0.03 | 0.03 | · | · | · | **0.77** | · | · | 0.18 | 39 |
| Pedestrian | · | · | · | · | · | · | · | **0.33** | · | 0.66 | 1840 |
| Short Trk | · | · | 0.07 | 0.01 | 0.07 | 0.02 | · | · | **0.53** | 0.31 | 595 |
| background FP | 3 | 3 | 1674 | 41 | 22 | 245 | 5 | 243 | 69 | | |

## Night infrared (grey frame) vs colour (Bus, Cyclist, LV, Long Trk, Medium Trk, Taxi, Motorcycle, Short Trk)

| | images | GT boxes | found (any class) | recall (right class) | precision |
|---|---|---|---|---|---|
| colour | 1623 | 10873 | 0.699 | 0.679 | 0.780 |
| IR / grey | 2931 | 3308 | 0.961 | 0.899 | 0.881 |

## Camera resolution (Bus, Cyclist, LV, Long Trk, Medium Trk, Taxi, Motorcycle, Short Trk)

| | images | GT boxes | found (any class) | recall (right class) | precision |
|---|---|---|---|---|---|
| 1600x900 | 713 | 5551 | 0.754 | 0.746 | 0.827 |
| 1920x1080 | 809 | 5189 | 0.634 | 0.605 | 0.725 |
| 2560x1440 | 3032 | 3441 | 0.960 | 0.895 | 0.877 |

## Box size at the model input (sqrt area, letterbox scale) (Bus, Cyclist, LV, Long Trk, Medium Trk, Taxi, Motorcycle, Short Trk)

| | images | GT boxes | found (any class) | recall (right class) | precision |
|---|---|---|---|---|---|
| <8 px |  | 1499 | 0.149 | 0.145 | – |
| 8-16 px |  | 2817 | 0.586 | 0.568 | – |
| 16-32 px |  | 3008 | 0.776 | 0.750 | – |
| 32-64 px |  | 3448 | 0.934 | 0.902 | – |
| 64-128 px |  | 2913 | 0.982 | 0.935 | – |
| >=128 px |  | 496 | 0.992 | 0.911 | – |

## Per test camera (Bus, Cyclist, LV, Long Trk, Medium Trk, Taxi, Motorcycle, Short Trk)

| | images | GT boxes | found (any class) | recall (right class) | precision |
|---|---|---|---|---|---|
| scene_079 | 749 | 5459 | 0.768 | 0.757 | 0.828 |
| scene_103 | 456 | 1776 | 0.647 | 0.601 | 0.697 |
| scene_025 | 105 | 1559 | 0.533 | 0.516 | 0.764 |
| scene_166 | 1049 | 1232 | 0.958 | 0.875 | 0.858 |
| scene_193 | 973 | 1060 | 0.992 | 0.949 | 0.907 |
| scene_064 | 48 | 812 | 0.728 | 0.714 | 0.691 |
| scene_014 | 70 | 768 | 0.609 | 0.589 | 0.726 |
| scene_208 | 681 | 767 | 0.928 | 0.877 | 0.883 |
| scene_113 | 131 | 277 | 0.913 | 0.859 | 0.829 |
| scene_059 | 85 | 247 | 0.587 | 0.547 | 0.767 |
| scene_169 | 207 | 224 | 0.942 | 0.871 | 0.874 |
