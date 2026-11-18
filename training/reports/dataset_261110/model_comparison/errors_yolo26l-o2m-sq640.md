# Error analysis: yolo26l-o2m-sq640.json at score >= 0.34

4554 images, 16021 GT boxes, matched class-agnostically at IoU >= 0.5.

| | mAP50-95 | mAP50 |
|---|---|---|
| class-aware | 0.474 | 0.699 |
| class-agnostic (any class counts) | 0.510 | 0.824 |

The gap between the two rows is what wrong class labels cost.

## Confusion (rows = true class, share of its GT boxes)

| true \ pred | Bus | Cyclist | LV | Long Trk | Medium Trk | Taxi | Motorcycle | Pedestrian | Short Trk | missed | GT |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Bus | **0.63** | · | · | 0.05 | 0.02 | 0.16 | · | · | · | 0.14 | 81 |
| Cyclist | · | **0.65** | · | · | · | · | 0.04 | 0.23 | · | 0.08 | 26 |
| LV | · | · | **0.73** | · | · | · | · | · | · | 0.27 | 10985 |
| Long Trk | · | · | · | **0.84** | 0.04 | · | · | · | 0.02 | 0.10 | 511 |
| Medium Trk | · | · | · | 0.04 | **0.64** | · | · | · | 0.17 | 0.14 | 248 |
| Taxi | · | · | 0.06 | · | · | **0.67** | · | · | · | 0.27 | 1696 |
| Motorcycle | · | 0.05 | 0.03 | · | · | · | **0.69** | · | · | 0.23 | 39 |
| Pedestrian | · | · | · | · | · | · | · | **0.31** | · | 0.69 | 1840 |
| Short Trk | · | · | 0.05 | 0.01 | 0.07 | 0.01 | · | · | **0.56** | 0.30 | 595 |
| background FP | 1 | 6 | 1256 | 32 | 24 | 233 | 6 | 191 | 63 | | |

## Night infrared (grey frame) vs colour (Bus, Cyclist, LV, Long Trk, Medium Trk, Taxi, Motorcycle, Short Trk)

| | images | GT boxes | found (any class) | recall (right class) | precision |
|---|---|---|---|---|---|
| colour | 1623 | 10873 | 0.674 | 0.659 | 0.816 |
| IR / grey | 2931 | 3308 | 0.959 | 0.901 | 0.892 |

## Camera resolution (Bus, Cyclist, LV, Long Trk, Medium Trk, Taxi, Motorcycle, Short Trk)

| | images | GT boxes | found (any class) | recall (right class) | precision |
|---|---|---|---|---|---|
| 1600x900 | 713 | 5551 | 0.739 | 0.731 | 0.855 |
| 1920x1080 | 809 | 5189 | 0.598 | 0.579 | 0.769 |
| 2560x1440 | 3032 | 3441 | 0.957 | 0.898 | 0.890 |

## Box size at the model input (sqrt area, letterbox scale) (Bus, Cyclist, LV, Long Trk, Medium Trk, Taxi, Motorcycle, Short Trk)

| | images | GT boxes | found (any class) | recall (right class) | precision |
|---|---|---|---|---|---|
| <8 px |  | 1499 | 0.112 | 0.110 | – |
| 8-16 px |  | 2817 | 0.514 | 0.506 | – |
| 16-32 px |  | 3008 | 0.762 | 0.744 | – |
| 32-64 px |  | 3448 | 0.939 | 0.910 | – |
| 64-128 px |  | 2913 | 0.982 | 0.937 | – |
| >=128 px |  | 496 | 0.992 | 0.915 | – |

## Per test camera (Bus, Cyclist, LV, Long Trk, Medium Trk, Taxi, Motorcycle, Short Trk)

| | images | GT boxes | found (any class) | recall (right class) | precision |
|---|---|---|---|---|---|
| scene_079 | 749 | 5459 | 0.754 | 0.744 | 0.854 |
| scene_103 | 456 | 1776 | 0.556 | 0.531 | 0.780 |
| scene_025 | 105 | 1559 | 0.543 | 0.527 | 0.767 |
| scene_166 | 1049 | 1232 | 0.950 | 0.878 | 0.873 |
| scene_193 | 973 | 1060 | 0.992 | 0.946 | 0.910 |
| scene_064 | 48 | 812 | 0.695 | 0.688 | 0.743 |
| scene_014 | 70 | 768 | 0.594 | 0.576 | 0.728 |
| scene_208 | 681 | 767 | 0.930 | 0.880 | 0.901 |
| scene_113 | 131 | 277 | 0.899 | 0.863 | 0.892 |
| scene_059 | 85 | 247 | 0.538 | 0.502 | 0.832 |
| scene_169 | 207 | 224 | 0.942 | 0.884 | 0.888 |
