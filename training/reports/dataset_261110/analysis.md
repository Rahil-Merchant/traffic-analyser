# Dataset analysis: `data/dataset_261110`

Layout: yolo · classes: 11 · object sizes measured after letterboxing to 640 px

## Splits as exported

| split | images | empty | instances | boxes/img mean | median | p95 | max |
|---|---|---|---|---|---|---|---|
| train | 25332 | 1179 | 121504 | 4.8 | 2.0 | 18.0 | 48 |
| valid | 12344 | 166 | 61025 | 4.9 | 2.0 | 17.0 | 59 |

## Instances per class

| class | total | % | images | train | valid |
|---|---|---|---|---|---|
| Animal drawn vehicle | 21 | 0.0 | 20 | 14 | 7 |
| Bus | 1844 | 1.0 | 1707 | 1222 | 622 |
| Cyclist | 231 | 0.1 | 208 | 138 | 93 |
| Light Vehicle | 108039 | 59.2 | 28861 | 73789 | 34250 |
| Long Truck | 10425 | 5.7 | 8713 | 6571 | 3854 |
| Medium Truck | 3366 | 1.8 | 3187 | 2234 | 1132 |
| Minibus Taxi | 26700 | 14.6 | 12892 | 17650 | 9050 |
| Motorcycle | 469 | 0.3 | 446 | 281 | 188 |
| Pedestrian | 23802 | 13.0 | 8844 | 14594 | 9208 |
| Person with wheel barrow | 57 | 0.0 | 50 | 35 | 22 |
| Short Truck | 7575 | 4.2 | 6618 | 4976 | 2599 |

Imbalance ratio (largest / smallest non-empty class): **5144.7**

## Image sizes

| width | height | images |
|---|---|---|
| 2560 | 1440 | 16798 |
| 1600 | 900 | 10795 |
| 1920 | 1080 | 10002 |
| 1632 | 917 | 2 |
| 1645 | 915 | 2 |
| 1594 | 873 | 1 |
| 1662 | 932 | 1 |
| 1677 | 924 | 1 |
| 1687 | 921 | 1 |
| 1663 | 922 | 1 |
| 1794 | 923 | 1 |
| 1656 | 924 | 1 |

## Object sizes at 640 px (sqrt of box area)

p5 5.2 · p25 12.4 · median 23.7 · p75 47.1 · p95 110.3 px

tiny (<16 px): 62192 · small (<32): 112556 · medium (32–96): 56901 · large (≥96): 13072

| class | n | sqrt-area p10 | median | p90 | median w×h | aspect | % small |
|---|---|---|---|---|---|---|---|
| Animal drawn vehicle | 21 | 31.0 | 44.1 | 67.9 | 59×42 | 1.79 | 14 |
| Bus | 1844 | 11.5 | 54.1 | 137.0 | 73×38 | 1.57 | 33 |
| Cyclist | 231 | 16.1 | 29.5 | 55.2 | 26×34 | 0.80 | 55 |
| Light Vehicle | 108039 | 6.1 | 21.0 | 70.1 | 26×17 | 1.45 | 66 |
| Long Truck | 10425 | 18.4 | 82.8 | 183.1 | 122×54 | 1.85 | 22 |
| Medium Truck | 3366 | 15.2 | 66.1 | 146.9 | 86×50 | 1.60 | 25 |
| Minibus Taxi | 26700 | 7.4 | 25.6 | 83.7 | 30×22 | 1.25 | 58 |
| Motorcycle | 469 | 12.7 | 25.6 | 49.7 | 25×27 | 0.87 | 69 |
| Pedestrian | 23802 | 9.8 | 19.6 | 39.6 | 14×28 | 0.46 | 82 |
| Person with wheel barrow | 57 | 12.5 | 32.1 | 63.2 | 27×39 | 0.76 | 49 |
| Short Truck | 7575 | 15.3 | 51.6 | 126.3 | 68×39 | 1.50 | 29 |

## Label problems

images with issues: 0
- none

## Duplicates and recordings

- unique source frames: 37676; frames with several Roboflow copies: 0 (+0 extra images); copies of one frame spread over different splits: **0**
- recordings (groups) recognised from file names: 31443 (singletons 31375, median size 1.0, largest [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]); frames with a parsable frame number: 6301
- recordings present in more than one split of the export: **25** (2083 images)
  - {'group': '0-11236-2_b572fe', 'train': 94, 'valid': 6}
  - {'group': '0-3753-2_94b45c', 'train': 4, 'valid': 1}
  - {'group': '10593_c9c4e5', 'train': 99, 'valid': 1}
  - {'group': '10928_873971', 'train': 97, 'valid': 3}
  - {'group': '11306_9afd83', 'train': 99, 'valid': 1}
  - {'group': '11477_388331', 'train': 96, 'valid': 4}
  - {'group': '11702_ec73f0', 'train': 98, 'valid': 2}
  - {'group': '12145-2_48b4a5', 'train': 63, 'valid': 7}
