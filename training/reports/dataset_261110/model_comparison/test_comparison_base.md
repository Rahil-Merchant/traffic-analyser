# Test-split comparison (4554 images, 16021 boxes)

COCO metrics from pycocotools (maxDets 100); operating point = score >= 0.25, IoU >= 0.5.

| model | mAP50-95 | mAP50 | mAP75 | AP small | AR100 | P@op | R@op | ms/img (model) |
|---|---|---|---|---|---|---|---|---|
| yolo26l | 0.457 | 0.678 | 0.504 | 0.051 | 0.553 | 0.726 | 0.739 | 20.45 |
| rfdetr-m | 0.479 | 0.733 | 0.533 | 0.042 | 0.644 | 0.655 | 0.782 | 16.84 |
| deimv2-l | 0.499 | 0.754 | 0.558 | 0.076 | 0.672 | 0.534 | 0.856 | 14.14 |

## Each model at its own best-F1 threshold

Score scales differ between model families (DETR scores run lower), so a shared threshold favours one of
them; this is the fairer operating-point comparison. The threshold is the one to give the tracker.

| model | threshold | precision | recall | F1 | R Bus | R Cyclist | R Light Vehicle | R Long Truck | R Medium Truck | R Minibus Taxi | R Motorcycle | R Pedestrian | R Short Truck |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| yolo26l | 0.34 | 0.818 | 0.681 | 0.743 | 0.59 | 0.69 | 0.74 | 0.85 | 0.64 | 0.67 | 0.77 | 0.32 | 0.57 |
| rfdetr-m | 0.37 | 0.789 | 0.710 | 0.748 | 0.59 | 0.69 | 0.78 | 0.87 | 0.63 | 0.66 | 0.72 | 0.38 | 0.58 |
| deimv2-l | 0.41 | 0.804 | 0.727 | 0.763 | 0.64 | 0.81 | 0.79 | 0.86 | 0.68 | 0.70 | 0.77 | 0.42 | 0.59 |

## AP50-95 per class

| class | GT boxes | yolo26l | rfdetr-m | deimv2-l |
|---|---|---|---|---|
| Bus | 81 | 0.482 | 0.533 | 0.571 |
| Cyclist | 26 | 0.453 | 0.500 | 0.509 |
| Light Vehicle | 10985 | 0.506 | 0.488 | 0.516 |
| Long Truck | 511 | 0.620 | 0.645 | 0.651 |
| Medium Truck | 248 | 0.401 | 0.464 | 0.465 |
| Minibus Taxi | 1696 | 0.458 | 0.440 | 0.473 |
| Motorcycle | 39 | 0.560 | 0.551 | 0.595 |
| Pedestrian | 1840 | 0.239 | 0.243 | 0.262 |
| Short Truck | 595 | 0.394 | 0.442 | 0.450 |

## Precision / recall per class at score >= 0.25

| class | yolo26l P | yolo26l R | rfdetr-m P | rfdetr-m R | deimv2-l P | deimv2-l R |
|---|---|---|---|---|---|---|
| Bus | 0.889 | 0.593 | 0.841 | 0.654 | 0.803 | 0.704 |
| Cyclist | 0.562 | 0.692 | 0.760 | 0.731 | 0.600 | 0.808 |
| Light Vehicle | 0.733 | 0.806 | 0.688 | 0.823 | 0.563 | 0.892 |
| Long Truck | 0.825 | 0.875 | 0.682 | 0.900 | 0.575 | 0.926 |
| Medium Truck | 0.634 | 0.649 | 0.470 | 0.706 | 0.369 | 0.851 |
| Minibus Taxi | 0.721 | 0.732 | 0.650 | 0.748 | 0.509 | 0.833 |
| Motorcycle | 0.732 | 0.769 | 0.682 | 0.769 | 0.647 | 0.846 |
| Pedestrian | 0.639 | 0.373 | 0.483 | 0.584 | 0.423 | 0.699 |
| Short Truck | 0.679 | 0.587 | 0.630 | 0.691 | 0.477 | 0.724 |
