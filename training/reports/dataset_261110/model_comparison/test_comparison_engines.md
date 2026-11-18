# Test-split comparison (4554 images, 16021 boxes)

COCO metrics from pycocotools (maxDets 100); operating point = score >= 0.25, IoU >= 0.5.

| model | mAP50-95 | mAP50 | mAP75 | AP small | AR100 | P@op | R@op | ms/img (model) |
|---|---|---|---|---|---|---|---|---|
| deimv2-cctv-torch | 0.503 | 0.763 | 0.548 | 0.095 | 0.677 | 0.542 | 0.860 | 14.26 |
| deimv2-cctv-trt | 0.504 | 0.763 | 0.549 | 0.095 | 0.677 | 0.542 | 0.860 | 6.99 |
| yolo-sq640-torch | 0.474 | 0.699 | 0.525 | 0.069 | 0.576 | 0.717 | 0.761 | 9.02 |
| yolo-sq640-trt | 0.473 | 0.699 | 0.526 | 0.067 | 0.576 | 0.717 | 0.762 | 7.58 |

## Each model at its own best-F1 threshold

Score scales differ between model families (DETR scores run lower), so a shared threshold favours one of
them; this is the fairer operating-point comparison. The threshold is the one to give the tracker.

| model | threshold | precision | recall | F1 | R Bus | R Cyclist | R Light Vehicle | R Long Truck | R Medium Truck | R Minibus Taxi | R Motorcycle | R Pedestrian | R Short Truck |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deimv2-cctv-torch | 0.42 | 0.803 | 0.730 | 0.765 | 0.64 | 0.73 | 0.79 | 0.86 | 0.69 | 0.72 | 0.82 | 0.40 | 0.61 |
| deimv2-cctv-trt | 0.42 | 0.804 | 0.731 | 0.766 | 0.64 | 0.73 | 0.79 | 0.86 | 0.69 | 0.72 | 0.82 | 0.40 | 0.61 |
| yolo-sq640-torch | 0.30 | 0.785 | 0.718 | 0.750 | 0.63 | 0.69 | 0.78 | 0.86 | 0.66 | 0.73 | 0.79 | 0.34 | 0.59 |
| yolo-sq640-trt | 0.30 | 0.786 | 0.719 | 0.751 | 0.63 | 0.69 | 0.78 | 0.86 | 0.66 | 0.73 | 0.79 | 0.34 | 0.59 |

## AP50-95 per class

| class | GT boxes | deimv2-cctv-torch | deimv2-cctv-trt | yolo-sq640-torch | yolo-sq640-trt |
|---|---|---|---|---|---|
| Bus | 81 | 0.580 | 0.582 | 0.498 | 0.492 |
| Cyclist | 26 | 0.501 | 0.501 | 0.456 | 0.455 |
| Light Vehicle | 10985 | 0.526 | 0.526 | 0.520 | 0.519 |
| Long Truck | 511 | 0.656 | 0.657 | 0.627 | 0.626 |
| Medium Truck | 248 | 0.487 | 0.486 | 0.418 | 0.419 |
| Minibus Taxi | 1696 | 0.475 | 0.475 | 0.483 | 0.483 |
| Motorcycle | 39 | 0.585 | 0.585 | 0.598 | 0.595 |
| Pedestrian | 1840 | 0.260 | 0.260 | 0.251 | 0.250 |
| Short Truck | 595 | 0.461 | 0.461 | 0.415 | 0.416 |

## Precision / recall per class at score >= 0.25

| class | deimv2-cctv-torch P | deimv2-cctv-torch R | deimv2-cctv-trt P | deimv2-cctv-trt R | yolo-sq640-torch P | yolo-sq640-torch R | yolo-sq640-trt P | yolo-sq640-trt R |
|---|---|---|---|---|---|---|---|---|
| Bus | 0.693 | 0.753 | 0.693 | 0.753 | 0.927 | 0.630 | 0.944 | 0.630 |
| Cyclist | 0.583 | 0.808 | 0.583 | 0.808 | 0.643 | 0.692 | 0.643 | 0.692 |
| Light Vehicle | 0.573 | 0.895 | 0.574 | 0.895 | 0.724 | 0.830 | 0.723 | 0.831 |
| Long Truck | 0.597 | 0.939 | 0.596 | 0.939 | 0.826 | 0.867 | 0.826 | 0.865 |
| Medium Truck | 0.422 | 0.879 | 0.420 | 0.879 | 0.631 | 0.681 | 0.631 | 0.681 |
| Minibus Taxi | 0.485 | 0.841 | 0.484 | 0.841 | 0.677 | 0.783 | 0.680 | 0.784 |
| Motorcycle | 0.654 | 0.872 | 0.642 | 0.872 | 0.711 | 0.821 | 0.711 | 0.821 |
| Pedestrian | 0.434 | 0.686 | 0.433 | 0.687 | 0.681 | 0.365 | 0.682 | 0.367 |
| Short Truck | 0.483 | 0.745 | 0.485 | 0.746 | 0.689 | 0.607 | 0.692 | 0.607 |
