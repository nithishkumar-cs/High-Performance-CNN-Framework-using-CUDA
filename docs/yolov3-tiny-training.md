# YOLOv3-tiny training on VOC2007

This page records the historical baseline trained with the original [pjreddie/darknet](https://github.com/pjreddie/darknet) GPU trainer. The independently written C++/CUDA framework also trains YOLOv3-tiny on the laptop GPU. Its 71-epoch run initialized from the official COCO backbone reached 0.25712 VOC2007 validation mAP@0.5, compared with 0.57612 for the external-Darknet checkpoint on the same 501 images and native evaluator. The native trainer still differs in optimizer, augmentation, and parts of the YOLO loss. A later correction to background objectness weighting has passed smoke tests but has not yet had a full retrain; short fine-tuning of the older checkpoint with the new weighting reduced validation mAP, so the correction's effect on training from the start remains unmeasured.

## Provenance and attribution

The external trainer was the unmodified Darknet source at commit [`f6afaabcdf85f77e7aff2ec55c020c0e297c77f9`](https://github.com/pjreddie/darknet/commit/f6afaabcdf85f77e7aff2ec55c020c0e297c77f9). `native/tools/prepare_darknet_voc.py` reads the upstream [`cfg/yolov3-tiny.cfg`](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-tiny.cfg) from outside this repository and writes a 20-class adaptation outside Git with an attribution header. Consult the upstream [LICENSE](https://github.com/pjreddie/darknet/blob/master/LICENSE) and [MIT notice](https://github.com/pjreddie/darknet/blob/master/LICENSE.mit) before redistributing the generated config or upstream files. No Darknet source or original config is vendored here.

The official COCO `yolov3-tiny.weights` used for initialization had SHA-256 `dccea06f59b781ec1234ddf8d1e94b9519a97f4245748a7d4db75d5b7080a42c`. The adapted 20-class config has SHA-256 `fc1c950433a6db1354ec1d4bfa81ba6d86a4b6aac19d607cc249dfc7e525eb4b`. The selected final checkpoint has SHA-256 `839fe2abd9a0e113104ff5acb3ff750f953d849ca3b0a8782536d330466a5c35` and its Darknet header records 320,000 images seen. The config, final weights, four intermediate validation checkpoints (batches 2,000/5,000/8,000/9,000), class names, and logs are in the locally ignored `runs/yolov3-tiny-voc/` directory; the Jetson run used `/tmp/darknet-validation.koQJO48n/voc-tiny-run/`, which is temporary and must not be treated as durable storage.

## Data and training

The source is VOC2007. Its trainval IDs were shuffled with seed 42 and divided into 4,510 train and 501 validation images; the separate 4,952-image VOC2007 test split was held out until checkpoint selection. Train, validation, and test image names were checked for overlap and none was found. The YOLO label conversion excluded objects marked `difficult` (5,998 exclusions across all splits). There were 1,305 non-difficult validation objects and 12,032 non-difficult test objects. This run uses less training data than the [Darknet VOC example](https://pjreddie.com/darknet/yolo/), which combines VOC2007 and VOC2012 train/validation data.

Recreate the converted labels from a new/empty output directory using the project's own converter (the source is the extracted official `VOCdevkit/VOC2007` directory):

```sh
python3 native/tools/convert_voc2007.py \
  --source /data/VOCdevkit/VOC2007 --output /data/voc2007 \
  --seed 42 --val-fraction 0.10
```

This produces `images/{train,val,test}`, matching `labels/{train,val,test}`, class names, and `preparation.json`. It refuses to overwrite a nonempty output directory.

For an already-converted dataset, add `--verify-existing` to the same command. This compares **every** image/label ID and label contents against the source XML without writes; it passed on all 9,963 images in this run. The preparation script checks each train/validation image-label pair and bounding box, and writes absolute-path lists, class names, a Darknet `.data` file, and two configs. The full config retains the original tiny layers and anchors, changes both heads to 20 classes and the preceding convolutions to 75 filters, and uses fixed 416x416 input, batch 32, subdivisions 16, burn-in 1,000, `max_batches=10000`, learning-rate steps at 8,000/9,000, and the upstream color/jitter augmentation. Fixed size disables multiscale resizing to keep Jetson memory predictable.

The original trainer was built on Jetson Orin with `make -j4 GPU=1 CUDNN=0 NVCC=/usr/local/cuda/bin/nvcc ARCH='-gencode arch=compute_87,code=sm_87'`. Its cuDNN path did not compile against the installed cuDNN 9 because it calls removed APIs; the successful run used the GPU path without cuDNN, with no upstream source changes. The board's existing 25W power mode was not changed.

```sh
python3 native/tools/prepare_darknet_voc.py \
  --upstream-cfg /external/darknet/cfg/yolov3-tiny.cfg \
  --train-images /data/voc2007/images/train --train-labels /data/voc2007/labels/train \
  --val-images /data/voc2007/images/val --val-labels /data/voc2007/labels/val \
  --output-dir /work/voc-tiny-run

/external/darknet/darknet partial /external/darknet/cfg/yolov3-tiny.cfg \
  /external/yolov3-tiny.weights /work/voc-tiny-run/yolov3-tiny.conv.15 15

/external/darknet/darknet detector train \
  /work/voc-tiny-run/voc.data /work/voc-tiny-run/yolov3-tiny-voc-preflight.cfg \
  /work/voc-tiny-run/yolov3-tiny.conv.15

/external/darknet/darknet detector train \
  /work/voc-tiny-run/voc.data /work/voc-tiny-run/yolov3-tiny-voc.cfg \
  /work/voc-tiny-run/backup/yolov3-tiny-voc-preflight_final.weights
```

The `partial` step transfers the first 15 compatible convolutional layers from the official 80-class model; the new 20-class heads initialize separately. The four-batch preflight produced the resumable start checkpoint. The full run finished at batch 10,000, representing 320,000 sampled training images (about 71 dataset-size equivalents, **not** 71 exact epochs because Darknet samples with replacement).

## Checkpoint selection and held-out evaluation

The native evaluator uses confidence 0.01, class-aware NMS at IoU 0.45, one-to-one matching at IoU 0.5, and all-point interpolated AP; mAP is the mean over classes present in the ground truth. All 20 VOC classes were present in validation and test. Exact milestone snapshots were taken by `native/tools/monitor_darknet_run.py`, which verifies the Darknet header's image count before copying a checkpoint. The best of the measured validation checkpoints was the final one:

| Batch | Validation images | Validation mAP@0.5 |
| ---: | ---: | ---: |
| 400 | 501 | 0.00372 |
| 900 | 501 | 0.16805 |
| 2,000 | 501 | 0.47951 |
| 5,000 | 501 | 0.52579 |
| 8,000 | 501 | 0.54391 |
| 9,000 | 501 | 0.57218 |
| 10,000 (selected) | 501 | **0.57379** |

The selected final model was evaluated once on all 4,952 untouched test images: **mAP@0.5 = 0.55534**, with 12,032 labeled objects and 29,863 predictions at confidence 0.01. See `runs/yolov3-tiny-voc/test-final.log` for per-class AP and the full training/validation logs in that directory. The test split was not used to choose the checkpoint. The same final weights loaded in the native Windows CUDA build and produced a `person` detection at score 0.9967 on a held-out VOC2007 test image. A batch-900 qualitative check on one validation image had earlier shown similar visible `person` scores in original Darknet (81%, 28%) and native CUDA (0.8127, 0.2788); that is not raw-tensor parity proof.

```sh
./build/cuda_detector_eval /work/voc-tiny-run/yolov3-tiny-voc.cfg \
  /work/voc-tiny-run/backup/yolov3-tiny-voc_final.weights \
  /data/voc2007/images/test /data/voc2007/labels/test \
  --backend cuda --confidence 0.01
```

This is **not** the official VOC metric: the converted labels omit `difficult` objects, but this evaluator has no ignore-region logic, so detections on those objects may count as false positives. It also uses all-point AP rather than VOC2007's 11-point convention. It must not be presented as COCO mAP@[.5:.95] or compared directly with published VOC numbers.

On Jetson Orin at its existing 25W setting, the final model's 416x416 native CUDA benchmark used five warmups and 30 timed iterations on the same PPM image as the earlier inference baseline. Mean network-forward time was **91.204 ms** (p50 91.154, p95 91.428); the mean in-memory pipeline time was **100.786 ms** (p50 100.918, p95 101.076). The pipeline includes letterbox, forward, decode, NMS, and inverse letterbox; both timings exclude image file I/O, config/weight loading, and CUDA session creation. Do not infer an end-to-end camera frame rate or a speedup over original Darknet from these timings.
