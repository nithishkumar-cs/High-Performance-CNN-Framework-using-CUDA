# Engineering log (resume / interview reference)

Chronological record of what was built, measured and debugged in this project, derived from
the git history and `DESIGN.yaml`. Use this to recall talking points for a resume bullet or an
interview; it is not re-derived automatically, so re-run `git log` / check `DESIGN.yaml` for
anything after the date below.

Generated: 2026-10-22.

## Project in one line

Independently written (not forked) C++/CUDA neural-network framework, inspired by Darknet,
supporting configurable training/inference for classifiers and YOLO-style detectors, with
native CUDA + TensorRT deployment on a Jetson Orin Nano.

## Milestones

**2025-07-20 — Starting point.** Original fixed-shape MNIST CUDA CNN (hand-written CUDA kernels,
no framework) kept as project history.

**2026-09-22/23 — Early exploration.** Added PyTorch/ONNX YOLO inference and a Jetson benchmark
tool; trained a VOC2007 detector; later reverted (see "Scope correction" below) once the project
goal was clarified as a from-scratch C++/CUDA framework, not a PyTorch wrapper.

**2026-09-26 — Independent tiny-YOLO detector vs Jetson.** Built and compared an independent
YOLO inference path.

**2026-10-01 — Scope correction + framework foundations.**
- Reverted the PyTorch/ONNX tooling; restored the original CUDA CNN as the historical baseline.
- Added shortcut (residual-add) layers to the CPU/CUDA graph — the layer required to parse full
  YOLOv3/Darknet-53-style configs.
- Added trainable batch normalization, branched-graph backprop/minibatching, a burn-in +
  step-decay LR schedule, and flip/exposure augmentation to the native CPU YOLO trainer.

**2026-10-05 — Native CUDA training backend; first real-scale retrain; near-zero mAP.**
- Committed the full native C++/CUDA detector/classifier framework and added a CUDA/GPU training
  backend (`CudaTrainer`) mirroring the CPU trainer, with a CPU/GPU parity test harness
  (matched-seed CPU and GPU runs must agree on every weight to ~1e-6).
  - **Bug found via parity testing:** `batch_norm_apply`/`batch_norm_backward_dz` CUDA kernels
    were missing a thread-bounds check, so overhang threads from the rounded-up launch grid wrote
    past buffer ends and silently corrupted unrelated device buffers (one weight diverged to ~0
    after a single step). Fixed by adding the missing bounds check.
- First full-scale GPU retrain of YOLOv3-tiny on VOC2007 converged training loss cleanly but left
  validation mAP@0.5 at ~0.002 — essentially a non-functional detector.
- Added `native/tests/overfit_smoke.cpp`: fit trivial synthetic examples for thousands of steps to
  prove the anchor-assignment/loss/decode round trip *can* learn correct, confident boxes from
  scratch, ruling out a hard correctness bug as the cause of the near-zero mAP.
- Added `ignore_thresh` (Darknet's near-miss suppression) and full crop/HSV augmentation
  (exposure/saturation/hue/crop jitter) — re-ran the same retrain: mAP@0.5 barely moved
  (0.00118), a second piece of evidence against "missing fidelity" as the dominant cause.

**2026-10-06 — True minibatch batch-norm statistics.**
- Restructured both `Trainer::step` and `CudaTrainer::step` to take a genuine minibatch (not
  one sample with cross-call accumulation), computing BN mean/variance across the whole batch —
  matching Darknet's batched statistics. Verified with a new finite-difference gradient check and
  a CPU/CUDA parity case; re-ran the retrain: mAP@0.5 still near zero (0.00028) — third piece of
  evidence that the architecture/loss machinery was not the bottleneck; the real gap was training
  budget and/or a (soon-to-be-found) loss-weighting bug.

**2026-10-07 — Pretrained-backbone initialization.**
- Implemented Darknet `partial N` weight loading semantics (`--init-layers`) so a differently
  shaped source file (official COCO `yolov3-tiny.weights`) can seed a matching-shape backbone
  prefix. Launched the first full-budget (71-epoch, 320,210-image) retrain from a COCO-pretrained
  backbone.

**2026-10-08 — Decisive fix: Darknet loss-fidelity bugs.**
- Pretrained-backbone checkpoint scored mAP@0.5 = 0.25712 — moved off the near-zero floor but far
  below the historical external-Darknet baseline (0.57612) on identical evaluator/data.
- **Root-cause audit found three loss bugs relative to original Darknet's `yolo_layer.c`:**
  1. Background/non-matched objectness gradient was scaled by `0.5 / (num_anchors*cells)`
     (~0.000197 per candidate) instead of Darknet's unscaled unit delta.
  2. Matched-box coordinate loss was missing Darknet's `(2 - truth.w*truth.h)` box-size scale
     (which up-weights small objects).
  3. An unintended `loss /= num_targets` per-image normalization was being applied; Darknet sums
     every target's full-scale delta with no such division.
  4. Added SGD momentum (default 0.9, matching Darknet's cfg) — previously plain SGD only.
- Fixed all four in the shared `yolo_loss.cpp` (used by both CPU and CUDA trainers), added
  regression tests, rescaled three momentum-sensitive test learning rates (momentum's
  ~1/(1-momentum) steady-state step amplification had destabilized them), and launched a fresh
  full-budget retrain with the fix.

**2026-10-09/11 — Payoff: mAP jumps above the external baseline.**
- Corrected full-budget retrain finished; evaluated on 501 validation images:
  **mAP@0.5 = 0.59801**, vs. 0.25712 before the fix and 0.57612 for the historical
  external-Darknet checkpoint on identical data/evaluator — now *above* the external baseline,
  with 3,380 total predictions vs. 14.2M before (the unscaled-background fix suppressed a flood
  of spurious detections). This is the single highest-leverage fix in the project: four
  Darknet-fidelity corrections took native training from "not useful" to "better than the
  external reference implementation."

**2026-10-12 — Test-set confirmation + Jetson deployment.**
- Same checkpoint on the held-out VOC2007 test split (4,952 images, 12,032 objects):
  mAP@0.5 = 0.58182, vs. 0.55534 for the historical external baseline — confirms generalization,
  not validation-set overfitting.
- Closed the last named Darknet fidelity gap: Darknet-style multiscale/resize-jitter training
  (`Network::resize`, `CudaTrainer` device-buffer reallocation that preserves momentum/BN state
  across a resize, `--multiscale-min/-max/-interval` CLI flags).
- Implemented dropout and reorg (space-to-depth) layers for CPU/CUDA inference and training.
- Deployed the corrected checkpoint to the Jetson Orin Nano: exported ONNX, built FP32/FP16
  TensorRT engines, and confirmed native CUDA / TensorRT FP32 / TensorRT FP16 all land within
  noise of the laptop numbers (~0.60 val / ~0.58 test mAP@0.5) while TensorRT FP16 cuts per-image
  pipeline latency from ~100 ms (native CUDA) to ~12 ms.

**2026-10-13 — Second model architecture: YOLOv2/Darknet19.**
- Implemented Darknet's `[region]` layer (YOLOv2's detection head: softmax class loss, rescore,
  object/noobject/class/coord scale knobs), reusing the shared yolo/region anchor-assignment code.
- Assembling a real YOLOv2 cfg surfaced a second real bug: the route-layer shape validator
  compared each concatenated branch's spatial shape against the *preceding sequential layer*
  instead of against the other branches — this happened to work for every route pattern used so
  far but rejected Darknet's standard standalone passthrough route (`route layers=-9`, jumping
  back to an earlier, differently-shaped layer). Fixed by validating branches against each other.
- Fetched and verified a pretrained ImageNet `darknet19_448.conv.23` backbone (from a community
  mirror after pjreddie's own file server returned 404) to seed training.
- Found batch size 32 (used for the first architecture) VRAM-overflows on this architecture at
  >3s/image and climbing (Windows shared-memory fallback); batch 16 is stable at ~0.62s/image.
  Launched the YOLOv2 full-budget retrain (71 epochs, 320,210 images, batch 16).

**2026-10-14 — Resume support, needed in practice almost immediately.**
- Added real step/epoch-resume (`--resume-from`, a `.step` sidecar file) so a crashed/killed run
  continues its LR schedule and epoch count instead of restarting burn-in at iteration 0 against
  already-trained weights — exactly the problem that had already bitten the in-flight retrain via
  an OS crash/update restart.

**2026-10-15/21 — Multiple silent interruptions; monitoring.**
- The YOLOv2 retrain was interrupted repeatedly (an OS crash + Windows Update restart, a user stop
  to free the GPU, an unexplained silent exit) and resumed each time using the new resume support.
- Progress checks tracked a worsening per-epoch rate (up to ~12.4h/epoch), motivating a
  performance investigation.

**2026-10-21 — CPU-heating bug in the CUDA training loop.**
- Found and fixed a host-thread busy-wait spin in the CUDA training loop that was pegging the CPU
  and worsening GPU thermal throttling during the retrain (a classic "the CPU side is supposed to
  be idle while the GPU works" bug — a blocking/synchronizing wait had been implemented as a
  spin-loop).

**2026-10-22 — GPU kernel-level performance optimization.**
- Nsight Systems profiling pinpointed the real bottleneck: direct nested-loop convolution kernels
  (75.7% of GPU kernel time) and serial per-channel batch-norm reductions (23.7%).
- Rewrote convolutional forward/backward to use im2col/col2im + cuBLAS SGEMM, and batch-norm
  mean/variance/backward reductions to tiled double-precision partial sums.
- Measured on a same-seed 160-image pilot on an unchanged checkpoint: **86.58s → 9.93s**
  (≈8.7x), with identical loss/positive counts and checkpoint floats matching the unoptimized
  path to within 7.63e-6 absolute / no difference above 1e-3 across 67M floats — a performance
  rewrite verified numerically equivalent, not just "looks right."
- Cut the retrain's measured epoch time from ~12.4h to ~4.46h with the earlier busy-wait fix, then
  further with this kernel rewrite — observed ~3-4 minutes/epoch on the resumed run.

**2026-10-25 — A second useful native detector architecture.**
- The full-budget YOLOv2/Darknet19 retrain completed and scored 0.56054 validation / 0.57215
  test mAP@0.5 on VOC2007 with this project's CUDA evaluator. This confirmed that the framework
  trains a second, structurally different detector rather than only YOLOv3-tiny.

**2026-10-26 — Full YOLOv3/Darknet-53 training and validation.**
- Parsed and trained the real 106-layer, three-head YOLOv3 graph using its ImageNet-pretrained
  Darknet-53 prefix. An initial learning rate of 0.001 diverged in epoch 2; 0.0001 completed all
  71 epochs at batch 8 on the 8 GB laptop GPU.
- The resulting native checkpoint scored **0.62098 validation / 0.61593 test mAP@0.5**, the best
  of this project's three completed detector architectures. The full validation result was also
  reproduced with the new cuDNN inference path (501 images, 1,305 objects, 2,853 predictions).

**2026-10-26 — cuDNN integration measured against the existing cuBLAS path.**
- Staged cuDNN from the no-account pip wheel and generated the missing Windows import library
  from the shipped DLL exports, making the build reproducible without a separate SDK download.
- Wired cached cuDNN convolution forward/backward and spatial batch normalization into optional
  CUDA inference and training. Forward/backward direct-sum checks and CPU/CUDA minibatch parity,
  including different samples and resize, passed.
- A same-seed YOLOv2 320-image batch-8 pilot took 11.4s with batched cuDNN versus 12.8s with
  cuBLAS at LR 0.0001. This is one short throughput comparison; the default remains cuBLAS while
  cuDNN is selected by `DETECTOR_CUDNN=ON`. The loss trajectories were close but not identical.

**2026-10-26 — RNN/LSTM/GRU forward and backpropagation through time.**
- Implemented all three recurrent cell kinds independently on CPU and CUDA. CPU finite-difference
  checks cover gate parameters; the CUDA test compares every timestep output, input gradient and
  parameter gradient with the CPU path.

**2026-10-27 — RNN/LSTM/GRU cfg integration completed on CPU and CUDA.**
- All three kinds run as cfg layers with one timestep per `classify` call and explicit sequence
  reset. CPU and CUDA inference carry hidden and LSTM cell state across calls.
- CPU and CUDA classifier trainers support truncated BPTT chunks; convolution, maxpool and dense
  layers around the recurrent layer stay in the CUDA graph and receive one averaged SGD update
  per chunk. CUDA recurrent parameter updates match the CPU trainer's unaveraged BPTT rule.
- CPU/CUDA parity covers stateful inference, two consecutive training chunks, every learnable
  parameter, and a convolution/maxpool prefix. A parity test found and fixed a maxpool path that
  incorrectly fell through to a weight update after backpropagation.

**2026-10-27 — CUDA detector data parallelism.**
- Added `--gpus 0,1` to run independent detector replicas on selected GPUs with balanced shards
  of the total `--batch`. Each global step synchronizes learned weights, momentum and batch norm
  running state, weighted by the actual samples on each device. Smaller final batches can leave a
  replica idle without losing its step count or optimizer state.
- A one-GPU test uses two logical CUDA trainers to compare a 1+2 shard split against an unsplit
  three-sample minibatch through multiple momentum steps, then an idle-replica partial batch.
  Separate checks cover batch norm state, explicit `--gpus 0` CLI training, and duplicate-device
  rejection. This laptop has only one GPU; actual two-device throughput and behavior remain to be
  measured on a multi-GPU host. Synchronization currently gathers tensors through host memory.

**2026-10-27 — Deeper CNN classifier trained end to end on the laptop GPU.**
- Added a 28×28 MNIST cfg with two convolution/maxpool stages and a 64-unit hidden dense layer.
  The native CUDA classifier trainer completed three fixed epochs over all 60,000 training images
  (seed 2026, LR 0.01, sample-wise SGD). Mean loss fell 0.131075 → 0.053979 → 0.038460.
- Its ignored checkpoint scored 9,849/10,000 = **98.49%** on the held-out test split with both
  native CUDA and CPU inference. This is above the earlier four-filter, one-conv classifier's
  97.27% test result. Laptop evaluation mean times were 0.131536 ms/sample CUDA and
  0.604649 ms/sample CPU for IDX sample conversion plus forward, excluding file I/O and setup;
  these are not Jetson timings.
- Checkpoint: `runs/mnist/deep-cnn-full-20261027.weights`, SHA-256
  `aeca7f625a307cb9012e4821814c71c818cfa74c0a1f7234d204be8bd049920`. Training and
  evaluation output is under `.agent-work/logs/deep_cnn_*`.

## Bugs found and fixed (debugging skills, concrete)

1. **CUDA buffer overrun** — missing thread-bounds check in batch-norm kernels corrupted
   unrelated device memory; caught via a cross-platform (CPU/GPU) numerical parity test, not by
   a crash.
2. **Loss-scaling bug (x3, found via root-cause audit against the reference implementation's
   source)** — unscaled-vs-Darknet background objectness weight, missing box-size loss scale,
   spurious per-target loss normalization. Fixing all three (plus adding missing SGD momentum)
   took validation mAP@0.5 from 0.257 to 0.598, surpassing the external reference baseline.
3. **Route-layer shape-validation bug** — validated against the wrong reference shape; invisible
   until a config exercised Darknet's standard passthrough route pattern.
4. **CPU busy-wait spin** in the CUDA training host loop — silent performance/thermal bug with no
   functional symptom, found by observing CPU utilization during a long-running job.
5. **GPU kernel hotspot** — direct convolution and serial BN reductions, found by profiling
   (Nsight Systems) rather than guessing; fixed with cuBLAS SGEMM + tiled parallel reductions,
   verified numerically equivalent before being trusted on a multi-day run.
6. **Deep-head training divergence** — the first full YOLOv3/Darknet-53 run reached non-finite
   loss in epoch 2 at LR 0.001 with momentum 0.9. A controlled tenfold LR reduction completed
   the 71-epoch run and yielded the strongest detector checkpoint.

## Methodology/skills demonstrated

- Root-cause isolation of a "trains but doesn't work" ML bug by elimination: correctness proofs
  (overfit tests) ruling out the loss/encode/decode math, then a systematic audit against the
  reference implementation's source to find the actual numerical discrepancies.
- CPU/GPU numerical parity testing as a correctness oracle for CUDA kernel work (tight relative
  tolerance across every learned parameter).
- Profiling-driven GPU performance work (Nsight Systems) over guesswork; cuBLAS SGEMM via
  im2col/col2im; verifying an optimization is numerically equivalent (not just "faster") before
  trusting it on a long unattended run.
- Operating a multi-day unattended training job through real-world failures (OS crash, Windows
  Update restart, process interruption) by building proper checkpoint/resume support rather than
  re-running from scratch each time.
- Faithful reproduction of a published reference system (Darknet/YOLO) from its config format and
  source semantics, without copying its source, including reproducing subtle loss-function
  details that materially change accuracy.
- GPU backend changes validated at several scales: independent convolution sums, CPU/CUDA
  parameter parity, a real 320-image throughput pilot, and same-evaluator full validation mAP.
