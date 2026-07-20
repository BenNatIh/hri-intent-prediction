# Prediction of Intent for Human-Robot Interaction

Final project for **AER1515 (Perception for Robotics)**, University of Toronto, by Ben Natra and Tian Yu. Full writeup: [report.pdf](report.pdf).

## Overview

A robot that can tell whether a nearby person intends to interact with it can decide when to approach, when to stay out of the way, and when to switch a display from idle to a selection menu. We built a two-stream intent prediction pipeline that fuses **body pose** and **gaze direction**, evaluated on real video:

- **Pose stream**: YOLOv11 keypoints → 2s sequence window → LSTM or Transformer classifier → binary "will interact" prediction.
- **Gaze stream**: MediaPipe face landmarks → MLP → pitch/yaw → binary "looking at the robot" prediction.
- The two binary outputs are combined with a simple voting table (`seek interaction` / `receptive` / `avoid`) to produce the robot's recommended action.

Splitting pose and gaze into two independently-trained models let us pull in more training data for each: four public datasets (Yale Shutter Interaction, MINT-RVAE, MPIIGaze, Columbia Gaze) were combined for training, with a fifth (JPL Interaction) used to qualitatively validate the full pipeline on unseen video.

**Demo:** [`media/demo.mp4`](media/demo.mp4), one sequence with three views side by side: raw video, the LSTM's live intent prediction (interacting / not interacting), and the predicted gaze direction.

<video src="media/demo.mp4" controls muted width="640"></video>

## My contributions

This was a two-person project. I built:

- **Gaze dataset combination**: merging MPIIGaze and Columbia Gaze into a single training set for the gaze model.
- **Gaze prediction model and training** (`gaze_mlp.py`): the MediaPipe-landmark → MLP pipeline that regresses pitch/yaw and thresholds it into an "at the robot" signal.
- **Transformer model and training** (`transformer_model.py`): the pose-sequence transformer classifier used as the alternative to the LSTM for intent classification.
- Contributed to the **Yale Shutter + MINT-RVAE dataset combination and coordinate-frame transform** (pose-dataset side), alongside Tian Yu, who took the lead on getting this working and refined it further.

Tian Yu built the YOLOv11 pose extraction/trajectory pipeline, the LSTM baseline, and the final voting/action logic.

## Results

Pose-based intent classifier, compared against the MINT-RVAE reference (sequence-level F1, AUC):

| Source | Architecture | F1 | AUC |
|---|---|---|---|
| MINT-RVAE (reference) | Transformer | 0.899 | 0.951 |
| MINT-RVAE (reference) | LSTM | 0.876 | 0.934 |
| **This project** | **Transformer** | **0.938** | **0.975** |
| This project | LSTM | 0.933 | 0.969 |

Gaze model, mean angular error vs. published baselines:

| Source | Architecture | Mean error (°) |
|---|---|---|
| MPIIGaze (reference) | GazeNet | 5.5 |
| MPIIGaze (reference) | GazeNet+ | 5.4 |
| **This project** | **MediaPipe landmarks + MLP** | **6.2** |

The transformer matched or slightly exceeded both the LSTM and the published MINT-RVAE reference, but this isn't an apples-to-apples architecture comparison: the reference was trained on a single dataset, while ours combined Yale Shutter + MINT-RVAE for ~10x the training sequences, which is the more likely driver of the gap. The label distribution (~64% positive) also likely inflates F1/accuracy somewhat.

**Important caveat:** the numbers above describe the pose classifier and gaze regressor in isolation. The combined pose+gaze fusion pipeline, the thing actually being pitched here, was only evaluated qualitatively on the JPL Interaction video (no ground truth labels available), not benchmarked quantitatively. Qualitatively, fusing gaze with pose did reduce flicker in the pose model's predictions: gaze stayed reliable up close where pose keypoints got noisy, and pose stayed reliable at a distance where MediaPipe's face mesh dropped out, but there's no end-to-end accuracy number to back that up.

## Repo structure

```
transformer_model.py                      # Pose-sequence transformer classifier
gaze_mlp.py                               # Gaze MLP (landmarks -> pitch/yaw)
data_utils.py                             # Dataset windowing/loading utilities
face_landmark.py                          # MediaPipe face landmark extraction
pretrain.py                               # Self-supervised pretraining for the pose encoder
scripts/
  process_all_data.py                     # Unified dataset -> NPZ preprocessing
  train_and_eval.py                       # Training/evaluation harness
  convert_to_npz.py, convert_dataset_approach.py,
  preprocess_tracks.py, relabel_npz_future.py,
  audit_windows.py, inspect_npz.py         # Dataset conversion, labeling, and inspection utilities
  yolo_pose.py                            # YOLOv11 pose extraction wrapper
lstm_pipeline/
  lstm_model.py                           # LSTM intent classifier
  gaze_mlp.py                             # Gaze MLP (pipeline-local copy)
  data_processing.py                      # Sequence windowing for the LSTM/transformer pipeline
  extract_gaze_from_video.py              # Gaze extraction over video
  preprocess_vis_data.py                  # Dataset visualization/preprocessing
  train_model.ipynb                       # Training/evaluation notebook
  video_pipeline.ipynb                    # End-to-end video inference notebook
```

Trained weights, processed dataset caches, and generated visualizations are not checked into this repo (see `.gitignore`); they're either downloaded automatically (YOLOv11/MediaPipe pretrained weights) or reproducible from the scripts above.

## Setup

```
pip install -r requirements.txt
```

Also requires `ultralytics`, `mediapipe`, and `scikit-learn` for pose extraction, face landmarks, and evaluation metrics (not pinned in `requirements.txt`). YOLOv11 pose weights and the MediaPipe face landmarker model are downloaded automatically on first run.

Datasets (Yale Shutter Interaction, MINT-RVAE, MPIIGaze, Columbia Gaze, JPL Interaction) are third-party and not redistributed here; see `report.pdf` §III-C for sources. Once obtained, preprocess with `scripts/process_all_data.py` and train with `scripts/train_and_eval.py` (see script `--help` for options).

## Limitations & next steps

- Full-pipeline evaluation was qualitative only (no ground truth on the JPL validation set); labeling a held-out video set would allow quantitative end-to-end evaluation.
- YOLOv11's facing-direction estimate is occasionally unreliable (confuses front/back), which can be mitigated by falling back to the gaze model's facing estimate when available.
- A natural extension is predicting specific actions (waving, pointing) using the same pose/face-mesh features, rather than a single binary intent signal.

---
Cloned and cleaned from the original collaborative repo for portfolio purposes; original at [github.com/TianmYu/aer1515](https://github.com/TianmYu/aer1515) (`model-lstm` branch).
