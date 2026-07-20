# Prediction of Intent for Human-Robot Interaction

Final project for **AER1515 (Perception for Robotics)**, University of Toronto, by Ben Natra and Tian Yu. Full writeup: [report.pdf](report.pdf).

## Overview

A robot that can tell whether a nearby person intends to interact with it can decide when to approach, when to stay out of the way, and when to switch a display from idle to a selection menu. We built a two-stream intent prediction pipeline that fuses **body pose** and **gaze direction**, evaluated on real video:

- **Pose stream**: YOLOv11 keypoints → 2s sequence window → LSTM or Transformer classifier → binary "will interact" prediction.
- **Gaze stream**: MediaPipe face landmarks → MLP → pitch/yaw → binary "looking at the robot" prediction.
- The two binary outputs are combined with a simple voting table (`seek interaction` / `receptive` / `avoid`) to produce the robot's recommended action.

Splitting pose and gaze into two independently-trained models let us pull in more training data for each: four public datasets (Yale Shutter Interaction, MINT-RVAE, MPIIGaze, Columbia Gaze) were combined for training, with a fifth (JPL Interaction) used to qualitatively validate the full pipeline on unseen video.

## My contributions

This was a two-person project. I built:

- **Gaze dataset combination** — merging MPIIGaze and Columbia Gaze into a single training set for the gaze model.
- **Gaze prediction model and training** — the MediaPipe-landmark → MLP pipeline that regresses pitch/yaw and thresholds it into an "at the robot" signal.
- **Transformer model and training** — the temporal transformer (encoding + downsampling + sinusoidal positional encoding + learned CLS token + 6-head/3-layer encoder) used as the alternative to the LSTM for pose-based intent classification.

Tian Yu built the YOLOv11 pose extraction/trajectory pipeline, the LSTM baseline, the pose-dataset combination (Yale Shutter + MINT-RVAE) and coordinate-frame transforms, and the final voting/action logic.

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

The transformer outperformed both the LSTM and the published reference despite a smaller feature set, largely due to combining Yale Shutter + MINT-RVAE for ~10x the training sequences. Qualitatively, fusing gaze with pose reduced flicker in the pose model's predictions — gaze stayed reliable up close where pose keypoints got noisy, and pose stayed reliable at a distance where MediaPipe's face mesh dropped out.

## Repo structure

```
face_landmark.py                          # MediaPipe face landmark extraction
yolo_pose.py                              # YOLOv11 pose extraction wrapper
lstm_pipeline/
  bilstm_model.py                         # LSTM intent classifier
  transformer_model.py                    # Transformer intent classifier
  extract_gaze_from_video_lstm.py         # Gaze extraction over video
  facing_estimator.py                     # Facing-direction heuristic from pose
  resample_dataverse_to_30hz.py           # Dataset preprocessing
  save_dataverse_animations.py            # Dataset visualization export
  train_model.ipynb                       # Training/evaluation notebook
  video_pipeline.ipynb                    # End-to-end video inference notebook
testing.ipynb                             # Pipeline testing notebook
```

## Setup

```
pip install -r requirements.txt
```

Also requires `torch`, `numpy`, and `scikit-learn` for model training/evaluation (not pinned in `requirements.txt`). YOLOv11 pose weights and the MediaPipe `face_landmarker.task` model are downloaded automatically by `ultralytics`/`mediapipe` on first run and are not checked into this repo.

Datasets (Yale Shutter Interaction, MINT-RVAE, MPIIGaze, Columbia Gaze, JPL Interaction) are third-party and not redistributed here — see `report.pdf` §III-C for sources.

## Limitations & next steps

- Full-pipeline evaluation was qualitative only (no ground truth on the JPL validation set); labeling a held-out video set would allow quantitative end-to-end evaluation.
- YOLOv11's facing-direction estimate is occasionally unreliable (confuses front/back), which can be mitigated by falling back to the gaze model's facing estimate when available.
- A natural extension is predicting specific actions (waving, pointing) using the same pose/face-mesh features, rather than a single binary intent signal.

---
Cloned and cleaned from the original collaborative repo for portfolio purposes; original at [github.com/TianmYu/aer1515](https://github.com/TianmYu/aer1515) (`mint_and_yale-2` branch).
