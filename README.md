# Synthetic-to-Real Object Detection Challenge

[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-blue)](https://www.kaggle.com/competitions/synthetic-2-real-object-detection-challenge)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red)
![Python](https://img.shields.io/badge/Python-3.10-green)

My solo entry in the [Synthetic to Real Object Detection Challenge](https://www.kaggle.com/competitions/synthetic-2-real-object-detection-challenge) on Kaggle. The task is to train a detector on synthetic images of a Cheerios box only, then find the box in real photos.

Case study: https://adityajadhav.dev/projects/synthetic-to-real/

## Results

Kaggle scores entries by mAP@50 (a predicted box counts when it overlaps the true box by at least half). The public leaderboard scores half of the real test images. The private leaderboard scores the other half and sets the final ranking.

| Leaderboard | mAP@50 |
|---|---|
| Private (final ranking) | 0.91750 |
| Public | 0.90738 |

On the 69 synthetic validation images, the last epoch reached mAP@50 0.9935 and mAP@50-95 0.9746 (`runs/detect/train/results.csv`). The drop to 0.9175 on real photos is the sim-to-real gap the challenge measures.

## What is here

The organizers gave every entrant a starter kit: a YOLOv8 training script, a prediction script, a script that turns predictions into the submission file, and batch files that set up a Conda environment. This repository holds my versions of those files, the training run behind my submission, its predictions and the submitted CSV.

```
├─ data/
│  ├─ train/                  # 431 synthetic images and labels
│  ├─ val/                    # 69 synthetic images and labels
│  └─ test/images/            # real photos, no labels
├─ ENV_SETUP/                 # Windows batch files that create the Conda environment
├─ runs/detect/train/         # the saved training run: weights, args.yaml, results.csv
├─ predictions/labels/        # one .txt of predicted boxes per test photo
├─ classes.txt                # the single class name
├─ yolo_params.yaml           # dataset paths and class names
├─ train.py                   # fine-tunes YOLOv8s
├─ predict.py                 # runs the best checkpoint over the test photos
├─ convert_preds_to_csv.py    # builds the Kaggle submission file
├─ submission.csv             # the file I submitted
└─ yolov8s.pt                 # pretrained YOLOv8s weights
```

## The run

`runs/detect/train/args.yaml` records the settings of the submitted run:

- YOLOv8s from its pretrained weights, every object treated as one class
- 8 epochs on a CPU at 640 pixels, batch size 8, about 103 minutes
- Adam optimizer, learning rate 0.0001
- Ultralytics augmentation: mosaic 0.3, mixup 0.1, hue, saturation and brightness shifts, translation, scaling and horizontal flips. Rotation stayed at 0 degrees.

Prediction uses a confidence threshold of 0.015. mAP ranks boxes by confidence, so a weak box that turns out right still counts.

## Setup

The environment scripts are Windows batch files and need Miniconda or Anaconda.

```bat
git clone https://github.com/AdityaJadhav17/Synthetic-to-Real-Object-Detection.git
cd Synthetic-to-Real-Object-Detection\ENV_SETUP
setup_env.bat
conda activate EDU
```

`setup_env.bat` creates a Conda environment named `EDU` with Python 3.10 and installs PyTorch, Torchvision, Ultralytics and OpenCV. On another system, install those four packages yourself.

The dataset is already in `data/`. `yolo_params.yaml` points the scripts at it.

## Usage

Train. Every setting has a default in `train.py`, and each can be overridden on the command line:

```bash
python train.py
python train.py --epochs 8 --mosaic 0.3 --mixup 0.1 --lr0 0.0001
```

Predict. The script takes no arguments. It loads `best.pt` from the training run under `runs/detect/`, reads the test path from `yolo_params.yaml`, and writes one `.txt` per photo to `predictions/labels/`:

```bash
python predict.py
```

Build the submission file:

```bash
python convert_preds_to_csv.py --preds_folder predictions/labels --test_images_folder data/test/images --output_csv submission.csv
```

Upload `submission.csv` to Kaggle.

## Limits

- The validation images are synthetic, so the score during training says little about real photos. There are no labelled real images here to check against before submitting.
- I also tried YOLOv8n and YOLOv8m and stronger augmentation. Only the YOLOv8s run was kept, so this repository cannot show what those experiments found.
- The organizers' original starter scripts are not kept here, so the history does not show which lines of the three scripts are mine.
- Eight epochs on a CPU is a short run, and with one run there is no measure of how much the score moves between runs.

## License

MIT. See [LICENSE](LICENSE).

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- Duality AI, for organizing the competition and providing the dataset and starter scripts
