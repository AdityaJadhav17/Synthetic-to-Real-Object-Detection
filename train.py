import argparse
from ultralytics import YOLO
import os
import sys
import torch

# Default hyperparameters
EPOCHS = 8
MOSAIC = 0.5 # Probability of mosaic augmentation (image mixing)
OPTIMIZER = 'Adam' # Optimization algorithm
MOMENTUM = 0.937
LR0 = 0.0001 # Initial learning rate
LRF = 0.0001 # Final learning rate
SINGLE_CLS = True # Treats all objects as the same class
BATCH_SIZE = 8  # change to 16, 32 or 64 if based on GPU memory availabilty
AMP = False  # Automatic Mixed Precision (faster training with lower memory usage)
MIXUP = 0.1  # Helps in occlusion scenarios
# WEIGHT_DECAY = 0.0008  # to Prevent overfitting




# Detect GPU availability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Train YOLOv8 model for Cheerio detection.")
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size (auto-adjusted)')
    parser.add_argument('--mosaic', type=float, default=MOSAIC, help='Mosaic augmentation probability')
    parser.add_argument('--mixup', type=float, default=MIXUP, help='Mixup augmentation probability')
    parser.add_argument('--optimizer', type=str, default=OPTIMIZER, help='Optimizer (Adam, SGD, etc.)')
    parser.add_argument('--momentum', type=float, default=MOMENTUM, help='Momentum for optimizer')
    # parser.add_argument('--weight_decay', type=float, default=WEIGHT_DECAY, help='Weight decay for regularization')
    parser.add_argument('--lr0', type=float, default=LR0, help='Initial learning rate')
    parser.add_argument('--lrf', type=float, default=LRF, help='Final learning rate')
    parser.add_argument('--single_cls', type=bool, default=SINGLE_CLS, help='Train for a single class')
    parser.add_argument('--amp', type=bool, default=AMP, help='Enable Automatic Mixed Precision (AMP)')
    parser.add_argument('--device', type=str, default=DEVICE, help='Device to run training on (cuda or cpu)')
    return parser.parse_args()


def train_model(args):
    """Train YOLOv8 model with the given arguments"""
    # Set working directory
    this_dir = os.path.dirname(__file__)
    os.chdir(this_dir)

    # Load YOLOv8 model
    model = YOLO(os.path.join(this_dir, "yolov8s.pt"))

    # Train the model
    results = model.train(
        data=os.path.join(this_dir, "yolo_params.yaml"), 
        epochs=args.epochs,
        batch=args.batch_size, 
        device=args.device,  # Ensure model runs on GPU if available
        single_cls=args.single_cls, 
        mosaic=args.mosaic,
        mixup=args.mixup,
        optimizer=args.optimizer, 
        # weight_decay=args.weight_decay,
        lr0=args.lr0, 
        lrf=args.lrf, 
        momentum=args.momentum,  
        val=True,                # Enable validation during training
        amp=args.amp            
    )
    return results

if __name__ == '__main__': 
    args = parse_args()
    train_model(args)
'''
Mixup boost val pred but reduces test pred
Mosaic shouldn't be 1.0  
'''


'''
                   from  n    params  module                                       arguments
  0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]
  1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]
  2                  -1  1      7360  ultralytics.nn.modules.block.C2f             [32, 32, 1, True]
  3                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]
  4                  -1  2     49664  ultralytics.nn.modules.block.C2f             [64, 64, 2, True]
  5                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]
  6                  -1  2    197632  ultralytics.nn.modules.block.C2f             [128, 128, 2, True]
  7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]
  8                  -1  1    460288  ultralytics.nn.modules.block.C2f             [256, 256, 1, True]
  9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5]
 10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 12                  -1  1    148224  ultralytics.nn.modules.block.C2f             [384, 128, 1]
 13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 14             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 15                  -1  1     37248  ultralytics.nn.modules.block.C2f             [192, 64, 1]
 16                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]
 17            [-1, 12]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 18                  -1  1    123648  ultralytics.nn.modules.block.C2f             [192, 128, 1]
 19                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]
 20             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 21                  -1  1    493056  ultralytics.nn.modules.block.C2f             [384, 256, 1]
 22        [15, 18, 21]  1    751507  ultralytics.nn.modules.head.Detect           [1, [64, 128, 256]]
Model summary: 225 layers, 3,011,043 parameters, 3,011,027 gradients, 8.2 GFLOPs
'''