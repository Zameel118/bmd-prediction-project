# Usage Guide

This guide provides detailed instructions for using the BMD Prediction project.

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Running the Code](#running-the-code)
- [Understanding Outputs](#understanding-outputs)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/yourusername/bmd-prediction-project.git
cd bmd-prediction-project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset

Organize your data in this structure:

```
data/
├── X-ray Images/
│   ├── train/    # 377 PNG images
│   ├── val/      # 54 PNG images
│   └── test/     # 108 PNG images
└── CSV Files/
    ├── train_groundtruth_BMD_only.csv
    ├── val_groundtruth_BMD_only.csv
    └── test_public_new.csv
```

### 3. Update Configuration

Edit `src/BMD_Prediction.py` at line 68:

```python
BASE_PATH = "/path/to/your/dataset"  # Update this!
```

### 4. Run Training

```bash
python src/BMD_Prediction.py
```

## ⚙️ Configuration

### Key Parameters in `Config` Class

#### Dataset Paths
```python
BASE_PATH = "path/to/dataset"          # Main dataset directory
TRAIN_IMG_PATH = f"{BASE_PATH}/X-ray Images/train"
VAL_IMG_PATH = f"{BASE_PATH}/X-ray Images/val"
TEST_IMG_PATH = f"{BASE_PATH}/X-ray Images/test"
```

#### Training Hyperparameters
```python
IMG_SIZE = 224              # Input image size (224x224)
BATCH_SIZE = 16            # Batch size for training
NUM_EPOCHS = 50            # Maximum training epochs
LEARNING_RATE = 0.001      # Initial learning rate
```

#### Model Settings
```python
PRETRAINED_MODEL = 'resnet50'    # Backbone architecture
FREEZE_BACKBONE = True           # Start with frozen backbone
UNFREEZE_EPOCH = 3              # Unfreeze after N epochs
```

#### Loss Weights
```python
CLASS_LOSS_WEIGHT = 2.0    # Classification loss weight
REG_LOSS_WEIGHT = 1.0      # Regression loss weight
USE_FOCAL_LOSS = True      # Use focal loss for classification
```

#### Test-Time Augmentation
```python
TTA_FLIP = True                 # Horizontal flip TTA
TTA_VFLIP = True               # Vertical flip TTA
TTA_ROT_ANGLES = [-10, 10]     # Rotation angles for TTA
```

#### Device
```python
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

## 🏃 Running the Code

### Basic Execution

```bash
python src/BMD_Prediction.py
```

### Using Jupyter Notebook

```bash
jupyter notebook notebooks/BMD_Prediction_Notebook.ipynb
```

### Execution Flow

The script performs these steps:

1. **Initialization** (~5 seconds)
   - Loads configuration
   - Sets random seeds
   - Creates output directories

2. **Data Loading** (~10 seconds)
   - Loads training, validation, and test datasets
   - Applies transformations
   - Creates data loaders

3. **CNN Training** (~10-15 minutes with GPU)
   - Trains for 50 epochs (or early stops)
   - Saves best model checkpoint
   - Logs training metrics

4. **Feature Extraction** (~2 minutes)
   - Extracts CNN features for SVM
   - Processes all datasets

5. **SVM Training** (~1 minute)
   - Trains SVR on extracted features
   - Saves trained model

6. **Evaluation** (~2 minutes)
   - Evaluates both models
   - Generates visualizations
   - Creates submission files

**Total Time**: 15-20 minutes (GPU) or 1-2 hours (CPU)

## 📊 Understanding Outputs

### Directory Structure

```
outputs/
├── models/
│   ├── best_cnn_model.pth          # Best CNN checkpoint (~95 MB)
│   ├── svm_model.pkl               # Trained SVM (~2 MB)
│   └── ridge_calibrator.pkl        # Ridge calibration (if enabled)
│
├── plots/
│   ├── training_history.png        # Loss and metrics over epochs
│   ├── confusion_matrix_cnn.png    # CNN confusion matrix
│   ├── confusion_matrix_svm.png    # SVM confusion matrix
│   ├── roc_curve.png              # ROC curves comparison
│   ├── model_comparison.png       # Side-by-side metrics
│   ├── accuracy_comparison.png    # Training vs validation
│   ├── prediction_scatter.png     # Predicted vs True BMD
│   └── residuals.png             # Residual analysis
│
└── results/
    ├── submission.csv                      # Best model (for Kaggle)
    ├── submission_cnn.csv                  # CNN predictions
    ├── submission_svm.csv                  # SVM predictions
    ├── cnn_detailed_predictions.csv        # CNN with T-scores
    ├── svm_detailed_predictions.csv        # SVM with T-scores
    ├── summary_report.txt                  # Overall metrics
    ├── cnn_classification_report.txt       # CNN classification details
    ├── svm_classification_report.txt       # SVM classification details
    └── classification_comparison.txt        # Model comparison
```

### Submission Files

#### submission.csv (Kaggle Format)
```csv
image,BMD
test_001.png,0.8234
test_002.png,1.0125
...
```

#### detailed_predictions.csv (Analysis Format)
```csv
image,BMD_pred,T_Score,Classification
test_001.png,0.8234,-0.3050,Normal
test_002.png,1.0125,1.2708,Normal
...
```

### Metrics Interpretation

#### Regression Metrics

- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and true BMD
  - Lower is better
  - Typical range: 0.08 - 0.15
  
- **RMSE (Root Mean Squared Error)**: Square root of average squared errors
  - Lower is better
  - More sensitive to large errors
  
- **R² Score**: Proportion of variance explained
  - Range: 0 to 1 (higher is better)
  - 0.2-0.3 is considered good for this task

#### Classification Metrics

- **Accuracy**: Percentage of correct classifications
  - Range: 0 to 1 (higher is better)
  - Target: >90%
  
- **Precision**: True positives / (True + False positives)
- **Recall**: True positives / (True + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **AUC**: Area under ROC curve
  - Range: 0.5 (random) to 1.0 (perfect)

## 🔬 Advanced Usage

### Custom Training

You can modify training by changing `Config` values:

```python
# Quick test run (2 epochs)
Config.NUM_EPOCHS = 2
Config.BATCH_SIZE = 8

# More aggressive training
Config.LEARNING_RATE = 0.01
Config.UNFREEZE_EPOCH = 1

# Disable TTA (faster inference)
Config.TTA_FLIP = False
Config.TTA_VFLIP = False
Config.TTA_ROT_ANGLES = []
```

### Using Pre-trained Models

To use saved models without retraining:

```python
# Load CNN model
model = BMDPredictionModel().to(Config.DEVICE)
model.load_state_dict(torch.load('outputs/models/best_cnn_model.pth'))
model.eval()

# Load SVM model
svm_model = joblib.load('outputs/models/svm_model.pkl')

# Make predictions
# ... (see code for details)
```

### Extracting Features Only

To extract features without training:

```python
from src.BMD_Prediction import extract_features

features = extract_features(
    model=model,
    dataloader=test_loader,
    device=Config.DEVICE
)
```

### Custom Evaluation

Evaluate on custom data:

```python
from src.BMD_Prediction import evaluate_model

metrics = evaluate_model(
    model=model,
    dataloader=custom_loader,
    device=Config.DEVICE
)

print(f"MAE: {metrics['mae']:.4f}")
print(f"Accuracy: {metrics['accuracy']:.4f}")
```

## 🐛 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
```python
# Reduce batch size
Config.BATCH_SIZE = 8  # or even 4

# Reduce image size
Config.IMG_SIZE = 128  # instead of 224

# Use gradient accumulation
# Modify training loop to accumulate gradients
```

#### 2. Dataset Not Found

**Error**: `FileNotFoundError: [Errno 2] No such file or directory`

**Solution**:
```python
# Verify paths exist
import os
print(os.path.exists(Config.BASE_PATH))
print(os.listdir(Config.BASE_PATH))

# Update BASE_PATH
Config.BASE_PATH = "correct/path/to/dataset"
```

#### 3. Slow Training on CPU

**Issue**: Training takes hours on CPU

**Solutions**:
- Use Google Colab (free GPU)
- Reduce `NUM_EPOCHS` for testing
- Use smaller `IMG_SIZE`
- Comment out TTA to speed up evaluation

#### 4. Module Not Found

**Error**: `ModuleNotFoundError: No module named 'torch'`

**Solution**:
```bash
# Activate virtual environment
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(torch.__version__)"
```

#### 5. Poor Model Performance

**Issue**: Validation accuracy < 70%

**Possible Causes & Solutions**:

1. **Insufficient Training**
   ```python
   Config.NUM_EPOCHS = 100  # Train longer
   ```

2. **Learning Rate Too High/Low**
   ```python
   Config.LEARNING_RATE = 0.0001  # Try different values
   ```

3. **Data Imbalance**
   ```python
   Config.USE_FOCAL_LOSS = True  # Enable focal loss
   ```

4. **Overfitting**
   ```python
   Config.WEIGHT_DECAY = 1e-3  # Increase regularization
   ```

### Performance Tips

#### Speed Up Training

1. **Use GPU**: Ensure CUDA is available
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Increase Batch Size**: If GPU memory allows
   ```python
   Config.BATCH_SIZE = 32
   ```

3. **Reduce Validation Frequency**: Validate less often
   ```python
   # In training loop, validate every N epochs instead of every epoch
   ```

4. **Disable TTA During Training**: Only use for final evaluation

#### Improve Accuracy

1. **Data Augmentation**: Already included, but can be tuned
2. **Longer Training**: Increase epochs with early stopping
3. **Ensemble Methods**: Average predictions from multiple models
4. **Hyperparameter Tuning**: Experiment with learning rates, etc.

### Getting Help

If you encounter issues:

1. Check existing GitHub Issues
2. Review error messages carefully
3. Verify all paths and file structures
4. Test with smaller dataset first
5. Create a new issue with:
   - Error message
   - System information
   - Steps to reproduce

## 📚 Additional Resources

### Tutorials
- [PyTorch Transfer Learning](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [Medical Image Analysis](https://github.com/topics/medical-image-analysis)

### Documentation
- [Project README](README.md)
- [Contributing Guide](CONTRIBUTING.md)
- [Assignment Brief](docs/assignment_brief.pdf)

### Support
- GitHub Issues: [Report a bug](https://github.com/yourusername/bmd-prediction-project/issues)
- Discussions: [Ask questions](https://github.com/yourusername/bmd-prediction-project/discussions)

---

**Need more help?** Open an issue on GitHub with the `question` label.
