# Bone Mineral Density (BMD) Prediction from X-ray Images

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Academic Project**: CSG2341 Intelligent Systems - Assignment 2  
> **Institution**: Edith Cowan University  
> **Semester**: 2, 2025

## 🎯 Project Overview

This project implements and compares **Machine Learning (SVM)** and **Deep Learning (CNN)** approaches to predict Bone Mineral Density (BMD) values from hand/wrist X-ray images. The system provides automated screening for osteoporosis risk by classifying bone health into:
- **Normal** (T-score ≥ -1.0)
- **Low BMD** (T-score < -1.0)

### Why This Matters

Osteoporosis affects millions, especially elderly women, causing increased fracture risk. Traditional diagnostic methods (DXA scans) are expensive and not widely accessible. This project explores a cost-effective alternative using standard X-ray images and artificial intelligence.

---

## 📊 Quick Results

### Model Performance (Validation Set)

| Metric | CNN (ResNet-50) | SVM (RBF) | Winner |
|--------|-----------------|-----------|--------|
| **MAE** | 0.1112 | **0.0985** | SVM ✓ |
| **RMSE** | 0.1387 | **0.1252** | SVM ✓ |
| **R² Score** | 0.0699 | **0.2420** | SVM ✓ |
| **Classification Accuracy** | **96.30%** | 90.74% | CNN ✓ |
| **AUC-ROC** | 0.7353 | **0.9510** | SVM ✓ |

**Conclusion**: SVM achieves superior regression performance for BMD prediction, while CNN excels at binary classification.

---

## 🔬 Technical Approach

### Models Implemented

#### 1. Convolutional Neural Network (CNN)
- **Architecture**: ResNet-50 (pre-trained on ImageNet)
- **Approach**: Multi-task learning (regression + classification)
- **Key Features**:
  - Transfer learning with progressive unfreezing
  - Focal Loss for class imbalance
  - Test-Time Augmentation (TTA)
  - Ridge regression calibration

#### 2. Support Vector Machine (SVM)
- **Type**: Support Vector Regression (SVR)
- **Kernel**: RBF (Radial Basis Function)
- **Features**: 
  - 2048-D CNN-extracted features (ResNet-50 penultimate layer)
  - 2-D metadata features (age, gender)

### Dataset

- **Total Images**: 539 hand/wrist X-rays
- **Training**: 377 images
- **Validation**: 54 images
- **Test**: 108 images (labels hidden for evaluation)
- **Format**: PNG images + CSV metadata (age, gender, BMD)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended, but CPU works)
- ~5 GB disk space

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/bmd-prediction-project.git
cd bmd-prediction-project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dataset Setup

Organize your data as follows:

```
data/
├── X-ray Images/
│   ├── train/    (377 images)
│   ├── val/      (54 images)
│   └── test/     (108 images)
└── CSV Files/
    ├── train_groundtruth_BMD_only.csv
    ├── val_groundtruth_BMD_only.csv
    └── test_public_new.csv
```

### Configuration

Update the dataset path in `src/BMD_Prediction.py` (line 68):

```python
BASE_PATH = "/path/to/your/dataset"  # Update this!
```

### Run Training

```bash
python src/BMD_Prediction.py
```

**Expected Runtime**:
- With GPU: 15-20 minutes
- CPU only: 1-2 hours

---

## 📈 Results & Visualizations

### Training History

<div align="center">
  <img src="outputs/plots/training_history.png" width="800" alt="CNN Training History"/>
  <p><em>CNN training progress over 50 epochs showing loss, metrics, and learning curves</em></p>
</div>

### Model Comparison

<table>
  <tr>
    <td align="center">
      <img src="outputs/plots/roc_curve.png" width="400" alt="ROC Curves"/>
      <br/>
      <strong>ROC Curves</strong>
      <br/>
      SVM: AUC = 0.9510
      <br/>
      CNN: AUC = 0.7353
    </td>
    <td align="center">
      <img src="outputs/plots/model_comparison.png" width="400" alt="Metrics Comparison"/>
      <br/>
      <strong>Performance Metrics</strong>
      <br/>
      Side-by-side comparison
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="outputs/plots/confusion_matrix_svm.png" width="400" alt="SVM Confusion Matrix"/>
      <br/>
      <strong>SVM Confusion Matrix</strong>
      <br/>
      Accuracy: 90.74%
    </td>
    <td align="center">
      <img src="outputs/plots/confusion_matrix_cnn.png" width="400" alt="CNN Confusion Matrix"/>
      <br/>
      <strong>CNN Confusion Matrix</strong>
      <br/>
      Accuracy: 96.30%
    </td>
  </tr>
</table>

### Prediction Analysis

<div align="center">
  <img src="outputs/plots/prediction_scatter.png" width="800" alt="Prediction Scatter Plots"/>
  <p><em>Predicted vs True BMD values for both models - SVM shows tighter correlation</em></p>
</div>

---

## 📁 Project Structure

```
bmd-prediction-project/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── LICENSE                           # MIT License
├── .gitignore                        # Git ignore rules
│
├── src/                              # Source code
│   ├── BMD_Prediction.py            # Main implementation (1,814 lines)
│   ├── models/                      # Model architectures
│   ├── training/                    # Training utilities
│   └── utils/                       # Helper functions
│
├── notebooks/                        # Jupyter notebooks
│   └── BMD_Prediction_Notebook.ipynb
│
├── data/                            # Dataset (not tracked)
│   ├── X-ray Images/               # X-ray images
│   └── CSV Files/                  # Metadata
│
├── outputs/                         # Generated outputs
│   ├── models/                     # Saved models
│   ├── plots/                      # Visualizations (8 images)
│   └── results/                    # Predictions & metrics
│
└── docs/                           # Documentation
    ├── assignment_brief.pdf
    ├── FAQ.pdf
    └── challenge_description.pdf
```

---

## 🔧 Key Features

### Data Processing
✅ Advanced image preprocessing (resize, normalize, augment)  
✅ Metadata encoding (age normalization, gender encoding)  
✅ Stratified train/val/test splits  

### Model Training
✅ Transfer learning with ResNet-50  
✅ Multi-task learning (regression + classification)  
✅ Focal Loss for class imbalance  
✅ Progressive backbone unfreezing  
✅ Early stopping with patience  
✅ Gradient clipping & regularization  

### Evaluation & Analysis
✅ Comprehensive metrics (MAE, RMSE, R², Accuracy, AUC)  
✅ Confusion matrices  
✅ ROC curves  
✅ Learning curves  
✅ Residual analysis  
✅ Test-Time Augmentation (TTA)  

---

## 📊 Detailed Results

### Regression Performance

**Mean Absolute Error (MAE)**:
- CNN: 0.1112 BMD units
- SVM: 0.0985 BMD units ⭐ (11% better)

**R² Score** (variance explained):
- CNN: 0.0699 (7%)
- SVM: 0.2420 (24%) ⭐ (3.5× better)

### Classification Performance

**Accuracy** (Normal vs Low BMD):
- CNN: 96.30% ⭐
- SVM: 90.74%

**Sensitivity** (recall for Low BMD):
- CNN: 66.7% (missed 1 in 3 cases)
- SVM: 100% ⭐ (caught all cases - better for screening)

**AUC-ROC** (discriminative power):
- CNN: 0.7353 (fair)
- SVM: 0.9510 ⭐ (excellent)

### Clinical Interpretation

For medical screening applications, **SVM is preferred** because:
1. Higher sensitivity (100%) - no false negatives for Low BMD
2. Better regression accuracy (MAE 0.0985)
3. Excellent discriminative power (AUC 0.9510)

Missing Low BMD cases (false negatives) is more dangerous than false positives in screening scenarios.

---

## 🛠️ Technologies Used

### Core Frameworks
- **PyTorch** 2.0+ - Deep learning framework
- **torchvision** - Pre-trained models & transforms
- **scikit-learn** - Machine learning algorithms
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation

### Visualization
- **Matplotlib** - Plotting library
- **Seaborn** - Statistical visualizations

### Others
- **Pillow** - Image processing
- **tqdm** - Progress bars
- **joblib** - Model serialization

---

## 📖 Documentation

- **[START_HERE.md](START_HERE.md)** - Navigation guide
- **[QUICKSTART.md](QUICKSTART.md)** - 5-minute setup
- **[USAGE.md](USAGE.md)** - Detailed usage instructions
- **[DATASET.md](DATASET.md)** - Dataset documentation
- **[RESULTS.md](RESULTS.md)** - In-depth analysis
- **[GITHUB_SETUP.md](GITHUB_SETUP.md)** - GitHub publishing guide
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guidelines

---

## 🔬 Methodology

### T-Score Calculation

BMD values are converted to T-scores for classification:

```
T-Score = (Patient BMD - Reference BMD) / Standard Deviation
Reference BMD = 0.86
Standard Deviation = 0.12
```

### WHO Classification

| T-Score | Classification | Action |
|---------|---------------|---------|
| ≥ -1.0 | Normal | Routine monitoring |
| -1.0 to -2.5 | Osteopenia | Lifestyle changes |
| ≤ -2.5 | Osteoporosis | Medical treatment |

This project uses **binary classification**: Normal (T ≥ -1.0) vs Low BMD (T < -1.0).

---

## 🎓 Academic Context

### Learning Outcomes Addressed

- **ULO1**: Identified appropriate ML and DL solutions for BMD prediction
- **ULO3**: Applied computational intelligence techniques (CNN, SVM)
- **ULO4**: Developed comprehensive intelligent system with documentation

### Assignment Deliverables

✅ Implementation of ML (SVM) and DL (CNN) models  
✅ Hyperparameter tuning and optimization  
✅ Performance evaluation with multiple metrics  
✅ Comparative analysis of approaches  
✅ Complete documentation and code  
✅ Kaggle-format predictions  

---

## 📝 Usage Examples

### Basic Prediction

```python
from src.BMD_Prediction import BMDPredictionModel
import torch

# Load trained model
model = BMDPredictionModel()
model.load_state_dict(torch.load('outputs/models/best_cnn_model.pth'))
model.eval()

# Make prediction
with torch.no_grad():
    bmd_pred, class_pred = model(image, metadata)
    t_score = (bmd_pred - 0.86) / 0.12
    
print(f"Predicted BMD: {bmd_pred:.3f}")
print(f"T-Score: {t_score:.3f}")
print(f"Classification: {'Normal' if class_pred == 1 else 'Low BMD'}")
```

### Custom Training

```python
# Modify hyperparameters in Config class
Config.BATCH_SIZE = 32
Config.LEARNING_RATE = 0.0001
Config.NUM_EPOCHS = 100

# Run training
python src/BMD_Prediction.py
```

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Ways to Contribute

- 🐛 Report bugs via GitHub Issues
- 💡 Suggest features or improvements
- 📝 Improve documentation
- 🔧 Submit pull requests
- ⭐ Star the repository

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Citation

If you use this project in your research or work, please cite:

```bibtex
@software{bmd_prediction_2025,
  author = {BMD Prediction Team},
  title = {Bone Mineral Density Prediction from X-ray Images},
  year = {2025},
  url = {https://github.com/yourusername/bmd-prediction-project},
  note = {Academic project for CSG2341 Intelligent Systems}
}
```

---

## 🙏 Acknowledgments

- **Dataset**: Provided by CSG2341 course coordinators, ECU
- **Pre-trained Models**: ResNet-50 from torchvision (ImageNet weights)
- **Inspiration**: Medical imaging research in osteoporosis detection
- **Institution**: Edith Cowan University, School of Science

### References

1. He, K., et al. (2016). Deep Residual Learning for Image Recognition. *CVPR*.
2. Lin, T. Y., et al. (2017). Focal Loss for Dense Object Detection. *ICCV*.
3. WHO. (1994). Assessment of fracture risk and its application to screening for postmenopausal osteoporosis.

---

## 🔮 Future Work

- [ ] Implement ensemble methods (CNN + SVM voting)
- [ ] Add attention mechanisms for interpretability (Grad-CAM)
- [ ] Extend to 3-class classification (Normal, Osteopenia, Osteoporosis)
- [ ] Experiment with other architectures (EfficientNet, Vision Transformer)
- [ ] Deploy as web application
- [ ] Multi-center validation study
- [ ] Integration with DICOM medical imaging standards

---

## 📊 Repository Statistics

- **Lines of Code**: 1,814 (Python)
- **Documentation**: 14,000+ words
- **Visualizations**: 8 comprehensive plots
- **Tests**: 539 images (377 train, 54 val, 108 test)
- **Model Performance**: 90.7% accuracy, 0.0985 MAE (SVM)

---

## ❓ FAQ

**Q: Can this replace DXA scans?**  
A: No. This is a screening tool for preliminary assessment. Clinical diagnosis requires DXA scans and physician evaluation.

**Q: Why is SVM better than CNN?**  
A: SVM uses high-quality features extracted by CNN and is better suited for this regression task with limited data.

**Q: Can I use my own X-ray images?**  
A: Yes, but ensure they're hand/wrist X-rays in PNG format with corresponding metadata.

**Q: What about data privacy?**  
A: All images are anonymized. No patient identifiers included.

**Q: Is this production-ready?**  
A: No. This is an academic project. Clinical deployment requires extensive validation, regulatory approval, and compliance.

---

## 📧 Contact

### For Questions or Collaboration

- **GitHub Issues**: [Create an issue](https://github.com/yourusername/bmd-prediction-project/issues)
- **GitHub Discussions**: [Start a discussion](https://github.com/yourusername/bmd-prediction-project/discussions)
- **Email**: your.email@university.edu

### Project Links

- **Repository**: https://github.com/yourusername/bmd-prediction-project
- **Documentation**: See docs/ folder
- **Assignment**: CSG2341, Semester 2, 2025

---

## ⭐ Show Your Support

If you found this project helpful:
- ⭐ Star this repository
- 🔗 Share with classmates
- 💬 Provide feedback via Issues
- 🤝 Contribute improvements

---

<div align="center">

**Made with ❤️ for CSG2341 Intelligent Systems**

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

**ECU School of Science** | **Semester 2, 2025**

</div>

---

## 🔄 Version History

See [CHANGELOG.md](CHANGELOG.md) for detailed version history.

**Current Version**: 1.0.0 (November 8, 2025)

---

**⚠️ Disclaimer**: This project is for educational and research purposes only. Not intended for clinical use without proper validation, regulatory approval, and medical supervision.
