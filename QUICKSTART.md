# Quick Start Guide

Get up and running with BMD Prediction in 5 minutes!

## ⚡ TL;DR

```bash
# 1. Clone and setup
git clone https://github.com/yourusername/bmd-prediction-project.git
cd bmd-prediction-project
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install
pip install -r requirements.txt

# 3. Configure
# Edit src/BMD_Prediction.py line 68:
# BASE_PATH = "/path/to/your/dataset"

# 4. Run
python src/BMD_Prediction.py

# 5. Submit
# Upload outputs/results/submission.csv to Kaggle
```

## 📁 Dataset Structure Required

```
your-dataset-folder/
├── X-ray Images/
│   ├── train/     (377 .png files)
│   ├── val/       (54 .png files)
│   └── test/      (108 .png files)
└── CSV Files/
    ├── train_groundtruth_BMD_only.csv
    ├── val_groundtruth_BMD_only.csv
    └── test_public_new.csv
```

## 🎯 Expected Outputs

After running (15-20 min with GPU):

```
outputs/
├── models/
│   ├── best_cnn_model.pth
│   └── svm_model.pkl
├── plots/
│   ├── training_history.png
│   ├── confusion_matrix_cnn.png
│   ├── confusion_matrix_svm.png
│   ├── roc_curve.png
│   ├── model_comparison.png
│   ├── accuracy_comparison.png
│   ├── prediction_scatter.png
│   └── residuals.png
└── results/
    ├── submission.csv          ← SUBMIT THIS TO KAGGLE
    ├── submission_cnn.csv
    ├── submission_svm.csv
    ├── cnn_detailed_predictions.csv
    ├── svm_detailed_predictions.csv
    └── summary_report.txt
```

## 🚨 Common Issues

### Issue: CUDA Out of Memory
```python
# In src/BMD_Prediction.py, change:
BATCH_SIZE = 8  # instead of 16
```

### Issue: Dataset Not Found
```python
# Verify path exists:
import os
print(os.path.exists("/path/to/dataset"))

# Update BASE_PATH in BMD_Prediction.py
```

### Issue: Slow on CPU
- Use Google Colab for free GPU
- Or reduce: `NUM_EPOCHS = 10`

## 📊 Expected Performance

| Model | MAE | Accuracy |
|-------|-----|----------|
| CNN | 0.111 | 96.3% |
| SVM | **0.099** | 90.7% |

**Best Model**: SVM (lower MAE)

## 🎓 For the Report

Key files to reference:
- `outputs/plots/*.png` - All visualizations
- `outputs/results/summary_report.txt` - Metrics
- `RESULTS.md` - Detailed analysis
- `DATASET.md` - Dataset description

## 🆘 Need Help?

1. Check [USAGE.md](USAGE.md) for detailed guide
2. See [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
3. Read [FAQ.pdf](docs/FAQ.pdf)
4. Open GitHub Issue

## ✅ Verification Checklist

Before submitting:
- [ ] Code runs without errors
- [ ] All 8 plots generated
- [ ] submission.csv created (108 rows)
- [ ] Models saved in outputs/models/
- [ ] Validation accuracy > 85%

## 🚀 Next Steps

After basic run:
1. Review visualizations in `outputs/plots/`
2. Check metrics in `outputs/results/summary_report.txt`
3. Submit `submission.csv` to Kaggle
4. Write report using RESULTS.md as reference

---

**Estimated Time**: 
- Setup: 5 min
- Training: 15-20 min (GPU) | 1-2 hours (CPU)
- Review: 10 min
- **Total**: ~30-40 minutes

**Good luck with your submission!** 🎉
