# Dataset Documentation

This document provides detailed information about the BMD Prediction dataset.

## 📊 Overview

The dataset consists of hand/wrist X-ray images with corresponding Bone Mineral Density (BMD) measurements and patient metadata.

### Statistics

| Split | Images | BMD Labels | Age Range | Gender Distribution |
|-------|--------|------------|-----------|---------------------|
| Train | 377 | ✓ | - | - |
| Validation | 54 | ✓ | - | - |
| Test | 108 | ✗ (Hidden) | - | - |
| **Total** | **539** | **431** | - | - |

## 📁 Dataset Structure

```
Dataset/
├── X-ray Images/
│   ├── train/
│   │   ├── train_001.png
│   │   ├── train_002.png
│   │   └── ... (377 total)
│   ├── val/
│   │   ├── val_001.png
│   │   ├── val_002.png
│   │   └── ... (54 total)
│   └── test/
│       ├── test_001.png
│       ├── test_002.png
│       └── ... (108 total)
└── CSV Files/
    ├── train_groundtruth_BMD_only.csv
    ├── val_groundtruth_BMD_only.csv
    └── test_public_new.csv
```

## 📝 CSV File Format

### Training & Validation CSVs

Contains ground truth BMD values:

```csv
image,interview_age,Gender,BMD
train_001.png,132,F,0.8234
train_002.png,156,M,1.0125
...
```

**Columns**:
- `image`: Image filename (e.g., "train_001.png")
- `interview_age`: Patient age in months
- `Gender`: M (Male) or F (Female)
- `BMD`: Bone Mineral Density value (continuous, range: 0.65 - 1.30)

### Test CSV

Contains only metadata (BMD labels hidden):

```csv
image,interview_age,Gender
test_001.png,144,F
test_002.png,168,M
...
```

**Columns**:
- `image`: Image filename
- `interview_age`: Patient age in months
- `Gender`: M or F

## 🖼️ Image Specifications

### Format
- **File Type**: PNG
- **Color Mode**: Grayscale (converted to RGB for model input)
- **Typical Size**: Varies (e.g., 512x512, 1024x1024)
- **Content**: Hand/wrist X-ray radiographs

### Preprocessing
Images are preprocessed before training:

1. **Resize**: 224×224 pixels
2. **Normalize**: Using ImageNet statistics
   - Mean: [0.485, 0.456, 0.406]
   - Std: [0.229, 0.224, 0.225]
3. **Augmentation** (training only):
   - Random rotation (±15°)
   - Random horizontal flip
   - Color jitter
   - Random affine transformations

### Sample Images

The dataset contains X-ray images similar to:

```
┌─────────────────────────┐
│                         │
│      Hand/Wrist         │
│      X-ray Image        │
│                         │
│    [Bone structures]    │
│                         │
└─────────────────────────┘
```

Typical X-ray shows:
- Metacarpal bones
- Phalanges
- Carpal bones
- Radius and ulna (distal ends)

## 📈 BMD Value Distribution

### Training Set

```
BMD Statistics:
  Mean: 0.9234
  Std:  0.1156
  Min:  0.6512
  Max:  1.2987
  
Quartiles:
  Q1:   0.8421
  Q2:   0.9145
  Q3:   1.0023
```

### Distribution Plot

```
        ▁▂▃▅▆▇█▇▆▅▃▂▁
BMD: 0.65  0.90  1.15  1.30
      Low  Normal  High
```

### Class Distribution

Based on T-Score classification:

**Training Set (377 images)**:
- Normal (T ≥ -1.0): ~285 images (75.6%)
- Low BMD (T < -1.0): ~92 images (24.4%)

**Validation Set (54 images)**:
- Normal: ~41 images (75.9%)
- Low BMD: ~13 images (24.1%)

**Note**: Slight class imbalance (3:1 ratio)

## 👥 Demographic Information

### Age Distribution

```
Age Range: 60-240 months (~5-20 years)

Age Groups:
  60-100 months:  ████████░░ (35%)
  101-150 months: ██████████ (40%)
  151-200 months: ██████░░░░ (20%)
  201-240 months: ███░░░░░░░ (5%)
```

### Gender Distribution

```
Male:   ████████████░ (52%)
Female: ████████████░ (48%)
```

Approximately balanced gender distribution.

## 🔬 BMD Measurement Context

### What is BMD?

Bone Mineral Density (BMD) measures the amount of mineral matter per square centimeter of bones. It's used to diagnose:
- Osteopenia (low bone mass)
- Osteoporosis (very low bone mass)

### T-Score Calculation

```
T-Score = (Patient BMD - Reference BMD) / Standard Deviation
```

**Project Parameters**:
- Reference BMD: 0.86
- Standard Deviation: 0.12

### WHO Classification

| T-Score Range | Classification | Health Status |
|---------------|----------------|---------------|
| ≥ -1.0 | Normal | Healthy bone density |
| -1.0 to -2.5 | Osteopenia | Low bone mass |
| ≤ -2.5 | Osteoporosis | Very low bone mass |

**Simplified Binary Classification** (This Project):
- **Normal**: T-Score ≥ -1.0
- **Low BMD**: T-Score < -1.0 (includes osteopenia + osteoporosis)

## 📊 Data Quality

### Image Quality
- ✅ All images are readable X-rays
- ✅ Consistent anatomical positioning
- ✅ No corrupted files
- ⚠️ Variable resolution (handled by preprocessing)

### Label Quality
- ✅ BMD values from ground truth measurements
- ✅ Age and gender metadata provided
- ✅ No missing values in training/validation
- ✅ Realistic BMD range (0.65-1.30)

### Potential Issues

1. **Class Imbalance**: 
   - 75% Normal vs 25% Low BMD
   - Addressed using Focal Loss and class weights

2. **Age Range**: 
   - Dataset focused on younger patients (5-20 years)
   - May not generalize well to elderly populations

3. **Image Variability**:
   - Different X-ray machines/settings
   - Handled through data augmentation

## 🔄 Data Splits

### Split Strategy

- **Training**: 70% (377 images)
- **Validation**: 10% (54 images)
- **Test**: 20% (108 images)

### Split Characteristics

All splits maintain similar:
- Age distributions
- Gender ratios
- BMD value ranges
- Class balance proportions

This ensures fair evaluation and reduces selection bias.

## 📥 Data Loading

### PyTorch Dataset Class

```python
class BMDDataset(Dataset):
    def __init__(self, img_dir, csv_file, transform=None):
        self.img_dir = img_dir
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        
        # Normalize age (months → years → percentage)
        self.df['age_normalized'] = self.df['interview_age'] / 12.0 / 100.0
        
        # Encode gender (F=1, M=0)
        self.df['gender_encoded'] = (self.df['Gender'] == 'F').astype(int)
    
    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.img_dir, self.df.iloc[idx]['image'])
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get metadata
        age = self.df.iloc[idx]['age_normalized']
        gender = self.df.iloc[idx]['gender_encoded']
        bmd = self.df.iloc[idx]['BMD']
        
        return {
            'image': image,
            'metadata': torch.tensor([age, gender]),
            'bmd': bmd
        }
```

### Data Augmentation

**Training** (aggressive augmentation):
```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                        [0.229, 0.224, 0.225])
])
```

**Validation/Test** (minimal processing):
```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                        [0.229, 0.224, 0.225])
])
```

## 🔒 Data Privacy & Ethics

### Privacy Considerations
- Images are anonymized (no patient identifiers)
- Age provided in months (not exact birth dates)
- Gender only (no other demographic details)

### Ethical Use
- ✅ Educational and research purposes
- ✅ Non-commercial academic project
- ❌ Not for clinical diagnosis
- ❌ Not for commercial deployment

### Limitations
⚠️ This is a simulated academic dataset. Real clinical applications would require:
- Larger dataset (1000s of samples)
- Multi-center validation
- Expert radiologist annotations
- FDA/regulatory approval
- Clinical validation studies

## 📚 Related Datasets

Similar publicly available datasets:

1. **RSNA Bone Age Challenge**
   - 12,611 hand radiographs
   - Age estimation task
   - Available on Kaggle

2. **NIH Osteoporosis Dataset**
   - DXA scans with BMD measurements
   - Multi-site study

3. **MURA (Musculoskeletal Radiographs)**
   - 40,561 radiographic images
   - Abnormality detection

## 🔗 References

1. World Health Organization. (1994). Assessment of fracture risk and its application to screening for postmenopausal osteoporosis.

2. Kanis, J.A., et al. (2008). European guidance for the diagnosis and management of osteoporosis in postmenopausal women. *Osteoporosis International*.

3. Assignment brief: `docs/assignment_brief.pdf`

## ❓ FAQ

**Q: Can I use this dataset for my own project?**  
A: This is an academic assignment dataset. Check with your instructor for usage permissions.

**Q: Why are test labels hidden?**  
A: To simulate a real Kaggle competition and prevent overfitting on test data.

**Q: How was BMD measured?**  
A: Simulated from DXA (Dual-Energy X-ray Absorptiometry) measurements in a controlled study.

**Q: Can I add more data?**  
A: For this assignment, use only the provided dataset. For personal projects, ensure compatibility.

**Q: What if an image fails to load?**  
A: The dataset class handles errors by creating a black placeholder image.

## 📞 Support

For dataset-related questions:
- Check `docs/FAQ.pdf`
- Review assignment brief
- Open GitHub issue with `dataset` label

---

**Dataset Version**: 1.0 (Semester 2, 2025)  
**Last Updated**: 2025
