import os
import sys
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision import transforms
import torchvision.transforms.functional as TF

import numpy as np
import pandas as pd
from PIL import Image

from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, 
                            accuracy_score, confusion_matrix, roc_curve, auc,
                            classification_report, precision_score, recall_score, f1_score)

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import joblib
class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in classification.
    Supports per-class alpha (weights) and focusing parameter gamma.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(weight=alpha, reduction='none')
    
    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)  # [N]
        with torch.no_grad():
            pt = torch.softmax(inputs, dim=1)[torch.arange(inputs.size(0)), targets]
        focal_factor = (1 - pt) ** self.gamma
        loss = focal_factor * ce_loss
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss

def calculate_bmd_stats(train_csv_file):
    """Compute mean and std of BMD on training set for normalization."""
    df = pd.read_csv(train_csv_file)
    mean = float(df['BMD'].mean())
    std = float(df['BMD'].std() if df['BMD'].std() > 1e-8 else 1.0)
    print(f"BMD stats -> mean: {mean:.4f}, std: {std:.4f}")
    return mean, std

# ================================================================================
# SECTION 1: CONFIGURATION
# ================================================================================

class Config:
    """Configuration and hyperparameters for the entire project"""
    
    # ============== UPDATE THIS PATH WITH YOUR DATASET LOCATION ==============
    BASE_PATH = "C:/Users/HP/Desktop/BMD Project 01"  # ⚠️ CHANGE THIS!
    # =========================================================================
    
    # Dataset paths
    TRAIN_IMG_PATH = f"{BASE_PATH}/X-ray Images/train"
    VAL_IMG_PATH = f"{BASE_PATH}/X-ray Images/val"
    TEST_IMG_PATH = f"{BASE_PATH}/X-ray Images/test"
    
    TRAIN_CSV = f"{BASE_PATH}/CSV Files/train_groundtruth_BMD_only.csv"
    VAL_CSV = f"{BASE_PATH}/CSV Files/val_groundtruth_BMD_only.csv"
    TEST_CSV = f"{BASE_PATH}/CSV Files/test_public_new.csv"
    
    # Output paths
    OUTPUT_DIR = "C:/Users/HP/Desktop/BMD Project 01/Outputs"
    MODEL_DIR = f"{OUTPUT_DIR}/models"
    RESULTS_DIR = f"{OUTPUT_DIR}/results"
    PLOTS_DIR = f"{OUTPUT_DIR}/plots"
    
    # T-Score parameters
    REFERENCE_BMD = 0.86
    STD_DEV = 0.12
    T_SCORE_THRESHOLD = -1.0
    
    # Model parameters
    IMG_SIZE = 224
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    
    # Classification training tweaks
    USE_FOCAL_LOSS = True
    CLASS_LOSS_WEIGHT = 2.0
    REG_LOSS_WEIGHT = 1.0
    UNFREEZE_EPOCH = 3  # unfreeze backbone after N epochs
    BACKBONE_LR = 1e-4  # lower LR for backbone when unfrozen
    
    # Regression training tweaks
    USE_BMD_NORMALIZATION = True
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP_NORM = 2.0
    
    # Inference
    TTA_FLIP = True  # average predictions with horizontal flip
    TTA_VFLIP = True
    TTA_ROT_ANGLES = [-10, 10]
    USE_RIDGE_CALIBRATION = True
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Random seed
    RANDOM_SEED = 42
    
    # CNN Architecture
    PRETRAINED_MODEL = 'resnet50'
    FREEZE_BACKBONE = True
    
    # SVM parameters
    SVM_C = 1.0
    SVM_KERNEL = 'rbf'
    SVM_GAMMA = 'scale'
    
    @staticmethod
    def calculate_t_score(bmd_value):
        """Calculate T-score from BMD value"""
        return (bmd_value - Config.REFERENCE_BMD) / Config.STD_DEV
    
    @staticmethod
    def classify_bone_health(bmd_value):
        """Classify as Normal (1) or Low BMD (0)"""
        t_score = Config.calculate_t_score(bmd_value)
        return 1 if t_score >= Config.T_SCORE_THRESHOLD else 0

# ================================================================================
# SECTION 2: DATASET AND DATA LOADING
# ================================================================================

class BMDDataset(Dataset):
    """Custom Dataset for BMD prediction with metadata"""
    
    def __init__(self, img_dir, csv_file, transform=None, is_test=False):
        self.img_dir = img_dir
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.is_test = is_test
        
        # Normalize age
        self.df['age_normalized'] = self.df['interview_age'] / 12.0 / 100.0
        
        # Encode gender
        self.df['gender_encoded'] = (self.df['Gender'] == 'F').astype(int)
        
        print(f"Loaded {len(self.df)} samples from {csv_file}")
        if not is_test:
            print(f"BMD range: {self.df['BMD'].min():.3f} - {self.df['BMD'].max():.3f}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['image']
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new('RGB', (Config.IMG_SIZE, Config.IMG_SIZE), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        age = torch.tensor(self.df.iloc[idx]['age_normalized'], dtype=torch.float32)
        gender = torch.tensor(self.df.iloc[idx]['gender_encoded'], dtype=torch.float32)
        metadata = torch.stack([age, gender])
        
        if self.is_test:
            return {
                'image': image,
                'metadata': metadata,
                'img_name': img_name
            }
        else:
            bmd = torch.tensor(self.df.iloc[idx]['BMD'], dtype=torch.float32)
            classification = torch.tensor(
                Config.classify_bone_health(self.df.iloc[idx]['BMD']), 
                dtype=torch.long
            )
            
            return {
                'image': image,
                'metadata': metadata,
                'bmd': bmd,
                'classification': classification,
                'img_name': img_name
            }

def get_transforms(is_training=True):
    """Get image transformations"""
    if is_training:
        return transforms.Compose([
            transforms.RandomResizedCrop(Config.IMG_SIZE, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

def calculate_class_weights(train_csv_file):
    """Calculate class weights for imbalanced dataset"""
    df = pd.read_csv(train_csv_file)
    classifications = [Config.classify_bone_health(bmd) for bmd in df['BMD']]
    class_counts = pd.Series(classifications).value_counts().sort_index()
    
    # Calculate weights inversely proportional to class frequency
    total_samples = len(classifications)
    num_classes = len(class_counts)
    weights = torch.tensor([total_samples / (num_classes * count) for count in class_counts], dtype=torch.float32)
    
    print(f"\nClass distribution in training set:")
    print(f"  Low BMD (0): {class_counts.get(0, 0)} samples")
    print(f"  Normal (1): {class_counts.get(1, 0)} samples")
    print(f"  Class weights: {weights}")
    
    return weights

def build_weighted_sampler(train_df):
    """Create a WeightedRandomSampler to balance classes each batch"""
    classifications = [Config.classify_bone_health(bmd) for bmd in train_df['BMD']]
    class_counts = pd.Series(classifications).value_counts().sort_index()
    total = len(classifications)
    num_classes = len(class_counts)
    class_weight = {cls: total / (num_classes * count) for cls, count in class_counts.items()}
    sample_weights = [class_weight[c] for c in classifications]
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler

def create_dataloaders():
    """Create train, validation, and test dataloaders"""
    
    train_dataset = BMDDataset(
        img_dir=Config.TRAIN_IMG_PATH,
        csv_file=Config.TRAIN_CSV,
        transform=get_transforms(is_training=True),
        is_test=False
    )
    
    val_dataset = BMDDataset(
        img_dir=Config.VAL_IMG_PATH,
        csv_file=Config.VAL_CSV,
        transform=get_transforms(is_training=False),
        is_test=False
    )
    
    test_dataset = BMDDataset(
        img_dir=Config.TEST_IMG_PATH,
        csv_file=Config.TEST_CSV,
        transform=get_transforms(is_training=False),
        is_test=True
    )
    
    # Use weighted sampler for imbalanced classification
    train_sampler = build_weighted_sampler(train_dataset.df)
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

# ================================================================================
# SECTION 3: CNN MODEL ARCHITECTURE
# ================================================================================

class CNNWithMetadata(nn.Module):
    """CNN model that combines image features with metadata"""
    
    def __init__(self, pretrained=True, freeze_backbone=True):
        super(CNNWithMetadata, self).__init__()
        
        # Load pretrained ResNet50
        self.backbone = models.resnet50(pretrained=pretrained)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Metadata processing
        self.metadata_fc = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Combined features
        combined_features = num_features + 16
        
        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(combined_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        
        # Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(combined_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
    
    def forward(self, image, metadata):
        # Extract image features
        image_features = self.backbone(image)
        
        # Process metadata
        metadata_features = self.metadata_fc(metadata)
        
        # Combine features
        combined = torch.cat([image_features, metadata_features], dim=1)
        
        # Get predictions
        bmd_pred = self.regression_head(combined).squeeze()
        class_pred = self.classification_head(combined)
        
        return bmd_pred, class_pred
    
    def extract_features(self, image, metadata):
        """Extract combined features for SVM"""
        with torch.no_grad():
            image_features = self.backbone(image)
            metadata_features = self.metadata_fc(metadata)
            combined = torch.cat([image_features, metadata_features], dim=1)
        return combined

# ================================================================================
# SECTION 4: CNN TRAINING
# ================================================================================

class CNNTrainer:
    """Trainer for CNN models"""
    
    def __init__(self, model, train_loader, val_loader, device=None, class_weights=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or Config.DEVICE
        self.model.to(self.device)
        
        # BMD normalization
        self.bmd_mean, self.bmd_std = calculate_bmd_stats(Config.TRAIN_CSV) if Config.USE_BMD_NORMALIZATION else (0.0, 1.0)
        self.ridge_calibrator = None
        
        # Loss functions
        self.regression_criterion = nn.SmoothL1Loss(beta=0.05)
        
        # Classification loss: focal loss (optional) or weighted cross-entropy
        if Config.USE_FOCAL_LOSS:
            self.classification_criterion = FocalLoss(
                alpha=class_weights.to(self.device) if class_weights is not None else None,
                gamma=2.0
            )
            if class_weights is not None:
                print(f"Using focal loss with alpha (class weights): {class_weights}")
        else:
            if class_weights is not None:
                class_weights = class_weights.to(self.device)
                print(f"Using class weights: {class_weights}")
                self.classification_criterion = nn.CrossEntropyLoss(weight=class_weights)
            else:
                self.classification_criterion = nn.CrossEntropyLoss()
        
        # Loss weights (configurable and adjustable during training)
        self.class_weight = Config.CLASS_LOSS_WEIGHT
        self.reg_weight = Config.REG_LOSS_WEIGHT
        
        # Optimizer
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # History
        self.history = {
            'train_loss': [], 'train_reg_loss': [], 'train_class_loss': [],
            'train_mae': [], 'train_rmse': [], 'train_r2': [], 'train_accuracy': [],
            'val_loss': [], 'val_reg_loss': [], 'val_class_loss': [],
            'val_mae': [], 'val_rmse': [], 'val_r2': [], 'val_accuracy': []
        }
        
        self.best_val_loss = float('inf')
        self.best_model_state = None
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        
        running_loss = 0.0
        running_reg_loss = 0.0
        running_class_loss = 0.0
        all_bmd_true = []
        all_bmd_pred = []
        all_class_true = []
        all_class_pred = []
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            images = batch['image'].to(self.device)
            metadata = batch['metadata'].to(self.device)
            bmd_true = batch['bmd'].to(self.device)
            class_true = batch['classification'].to(self.device)
            
            self.optimizer.zero_grad()
            bmd_pred, class_pred = self.model(images, metadata)
            
            # Normalize BMD for regression loss if enabled
            if Config.USE_BMD_NORMALIZATION:
                bmd_pred_norm = (bmd_pred - self.bmd_mean) / self.bmd_std
                bmd_true_norm = (bmd_true - self.bmd_mean) / self.bmd_std
                reg_loss = self.regression_criterion(bmd_pred_norm, bmd_true_norm)
            else:
                reg_loss = self.regression_criterion(bmd_pred, bmd_true)
            class_loss = self.classification_criterion(class_pred, class_true)
            # Weighted multi-task loss
            loss = self.reg_weight * reg_loss + self.class_weight * class_loss
            
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=Config.GRAD_CLIP_NORM)
            self.optimizer.step()
            
            running_loss += loss.item()
            running_reg_loss += reg_loss.item()
            running_class_loss += class_loss.item()
            
            all_bmd_true.extend(bmd_true.cpu().detach().numpy())
            all_bmd_pred.extend(bmd_pred.cpu().detach().numpy())
            all_class_true.extend(class_true.cpu().detach().numpy())
            all_class_pred.extend(class_pred.argmax(dim=1).cpu().detach().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_reg_loss = running_reg_loss / len(self.train_loader)
        epoch_class_loss = running_class_loss / len(self.train_loader)
        
        mae = mean_absolute_error(all_bmd_true, all_bmd_pred)
        rmse = np.sqrt(mean_squared_error(all_bmd_true, all_bmd_pred))
        r2 = r2_score(all_bmd_true, all_bmd_pred)
        accuracy = accuracy_score(all_class_true, all_class_pred)
        
        return {
            'loss': epoch_loss, 'reg_loss': epoch_reg_loss, 'class_loss': epoch_class_loss,
            'mae': mae, 'rmse': rmse, 'r2': r2, 'accuracy': accuracy
        }
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        
        running_loss = 0.0
        running_reg_loss = 0.0
        running_class_loss = 0.0
        all_bmd_true = []
        all_bmd_pred = []
        all_class_true = []
        all_class_pred = []
        all_class_probs = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                images = batch['image'].to(self.device)
                metadata = batch['metadata'].to(self.device)
                bmd_true = batch['bmd'].to(self.device)
                class_true = batch['classification'].to(self.device)
                
                bmd_pred, class_pred = self.model(images, metadata)
                
                if Config.USE_BMD_NORMALIZATION:
                    bmd_pred_norm = (bmd_pred - self.bmd_mean) / self.bmd_std
                    bmd_true_norm = (bmd_true - self.bmd_mean) / self.bmd_std
                    reg_loss = self.regression_criterion(bmd_pred_norm, bmd_true_norm)
                else:
                    reg_loss = self.regression_criterion(bmd_pred, bmd_true)
                class_loss = self.classification_criterion(class_pred, class_true)
                loss = self.reg_weight * reg_loss + self.class_weight * class_loss
                
                running_loss += loss.item()
                running_reg_loss += reg_loss.item()
                running_class_loss += class_loss.item()
                
                all_bmd_true.extend(bmd_true.cpu().numpy())
                all_bmd_pred.extend(bmd_pred.cpu().numpy())
                all_class_true.extend(class_true.cpu().numpy())
                all_class_pred.extend(class_pred.argmax(dim=1).cpu().numpy())
                all_class_probs.extend(torch.softmax(class_pred, dim=1)[:, 1].cpu().numpy())
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_reg_loss = running_reg_loss / len(self.val_loader)
        epoch_class_loss = running_class_loss / len(self.val_loader)
        
        # Ridge calibration (stacking) to improve regression fit
        preds_np = np.array(all_bmd_pred).reshape(-1, 1)
        trues_np = np.array(all_bmd_true)
        if Config.USE_RIDGE_CALIBRATION:
            self.ridge_calibrator = Ridge(alpha=1.0)
            self.ridge_calibrator.fit(preds_np, trues_np)
            all_bmd_pred_cal = self.ridge_calibrator.predict(preds_np)
        else:
            a, b = np.polyfit(all_bmd_pred, all_bmd_true, deg=1)
            self.calib_a, self.calib_b = float(a), float(b)
            all_bmd_pred_cal = a * np.array(all_bmd_pred) + b
        mae = mean_absolute_error(all_bmd_true, all_bmd_pred_cal)
        rmse = np.sqrt(mean_squared_error(all_bmd_true, all_bmd_pred_cal))
        r2 = r2_score(all_bmd_true, all_bmd_pred_cal)
        
        # Optimize CNN classification threshold using probs
        from sklearn.metrics import f1_score
        best_acc = 0.0
        best_thresh = 0.5
        probs = np.array(all_class_probs)
        for t in np.linspace(0.2, 0.8, 25):
            preds_t = (probs >= t).astype(int)
            acc_t = accuracy_score(all_class_true, preds_t)
            if acc_t > best_acc:
                best_acc = acc_t
                best_thresh = t
        self.best_cnn_threshold = best_thresh
        accuracy = best_acc
        
        return {
            'loss': epoch_loss, 'reg_loss': epoch_reg_loss, 'class_loss': epoch_class_loss,
            'mae': mae, 'rmse': rmse, 'r2': r2, 'accuracy': accuracy,
            'bmd_true': all_bmd_true, 'bmd_pred': all_bmd_pred,
            'class_true': all_class_true, 'class_pred': all_class_pred,
            'class_probs': all_class_probs
        }
    
    def train(self, num_epochs, save_path=None):
        """Train the model for multiple epochs"""
        print(f"\nStarting training on {self.device}...")
        
        for epoch in range(num_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*50}")
            
            # Adjust loss weights schedule to emphasize regression later
            if epoch + 1 >= Config.UNFREEZE_EPOCH:
                self.reg_weight = max(Config.REG_LOSS_WEIGHT * 2.5, 2.5)
                self.class_weight = max(0.5, Config.CLASS_LOSS_WEIGHT * 0.5)
            else:
                self.reg_weight = Config.REG_LOSS_WEIGHT
                self.class_weight = Config.CLASS_LOSS_WEIGHT

            # Optionally unfreeze backbone after a few epochs with lower LR
            if Config.FREEZE_BACKBONE and epoch + 1 == Config.UNFREEZE_EPOCH:
                print("\nUnfreezing backbone for fine-tuning...")
                for param in self.model.backbone.parameters():
                    param.requires_grad = True
                # Use different LR for backbone vs heads
                self.optimizer = optim.Adam([
                    { 'params': self.model.backbone.parameters(), 'lr': Config.BACKBONE_LR },
                    { 'params': list(self.model.metadata_fc.parameters()) +
                               list(self.model.regression_head.parameters()) +
                               list(self.model.classification_head.parameters()),
                      'lr': Config.LEARNING_RATE }
                ])

            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            
            self.scheduler.step(val_metrics['loss'])
            
            # Save history
            for key in ['loss', 'reg_loss', 'class_loss', 'mae', 'rmse', 'r2', 'accuracy']:
                self.history[f'train_{key}'].append(train_metrics[key])
                self.history[f'val_{key}'].append(val_metrics[key])
            
            print(f"\nTrain - Loss: {train_metrics['loss']:.4f} | MAE: {train_metrics['mae']:.4f} | Acc: {train_metrics['accuracy']:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f} | MAE: {val_metrics['mae']:.4f} | Acc: {val_metrics['accuracy']:.4f}")
            
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_model_state = self.model.state_dict().copy()
                print(f"\n✓ New best model saved!")
                
                if save_path:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': val_metrics['loss'],
                        'history': self.history
                    }, save_path)
        
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        return self.history
    
    def get_predictions(self, data_loader):
        """Get predictions for a dataset"""
        self.model.eval()
        
        all_bmd_pred = []
        all_class_pred = []
        all_class_probs = []
        all_img_names = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc='Predicting'):
                images = batch['image'].to(self.device)
                metadata = batch['metadata'].to(self.device)
                
                # TTA: average over multiple transforms
                bmd_logits = []
                class_logits = []
                
                def forward_imgs(imgs):
                    bp, cp = self.model(imgs, metadata)
                    return bp, cp
                
                # original
                bp, cp = forward_imgs(images)
                bmd_logits.append(bp)
                class_logits.append(cp)
                
                # horizontal flip
                if Config.TTA_FLIP:
                    imgs_h = torch.flip(images, dims=[3])
                    bp_h, cp_h = forward_imgs(imgs_h)
                    bmd_logits.append(bp_h)
                    class_logits.append(cp_h)
                
                # vertical flip
                if Config.TTA_VFLIP:
                    imgs_v = torch.flip(images, dims=[2])
                    bp_v, cp_v = forward_imgs(imgs_v)
                    bmd_logits.append(bp_v)
                    class_logits.append(cp_v)
                
                # small rotations
                for ang in Config.TTA_ROT_ANGLES:
                    imgs_r = TF.rotate(images, angle=ang)
                    bp_r, cp_r = forward_imgs(imgs_r)
                    bmd_logits.append(bp_r)
                    class_logits.append(cp_r)
                
                bmd_pred = torch.stack(bmd_logits).mean(dim=0)
                class_pred = torch.stack(class_logits).mean(dim=0)
                
                all_bmd_pred.extend(bmd_pred.cpu().numpy())
                probs = torch.softmax(class_pred, dim=1)[:, 1]
                if hasattr(self, 'best_cnn_threshold'):
                    preds = (probs >= self.best_cnn_threshold).int()
                else:
                    preds = class_pred.argmax(dim=1)
                all_class_pred.extend(preds.cpu().numpy())
                all_class_probs.extend(probs.cpu().numpy())
                all_img_names.extend(batch['img_name'])
        
        # Apply regression calibration if available (ridge preferred)
        bmd_np = np.array(all_bmd_pred)
        if self.ridge_calibrator is not None:
            bmd_np = self.ridge_calibrator.predict(bmd_np.reshape(-1, 1))
        elif hasattr(self, 'calib_a') and hasattr(self, 'calib_b'):
            bmd_np = self.calib_a * bmd_np + self.calib_b
        
        return {
            'img_names': all_img_names,
            'bmd_pred': bmd_np,
            'class_pred': np.array(all_class_pred),
            'class_probs': np.array(all_class_probs)
        }

# ================================================================================
# SECTION 5: SVM TRAINING
# ================================================================================

class SVMWithCNNFeatures:
    """SVM Regressor using CNN-extracted features"""
    
    def __init__(self, cnn_model, device=None):
        self.cnn_model = cnn_model
        self.device = device or Config.DEVICE
        self.cnn_model.to(self.device)
        self.cnn_model.eval()
        
        self.optimal_threshold = None  # Will be set during training
        self.svm_regressor = SVR(
            kernel=Config.SVM_KERNEL,
            C=Config.SVM_C,
            gamma=Config.SVM_GAMMA,
            epsilon=0.01
        )
        
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def extract_features(self, data_loader):
        """Extract features from data using CNN"""
        all_features = []
        all_bmd = []
        all_classifications = []
        all_img_names = []
        
        self.cnn_model.eval()
        with torch.no_grad():
            for batch in tqdm(data_loader, desc='Extracting features'):
                images = batch['image'].to(self.device)
                metadata = batch['metadata'].to(self.device)
                
                features = self.cnn_model.extract_features(images, metadata)
                all_features.append(features.cpu().numpy())
                
                if 'bmd' in batch:
                    all_bmd.append(batch['bmd'].numpy())
                    all_classifications.append(batch['classification'].numpy())
                
                if 'img_name' in batch:
                    all_img_names.extend(batch['img_name'])
        
        features = np.vstack(all_features)
        
        result = {'features': features, 'img_names': all_img_names}
        
        if all_bmd:
            result['bmd'] = np.concatenate(all_bmd)
            result['classifications'] = np.concatenate(all_classifications)
        
        return result
    
    def train(self, train_loader, val_loader=None):
        """Train SVM on extracted features"""
        print("\n" + "="*50)
        print("Training SVM with CNN Features")
        print("="*50)
        
        print("\nExtracting training features...")
        train_data = self.extract_features(train_loader)
        X_train = train_data['features']
        y_train = train_data['bmd']
        
        print("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        print(f"\nTraining SVM on {len(X_train)} samples...")
        self.svm_regressor.fit(X_train_scaled, y_train)
        
        train_pred = self.svm_regressor.predict(X_train_scaled)
        # Linear calibration for SVM regression
        a, b = np.polyfit(train_pred, y_train, deg=1)
        self.calib_a, self.calib_b = float(a), float(b)
        train_pred_cal = a * train_pred + b
        train_metrics = self._calculate_metrics(y_train, train_pred_cal, train_data['classifications'])
        
        print("\nTraining Metrics:")
        print(f"  MAE: {train_metrics['mae']:.4f}")
        print(f"  RMSE: {train_metrics['rmse']:.4f}")
        print(f"  R²: {train_metrics['r2']:.4f}")
        print(f"  Accuracy: {train_metrics['accuracy']:.4f}")
        
        val_metrics = None
        if val_loader:
            print("\nExtracting validation features...")
            val_data = self.extract_features(val_loader)
            X_val = val_data['features']
            y_val = val_data['bmd']
            
            X_val_scaled = self.scaler.transform(X_val)
            val_pred = self.svm_regressor.predict(X_val_scaled)
            val_pred = self.calib_a * val_pred + self.calib_b
            
            # Find optimal threshold for classification
            optimal_threshold = self._find_optimal_threshold(y_val, val_pred, val_data['classifications'])
            print(f"  Optimal T-score threshold: {optimal_threshold:.2f} (default: {Config.T_SCORE_THRESHOLD})")
            self.optimal_threshold = optimal_threshold
            
            val_metrics = self._calculate_metrics(y_val, val_pred, val_data['classifications'], optimal_threshold)
            
            print("\nValidation Metrics:")
            print(f"  MAE: {val_metrics['mae']:.4f}")
            print(f"  RMSE: {val_metrics['rmse']:.4f}")
            print(f"  R²: {val_metrics['r2']:.4f}")
            print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
            
            val_metrics['bmd_true'] = y_val
            val_metrics['bmd_pred'] = val_pred
            val_metrics['class_true'] = val_data['classifications']
        
        self.is_fitted = True
        
        return train_metrics, val_metrics
    
    def _find_optimal_threshold(self, y_true, y_pred, class_true):
        """Find optimal T-score threshold using balanced accuracy and class coverage.
        Avoids degenerate thresholds where only one class is predicted.
        """
        from sklearn.metrics import f1_score, balanced_accuracy_score
        
        # Try different T-score thresholds
        best_threshold = Config.T_SCORE_THRESHOLD
        best_score = -1.0
        
        # Calculate T-scores for predictions
        t_scores = np.array([Config.calculate_t_score(bmd) for bmd in y_pred])
        
        # Try thresholds from -2.5 to 0.5 in 0.1 steps
        for threshold in np.arange(-2.5, 0.6, 0.05):
            class_pred = (t_scores >= threshold).astype(int)
            # Ensure both classes appear in predictions
            if len(np.unique(class_pred)) < 2:
                continue
            score = 0.7 * balanced_accuracy_score(class_true, class_pred) + 0.3 * f1_score(class_true, class_pred, average='weighted')
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        # Fallback: if all thresholds were degenerate, set prevalence-matching threshold
        if best_score < 0:
            prevalence = np.mean(class_true)
            perc = 100 * (1 - prevalence)
            best_threshold = np.percentile(t_scores, perc)
        
        return best_threshold
    
    def _calculate_metrics(self, y_true, y_pred, class_true=None, optimal_threshold=None):
        """Calculate regression and classification metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # Use optimal threshold if provided, otherwise use default
        if optimal_threshold is not None:
            t_scores = np.array([Config.calculate_t_score(bmd) for bmd in y_pred])
            class_pred = (t_scores >= optimal_threshold).astype(int)
        else:
            class_pred = np.array([Config.classify_bone_health(bmd) for bmd in y_pred])
        
        if class_true is not None:
            accuracy = accuracy_score(class_true, class_pred)
        else:
            class_true_calc = np.array([Config.classify_bone_health(bmd) for bmd in y_true])
            accuracy = accuracy_score(class_true_calc, class_pred)
        
        return {
            'mae': mae, 'rmse': rmse, 'r2': r2, 'accuracy': accuracy,
            'class_pred': class_pred
        }
    
    def predict(self, data_loader):
        """Make predictions on new data"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        print("\nExtracting features for prediction...")
        data = self.extract_features(data_loader)
        X = data['features']
        
        X_scaled = self.scaler.transform(X)
        bmd_pred = self.svm_regressor.predict(X_scaled)
        if hasattr(self, 'calib_a') and hasattr(self, 'calib_b'):
            bmd_pred = self.calib_a * bmd_pred + self.calib_b
        
        # Use optimal threshold if available, otherwise use default
        if hasattr(self, 'optimal_threshold'):
            t_scores = np.array([Config.calculate_t_score(bmd) for bmd in bmd_pred])
            class_pred = (t_scores >= self.optimal_threshold).astype(int)
        else:
            class_pred = np.array([Config.classify_bone_health(bmd) for bmd in bmd_pred])
        
        result = {
            'img_names': data['img_names'],
            'bmd_pred': bmd_pred,
            'class_pred': class_pred
        }
        
        if 'bmd' in data:
            result['bmd_true'] = data['bmd']
            result['class_true'] = data['classifications']
        
        return result
    
    def save(self, filepath):
        """Save SVM model and scaler"""
        joblib.dump({
            'svm': self.svm_regressor,
            'scaler': self.scaler
        }, filepath)
        print(f"SVM model saved to {filepath}")

# ================================================================================
# SECTION 6: VISUALIZATIONS
# ================================================================================

class Visualizer:
    """Comprehensive visualization for model evaluation"""
    
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        sns.set_style('whitegrid')
    
    def plot_training_history(self, cnn_history, save_name='training_history.png'):
        """Plot training and validation metrics"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('CNN Training History', fontsize=16, fontweight='bold')
        
        epochs = range(1, len(cnn_history['train_loss']) + 1)
        
        # Loss
        axes[0, 0].plot(epochs, cnn_history['train_loss'], 'b-', label='Train', linewidth=2)
        axes[0, 0].plot(epochs, cnn_history['val_loss'], 'r-', label='Validation', linewidth=2)
        axes[0, 0].set_title('Total Loss', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # MAE
        axes[0, 1].plot(epochs, cnn_history['train_mae'], 'b-', label='Train', linewidth=2)
        axes[0, 1].plot(epochs, cnn_history['val_mae'], 'r-', label='Validation', linewidth=2)
        axes[0, 1].set_title('Mean Absolute Error', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # RMSE
        axes[0, 2].plot(epochs, cnn_history['train_rmse'], 'b-', label='Train', linewidth=2)
        axes[0, 2].plot(epochs, cnn_history['val_rmse'], 'r-', label='Validation', linewidth=2)
        axes[0, 2].set_title('Root Mean Squared Error', fontweight='bold')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('RMSE')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # R²
        axes[1, 0].plot(epochs, cnn_history['train_r2'], 'b-', label='Train', linewidth=2)
        axes[1, 0].plot(epochs, cnn_history['val_r2'], 'r-', label='Validation', linewidth=2)
        axes[1, 0].set_title('R² Score', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('R²')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1, 1].plot(epochs, cnn_history['train_accuracy'], 'b-', label='Train', linewidth=2)
        axes[1, 1].plot(epochs, cnn_history['val_accuracy'], 'r-', label='Validation', linewidth=2)
        axes[1, 1].set_title('Classification Accuracy', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Loss components
        axes[1, 2].plot(epochs, cnn_history['train_reg_loss'], 'b-', label='Train Reg', linewidth=2)
        axes[1, 2].plot(epochs, cnn_history['val_reg_loss'], 'r-', label='Val Reg', linewidth=2)
        axes[1, 2].plot(epochs, cnn_history['train_class_loss'], 'b--', label='Train Class', linewidth=2)
        axes[1, 2].plot(epochs, cnn_history['val_class_loss'], 'r--', label='Val Class', linewidth=2)
        axes[1, 2].set_title('Loss Components', fontweight='bold')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Loss')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to {save_path}")
        plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name, save_name):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Low BMD', 'Normal'],
                    yticklabels=['Low BMD', 'Normal'],
                    cbar_kws={'label': 'Count'})
        
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        accuracy = np.trace(cm) / np.sum(cm)
        plt.text(0.5, -0.15, f'Accuracy: {accuracy:.4f}', 
                ha='center', transform=plt.gca().transAxes, fontsize=12)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
        plt.close()
    
    def plot_roc_curve(self, y_true, y_probs_cnn, y_probs_svm=None, save_name='roc_curve.png'):
        """Plot ROC curve"""
        plt.figure(figsize=(10, 8))
        
        fpr_cnn, tpr_cnn, _ = roc_curve(y_true, y_probs_cnn)
        roc_auc_cnn = auc(fpr_cnn, tpr_cnn)
        plt.plot(fpr_cnn, tpr_cnn, 'b-', linewidth=2.5, 
                label=f'CNN (AUC = {roc_auc_cnn:.4f})')
        
        if y_probs_svm is not None:
            fpr_svm, tpr_svm, _ = roc_curve(y_true, y_probs_svm)
            roc_auc_svm = auc(fpr_svm, tpr_svm)
            plt.plot(fpr_svm, tpr_svm, 'r-', linewidth=2.5,
                    label=f'SVM (AUC = {roc_auc_svm:.4f})')
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title('ROC Curve - Model Comparison', fontsize=14, fontweight='bold', pad=20)
        plt.legend(loc='lower right', fontsize=11)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
        plt.close()
    
    def plot_model_comparison_bars(self, cnn_metrics, svm_metrics, save_name='model_comparison.png'):
        """Plot bar chart comparing models"""
        metrics = ['MAE', 'RMSE', 'R²', 'Accuracy']
        cnn_values = [cnn_metrics['mae'], cnn_metrics['rmse'], 
                     cnn_metrics['r2'], cnn_metrics['accuracy']]
        svm_values = [svm_metrics['mae'], svm_metrics['rmse'], 
                     svm_metrics['r2'], svm_metrics['accuracy']]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 7))
        bars1 = ax.bar(x - width/2, cnn_values, width, label='CNN', 
                      color='#3498db', edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x + width/2, svm_values, width, label='SVM', 
                      color='#e74c3c', edgecolor='black', linewidth=1.5)
        
        ax.set_xlabel('Metrics', fontsize=13, fontweight='bold')
        ax.set_ylabel('Values', fontsize=13, fontweight='bold')
        ax.set_title('Model Performance Comparison', fontsize=15, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison saved to {save_path}")
        plt.close()
    
    def plot_accuracy_comparison(self, cnn_train_acc, cnn_val_acc, 
                                svm_train_acc, svm_val_acc, 
                                save_name='accuracy_comparison.png'):
        """Plot accuracy comparison"""
        models = ['CNN', 'SVM']
        train_accs = [cnn_train_acc, svm_train_acc]
        val_accs = [cnn_val_acc, svm_val_acc]
        
        x = np.arange(len(models))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 7))
        bars1 = ax.bar(x - width/2, train_accs, width, label='Training',
                      color='#2ecc71', edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x + width/2, val_accs, width, label='Validation',
                      color='#f39c12', edgecolor='black', linewidth=1.5)
        
        ax.set_xlabel('Model', fontsize=13, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
        ax.set_title('Training vs Validation Accuracy', fontsize=15, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=12)
        ax.set_ylim([0, 1.1])
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Accuracy comparison saved to {save_path}")
        plt.close()
    
    def plot_prediction_scatter(self, y_true, y_pred_cnn, y_pred_svm, 
                               save_name='prediction_scatter.png'):
        """Plot scatter plots"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        axes[0].scatter(y_true, y_pred_cnn, alpha=0.6, s=50, edgecolor='black')
        axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                    'r--', linewidth=2, label='Perfect')
        axes[0].set_xlabel('True BMD', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Predicted BMD', fontsize=12, fontweight='bold')
        axes[0].set_title('CNN Predictions', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].scatter(y_true, y_pred_svm, alpha=0.6, s=50, edgecolor='black', color='orange')
        axes[1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                    'r--', linewidth=2, label='Perfect')
        axes[1].set_xlabel('True BMD', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Predicted BMD', fontsize=12, fontweight='bold')
        axes[1].set_title('SVM Predictions', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction scatter saved to {save_path}")
        plt.close()
    
    def plot_residuals(self, y_true, y_pred_cnn, y_pred_svm, save_name='residuals.png'):
        """Plot residual analysis"""
        residuals_cnn = y_true - y_pred_cnn
        residuals_svm = y_true - y_pred_svm
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        axes[0, 0].scatter(y_pred_cnn, residuals_cnn, alpha=0.6, s=40)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('Predicted BMD', fontsize=11, fontweight='bold')
        axes[0, 0].set_ylabel('Residuals', fontsize=11, fontweight='bold')
        axes[0, 0].set_title('CNN Residuals', fontsize=13, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].hist(residuals_cnn, bins=30, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('Residuals', fontsize=11, fontweight='bold')
        axes[0, 1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[0, 1].set_title('CNN Residuals Distribution', fontsize=13, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].scatter(y_pred_svm, residuals_svm, alpha=0.6, s=40, color='orange')
        axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[1, 0].set_xlabel('Predicted BMD', fontsize=11, fontweight='bold')
        axes[1, 0].set_ylabel('Residuals', fontsize=11, fontweight='bold')
        axes[1, 0].set_title('SVM Residuals', fontsize=13, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].hist(residuals_svm, bins=30, edgecolor='black', alpha=0.7, color='orange')
        axes[1, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('Residuals', fontsize=11, fontweight='bold')
        axes[1, 1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[1, 1].set_title('SVM Residuals Distribution', fontsize=13, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Residuals saved to {save_path}")
        plt.close()
    
    def create_summary_report(self, cnn_metrics, svm_metrics, save_name='summary_report.txt'):
        """Create text summary"""
        report = []
        report.append("="*60)
        report.append("BMD PREDICTION - MODEL PERFORMANCE SUMMARY")
        report.append("="*60)
        report.append("")
        
        report.append("CNN MODEL METRICS:")
        report.append("-" * 40)
        report.append(f"MAE:      {cnn_metrics['mae']:.4f}")
        report.append(f"RMSE:     {cnn_metrics['rmse']:.4f}")
        report.append(f"R²:       {cnn_metrics['r2']:.4f}")
        report.append(f"Accuracy: {cnn_metrics['accuracy']:.4f}")
        report.append("")
        
        report.append("SVM MODEL METRICS:")
        report.append("-" * 40)
        report.append(f"MAE:      {svm_metrics['mae']:.4f}")
        report.append(f"RMSE:     {svm_metrics['rmse']:.4f}")
        report.append(f"R²:       {svm_metrics['r2']:.4f}")
        report.append(f"Accuracy: {svm_metrics['accuracy']:.4f}")
        report.append("")
        
        report.append("="*60)
        
        report_text = "\n".join(report)
        
        save_path = os.path.join(self.save_dir, save_name)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\nSummary saved to {save_path}")

# ================================================================================
# SECTION 7: MAIN EXECUTION
# ================================================================================

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    print("\n" + "="*70)
    print(" "*15 + "BMD PREDICTION PROJECT")
    print(" "*10 + "CNN + SVM Implementation")
    print("="*70)
    
    # Set seed
    set_seed(Config.RANDOM_SEED)
    
    # Create directories
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    os.makedirs(Config.PLOTS_DIR, exist_ok=True)
    
    print(f"\nDevice: {Config.DEVICE}")
    print(f"Output directory: {Config.OUTPUT_DIR}")
    
    # Load data
    print("\n" + "="*70)
    print("STEP 1: Loading Data")
    print("="*70)
    
    train_loader, val_loader, test_loader = create_dataloaders()
    
    print(f"\n✓ Data loaded!")
    print(f"  Training: {len(train_loader.dataset)}")
    print(f"  Validation: {len(val_loader.dataset)}")
    print(f"  Test: {len(test_loader.dataset)}")
    
    # Train CNN
    print("\n" + "="*70)
    print("STEP 2: Training CNN Model")
    print("="*70)
    
    cnn_model = CNNWithMetadata(pretrained=True, freeze_backbone=Config.FREEZE_BACKBONE)
    print(f"\n✓ CNN model created")
    
    # Calculate class weights to handle imbalanced dataset
    class_weights = calculate_class_weights(Config.TRAIN_CSV)
    
    cnn_trainer = CNNTrainer(cnn_model, train_loader, val_loader, class_weights=class_weights)
    cnn_save_path = os.path.join(Config.MODEL_DIR, 'best_cnn_model.pth')
    cnn_history = cnn_trainer.train(num_epochs=Config.NUM_EPOCHS, save_path=cnn_save_path)
    
    cnn_val_results = cnn_trainer.validate()
    
    print("\n✓ CNN Training Complete!")
    print(f"  Best val loss: {cnn_trainer.best_val_loss:.4f}")
    print(f"  Final val MAE: {cnn_val_results['mae']:.4f}")
    print(f"  Final val accuracy: {cnn_val_results['accuracy']:.4f}")
    
    # Train SVM
    print("\n" + "="*70)
    print("STEP 3: Training SVM with CNN Features")
    print("="*70)
    
    svm_model = SVMWithCNNFeatures(cnn_model)
    svm_train_metrics, svm_val_metrics = svm_model.train(train_loader, val_loader)

    # Blending: learn ridge on validation combining CNN and SVM calibrated preds
    try:
        y_val_true = np.array(cnn_val_results['bmd_true'])
        cnn_val_pred = np.array(cnn_val_results['bmd_pred']).reshape(-1, 1)
        if cnn_trainer.ridge_calibrator is not None:
            cnn_val_pred = cnn_trainer.ridge_calibrator.predict(cnn_val_pred).reshape(-1, 1)
        elif hasattr(cnn_trainer, 'calib_a') and hasattr(cnn_trainer, 'calib_b'):
            cnn_val_pred = (cnn_trainer.calib_a * cnn_val_pred + cnn_trainer.calib_b)
        
        svm_val_pred = np.array(svm_val_metrics['bmd_pred']).reshape(-1, 1)
        X_blend = np.hstack([cnn_val_pred, svm_val_pred])
        blender = Ridge(alpha=1e-3)
        blender.fit(X_blend, y_val_true)
        blend_val_pred = blender.predict(X_blend)
        blend_mae = mean_absolute_error(y_val_true, blend_val_pred)
        blend_rmse = np.sqrt(mean_squared_error(y_val_true, blend_val_pred))
        blend_r2 = r2_score(y_val_true, blend_val_pred)
        print(f"\nBlended (CNN+SVM) on validation -> MAE: {blend_mae:.4f} | RMSE: {blend_rmse:.4f} | R²: {blend_r2:.4f}")
    except Exception as e:
        print(f"Blending failed: {e}")
        blender = None
    
    svm_save_path = os.path.join(Config.MODEL_DIR, 'svm_model.pkl')
    svm_model.save(svm_save_path)
    
    print("\n✓ SVM Training Complete!")
    
    # Visualizations
    print("\n" + "="*70)
    print("STEP 4: Generating Visualizations")
    print("="*70)
    
    visualizer = Visualizer(Config.PLOTS_DIR)
    
    print("\n1. Training history...")
    visualizer.plot_training_history(cnn_history)
    
    print("2. Confusion matrices...")
    visualizer.plot_confusion_matrix(
        cnn_val_results['class_true'], 
        cnn_val_results['class_pred'],
        'CNN', 'confusion_matrix_cnn.png'
    )
    
    visualizer.plot_confusion_matrix(
        svm_val_metrics['class_true'],
        svm_val_metrics['class_pred'],
        'SVM', 'confusion_matrix_svm.png'
    )
    
    print("3. ROC curves...")
    svm_bmd_normalized = (svm_val_metrics['bmd_pred'] - svm_val_metrics['bmd_pred'].min()) / \
                         (svm_val_metrics['bmd_pred'].max() - svm_val_metrics['bmd_pred'].min())
    
    visualizer.plot_roc_curve(
        cnn_val_results['class_true'],
        cnn_val_results['class_probs'],
        svm_bmd_normalized
    )
    
    print("4. Model comparison...")
    visualizer.plot_model_comparison_bars(cnn_val_results, svm_val_metrics)
    
    print("5. Accuracy comparison...")
    cnn_train_acc = cnn_history['train_accuracy'][-1]
    cnn_val_acc = cnn_history['val_accuracy'][-1]
    svm_train_acc = svm_train_metrics['accuracy']
    svm_val_acc = svm_val_metrics['accuracy']
    
    visualizer.plot_accuracy_comparison(
        cnn_train_acc, cnn_val_acc,
        svm_train_acc, svm_val_acc
    )
    
    print("6. Prediction scatter...")
    visualizer.plot_prediction_scatter(
        np.array(cnn_val_results['bmd_true']),
        np.array(cnn_val_results['bmd_pred']),
        svm_val_metrics['bmd_pred']
    )
    
    print("7. Residuals...")
    visualizer.plot_residuals(
        np.array(cnn_val_results['bmd_true']),
        np.array(cnn_val_results['bmd_pred']),
        svm_val_metrics['bmd_pred']
    )
    
    print("8. Summary report...")
    visualizer.create_summary_report(cnn_val_results, svm_val_metrics)
    
    # Generate Classification Reports
    print("9. Generating classification reports...")
    
    # CNN Classification Report
    cnn_class_report = classification_report(
        cnn_val_results['class_true'],
        cnn_val_results['class_pred'],
        target_names=['Low BMD', 'Normal'],
        digits=4
    )
    
    cnn_report_path = os.path.join(Config.RESULTS_DIR, 'cnn_classification_report.txt')
    with open(cnn_report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("CNN MODEL - CLASSIFICATION REPORT\n")
        f.write("="*70 + "\n\n")
        f.write("Task: Binary Classification (Normal vs Low BMD)\n")
        f.write("Threshold: T-score = -1.0\n")
        f.write(f"Dataset: Validation set ({len(cnn_val_results['class_true'])} samples)\n\n")
        f.write(cnn_class_report)
        f.write("\n" + "="*70 + "\n")
        f.write("DETAILED METRICS EXPLANATION\n")
        f.write("="*70 + "\n\n")
        f.write("Precision: Of all predicted positives, how many were correct?\n")
        f.write("Recall:    Of all actual positives, how many did we find?\n")
        f.write("F1-Score:  Harmonic mean of precision and recall\n")
        f.write("Support:   Number of actual occurrences in the dataset\n\n")
        
        # Additional metrics
        cnn_precision = precision_score(cnn_val_results['class_true'], 
                                        cnn_val_results['class_pred'], average='weighted')
        cnn_recall = recall_score(cnn_val_results['class_true'], 
                                  cnn_val_results['class_pred'], average='weighted')
        cnn_f1 = f1_score(cnn_val_results['class_true'], 
                         cnn_val_results['class_pred'], average='weighted')
        
        f.write("WEIGHTED AVERAGES:\n")
        f.write(f"  Precision: {cnn_precision:.4f}\n")
        f.write(f"  Recall:    {cnn_recall:.4f}\n")
        f.write(f"  F1-Score:  {cnn_f1:.4f}\n")
        f.write(f"  Accuracy:  {cnn_val_results['accuracy']:.4f}\n")
    
    print(f"   ✓ CNN classification report: {cnn_report_path}")
    
    # SVM Classification Report
    svm_class_report = classification_report(
        svm_val_metrics['class_true'],
        svm_val_metrics['class_pred'],
        target_names=['Low BMD', 'Normal'],
        digits=4
    )
    
    svm_report_path = os.path.join(Config.RESULTS_DIR, 'svm_classification_report.txt')
    with open(svm_report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("SVM MODEL - CLASSIFICATION REPORT\n")
        f.write("="*70 + "\n\n")
        f.write("Task: Binary Classification (Normal vs Low BMD)\n")
        f.write("Threshold: T-score = -1.0\n")
        f.write(f"Dataset: Validation set ({len(svm_val_metrics['class_true'])} samples)\n\n")
        f.write(svm_class_report)
        f.write("\n" + "="*70 + "\n")
        f.write("DETAILED METRICS EXPLANATION\n")
        f.write("="*70 + "\n\n")
        f.write("Precision: Of all predicted positives, how many were correct?\n")
        f.write("Recall:    Of all actual positives, how many did we find?\n")
        f.write("F1-Score:  Harmonic mean of precision and recall\n")
        f.write("Support:   Number of actual occurrences in the dataset\n\n")
        
        # Additional metrics
        svm_precision = precision_score(svm_val_metrics['class_true'], 
                                        svm_val_metrics['class_pred'], average='weighted')
        svm_recall = recall_score(svm_val_metrics['class_true'], 
                                  svm_val_metrics['class_pred'], average='weighted')
        svm_f1 = f1_score(svm_val_metrics['class_true'], 
                         svm_val_metrics['class_pred'], average='weighted')
        
        f.write("WEIGHTED AVERAGES:\n")
        f.write(f"  Precision: {svm_precision:.4f}\n")
        f.write(f"  Recall:    {svm_recall:.4f}\n")
        f.write(f"  F1-Score:  {svm_f1:.4f}\n")
        f.write(f"  Accuracy:  {svm_val_metrics['accuracy']:.4f}\n")
    
    print(f"   ✓ SVM classification report: {svm_report_path}")
    
    # Comparison Report
    comparison_report_path = os.path.join(Config.RESULTS_DIR, 'classification_comparison.txt')
    with open(comparison_report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("CLASSIFICATION PERFORMANCE COMPARISON\n")
        f.write("CNN vs SVM for Normal vs Low BMD Classification\n")
        f.write("="*70 + "\n\n")
        
        f.write("OVERALL METRICS COMPARISON:\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Metric':<20} {'CNN':<25} {'SVM':<25}\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Accuracy':<20} {cnn_val_results['accuracy']:<25.4f} {svm_val_metrics['accuracy']:<25.4f}\n")
        f.write(f"{'Precision':<20} {cnn_precision:<25.4f} {svm_precision:<25.4f}\n")
        f.write(f"{'Recall':<20} {cnn_recall:<25.4f} {svm_recall:<25.4f}\n")
        f.write(f"{'F1-Score':<20} {cnn_f1:<25.4f} {svm_f1:<25.4f}\n")
        f.write("-"*70 + "\n\n")
        
        # Per-class comparison
        cnn_cm = confusion_matrix(cnn_val_results['class_true'], cnn_val_results['class_pred'])
        svm_cm = confusion_matrix(svm_val_metrics['class_true'], svm_val_metrics['class_pred'])
        
        f.write("PER-CLASS PERFORMANCE:\n")
        f.write("-"*70 + "\n")
        
        # Low BMD class
        f.write("\nLow BMD Class (T-score < -1.0):\n")
        cnn_low_precision = cnn_cm[0,0] / (cnn_cm[0,0] + cnn_cm[1,0]) if (cnn_cm[0,0] + cnn_cm[1,0]) > 0 else 0
        cnn_low_recall = cnn_cm[0,0] / (cnn_cm[0,0] + cnn_cm[0,1]) if (cnn_cm[0,0] + cnn_cm[0,1]) > 0 else 0
        svm_low_precision = svm_cm[0,0] / (svm_cm[0,0] + svm_cm[1,0]) if (svm_cm[0,0] + svm_cm[1,0]) > 0 else 0
        svm_low_recall = svm_cm[0,0] / (svm_cm[0,0] + svm_cm[0,1]) if (svm_cm[0,0] + svm_cm[0,1]) > 0 else 0
        
        f.write(f"  CNN - Precision: {cnn_low_precision:.4f}, Recall: {cnn_low_recall:.4f}\n")
        f.write(f"  SVM - Precision: {svm_low_precision:.4f}, Recall: {svm_low_recall:.4f}\n")
        
        # Normal class
        f.write("\nNormal Class (T-score ≥ -1.0):\n")
        cnn_norm_precision = cnn_cm[1,1] / (cnn_cm[0,1] + cnn_cm[1,1]) if (cnn_cm[0,1] + cnn_cm[1,1]) > 0 else 0
        cnn_norm_recall = cnn_cm[1,1] / (cnn_cm[1,0] + cnn_cm[1,1]) if (cnn_cm[1,0] + cnn_cm[1,1]) > 0 else 0
        svm_norm_precision = svm_cm[1,1] / (svm_cm[0,1] + svm_cm[1,1]) if (svm_cm[0,1] + svm_cm[1,1]) > 0 else 0
        svm_norm_recall = svm_cm[1,1] / (svm_cm[1,0] + svm_cm[1,1]) if (svm_cm[1,0] + svm_cm[1,1]) > 0 else 0
        
        f.write(f"  CNN - Precision: {cnn_norm_precision:.4f}, Recall: {cnn_norm_recall:.4f}\n")
        f.write(f"  SVM - Precision: {svm_norm_precision:.4f}, Recall: {svm_norm_recall:.4f}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("CONFUSION MATRICES:\n")
        f.write("="*70 + "\n\n")
        
        f.write("CNN Confusion Matrix:\n")
        f.write(f"                Predicted Low BMD    Predicted Normal\n")
        f.write(f"Actual Low BMD       {cnn_cm[0,0]:<18} {cnn_cm[0,1]:<18}\n")
        f.write(f"Actual Normal        {cnn_cm[1,0]:<18} {cnn_cm[1,1]:<18}\n\n")
        
        f.write("SVM Confusion Matrix:\n")
        f.write(f"                Predicted Low BMD    Predicted Normal\n")
        f.write(f"Actual Low BMD       {svm_cm[0,0]:<18} {svm_cm[0,1]:<18}\n")
        f.write(f"Actual Normal        {svm_cm[1,0]:<18} {svm_cm[1,1]:<18}\n\n")
        
        # Winner
        f.write("="*70 + "\n")
        f.write("BEST CLASSIFIER:\n")
        f.write("="*70 + "\n")
        
        if cnn_val_results['accuracy'] > svm_val_metrics['accuracy']:
            winner = "CNN"
            margin = (cnn_val_results['accuracy'] - svm_val_metrics['accuracy']) * 100
        else:
            winner = "SVM"
            margin = (svm_val_metrics['accuracy'] - cnn_val_results['accuracy']) * 100
        
        f.write(f"\nBest Classifier: {winner}\n")
        f.write(f"Accuracy Advantage: {margin:.2f}%\n")
    
    print(f"   ✓ Classification comparison: {comparison_report_path}")
    
    # Print to console
    print("\n" + "="*70)
    print("CLASSIFICATION REPORTS GENERATED")
    print("="*70)
    print("\nCNN Classification Report:")
    print(cnn_class_report)
    print("\nSVM Classification Report:")
    print(svm_class_report)
    
    print("\n✓ All visualizations and reports generated!")
    
    # Test predictions
    print("\n" + "="*70)
    print("STEP 5: Test Set Predictions")
    print("="*70)
    
    print("\n1. CNN predictions...")
    cnn_test_results = cnn_trainer.get_predictions(test_loader)
    
    print("2. SVM predictions...")
    svm_test_results = svm_model.predict(test_loader)
    
    print("\n3. Creating submission files...")
    
    # Detailed CNN submission (with analysis)
    cnn_submission_detailed = pd.DataFrame({
        'image': cnn_test_results['img_names'],
        'BMD_pred': cnn_test_results['bmd_pred'],
        'T_Score': [Config.calculate_t_score(bmd) for bmd in cnn_test_results['bmd_pred']],
        'Classification': ['Normal' if c == 1 else 'Low BMD' for c in cnn_test_results['class_pred']]
    })
    
    cnn_detailed_path = os.path.join(Config.RESULTS_DIR, 'cnn_detailed_predictions.csv')
    cnn_submission_detailed.to_csv(cnn_detailed_path, index=False)
    print(f"   ✓ CNN detailed: {cnn_detailed_path}")
    
    # Detailed SVM submission (with analysis)
    svm_submission_detailed = pd.DataFrame({
        'image': svm_test_results['img_names'],
        'BMD_pred': svm_test_results['bmd_pred'],
        'T_Score': [Config.calculate_t_score(bmd) for bmd in svm_test_results['bmd_pred']],
        'Classification': ['Normal' if c == 1 else 'Low BMD' for c in svm_test_results['class_pred']]
    })
    
    svm_detailed_path = os.path.join(Config.RESULTS_DIR, 'svm_detailed_predictions.csv')
    svm_submission_detailed.to_csv(svm_detailed_path, index=False)
    print(f"   ✓ SVM detailed: {svm_detailed_path}")
    
    # OFFICIAL KAGGLE FORMAT (exactly as required)
    # CNN submission
    kaggle_cnn = pd.DataFrame({
        'image': cnn_test_results['img_names'],
        'BMD': cnn_test_results['bmd_pred']
    })
    
    # SVM submission
    kaggle_svm = pd.DataFrame({
        'image': svm_test_results['img_names'],
        'BMD': svm_test_results['bmd_pred']
    })
    
    # Blended submission (if blender available)
    if 'blender' in locals() and blender is not None:
        try:
            X_blend_test = np.hstack([
                np.array(cnn_test_results['bmd_pred']).reshape(-1, 1),
                np.array(svm_test_results['bmd_pred']).reshape(-1, 1)
            ])  # shape (N,2)
            blend_test_pred = blender.predict(X_blend_test)
            kaggle_blend = pd.DataFrame({ 'image': cnn_test_results['img_names'], 'BMD': blend_test_pred })
            kaggle_blend_path = os.path.join(Config.RESULTS_DIR, 'submission_blend.csv')
            kaggle_blend.to_csv(kaggle_blend_path, index=False)
            print(f"   ✓ Kaggle BLEND: {kaggle_blend_path}")
        except Exception as e:
            print(f"Blended submission failed: {e}")

    # Save with official Kaggle names
    kaggle_cnn_path = os.path.join(Config.RESULTS_DIR, 'submission_cnn.csv')
    kaggle_svm_path = os.path.join(Config.RESULTS_DIR, 'submission_svm.csv')
    
    kaggle_cnn.to_csv(kaggle_cnn_path, index=False)
    kaggle_svm.to_csv(kaggle_svm_path, index=False)
    
    print(f"   ✓ Kaggle CNN: {kaggle_cnn_path}")
    print(f"   ✓ Kaggle SVM: {kaggle_svm_path}")
    
    # Also create a single "submission.csv" using the best option by validation MAE
    options = []
    options.append(('CNN', np.array(cnn_val_results['bmd_true']), np.array(cnn_val_results['bmd_pred'])))
    options.append(('SVM', np.array(cnn_val_results['bmd_true']), np.array(svm_val_metrics['bmd_pred'])))
    if 'blender' in locals() and blender is not None:
        try:
            X_blend_val = np.hstack([
                np.array(cnn_val_results['bmd_pred']).reshape(-1,1) if cnn_trainer.ridge_calibrator is None else cnn_trainer.ridge_calibrator.predict(np.array(cnn_val_results['bmd_pred']).reshape(-1,1)).reshape(-1,1),
                np.array(svm_val_metrics['bmd_pred']).reshape(-1,1)
            ])
            blend_val_pred = blender.predict(X_blend_val)
            options.append(('BLEND', np.array(cnn_val_results['bmd_true']), blend_val_pred))
        except Exception as e:
            print(f"Blend comparison failed: {e}")
    # Evaluate options on validation
    best_name = 'CNN'
    best_mae = 1e9
    for name, y_true_opt, y_pred_opt in options:
        mae_opt = mean_absolute_error(y_true_opt, y_pred_opt)
        if mae_opt < best_mae:
            best_mae = mae_opt
            best_name = name
    
    if best_name == 'BLEND' and 'blender' in locals() and blender is not None:
        X_blend_test = np.hstack([
            np.array(cnn_test_results['bmd_pred']).reshape(-1, 1),
            np.array(svm_test_results['bmd_pred']).reshape(-1, 1)
        ])
        best_model_bmd = blender.predict(X_blend_test)
        best_model_name = 'BLEND'
    elif best_name == 'SVM':
        best_model_bmd = svm_test_results['bmd_pred']
        best_model_name = 'SVM'
    else:
        best_model_bmd = cnn_test_results['bmd_pred']
        best_model_name = 'CNN'
    
    kaggle_best = pd.DataFrame({
        'image': cnn_test_results['img_names'],
        'BMD': best_model_bmd
    })
    
    kaggle_best_path = os.path.join(Config.RESULTS_DIR, 'submission.csv')
    kaggle_best.to_csv(kaggle_best_path, index=False)
    print(f"   ✓ Best model ({best_model_name}): {kaggle_best_path} SUBMIT THIS TO KAGGLE!")
    
    # Final summary
    print("\n" + "="*70)
    print("PROJECT COMPLETE!")
    print("="*70)
    
    print("\n FINAL RESULTS:")
    print("-" * 70)
    print(f"{'Metric':<25} {'CNN':<20} {'SVM':<20}")
    print("-" * 70)
    print(f"{'MAE':<25} {cnn_val_results['mae']:<20.4f} {svm_val_metrics['mae']:<20.4f}")
    print(f"{'RMSE':<25} {cnn_val_results['rmse']:<20.4f} {svm_val_metrics['rmse']:<20.4f}")
    print(f"{'R²':<25} {cnn_val_results['r2']:<20.4f} {svm_val_metrics['r2']:<20.4f}")
    print(f"{'Accuracy':<25} {cnn_val_results['accuracy']:<20.4f} {svm_val_metrics['accuracy']:<20.4f}")
    print("-" * 70)
    
    if cnn_val_results['mae'] < svm_val_metrics['mae']:
        best_model = "CNN"
    else:
        best_model = "SVM"
    
    print(f"\n Best Model (by MAE): {best_model}")
    
    # Analyze test predictions
    print("\n" + "="*70)
    print("TEST SET PREDICTIONS ANALYSIS")
    print("="*70)
    
    # Best model predictions
    best_bmd = cnn_test_results['bmd_pred'] if best_model == 'CNN' else svm_test_results['bmd_pred']
    
    # Calculate statistics
    print(f"\nBest Model ({best_model}) Test Predictions:")
    print(f"  Mean BMD:   {np.mean(best_bmd):.4f}")
    print(f"  Min BMD:    {np.min(best_bmd):.4f}")
    print(f"  Max BMD:    {np.max(best_bmd):.4f}")
    print(f"  Std Dev:    {np.std(best_bmd):.4f}")
    
    # Calculate T-scores
    t_scores = [(bmd - Config.REFERENCE_BMD) / Config.STD_DEV for bmd in best_bmd]
    normal_count = sum(1 for t in t_scores if t >= Config.T_SCORE_THRESHOLD)
    low_bmd_count = len(t_scores) - normal_count
    
    print(f"\nClassification Summary (108 test images):")
    print(f"  Normal (T-score ≥ -1.0):   {normal_count} ({normal_count/len(t_scores)*100:.1f}%)")
    print(f"  Low BMD (T-score < -1.0):  {low_bmd_count} ({low_bmd_count/len(t_scores)*100:.1f}%)")
    
    print("\n OUTPUT FILES:")
    print(f"   Models: {Config.MODEL_DIR}")
    print(f"     - best_cnn_model.pth (~95 MB)")
    print(f"     - svm_model.pkl (~2 MB)")
    print(f"\n   Plots: {Config.PLOTS_DIR}")
    print(f"     - training_history.png")
    print(f"     - confusion_matrix_cnn.png")
    print(f"     - confusion_matrix_svm.png")
    print(f"     - roc_curve.png")
    print(f"     - model_comparison.png")
    print(f"     - accuracy_comparison.png")
    print(f"     - prediction_scatter.png")
    print(f"     - residuals.png")
    print(f"\n   Results: {Config.RESULTS_DIR}")
    print(f"     - submission.csv SUBMIT THIS TO KAGGLE!")
    print(f"     - submission_cnn.csv (CNN predictions)")
    print(f"     - submission_svm.csv (SVM predictions)")
    print(f"     - cnn_detailed_predictions.csv (with T-scores)")
    print(f"     - svm_detailed_predictions.csv (with T-scores)")
    print(f"     - summary_report.txt (metrics summary)")
    print(f"     - cnn_classification_report.txt For report!")
    print(f"     - svm_classification_report.txt For report!")
    print(f"     - classification_comparison.txt For report!")
    
    print("\n" + "="*70)
    print("KAGGLE SUBMISSION INSTRUCTIONS")
    print("="*70)
    print(f"\n1. Go to the Kaggle competition page")
    print(f"2. Click 'Submit Predictions'")
    print(f"3. Upload: {os.path.join(Config.RESULTS_DIR, 'submission.csv')}")
    print(f"4. This file contains {best_model} predictions (best validation MAE)")
    print(f"5. Expected format: image,BMD (108 rows)")
    print(f"6. Leaderboard metric: Mean Absolute Error (MAE)")
    
    print("\n Tips:")
    print("   - Lower MAE is better")
    print("   - You can also submit submission_cnn.csv or submission_svm.csv separately")
    print("   - Compare leaderboard score with validation MAE")
    print(f"   - Your validation MAE: {cnn_val_results['mae'] if best_model == 'CNN' else svm_val_metrics['mae']:.4f}")
    
    print("\n✓ All tasks completed successfully!")
    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n Error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
