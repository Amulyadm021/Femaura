"""
Train Tabular Model for PCOS Detection
This script trains a machine learning model using the PCOS dataset from Kaggle.
Dataset: https://www.kaggle.com/datasets/shreyasvedpathak/pcos-dataset/data

Usage:
1. Download the dataset from Kaggle and place it in the data/ directory
2. Run: python train_tabular_model.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

# Configuration
DATASET_PATH = 'data/pcos_dataset.csv'  # Update this path to your dataset
# Alternative common filenames to check:
ALTERNATIVE_PATHS = [
    'data/PCOS_data.csv',
    'data/PCOS_data_without_infertility.csv',
    'data/pcos_data.csv',
    'PCOS_data.csv',
    'PCOS_data_without_infertility.csv'
]
MODEL_PATH = 'pcos_tabular_model.pkl'
SCALER_PATH = 'pcos_scaler.pkl'
RANDOM_STATE = 42
TEST_SIZE = 0.2

def load_and_preprocess_data(csv_path):
    """Load and preprocess the PCOS dataset"""
    print("Loading dataset...")
    
    # Try to find the dataset in multiple locations
    actual_path = None
    if os.path.exists(csv_path):
        actual_path = csv_path
    else:
        # Try alternative paths
        for alt_path in ALTERNATIVE_PATHS:
            if os.path.exists(alt_path):
                actual_path = alt_path
                print(f"Found dataset at alternative location: {alt_path}")
                break
    
    if actual_path is None:
        print("\n" + "="*60)
        print("ERROR: Dataset not found!")
        print("="*60)
        print(f"\nSearched for dataset at:")
        print(f"  - {csv_path}")
        for alt_path in ALTERNATIVE_PATHS:
            print(f"  - {alt_path}")
        print("\n" + "="*60)
        print("SOLUTIONS:")
        print("="*60)
        print("\nOption 1: Download from Kaggle")
        print("  1. Go to: https://www.kaggle.com/datasets/shreyasvedpathak/pcos-dataset/data")
        print("  2. Download the dataset")
        print("  3. Extract the CSV file")
        print(f"  4. Place it as: {csv_path}")
        print("\nOption 2: Create sample dataset for testing")
        print("  Run: python create_sample_dataset.py")
        print("  This creates a minimal test dataset")
        print("\n" + "="*60)
        raise FileNotFoundError(
            f"Dataset not found. Please download from Kaggle or create sample dataset."
        )
    
    csv_path = actual_path
    
    # Load the dataset
    df = pd.read_csv(csv_path)
    
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nColumn names:\n{df.columns.tolist()}")
    
    # Display first few rows
    print("\nFirst few rows:")
    print(df.head())
    
    # Check for target variable (PCOS status)
    # Common column names for PCOS status:
    target_cols = ['PCOS (Y/N)', 'PCOS', 'pcos', 'PCOS(Y/N)', 'PCOS(Y/N)']
    target_col = None
    
    for col in target_cols:
        if col in df.columns:
            target_col = col
            break
    
    if target_col is None:
        # Try to find any column with PCOS in name
        pcos_cols = [col for col in df.columns if 'pcos' in col.lower()]
        if pcos_cols:
            target_col = pcos_cols[0]
            print(f"\nUsing target column: {target_col}")
        else:
            raise ValueError(
                "Could not find PCOS target column. "
                "Please ensure the dataset has a column indicating PCOS status."
            )
    
    # Separate features and target
    # Exclude non-feature columns
    exclude_cols = [
        target_col, 'Patient File No.', 'Patient File No', 'Sl. No', 'Sl No',
        'Unnamed: 0', 'index'
    ]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Handle missing values
    print("\nHandling missing values...")
    missing_before = X.isnull().sum().sum()
    if missing_before > 0:
        print(f"Found {missing_before} missing values")
        # Fill numeric columns with median
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
        
        # Fill categorical columns with mode
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if X[col].isnull().sum() > 0:
                X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 'Unknown')
    
    # Encode categorical variables
    print("\nEncoding categorical variables...")
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        # Try label encoding first
        unique_vals = X[col].unique()
        if len(unique_vals) <= 10:  # If few unique values, use label encoding
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        else:
            # For many categories, drop or use target encoding
            print(f"Dropping column {col} due to too many categories ({len(unique_vals)})")
            X = X.drop(columns=[col])
    
    # Convert target to binary (0/1)
    if y.dtype == 'object':
        y = y.map({'Yes': 1, 'Y': 1, 'yes': 1, 'No': 0, 'N': 0, 'no': 0, 1: 1, 0: 0})
    y = y.astype(int)
    
    # Remove any remaining non-numeric columns
    X = X.select_dtypes(include=[np.number])
    
    print(f"\nFinal feature set: {X.shape[1]} features")
    print(f"Feature names: {X.columns.tolist()}")
    print(f"\nTarget distribution:")
    print(y.value_counts())
    
    return X, y, feature_cols

def train_model(X, y):
    """Train a Random Forest classifier"""
    print("\n" + "="*50)
    print("Training Model...")
    print("="*50)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    print("\nTraining Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    print("\nEvaluating model...")
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    print(f"\nTraining Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_pred_test, target_names=['No PCOS', 'PCOS']))
    
    print("\nConfusion Matrix (Test Set):")
    print(confusion_matrix(y_test, y_pred_test))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    return model, scaler, X.columns.tolist()

def save_model(model, scaler, feature_names, model_path, scaler_path):
    """Save the trained model and scaler"""
    print(f"\nSaving model to {model_path}...")
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'feature_names': feature_names,
            'scaler': scaler
        }, f)
    
    print(f"Model saved successfully!")
    print(f"Feature names saved: {len(feature_names)} features")
    
    # Also save scaler separately for convenience
    print(f"\nSaving scaler to {scaler_path}...")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print("Scaler saved successfully!")

def main():
    """Main training function"""
    print("="*50)
    print("PCOS Tabular Model Training")
    print("="*50)
    
    try:
        # Load and preprocess data
        X, y, feature_cols = load_and_preprocess_data(DATASET_PATH)
        
        # Train model
        model, scaler, feature_names = train_model(X, y)
        
        # Save model
        save_model(model, scaler, feature_names, MODEL_PATH, SCALER_PATH)
        
        print("\n" + "="*50)
        print("Training Complete!")
        print("="*50)
        print(f"\nModel saved to: {MODEL_PATH}")
        print(f"Scaler saved to: {SCALER_PATH}")
        print("\nYou can now use this model in your Flask application.")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())

