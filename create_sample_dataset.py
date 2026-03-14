"""
Create a Sample PCOS Dataset for Testing
This script creates a minimal sample dataset with the required columns.
NOTE: This is only for testing purposes. For real predictions, use the actual Kaggle dataset.
"""

import pandas as pd
import numpy as np
import os

def create_sample_dataset():
    """Create a sample dataset with required columns"""
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Define sample data (100 rows for testing)
    n_samples = 100
    
    # Generate sample data with realistic PCOS features
    np.random.seed(42)
    
    data = {
        # Basic Information
        'Age (yrs)': np.random.randint(18, 45, n_samples),
        'Weight (Kg)': np.random.uniform(45, 100, n_samples),
        'Height(Cm) ': np.random.uniform(150, 180, n_samples),
        
        # Menstrual Cycle
        'Cycle(R/I)': np.random.choice([0, 1], n_samples),  # 0=Irregular, 1=Regular
        'Cycle length(days)': np.random.uniform(20, 40, n_samples),
        
        # Lifestyle
        'Fast food (Y/N)': np.random.choice([0, 1], n_samples),
        'Pregnant(Y/N)': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'No. of aborts': np.random.randint(0, 3, n_samples),
        
        # Ultrasound Features (if available)
        'Follicle No. (R)': np.random.randint(5, 25, n_samples),
        'Follicle No. (L)': np.random.randint(5, 25, n_samples),
        
        # Symptoms
        'Hair growth(Y/N)': np.random.choice([0, 1], n_samples),
        'Skin darkening (Y/N)': np.random.choice([0, 1], n_samples),
        'Hair loss(Y/N)': np.random.choice([0, 1], n_samples),
        'Pimples(Y/N)': np.random.choice([0, 1], n_samples),
        'Weight gain(Y/N)': np.random.choice([0, 1], n_samples),
        
        # Lifestyle
        'Exercise(Y/N)': np.random.choice([0, 1], n_samples),
        'Regular exercise(Y/N)': np.random.choice([0, 1], n_samples),
        
        # Target Variable (PCOS - 0=No PCOS, 1=PCOS)
        'PCOS (Y/N)': np.random.choice([0, 1], n_samples, p=[0.5, 0.5])
    }
    
    # Calculate BMI
    data['BMI'] = data['Weight (Kg)'] / ((data['Height(Cm) '] / 100) ** 2)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    output_path = 'data/pcos_dataset.csv'
    df.to_csv(output_path, index=False)
    
    print("="*50)
    print("Sample Dataset Created Successfully!")
    print("="*50)
    print(f"\nDataset saved to: {output_path}")
    print(f"Number of samples: {n_samples}")
    print(f"Number of features: {len(df.columns) - 1}")  # Excluding target
    print(f"Target variable: PCOS (Y/N)")
    print("\nColumn names:")
    print(df.columns.tolist())
    print("\nFirst few rows:")
    print(df.head())
    print("\n⚠️  WARNING: This is sample/test data only!")
    print("For accurate predictions, download the real dataset from Kaggle:")
    print("https://www.kaggle.com/datasets/shreyasvedpathak/pcos-dataset/data")
    print("\nYou can now run: python train_tabular_model.py")

if __name__ == '__main__':
    try:
        create_sample_dataset()
    except Exception as e:
        print(f"Error creating sample dataset: {e}")
        import traceback
        traceback.print_exc()



