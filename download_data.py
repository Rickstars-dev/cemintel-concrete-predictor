"""
Download Concrete Compressive Strength Dataset
UCI Machine Learning Repository
"""
import pandas as pd
import os

print("=" * 60)
print("DOWNLOADING CONCRETE DATASET")
print("=" * 60)

# Create data directory
os.makedirs('data', exist_ok=True)

# Download from UCI repository
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls'

print(f"\n📥 Downloading from UCI repository...")
print(f"   URL: {url}")

try:
    # Read Excel file directly from URL
    df = pd.read_excel(url)
    
    print(f"\n✅ Dataset downloaded successfully!")
    print(f"   Samples: {len(df)}")
    print(f"   Features: {len(df.columns)}")
    
    # Rename columns for easier use
    df.columns = [
        'cement',
        'blast_furnace_slag',
        'fly_ash',
        'water',
        'superplasticizer',
        'coarse_aggregate',
        'fine_aggregate',
        'age',
        'compressive_strength'
    ]
    
    # Save to CSV
    df.to_csv('data/concrete_data.csv', index=False)
    print(f"\n💾 Saved to: data/concrete_data.csv")
    
    # Display info
    print("\n" + "=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)
    print(f"\n{df.head()}")
    print(f"\n📊 Statistics:")
    print(df.describe())
    
    print(f"\n🔍 Missing values:")
    print(df.isnull().sum())
    
    print(f"\n📈 Target variable (Compressive Strength):")
    print(f"   Min: {df['compressive_strength'].min():.2f} MPa")
    print(f"   Max: {df['compressive_strength'].max():.2f} MPa")
    print(f"   Mean: {df['compressive_strength'].mean():.2f} MPa")
    print(f"   Median: {df['compressive_strength'].median():.2f} MPa")
    
    print("\n" + "=" * 60)
    print("✅ DATASET READY FOR TRAINING!")
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ Error downloading dataset: {e}")
    print("\n💡 Alternative: Manually download from:")
    print("   https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength")
    print("   or")
    print("   https://www.kaggle.com/datasets/maajdl/yeh-concret-data")
