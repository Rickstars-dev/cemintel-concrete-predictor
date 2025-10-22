import pandas as pd
import numpy as np

df = pd.read_csv('data/concrete_data.csv')

print('=== DATASET STATS ===')
print('Total samples:', len(df))
print('Unique ages:', df['age'].nunique())
print('Age range:', df['age'].min(), '-', df['age'].max(), 'days')

print('\n=== STRENGTH RANGE ===')
print(f'Min: {df["compressive_strength"].min():.2f} MPa')
print(f'Max: {df["compressive_strength"].max():.2f} MPa')
print(f'Range: {df["compressive_strength"].max() - df["compressive_strength"].min():.2f} MPa')

print('\n=== MIX DIVERSITY ===')
scm_mixes = len(df[(df['blast_furnace_slag'] > 0) | (df['fly_ash'] > 0)])
print(f'Mixes with SCMs (slag/fly ash): {scm_mixes} ({scm_mixes/len(df)*100:.0f}%)')

sp_mixes = len(df[df['superplasticizer'] > 0])
print(f'Mixes with superplasticizer: {sp_mixes} ({sp_mixes/len(df)*100:.0f}%)')

print('\n=== VERIFIABLE METRICS ===')
print(f'1. Training samples: {len(df)}')
print(f'2. Age coverage: {df["age"].nunique()} unique test ages')
print(f'3. Strength spectrum: {df["compressive_strength"].max() - df["compressive_strength"].min():.0f} MPa range')
print(f'4. SCM representation: {scm_mixes/len(df)*100:.0f}% of dataset uses eco-friendly additives')
