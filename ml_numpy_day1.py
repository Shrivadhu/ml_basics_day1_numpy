"""
DAY 1: ML FUNDAMENTALS - NumPy & Pandas
Master the tools that power 99% of ML code

Author: Your ML Teacher
Goal: Build foundation for real ML projects
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("=" * 60)
print("LESSON 1: NUMPY - The Math Engine of ML")
print("=" * 60)

# ============================================================================
# PART 1: NUMPY BASICS - Why we need it
# ============================================================================
"""
Why NumPy?
- 50x faster than Python lists
- All ML libraries (scikit-learn, TensorFlow, PyTorch) use NumPy
- Handles matrix operations needed for ML algorithms
"""

# Creating arrays (the ML data structure)
python_list = [1, 2, 3, 4, 5]
numpy_array = np.array([1, 2, 3, 4, 5])

print("\n1. ARRAYS - Your ML Data Container")
print(f"Python List: {python_list}")
print(f"NumPy Array: {numpy_array}")
print(f"Array shape: {numpy_array.shape}")
print(f"Array type: {numpy_array.dtype}")

# Multi-dimensional arrays (how images, data tables work)
matrix_2d = np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])

print("\n2. 2D ARRAYS - Like a spreadsheet or image")
print(matrix_2d)
print(f"Shape: {matrix_2d.shape} (3 rows, 3 columns)")

# Real ML example: Image representation
# An image is just a matrix of pixel values!
fake_image = np.random.randint(0, 255, size=(28, 28))  # 28x28 grayscale image
print(f"\n3. IMAGE AS ARRAY")
print(f"Image shape: {fake_image.shape}")
print(f"First 5x5 pixels:\n{fake_image[:5, :5]}")

# ============================================================================
# PART 2: NUMPY OPERATIONS - The ML Math
# ============================================================================

print("\n" + "=" * 60)
print("NUMPY OPERATIONS - Math that Powers ML")
print("=" * 60)

# Element-wise operations (super fast!)
arr = np.array([1, 2, 3, 4, 5])

print("\n4. VECTORIZED OPERATIONS (Why NumPy is fast)")
print(f"Original: {arr}")
print(f"Add 10: {arr + 10}")
print(f"Multiply by 2: {arr * 2}")
print(f"Square: {arr ** 2}")

# Matrix operations (core of neural networks!)
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

print("\n5. MATRIX OPERATIONS - Neural Network Math")
print(f"Matrix A:\n{A}")
print(f"Matrix B:\n{B}")
print(f"\nMatrix Multiplication (A @ B):\n{A @ B}")
print("^ This is how neural networks process data!")

# Statistical operations (feature engineering)
data = np.random.randn(1000)  # Generate 1000 random numbers

print("\n6. STATISTICS - Understanding Your Data")
print(f"Mean: {data.mean():.2f}")
print(f"Std Dev: {data.std():.2f}")
print(f"Min: {data.min():.2f}")
print(f"Max: {data.max():.2f}")
print(f"Median: {np.median(data):.2f}")

# ============================================================================
# PART 3: PANDAS - Real World Data Handling
# ============================================================================

print("\n" + "=" * 60)
print("LESSON 2: PANDAS - Working with Real Data")
print("=" * 60)

# Creating a dataset (like what you'd get from a company)
data_dict = {
    'customer_id': [1, 2, 3, 4, 5, 6, 7, 8],
    'age': [25, 35, 45, 23, 34, 56, 28, 41],
    'income': [50000, 75000, 90000, 45000, 65000, 120000, 55000, 80000],
    'purchases': [2, 5, 8, 1, 4, 12, 3, 7],
    'region': ['North', 'South', 'North', 'West', 'South', 'North', 'West', 'South']
}

df = pd.DataFrame(data_dict)

print("\n7. DATAFRAME - Your ML Dataset")
print(df)
print(f"\nShape: {df.shape} (8 customers, 5 features)")

# Essential Pandas operations for ML
print("\n8. DATA EXPLORATION - First Step in Any ML Project")
print("\nBasic Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\n9. DATA SELECTION - Getting What You Need")
print("\nSelect 'age' column:")
print(df['age'].head())

print("\nSelect customers over 30:")
print(df[df['age'] > 30])

print("\nSelect multiple columns:")
print(df[['age', 'income']].head())

# ============================================================================
# PART 4: DATA CLEANING - The Real ML Work
# ============================================================================

print("\n" + "=" * 60)
print("DATA CLEANING - 80% of ML Job Is This!")
print("=" * 60)

# Create messy data (real world is always messy!)
messy_data = {
    'name': ['Alice', 'Bob', 'Charlie', None, 'Eve'],
    'score': [85, 90, None, 78, 92],
    'grade': ['A', 'A', 'B', 'C', 'A'],
    'attendance': [95, 88, None, 92, 97]
}

messy_df = pd.DataFrame(messy_data)

print("\n10. HANDLING MISSING DATA")
print("Original (with missing values):")
print(messy_df)

print("\nMissing values per column:")
print(messy_df.isnull().sum())

# Fill missing values
messy_df['score'].fillna(messy_df['score'].mean(), inplace=True)
messy_df['attendance'].fillna(messy_df['attendance'].mean(), inplace=True)
messy_df.dropna(inplace=True)  # Remove rows with any missing values

print("\nCleaned data:")
print(messy_df)

# ============================================================================
# PART 5: FEATURE ENGINEERING - Creating ML Features
# ============================================================================

print("\n" + "=" * 60)
print("FEATURE ENGINEERING - Making Data ML-Ready")
print("=" * 60)

# Going back to our customer data
print("\n11. CREATING NEW FEATURES")
print("Original features:")
print(df[['income', 'purchases']].head())

# Create new feature: average purchase value
df['avg_purchase_value'] = df['income'] / df['purchases']
print("\nNew feature - Average Purchase Value:")
print(df[['income', 'purchases', 'avg_purchase_value']].head())

# Encoding categorical variables (ML needs numbers!)
print("\n12. ENCODING CATEGORIES - ML Only Understands Numbers")
print("Original regions:", df['region'].unique())

# One-hot encoding
df_encoded = pd.get_dummies(df, columns=['region'], prefix='region')
print("\nAfter encoding:")
print(df_encoded.head())

# ============================================================================
# PRACTICAL EXERCISE: Mini ML Preprocessing Pipeline
# ============================================================================

print("\n" + "=" * 60)
print("HANDS-ON EXERCISE: Complete ML Preprocessing")
print("=" * 60)

# Simulate real customer data
np.random.seed(42)
n_samples = 100

customer_data = pd.DataFrame({
    'age': np.random.randint(18, 70, n_samples),
    'income': np.random.randint(30000, 150000, n_samples),
    'credit_score': np.random.randint(300, 850, n_samples),
    'num_products': np.random.randint(1, 6, n_samples),
    'is_active': np.random.choice([0, 1], n_samples),
    'will_churn': np.random.choice([0, 1], n_samples)  # Target variable
})

print("\n13. REAL ML DATASET - Customer Churn Prediction")
print(customer_data.head(10))

print("\nDataset statistics:")
print(customer_data.describe())

# Separate features (X) and target (y) - Standard ML practice
X = customer_data.drop('will_churn', axis=1)
y = customer_data['will_churn']

print("\n14. SPLITTING FEATURES AND TARGET")
print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")
print(f"\nFirst 5 samples:")
print("Features:")
print(X.head())
print("\nTarget:")
print(y.head())

# Feature scaling (essential for many ML algorithms!)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

print("\n15. FEATURE SCALING - Making All Features Comparable")
print("Before scaling:")
print(X.head())
print("\nAfter scaling (standardized):")
print(X_scaled_df.head())

# ============================================================================
# KEY TAKEAWAYS
# ============================================================================

print("\n" + "=" * 60)
print("KEY TAKEAWAYS - What You Learned")
print("=" * 60)

takeaways = """
✓ NumPy: Fast array operations, matrix math for ML
✓ Pandas: Loading, exploring, cleaning real-world data
✓ Data Cleaning: Handling missing values (80% of the job!)
✓ Feature Engineering: Creating useful features from raw data
✓ Encoding: Converting categories to numbers
✓ Scaling: Normalizing features for ML algorithms
✓ X/y split: Separating features from target variable

NEXT STEPS:
1. Practice these operations on real datasets (Kaggle.com)
2. Tomorrow: Build your first ML model (Linear Regression)
3. Start thinking: "What patterns could ML find in this data?"
"""

print(takeaways)

# ============================================================================
# HOMEWORK CHALLENGE
# ============================================================================

print("=" * 60)
print("HOMEWORK CHALLENGE")
print("=" * 60)

homework = """
Create a dataset about house prices with these features:
- size (sq ft)
- bedrooms
- age
- location (categorical: urban/suburban/rural)
- price (target)

Tasks:
1. Create 50 samples using NumPy
2. Load into Pandas DataFrame
3. Check for missing values
4. Encode the location feature
5. Scale all numeric features
6. Split into X (features) and y (price)

This is EXACTLY what you'll do in ML interviews!
"""

print(homework)

print("\n" + "=" * 60)
print("Great job! You've completed Day 1!")
print("Run this script, understand each section, then modify it.")
print("=" * 60)