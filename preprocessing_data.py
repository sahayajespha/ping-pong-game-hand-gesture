import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Load dataset
file_path = "C:/Users/sahaj/OneDrive/Desktop/pp_game/dataset/hand_landmarks_dataset.csv" 
df = pd.read_csv(file_path)

# Check for missing values
print("Missing Values:\n", df.isnull().sum())

# Normalize x, y, z values between 0 and 1
scaler = MinMaxScaler()
landmark_cols = [col for col in df.columns if col.startswith(('x', 'y', 'z'))]
df[landmark_cols] = scaler.fit_transform(df[landmark_cols])

# Encode categorical variables
le_hand = LabelEncoder()
df['Hand'] = le_hand.fit_transform(df['Hand'])  # LEFT = 0, RIGHT = 1

le_gesture = LabelEncoder()
df['Gesture'] = le_gesture.fit_transform(df['Gesture'])  # Convert gesture names to numbers

# Save processed dataset
processed_file_path = "C:/Users/sahaj/OneDrive/Desktop/pp_game/dataset/hand_landmarks_preprocessed.csv"
df.to_csv(processed_file_path, index=False)
print(f"Preprocessed dataset saved at: {processed_file_path}")

# Display dataset summary
print(df.head())

import seaborn as sns
import matplotlib.pyplot as plt

# Plot boxplots for a few landmark coordinates
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[['x0', 'y0', 'z0', 'x5', 'y5', 'z5']])  # Random selection
plt.title("Boxplot of Selected Landmark Coordinates")
plt.show()

print(df['Hand'].value_counts())    # Check Left vs Right balance
print(df['Gesture'].value_counts()) # Check gesture class distribution

