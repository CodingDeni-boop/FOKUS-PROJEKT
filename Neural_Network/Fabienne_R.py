from tensorflow import keras
from tensorflow.keras import layers, regularizers, optimizers
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Load data
X = pd.read_csv('../model/processed_features.csv', index_col=['video_id', 'frame'])
y = pd.read_csv('../model/processed_labels.csv', index_col=['video_id', 'frame'])

# Convert labels to categorical (one-hot encoding)
from sklearn.preprocessing import LabelEncoder


# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set dimensions
N_input = X_train.shape[1]
N_features = y.shape[1]


# Assuming MLData is a dictionary with parameters
# You'll need to define N_input and N_features values
# Example: N_input = 100, N_features = 4

N_input = 100

N_features = 4

model = keras.Sequential([
    layers.Dense(units=1024, activation='relu',
                 input_shape=(N_input,), kernel_regularizer=regularizers.l2(0)),
    layers.Dropout(rate=0.4),
    layers.Dense(units=256, activation='relu', kernel_regularizer=regularizers.l2(0)),
    layers.Dropout(rate=0.4),
    layers.Dense(units=N_features, activation='softmax')
])

# define optimizer
model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizers.RMSprop(),
    metrics=['f1_macro']
)