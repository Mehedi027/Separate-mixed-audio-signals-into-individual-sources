import os
import pandas as pd
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Step 1: Load Metadata CSV Files
train_metadata_path = 'C:/Users/mehed/OneDrive/Desktop/RME/Sem1/Ayesha/ADSP/Dataset/Metadata_Train.csv'
test_metadata_path = 'C:/Users/mehed/OneDrive/Desktop/RME/Sem1/Ayesha/ADSP/Dataset/Metadata_Test.csv'

train_metadata = pd.read_csv(train_metadata_path)
test_metadata = pd.read_csv(test_metadata_path)

print(train_metadata.head())  # Check the first few rows
print(test_metadata.head())


# Step 2: Extract Features from Audio Files
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    # Extracting MFCC (Mel-frequency Cepstral Coefficients) features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    # Averaging the MFCCs over time
    mfcc = np.mean(mfcc.T, axis=0)
    return mfcc


# Step 3: Prepare the Dataset
def prepare_dataset(metadata, folder_path):
    features = []
    labels = []

    for index, row in metadata.iterrows():
        # Construct the full file path
        file_path = os.path.join(folder_path, row['FileName'])

        # Extract features
        mfcc = extract_features(file_path)
        features.append(mfcc)

        # Convert label to integer (you might need to map classes to integers)
        label = row['Class']
        labels.append(label)

    return np.array(features), np.array(labels)


# Prepare training and test datasets
X_train, y_train = prepare_dataset(train_metadata,
                                   'C:/Users/mehed/OneDrive/Desktop/RME/Sem1/Ayesha/ADSP/Dataset/Train_submission')
X_test, y_test = prepare_dataset(test_metadata,
                                 'C:/Users/mehed/OneDrive/Desktop/RME/Sem1/Ayesha/ADSP/Dataset/Test_submission')

print(X_train.shape, y_train.shape)  # Check the shape of the data

# Step 4: Encode Labels
encoder = LabelEncoder()

# Fit the encoder to the training labels and transform them
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)

print(np.unique(y_train_encoded))  # Check the unique class labels after encoding

# Step 5: Build the Neural Network Model
model = models.Sequential([
    layers.InputLayer(input_shape=(13,)),  # 13 MFCC coefficients
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(len(np.unique(y_train_encoded)), activation='softmax')  # Number of unique labels
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()

# Step 6: Train the Model
history = model.fit(X_train, y_train_encoded, validation_data=(X_test, y_test_encoded), epochs=10, batch_size=32)

# Plot the training and validation accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Step 7: Evaluate the Model
test_loss, test_acc = model.evaluate(X_test, y_test_encoded, verbose=2)
print(f'Test accuracy: {test_acc}')


# Step 8: Classify New Audio File
def predict_audio(file_path):
    features = extract_features(file_path).reshape(1, -1)  # Reshape for prediction
    prediction = model.predict(features)
    predicted_class = encoder.inverse_transform([np.argmax(prediction)])
    return predicted_class[0]


# Test with a new file (replace with the actual path of your local file)
new_audio_file = 'C:/Users/mehed/OneDrive/Desktop/RME/Sem1/Ayesha/ADSP/Drum_Test.wav'  # Example: 'C:/path/to/your/file.wav'
predicted_class = predict_audio(new_audio_file)
print(f'Predicted Class: {predicted_class}')
