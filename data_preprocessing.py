import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_and_preprocess_data(train_path, test_path):
    # Load the training and test feature files (assuming CSV format)
    print("Loading data...")
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # Separate features (X) and labels (y) for training and testing sets
    X_train = train_data.drop('label', axis=1)
    y_train = train_data['label']
    X_test = test_data.drop('label', axis=1)
    y_test = test_data['label']

    # Combine train and test labels to fit the encoder on all possible labels
    all_labels = pd.concat([y_train, y_test])

    # Step 1: Scale the features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Step 2: Encode the labels (fit on all labels in the combined dataset)
    print("Encoding labels...")
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)  # Fit on all labels, combining train and test

    # Transform both training and testing labels
    y_train = label_encoder.transform(y_train)
    y_test = label_encoder.transform(y_test)

    return X_train, X_test, y_train, y_test, label_encoder
