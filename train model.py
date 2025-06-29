import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

# 1. Get labels from directory structure
def get_labels_from_directory(csv_path, base_dir):
    df = pd.read_csv(csv_path)
    
    # Create label mapping from directory structure
    label_map = {
        'Normal': 0,
        'diabetic_retinopathy': 1,
        'glaucoma': 2
    }
    
    # Extract labels from file paths
    labels = []
    for filename in df['filename']:
        for class_name in label_map.keys():
            if class_name.lower() in filename.lower():
                labels.append(class_name)
                break
        else:
            labels.append('Unknown')
    
    df['label'] = labels
    return df[df['label'] != 'Unknown']  # Remove unclassified samples

# 2. Create and train model
def train_model(csv_path, base_dir):
    # Load data with labels
    df = get_labels_from_directory(csv_path, base_dir)
    
    # Prepare features and labels
    features = df.drop(['filename', 'label'], axis=1).values
    labels = df['label'].values
    
    # Encode labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    class_names = le.classes_
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels_encoded, 
        test_size=0.2, 
        stratify=labels_encoded,
        random_state=30
    )
    
    # Model architecture
    model = Sequential([
        Dense(512, activation='relu', input_shape=(features.shape[1],)),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(len(class_names), activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_accuracy', mode='max')
    ]
    
    # Training
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=16,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate best model on the validation set
    best_epoch = np.argmax(history.history['val_accuracy']) + 1
    _, train_acc = model.evaluate(X_train, y_train, verbose=0)
    _, val_acc = model.evaluate(X_val, y_val, verbose=0)

    print(f"Best Epoch: {best_epoch}")
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")

    return model, le

# 3. Main execution
if __name__ == "__main__":
    # Path configurations
    csv_path = r"C:\Users\abdal\Desktop\CPE 598\Shafflefeatures.csv"  # Path to your features CSV
    base_image_dir = r'C:\Users\abdal\Desktop\CPE 598\Training dataset\dataset\all'
    
    # Train model
    model, label_encoder = train_model(csv_path, base_image_dir)
    
    # Save final model and label encoder
    model.save('retinal_classifier.h5')
    np.save('label_encoder_classes.npy', label_encoder.classes_)
    print("Model and label encoder saved successfully!")