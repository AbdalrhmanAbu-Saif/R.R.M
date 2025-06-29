import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model

# 1. Initialize pre-trained model (VGG16 without top layers)
base_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)

# 2. Image preprocessing function
def preprocess_image(img_path, target_size=(224, 224)):
    try:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        h, w = img.shape[:2]
        scale = min(target_size[0]/h, target_size[1]/w)
        new_h, new_w = int(h*scale), int(w*scale)
        resized_img = cv2.resize(img, (new_w, new_h))
        
        delta_h = target_size[0] - new_h
        delta_w = target_size[1] - new_w
        top, bottom = delta_h//2, delta_h - (delta_h//2)
        left, right = delta_w//2, delta_w - (delta_w//2)
        
        padded_img = cv2.copyMakeBorder(
            resized_img, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=[0,0,0]
        )
        
        padded_img = preprocess_input(padded_img)
        return padded_img
    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")
        return None

# 3. Feature extraction function
def extract_features(img_path):
    img = preprocess_image(img_path)
    if img is None:
        return None
    img = np.expand_dims(img, axis=0)
    features = model.predict(img, verbose=0)
    return features.flatten()

# 4. Process all images in folder
def process_images(folder_path, output_csv='features.csv'):
    features_list = []
    filenames = []
    
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        if not os.path.isfile(img_path):
            continue
            
        features = extract_features(img_path)
        if features is not None:
            features_list.append(features)
            filenames.append(img_name)
            print(f"Processed: {img_name}")
    
    df = pd.DataFrame(features_list)
    df.insert(0, 'filename', filenames)
    
    df.to_csv(output_csv, index=False)
    print(f"Features saved to {output_csv}")

# 5. Main execution with hardcoded path
if __name__ == "__main__":
    image_directory = r'C:\Users\abdal\Desktop\CPE 598\Training dataset\dataset\all'
    output_filename = 'features.csv'
    process_images(image_directory, output_filename)