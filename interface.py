import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

class RetinalDiseaseClassifier:
    def __init__(self, root):
        self.root = root
        self.root.title("Retinal Disease Classifier")
        self.root.geometry("1000x700")
        self.root.configure(bg="#f0f2f5")
        
        # Disease information database
        self.disease_info = {
            'normal': {
                'description': "Healthy retinal scan with no signs of abnormalities.",
                'symptoms': ["No microaneurysms", "Normal blood vessel structure", "Healthy optic disc"],
                'recommendation': "Continue regular eye checkups every 2 years."
            },
            'diabetic_retinopathy': {
                'description': "Diabetes-induced damage to the retinal blood vessels.",
                'symptoms': ["Microaneurysms", "Retinal hemorrhages", "Hard exudates", "Macular edema"],
                'recommendation': "1. Immediate ophthalmologist consultation\n2. Strict blood sugar control\n3. Regular fundus examinations"
            },
            'glaucoma': {
                'description': "Optic nerve damage typically associated with elevated intraocular pressure.",
                'symptoms': ["Increased eye pressure", "Optic nerve cupping", "Peripheral vision loss"],
                'recommendation': "1. Urgent specialist referral\n2. Possible pressure-lowering drops\n3. Regular visual field testing"
            }
        }

        # Load trained model and label encoder
        self.model = load_model('retinal_classifier.h5')
        self.label_encoder = np.load('label_encoder_classes.npy', allow_pickle=True)
        
        # Initialize VGG16 for feature extraction
        self.vgg_model = tf.keras.applications.VGG16(
            weights='imagenet', 
            include_top=False, 
            pooling='avg'
        )
        
        # GUI Components
        self.create_widgets()
        self.current_image_path = None

    def create_widgets(self):
        # Main container
        main_frame = tk.Frame(self.root, bg="#f0f2f5")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Left panel - Image display
        self.image_frame = tk.Frame(main_frame, bg="white", bd=2, relief=tk.GROOVE)
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.image_label = tk.Label(self.image_frame, bg="white")
        self.image_label.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Right panel - Controls and results
        control_frame = tk.Frame(main_frame, bg="#f0f2f5")
        control_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Upload section
        upload_frame = tk.Frame(control_frame, bg="#f0f2f5")
        upload_frame.pack(pady=10, fill=tk.X)
        
        self.upload_btn = ttk.Button(
            upload_frame,
            text="📁 Upload Image",
            command=self.upload_image,
            style="Primary.TButton"
        )
        self.upload_btn.pack(fill=tk.X, ipady=8)

        # Detection section
        detect_frame = tk.Frame(control_frame, bg="#f0f2f5")
        detect_frame.pack(pady=10, fill=tk.X)
        
        self.detect_btn = ttk.Button(
            detect_frame,
            text="🔍 Detect Disease",
            command=self.detect_image,
            style="Success.TButton",
            state=tk.DISABLED
        )
        self.detect_btn.pack(fill=tk.X, ipady=8)

        # Results panel
        results_frame = tk.Frame(control_frame, bg="white", bd=2, relief=tk.GROOVE)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        tk.Label(results_frame, 
                text="Analysis Results", 
                font=("Arial", 12, "bold"), 
                bg="white").pack(pady=10)

        # Diagnosis labels
        diagnosis_container = tk.Frame(results_frame, bg="white")
        diagnosis_container.pack(fill=tk.X, padx=10)
        
        self.diagnosis_label = tk.Label(
            diagnosis_container,
            text="Diagnosis: N/A",
            font=("Arial", 14),
            bg="white",
            width=20
        )
        self.diagnosis_label.pack(side=tk.LEFT)

        self.confidence_label = tk.Label(
            diagnosis_container,
            text="Confidence: N/A",
            font=("Arial", 12),
            fg="#666666",
            bg="white"
        )
        self.confidence_label.pack(side=tk.RIGHT)

        # Information box
        info_frame = tk.Frame(results_frame, bg="white")
        info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.info_text = tk.Text(
            info_frame,
            wrap=tk.WORD,
            font=("Arial", 15),
            bg="#f8f9fa",
            padx=15,
            pady=15,
            height=8
        )
        scrollbar = ttk.Scrollbar(info_frame, command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.info_text.pack(fill=tk.BOTH, expand=True)
        
        self.info_text.insert(tk.END, "Disease information will appear here...")
        self.info_text.config(state=tk.DISABLED)

        # Progress bar
        self.progress = ttk.Progressbar(
            control_frame,
            orient=tk.HORIZONTAL,
            mode='indeterminate',
            length=200
        )
        self.progress.pack(pady=10)

        # Status bar
        self.status_bar = tk.Label(
            self.root,
            text="Ready",
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W,
            font=("Arial", 10),
            fg="#666666"
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Configure styles
        style = ttk.Style()
        style.configure("Primary.TButton", foreground="Black", background="#2196F3", font=("Arial", 12))
        style.configure("Success.TButton", foreground="black", background="#4CAF50", font=("Arial", 12))

    def update_status(self, message):
        self.status_bar.config(text=message)
        self.root.update_idletasks()

    def preprocess_image(self, img_path):
        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize with aspect ratio preservation
            h, w = img.shape[:2]
            target_size = (224, 224)
            scale = min(target_size[0]/h, target_size[1]/w)
            new_h, new_w = int(h*scale), int(w*scale)
            resized_img = cv2.resize(img, (new_w, new_h))
            
            # Add padding
            delta_h = target_size[0] - new_h
            delta_w = target_size[1] - new_w
            top, bottom = delta_h//2, delta_h - (delta_h//2)
            left, right = delta_w//2, delta_w - (delta_w//2)
            
            padded_img = cv2.copyMakeBorder(
                resized_img, top, bottom, left, right,
                cv2.BORDER_CONSTANT, value=[0,0,0]
            )
            
            # Preprocess for VGG16
            processed_img = tf.keras.applications.vgg16.preprocess_input(padded_img)
            return processed_img
        except Exception as e:
            self.update_status(f"Error processing image: {str(e)}")
            return None

    def extract_features(self, img_array):
        img_array = np.expand_dims(img_array, axis=0)
        features = self.vgg_model.predict(img_array, verbose=0)
        return features.flatten()

    def predict_image(self):
        if not self.current_image_path:
            return None, None
        
        try:
            self.progress.start()
            self.update_status("Processing image...")
            
            processed_img = self.preprocess_image(self.current_image_path)
            if processed_img is None:
                return None, None
                
            features = self.extract_features(processed_img)
            
            prediction = self.model.predict(features.reshape(1, -1), verbose=0)
            class_idx = np.argmax(prediction)
            confidence = np.max(prediction)
            class_name = self.label_encoder[class_idx]
            
            return class_name, confidence
            
        except Exception as e:
            self.update_status(f"Error during prediction: {str(e)}")
            return None, None
        finally:
            self.progress.stop()
            self.update_status("Ready")

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
        )
        
        if file_path:
            try:
                self.current_image_path = file_path
                img = Image.open(file_path)
                
                # Maintain aspect ratio
                max_size = (400, 400)
                img.thumbnail(max_size, Image.LANCZOS)
                
                photo = ImageTk.PhotoImage(img)
                self.image_label.config(image=photo)
                self.image_label.image = photo
                self.detect_btn.config(state=tk.NORMAL)
                self.update_status("Image uploaded successfully")
                
                # Clear previous results
                self.diagnosis_label.config(text="Diagnosis: N/A", fg="black")
                self.confidence_label.config(text="Confidence: N/A")
                self.info_text.config(state=tk.NORMAL)
                self.info_text.delete(1.0, tk.END)
                self.info_text.insert(tk.END, "Disease information will appear here...")
                self.info_text.config(state=tk.DISABLED)
                
            except Exception as e:
                self.update_status(f"Error loading image: {str(e)}")

    def detect_image(self):
        class_name, confidence = self.predict_image()
        if class_name and confidence:
            # Format diagnosis name
            formatted_name = class_name.replace('_', ' ').title()
            
            # Set diagnosis and confidence
            color = "#4CAF50" if class_name.lower() == "normal" else "#f44336"
            self.diagnosis_label.config(text=f"Diagnosis: {formatted_name}", fg=color)
            self.confidence_label.config(text=f"Confidence: {confidence:.2%}")

            # Update disease information
            self.info_text.config(state=tk.NORMAL)
            self.info_text.delete(1.0, tk.END)
            
            disease = self.disease_info.get(class_name.lower(), {})
            if disease:
                self.info_text.insert(tk.END, f"Description:\n{disease['description']}\n\n")
                self.info_text.insert(tk.END, "Key Symptoms:\n• " + "\n• ".join(disease['symptoms']) + "\n\n")
                self.info_text.insert(tk.END, f"Recommendations:\n{disease['recommendation']}")
            else:
                self.info_text.insert(tk.END, "No additional information available for this diagnosis.")
            
            self.info_text.config(state=tk.DISABLED)
            self.update_status("Analysis complete")
        else:
            self.diagnosis_label.config(text="Diagnosis: Error", fg="#f44336")
            self.confidence_label.config(text="Confidence: N/A")
            self.info_text.config(state=tk.NORMAL)
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(tk.END, "Could not retrieve disease information.")
            self.info_text.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = RetinalDiseaseClassifier(root)
    root.mainloop()