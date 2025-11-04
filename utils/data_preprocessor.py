import os
import cv2
import numpy as np
from PIL import Image
import shutil

def organize_dataset(source_dir, target_dir):
    """Organize dataset into train/test folders"""
    
    # Buat struktur folder
    for split in ['train', 'test']:
        for class_name in ['normal', 'benign', 'malignant']:
            os.makedirs(os.path.join(target_dir, split, class_name), exist_ok=True)
    
    # Organize files (contoh sederhana - sesuaikan dengan struktur data Anda)
    # Ini adalah template, sesuaikan dengan struktur data asli Anda
    pass

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess individual image"""
    try:
        # Baca gambar
        image = cv2.imread(image_path)
        if image is None:
            return None
            
        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image = cv2.resize(image, target_size)
        
        # Normalisasi
        image = image / 255.0
        
        return image
        
    except Exception as e:
        print(f"Error preprocessing {image_path}: {str(e)}")
        return None

def augment_dataset(data_dir, output_dir):
    """Augment dataset untuk meningkatkan jumlah data"""
    # Implementasi data augmentation bisa ditambahkan di sini
    pass