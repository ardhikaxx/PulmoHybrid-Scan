import os
import shutil
from pathlib import Path
import cv2
import numpy as np
from collections import defaultdict
import hashlib

def calculate_image_hash(image_path):
    """Menghitung hash MD5 untuk gambar"""
    try:
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception as e:
        print(f"Error calculating hash for {image_path}: {e}")
        return None

def find_duplicate_images(base_dir):
    """Mencari gambar duplikat di seluruh dataset"""
    image_hashes = defaultdict(list)
    
    # Direktori yang akan diperiksa
    directories = ['train', 'val', 'test']
    categories = ['normal', 'benign', 'malignant']
    
    total_images = 0
    for dir_name in directories:
        for category in categories:
            dir_path = Path(base_dir) / dir_name / category
            if dir_path.exists():
                for image_file in dir_path.glob('*.*'):
                    if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        image_hash = calculate_image_hash(image_file)
                        if image_hash:
                            image_hashes[image_hash].append(image_file)
                            total_images += 1
    
    print(f"Total images processed: {total_images}")
    
    # Filter hanya hash yang memiliki duplikat
    duplicates = {hash_val: paths for hash_val, paths in image_hashes.items() 
                 if len(paths) > 1}
    
    print(f"Found {len(duplicates)} duplicate groups")
    
    return duplicates

def get_current_file_count(directory):
    """Menghitung jumlah file gambar saat ini di direktori"""
    count = 0
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        count += len(list(directory.glob(ext)))
    return count

def get_next_available_filename(directory, base_name, extension, max_files=1000):
    """Mencari nama file berikutnya yang tersedia dengan format yang sesuai"""
    # Cek jumlah file saat ini di direktori
    current_count = get_current_file_count(directory)
    if current_count >= max_files:
        return None  # Sudah mencapai batas maksimal
    
    counter = 1
    while True:
        # Format: "Base name (counter).extension"
        if counter == 1:
            potential_name = f"{base_name}{extension}"
        else:
            potential_name = f"{base_name} ({counter}){extension}"
        
        full_path = directory / potential_name
        
        if not full_path.exists():
            return full_path
        counter += 1
        
        # Safety check untuk mencegah infinite loop
        if counter > max_files * 2:
            return None

def extract_base_name(filename):
    """Mengekstrak nama base dari file dengan format 'Bengin case (1).jpg'"""
    stem = filename.stem  # 'Bengin case (1)'
    extension = filename.suffix  # '.jpg'
    
    # Jika ada pattern (number) di akhir, kita ekstrak base name-nya
    if ' (' in stem and stem.endswith(')'):
        base_part = stem.rsplit(' (', 1)[0]
        # Cek jika bagian dalam kurung adalah angka
        number_part = stem.rsplit(' (', 1)[1][:-1]
        if number_part.isdigit():
            return base_part, extension
    
    return stem, extension

def apply_augmentations(image):
    """Menerapkan berbagai augmentasi pada gambar"""
    augmented_images = []
    
    # 1. Original dengan kontras berbeda
    alpha_values = [0.8, 1.2, 1.5]  # Faktor kontras
    for alpha in alpha_values:
        augmented = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
        augmented_images.append(augmented)
    
    # 2. Flip horizontal
    flipped = cv2.flip(image, 1)
    augmented_images.append(flipped)
    
    # 3. Flip horizontal dengan kontras berbeda
    for alpha in [0.9, 1.3]:
        flipped_contrast = cv2.convertScaleAbs(flipped, alpha=alpha, beta=0)
        augmented_images.append(flipped_contrast)
    
    # 4. Brightness variations
    beta_values = [-30, 30, 50]  # Perubahan brightness
    for beta in beta_values:
        bright = cv2.convertScaleAbs(image, alpha=1.0, beta=beta)
        augmented_images.append(bright)
    
    # 5. Brightness + contrast combinations
    for alpha in [0.7, 1.4]:
        for beta in [-20, 20]:
            combo = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            augmented_images.append(combo)
    
    return augmented_images

def get_augmentation_type_name(index):
    """Mengembalikan nama untuk tipe augmentasi"""
    types = [
        "contrast_low", "contrast_medium", "contrast_high",
        "flip_horizontal",
        "flip_contrast_low", "flip_contrast_high",
        "brightness_low", "brightness_medium", "brightness_high",
        "combo_low_contrast_low_bright", "combo_low_contrast_high_bright",
        "combo_high_contrast_low_bright", "combo_high_contrast_high_bright"
    ]
    return types[index] if index < len(types) else f"aug_{index+1}"

def process_duplicates(duplicates, base_dir, max_files_per_class=1000):
    """Memproses duplikat dan membuat augmented versions dengan batas maksimal"""
    base_path = Path(base_dir)
    processed_files = set()
    augmentation_stats = defaultdict(int)
    
    for hash_val, file_paths in duplicates.items():
        print(f"\nProcessing duplicate group with {len(file_paths)} files:")
        for path in file_paths:
            print(f"  - {path}")
        
        # Pilih satu file sebagai master (yang pertama)
        master_file = file_paths[0]
        
        if master_file in processed_files:
            continue
            
        # Baca gambar master
        try:
            image = cv2.imread(str(master_file))
            if image is None:
                print(f"Warning: Could not read image {master_file}")
                continue
        except Exception as e:
            print(f"Error reading image {master_file}: {e}")
            continue
        
        # Terapkan augmentasi
        augmented_images = apply_augmentations(image)
        
        # Tentukan direktori target (gunakan direktori dari file master)
        target_dir = master_file.parent
        category_name = target_dir.name
        
        # Cek jumlah file saat ini di direktori target
        current_count = get_current_file_count(target_dir)
        remaining_slots = max_files_per_class - current_count
        
        if remaining_slots <= 0:
            print(f"⚠️  Directory {target_dir} already has {current_count} files (max: {max_files_per_class}). Skipping...")
            continue
        
        print(f"📁 Directory {category_name}: {current_count}/{max_files_per_class} files, {remaining_slots} slots available")
        
        # Ekstrak base name
        base_name, extension = extract_base_name(master_file)
        
        print(f"Creating augmented versions for: {base_name}")
        
        # Simpan gambar yang di-augmentasi (maksimal sesuai remaining_slots)
        aug_count = 0
        for i, aug_image in enumerate(augmented_images):
            if aug_count >= remaining_slots:
                print(f"⚠️  Reached maximum file limit for {category_name}. Stopping augmentation.")
                break
                
            # Cari nama file berikutnya yang tersedia
            aug_base_name = f"{base_name}"
            new_file_path = get_next_available_filename(target_dir, aug_base_name, extension, max_files_per_class)
            
            if new_file_path is None:
                print(f"⚠️  Cannot create more files in {target_dir}. Limit reached.")
                break
            
            try:
                # Simpan gambar yang di-augmentasi
                success = cv2.imwrite(str(new_file_path), aug_image)
                if success:
                    aug_type = get_augmentation_type_name(i)
                    print(f"  ✅ Created: {new_file_path.name} [{aug_type}]")
                    aug_count += 1
                    augmentation_stats[category_name] += 1
                else:
                    print(f"  ❌ Failed to save: {new_file_path}")
            except Exception as e:
                print(f"  ❌ Error saving augmented image: {e}")
        
        processed_files.add(master_file)
        print(f"  📊 Created {aug_count} augmented images for this duplicate")
    
    return augmentation_stats

def analyze_dataset_structure(base_dir, max_files_per_class=1000):
    """Menganalisis struktur dataset"""
    print("Analyzing dataset structure...")
    base_path = Path(base_dir)
    stats = {}
    
    for split in ['train', 'val', 'test']:
        split_path = base_path / split
        if split_path.exists():
            print(f"\n{split.upper()} split:")
            stats[split] = {}
            for category in ['normal', 'benign', 'malignant']:
                category_path = split_path / category
                if category_path.exists():
                    image_count = len(list(category_path.glob('*.*')))
                    stats[split][category] = image_count
                    status = "✅ OK" if image_count <= max_files_per_class else "⚠️ OVER LIMIT"
                    print(f"  {category}: {image_count} images {status}")
    
    return stats

def main():
    base_directory = r"D:\PulmoHybrid-Scan\data"
    MAX_FILES_PER_CLASS = 1000
    
    print("=== Data Leakage Check and Augmentation ===")
    print(f"Maximum files per class: {MAX_FILES_PER_CLASS}")
    
    # Analisis struktur dataset
    dataset_stats = analyze_dataset_structure(base_directory, MAX_FILES_PER_CLASS)
    
    # Cari duplikat
    print("\nSearching for duplicate images...")
    duplicate_groups = find_duplicate_images(base_directory)
    
    if duplicate_groups:
        print(f"\nFound {len(duplicate_groups)} groups of duplicate images!")
        
        # Tampilkan summary
        total_duplicates = sum(len(paths) for paths in duplicate_groups.values())
        unique_duplicates = len(duplicate_groups)
        print(f"Total duplicate files: {total_duplicates}")
        print(f"Unique duplicate groups: {unique_duplicates}")
        
        # Proses duplikat
        print("\nProcessing duplicates and creating augmented versions...")
        augmentation_stats = process_duplicates(duplicate_groups, base_directory, MAX_FILES_PER_CLASS)
        
        print("\n=== PROCESSING COMPLETED ===")
        print(f"Processed {len(duplicate_groups)} duplicate groups")
        
        # Tampilkan statistik augmentasi
        if augmentation_stats:
            print("\n📊 Augmentation Statistics:")
            for category, count in augmentation_stats.items():
                print(f"  {category}: {count} augmented images created")
        
        # Tampilkan statistik akhir
        print("\n📈 Final Dataset Statistics:")
        final_stats = analyze_dataset_structure(base_directory, MAX_FILES_PER_CLASS)
        
    else:
        print("\nNo duplicate images found!")

if __name__ == "__main__":
    main()