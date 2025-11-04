import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import class hybrid dari training.py
from training import HybridPulmoClassifier

def test_hybrid_system():
    """Test sistem hybrid dengan dataset test"""
    
    # Load hybrid classifier
    hybrid_classifier = HybridPulmoClassifier()
    
    try:
        hybrid_classifier.load_models()
        print("✅ Sistem Hybrid berhasil dimuat")
    except Exception as e:
        print(f"❌ Gagal memuat sistem hybrid: {e}")
        print("💡 Pastikan training sudah dijalankan dan file model ada")
        return None
    
    # Load test data
    test_dir = "data/test"
    if not os.path.exists(test_dir):
        print("📁 Folder test tidak ditemukan.")
        print("💡 Buat folder data/test/ dengan subfolder normal, benign, malignant")
        return None
    
    # Evaluate hybrid system
    try:
        hybrid_accuracy = hybrid_classifier.evaluate_hybrid(test_dir)
        return hybrid_accuracy
        
    except Exception as e:
        print(f"❌ Error selama evaluasi hybrid: {e}")
        return None

def predict_single_image(image_path):
    """Prediksi single image dengan sistem hybrid"""
    
    if not os.path.exists(image_path):
        print(f"❌ File {image_path} tidak ditemukan")
        return None
    
    # Load hybrid classifier
    hybrid_classifier = HybridPulmoClassifier()
    
    try:
        hybrid_classifier.load_models()
        print("✅ Sistem Hybrid berhasil dimuat")
    except Exception as e:
        print(f"❌ Gagal memuat sistem hybrid: {e}")
        return None
    
    # Prediksi dengan sistem hybrid
    result = hybrid_classifier.predict_hybrid(image_path)
    return result

def batch_predict(folder_path):
    """Prediksi semua gambar dalam folder dengan sistem hybrid"""
    
    if not os.path.exists(folder_path):
        print(f"❌ Folder {folder_path} tidak ditemukan")
        return
    
    # Load hybrid classifier
    hybrid_classifier = HybridPulmoClassifier()
    
    try:
        hybrid_classifier.load_models()
        print("✅ Sistem Hybrid berhasil dimuat")
    except Exception as e:
        print(f"❌ Gagal memuat sistem hybrid: {e}")
        return
    
    # Prediksi batch
    hybrid_classifier.batch_predict_hybrid(folder_path)

def check_model_availability():
    """Cek ketersediaan model"""
    print("\n🔍 CHECK KETERSEDIAAN MODEL:")
    
    nb_available = os.path.exists('models/pulmo_naive_bayes.pkl')
    cnn_available = os.path.exists('models/pulmo_cnn.pth')
    hybrid_available = nb_available and cnn_available
    
    print(f"   Naive Bayes: {'✅ Tersedia' if nb_available else '❌ Tidak tersedia'}")
    print(f"   CNN: {'✅ Tersedia' if cnn_available else '❌ Tidak tersedia'}")
    print(f"   Hybrid System: {'✅ Tersedia' if hybrid_available else '❌ Tidak tersedia'}")
    
    return hybrid_available

if __name__ == "__main__":
    print("🚀 HYBRID PULMO CLASSIFIER - TESTING SYSTEM")
    print("="*60)
    print("Sistem Hybrid CNN-Naive Bayes untuk Klasifikasi Kanker Paru-paru")
    print("="*60)
    
    # Cek ketersediaan model
    hybrid_available = check_model_availability()
    
    if not hybrid_available:
        print("\n❌ Sistem Hybrid tidak tersedia. Jalankan training.py terlebih dahulu.")
        exit()
    
    while True:
        print("\nPilih opsi testing:")
        print("1. Test Hybrid System dengan dataset")
        print("2. Prediksi single image")
        print("3. Prediksi batch folder")
        print("4. Cek ketersediaan model")
        print("5. Keluar")
        
        choice = input("\nMasukkan pilihan (1-5): ").strip()
        
        if choice == '1':
            test_hybrid_system()
        
        elif choice == '2':
            image_path = input("Masukkan path gambar: ").strip()
            predict_single_image(image_path)
        
        elif choice == '3':
            folder_path = input("Masukkan path folder: ").strip()
            batch_predict(folder_path)
        
        elif choice == '4':
            check_model_availability()
        
        elif choice == '5':
            print("👋 Terima kasih menggunakan HybridPulmoClassifier!")
            break
        
        else:
            print("❌ Pilihan tidak valid")