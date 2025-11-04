import os
import numpy as np
import joblib
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import classes dari training.py yang sudah diperbarui
from training import PulmoNaiveBayesClassifier, PulmoCNNClassifier, HybridPulmoClassifier

def test_naive_bayes_with_test_dataset():
    """Test model Naive Bayes dengan dataset test yang terpisah"""
    
    # Load model Naive Bayes
    classifier = PulmoNaiveBayesClassifier()
    
    try:
        classifier.load_model('models/pulmo_naive_bayes.pkl')
        print("✅ Model Naive Bayes berhasil dimuat")
    except Exception as e:
        print(f"❌ Gagal memuat model Naive Bayes: {e}")
        print("💡 Pastikan training sudah dijalankan dan file model ada")
        return None
    
    # Load test data jika ada
    test_dir = "data/test"
    if not os.path.exists(test_dir):
        print("📁 Folder test tidak ditemukan. Membuat struktur folder...")
        os.makedirs('data/test/normal', exist_ok=True)
        os.makedirs('data/test/benign', exist_ok=True)
        os.makedirs('data/test/malignant', exist_ok=True)
        print("✅ Struktur folder test dibuat.")
        print("📝 Silakan tambahkan gambar test ke folder data/test/")
        return None
    
    X_test, y_test, file_paths = classifier.load_dataset(test_dir)
    
    if len(X_test) == 0:
        print("❌ Tidak ada data test yang ditemukan")
        print("💡 Tambahkan gambar ke folder:")
        print("   - data/test/normal/")
        print("   - data/test/benign/")
        print("   - data/test/malignant/")
        return None
    
    # Predict dengan Naive Bayes
    print("Memproses prediksi Naive Bayes...")
    y_pred = classifier.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print("\n" + "="*60)
    print("📊 HASIL TESTING NAIVE BAYES DENGAN DATA BARU")
    print("="*60)
    print(f"✅ Test Accuracy: {accuracy * 100:.2f}%")
    print(f"📈 Jumlah Sample Test: {len(X_test)}")
    print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=classifier.classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
               xticklabels=classifier.classes,
               yticklabels=classifier.classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('HybridPulmoClassifier - Confusion Matrix Naive Bayes (Test Data)')
    plt.tight_layout()
    plt.savefig('test_confusion_matrix_naive_bayes.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return accuracy

def test_cnn_with_test_dataset():
    """Test model CNN dengan dataset test yang terpisah"""
    
    # Load model CNN
    cnn_classifier = PulmoCNNClassifier()
    
    try:
        cnn_classifier.load_model('models/pulmo_cnn.pth')
        print("✅ Model CNN berhasil dimuat")
    except Exception as e:
        print(f"❌ Gagal memuat model CNN: {e}")
        print("💡 Pastikan training CNN sudah dijalankan dan file model ada")
        return None
    
    # Load test data
    test_dir = "data/test"
    if not os.path.exists(test_dir):
        print("📁 Folder test tidak ditemukan.")
        return None
    
    # Evaluate CNN model
    try:
        y_true, y_pred, probs, accuracy = cnn_classifier.evaluate_model(test_dir, batch_size=8)
        
        # Plot confusion matrix untuk CNN
        cm = confusion_matrix(y_true, y_pred, labels=cnn_classifier.classes)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                   xticklabels=cnn_classifier.classes,
                   yticklabels=cnn_classifier.classes)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('HybridPulmoClassifier - Confusion Matrix CNN (Test Data)')
        plt.tight_layout()
        plt.savefig('test_confusion_matrix_cnn.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return accuracy
        
    except Exception as e:
        print(f"❌ Error selama evaluasi CNN: {e}")
        return None

def test_hybrid_with_test_dataset():
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
        return None
    
    # Evaluate hybrid system
    try:
        hybrid_accuracy = hybrid_classifier.evaluate_hybrid(test_dir)
        return hybrid_accuracy
        
    except Exception as e:
        print(f"❌ Error selama evaluasi hybrid: {e}")
        return None

def compare_all_models():
    """Bandingkan performa ketiga pendekatan"""
    print("\n" + "="*70)
    print("🔄 MEMBANDINGKAN SEMUA MODEL")
    print("="*70)
    
    accuracies = {}
    
    print("\n🧪 Testing Naive Bayes...")
    nb_accuracy = test_naive_bayes_with_test_dataset()
    if nb_accuracy is not None:
        accuracies['Naive Bayes'] = nb_accuracy
    
    print("\n🧠 Testing CNN...")
    cnn_accuracy = test_cnn_with_test_dataset()
    if cnn_accuracy is not None:
        accuracies['CNN'] = cnn_accuracy
    
    print("\n🤖 Testing Hybrid System...")
    hybrid_accuracy = test_hybrid_with_test_dataset()
    if hybrid_accuracy is not None:
        accuracies['Hybrid System'] = hybrid_accuracy
    
    if accuracies:
        print("\n" + "="*70)
        print("📈 PERBANDINGAN AKURASI SEMUA MODEL")
        print("="*70)
        
        # Tampilkan hasil
        for model_name, accuracy in accuracies.items():
            print(f"🎯 {model_name} Accuracy: {accuracy * 100:.2f}%")
        
        # Cari model terbaik
        best_model = max(accuracies, key=accuracies.get)
        best_accuracy = accuracies[best_model]
        
        print(f"\n🏆 MODEL TERBAIK: {best_model} ({best_accuracy * 100:.2f}%)")
        
        # Plot comparison
        models = list(accuracies.keys())
        accs = list(accuracies.values())
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, accs, color=['#3498db', '#9b59b6', '#2ecc71'])
        plt.ylim(0, 1)
        plt.ylabel('Accuracy')
        plt.title('HybridPulmoClassifier - Model Accuracy Comparison (Test Data)')
        
        # Tambah nilai di atas bar
        for bar, accuracy in zip(bars, accs):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{accuracy:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('test_accuracy_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    else:
        print("\n❌ Tidak ada model yang berhasil di-test")

def predict_single_image_hybrid(image_path):
    """Prediksi single image dengan sistem hybrid"""
    
    if not os.path.exists(image_path):
        print(f"❌ File {image_path} tidak ditemukan")
        return None
    
    print("\n" + "="*70)
    print(f"🔍 HYBRID ANALISIS GAMBAR: {os.path.basename(image_path)}")
    print("="*70)
    
    # Load hybrid classifier
    hybrid_classifier = HybridPulmoClassifier()
    
    try:
        hybrid_classifier.load_models()
        print("✅ Sistem Hybrid berhasil dimuat")
    except Exception as e:
        print(f"❌ Gagal memuat sistem hybrid: {e}")
        print("💡 Gunakan prediksi individual sebagai alternatif")
        return predict_single_image_individual(image_path)
    
    # Prediksi dengan sistem hybrid
    result = hybrid_classifier.predict_hybrid(image_path)
    
    if result:
        # Tampilkan rekomendasi berdasarkan hasil hybrid
        final_prediction = result['hybrid_prediction']
        confidence = max(result['hybrid_probabilities'].values())
        
        print(f"\n💡 REKOMENDASI HYBRID (Confidence: {confidence*100:.1f}%):")
        if final_prediction == 'normal':
            print("✅ STATUS: NORMAL - Paru-paru dalam kondisi sehat")
            print("   ✅ Tidak diperlukan tindakan khusus")
            print("   💡 Tetap jaga kesehatan dengan pola hidup sehat")
        elif final_prediction == 'benign':
            print("⚠️ STATUS: JINAK - Tumor jinak terdeteksi")
            print("   📋 Disarankan konsultasi rutin dengan dokter")
            print("   🔍 Monitoring berkala diperlukan")
        elif final_prediction == 'malignant':
            print("🚨 STATUS: GANAS - Kanker ganas terdeteksi")
            print("   🏥 Segera konsultasi dengan dokter spesialis onkologi")
            print("   ⚡ Penanganan medis segera diperlukan")
    
    return result

def predict_single_image_individual(image_path):
    """Prediksi single image dengan masing-masing model (fallback)"""
    
    if not os.path.exists(image_path):
        print(f"❌ File {image_path} tidak ditemukan")
        return None
    
    print("\n" + "="*70)
    print(f"🔍 INDIVIDUAL ANALISIS GAMBAR: {os.path.basename(image_path)}")
    print("="*70)
    
    # Prediksi dengan Naive Bayes
    print("\n🧪 NAIVE BAYES PREDICTION:")
    print("-" * 40)
    
    nb_classifier = PulmoNaiveBayesClassifier()
    nb_prediction = None
    nb_probabilities = None
    
    try:
        nb_classifier.load_model('models/pulmo_naive_bayes.pkl')
        
        features = nb_classifier.feature_extractor.extract_features(image_path)
        if features is not None:
            features = features.reshape(1, -1)
            nb_prediction = nb_classifier.predict(features)[0]
            nb_probabilities = nb_classifier.predict_proba(features)[0]
            
            print(f"🎯 Klasifikasi: {nb_prediction.upper()}")
            print("📊 Probabilitas:")
            for class_name, prob in zip(nb_classifier.classes, nb_probabilities):
                percentage = prob * 100
                print(f"   {class_name.upper()}: {percentage:.2f}%")
        else:
            print("❌ Gagal memproses gambar dengan Naive Bayes")
    except Exception as e:
        print(f"❌ Model Naive Bayes tidak tersedia: {e}")
    
    # Prediksi dengan CNN
    print("\n🧠 CNN TRANSFER LEARNING PREDICTION:")
    print("-" * 40)
    
    cnn_classifier = PulmoCNNClassifier()
    cnn_prediction = None
    cnn_probabilities = None
    
    try:
        cnn_classifier.load_model('models/pulmo_cnn.pth')
        
        cnn_prediction, cnn_probabilities = cnn_classifier.predict_single_image(image_path)
        
        if cnn_prediction is not None:
            print(f"🎯 Klasifikasi: {cnn_prediction.upper()}")
            print("📊 Probabilitas:")
            for class_name, prob in zip(cnn_classifier.classes, cnn_probabilities):
                percentage = prob * 100
                print(f"   {class_name.upper()}: {percentage:.2f}%")
        else:
            print("❌ Gagal memproses gambar dengan CNN")
    except Exception as e:
        print(f"❌ Model CNN tidak tersedia: {e}")
    
    # Kesimpulan
    print("\n" + "="*70)
    print("🎯 KESIMPULAN FINAL")
    print("="*70)
    
    final_prediction = None
    confidence = 0.0
    
    if nb_prediction and cnn_prediction:
        if nb_prediction == cnn_prediction:
            print(f"✅ KONSISTEN: Kedua model sepakat - {nb_prediction.upper()}")
            final_prediction = nb_prediction
            # Ambil confidence dari CNN (biasanya lebih akurat)
            confidence = max(cnn_probabilities) if cnn_probabilities is not None else 0.0
        else:
            print(f"⚠️  KONFLIK: Model berbeda")
            print(f"   Naive Bayes: {nb_prediction.upper()}")
            print(f"   CNN: {cnn_prediction.upper()}")
            # Prioritaskan CNN karena biasanya lebih akurat
            final_prediction = cnn_prediction
            confidence = max(cnn_probabilities) if cnn_probabilities is not None else 0.0
            print(f"   🎯 Menggunakan prediksi CNN: {final_prediction.upper()}")
    elif nb_prediction:
        final_prediction = nb_prediction
        confidence = max(nb_probabilities) if nb_probabilities is not None else 0.0
        print(f"✅ Menggunakan Naive Bayes: {final_prediction.upper()}")
    elif cnn_prediction:
        final_prediction = cnn_prediction
        confidence = max(cnn_probabilities) if cnn_probabilities is not None else 0.0
        print(f"✅ Menggunakan CNN: {final_prediction.upper()}")
    else:
        print("❌ Tidak ada model yang berhasil memprediksi")
        return None
    
    # Tampilkan rekomendasi
    print(f"\n💡 REKOMENDASI (Confidence: {confidence*100:.1f}%):")
    if final_prediction == 'normal':
        print("✅ STATUS: NORMAL - Paru-paru dalam kondisi sehat")
        print("   ✅ Tidak diperlukan tindakan khusus")
        print("   💡 Tetap jaga kesehatan dengan pola hidup sehat")
    elif final_prediction == 'benign':
        print("⚠️ STATUS: JINAK - Tumor jinak terdeteksi")
        print("   📋 Disarankan konsultasi rutin dengan dokter")
        print("   🔍 Monitoring berkala diperlukan")
    elif final_prediction == 'malignant':
        print("🚨 STATUS: GANAS - Kanker ganas terdeteksi")
        print("   🏥 Segera konsultasi dengan dokter spesialis onkologi")
        print("   ⚡ Penanganan medis segera diperlukan")
    
    return {
        'final_prediction': final_prediction,
        'confidence': confidence,
        'naive_bayes': nb_prediction,
        'cnn': cnn_prediction,
        'agreement': nb_prediction == cnn_prediction if nb_prediction and cnn_prediction else None
    }

def batch_predict_hybrid(folder_path):
    """Prediksi semua gambar dalam folder dengan sistem hybrid"""
    
    if not os.path.exists(folder_path):
        print(f"❌ Folder {folder_path} tidak ditemukan")
        return
    
    print(f"\n🔍 Memproses gambar di folder: {folder_path}")
    
    # Load hybrid classifier
    hybrid_classifier = HybridPulmoClassifier()
    
    try:
        hybrid_classifier.load_models()
        print("✅ Sistem Hybrid berhasil dimuat")
    except Exception as e:
        print(f"❌ Gagal memuat sistem hybrid: {e}")
        print("💡 Menggunakan prediksi individual sebagai alternatif")
        return batch_predict_individual(folder_path)
    
    results = []
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    
    file_list = [f for f in os.listdir(folder_path) 
                if f.lower().endswith(supported_formats)]
    
    if not file_list:
        print("❌ Tidak ada gambar yang ditemukan dalam folder")
        return
    
    print(f"📁 Ditemukan {len(file_list)} gambar")
    
    for i, filename in enumerate(file_list, 1):
        image_path = os.path.join(folder_path, filename)
        
        print(f"\n[{i}/{len(file_list)}] Memproses: {filename}")
        
        try:
            result = hybrid_classifier.predict_hybrid(image_path)
            
            if result:
                results.append({
                    'file': filename,
                    'hybrid_prediction': result['hybrid_prediction'],
                    'hybrid_probabilities': result['hybrid_probabilities'],
                    'nb_prediction': result['naive_bayes_prediction'],
                    'cnn_prediction': result['cnn_prediction']
                })
                
                final_pred = result['hybrid_prediction']
                confidence = max(result['hybrid_probabilities'].values())
                print(f"   🎯 HYBRID: {final_pred.upper()} ({confidence*100:.1f}%)")
            else:
                print("   ❌ Gagal memproses dengan sistem hybrid")
                results.append({
                    'file': filename,
                    'hybrid_prediction': 'ERROR',
                    'hybrid_probabilities': {},
                    'nb_prediction': None,
                    'cnn_prediction': None
                })
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({
                'file': filename,
                'hybrid_prediction': 'ERROR',
                'hybrid_probabilities': {},
                'nb_prediction': None,
                'cnn_prediction': None
            })
    
    # Summary
    if results:
        print(f"\n📈 HYBRID SUMMARY: {len(results)} gambar diproses")
        
        # Count by final prediction
        print("\n🎯 DISTRIBUSI HASIL HYBRID:")
        for class_name in ['normal', 'benign', 'malignant']:
            count = sum(1 for r in results if r['hybrid_prediction'] == class_name)
            percentage = (count / len(results)) * 100
            print(f"   {class_name.upper()}: {count} gambar ({percentage:.1f}%)")
        
        # Agreement analysis
        agreement_count = sum(1 for r in results if r['nb_prediction'] and r['cnn_prediction'] and r['nb_prediction'] == r['cnn_prediction'])
        conflict_count = sum(1 for r in results if r['nb_prediction'] and r['cnn_prediction'] and r['nb_prediction'] != r['cnn_prediction'])
        
        print(f"\n🤝 ANALISIS KESEPAKATAN MODEL:")
        print(f"   Sepakat: {agreement_count} gambar")
        print(f"   Berbeda: {conflict_count} gambar")
        if (agreement_count + conflict_count) > 0:
            agreement_rate = (agreement_count/(agreement_count + conflict_count))*100
            print(f"   Tingkat Kesepakatan: {agreement_rate:.1f}%")
            
            if agreement_rate > 80:
                print("   ✅ Tingkat kesepakatan sangat baik")
            elif agreement_rate > 60:
                print("   ⚠️  Tingkat kesepakatan cukup")
            else:
                print("   ❌ Tingkat kesepakatan rendah")

def batch_predict_individual(folder_path):
    """Prediksi batch dengan model individual (fallback)"""
    
    if not os.path.exists(folder_path):
        print(f"❌ Folder {folder_path} tidak ditemukan")
        return
    
    print(f"\n🔍 Memproses gambar di folder: {folder_path} (Individual Models)")
    
    # Load models
    nb_classifier = PulmoNaiveBayesClassifier()
    cnn_classifier = PulmoCNNClassifier()
    
    nb_available = False
    cnn_available = False
    
    try:
        nb_classifier.load_model('models/pulmo_naive_bayes.pkl')
        nb_available = True
        print("✅ Model Naive Bayes siap")
    except Exception as e:
        print(f"❌ Model Naive Bayes tidak tersedia: {e}")
    
    try:
        cnn_classifier.load_model('models/pulmo_cnn.pth')
        cnn_available = True
        print("✅ Model CNN siap")
    except Exception as e:
        print(f"❌ Model CNN tidak tersedia: {e}")
    
    if not nb_available and not cnn_available:
        print("❌ Tidak ada model yang tersedia")
        return
    
    results = []
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    
    file_list = [f for f in os.listdir(folder_path) 
                if f.lower().endswith(supported_formats)]
    
    if not file_list:
        print("❌ Tidak ada gambar yang ditemukan dalam folder")
        return
    
    print(f"📁 Ditemukan {len(file_list)} gambar")
    
    for i, filename in enumerate(file_list, 1):
        image_path = os.path.join(folder_path, filename)
        
        print(f"\n[{i}/{len(file_list)}] Memproses: {filename}")
        
        nb_prediction = None
        cnn_prediction = None
        final_prediction = None
        confidence = 0.0
        
        # Naive Bayes prediction
        if nb_available:
            try:
                features = nb_classifier.feature_extractor.extract_features(image_path)
                if features is not None:
                    features = features.reshape(1, -1)
                    nb_prediction = nb_classifier.predict(features)[0]
                    print(f"   Naive Bayes: {nb_prediction.upper()}")
                else:
                    print("   Naive Bayes: Gagal ekstraksi fitur")
            except Exception as e:
                print(f"   Naive Bayes Error: {e}")
        
        # CNN prediction
        if cnn_available:
            try:
                cnn_prediction, cnn_probs = cnn_classifier.predict_single_image(image_path)
                if cnn_prediction is not None:
                    confidence = max(cnn_probs) if cnn_probs is not None else 0.0
                    print(f"   CNN: {cnn_prediction.upper()} ({confidence*100:.1f}%)")
                else:
                    print("   CNN: Gagal prediksi")
            except Exception as e:
                print(f"   CNN Error: {e}")
        
        # Determine final prediction
        if nb_prediction and cnn_prediction:
            if nb_prediction == cnn_prediction:
                final_prediction = nb_prediction
                agreement = "✅"
            else:
                final_prediction = cnn_prediction  # Prioritize CNN
                agreement = "⚠️"
        elif nb_prediction:
            final_prediction = nb_prediction
            agreement = "NB"
        elif cnn_prediction:
            final_prediction = cnn_prediction
            agreement = "CNN"
        else:
            final_prediction = "ERROR"
            agreement = "❌"
        
        results.append({
            'file': filename,
            'nb_prediction': nb_prediction,
            'cnn_prediction': cnn_prediction,
            'final_prediction': final_prediction,
            'confidence': confidence,
            'agreement': agreement
        })
        
        print(f"   🎯 FINAL: {final_prediction.upper()} [{agreement}]")
    
    # Summary
    if results:
        print(f"\n📈 SUMMARY: {len(results)} gambar diproses")
        
        # Count by final prediction
        print("\n🎯 DISTRIBUSI HASIL FINAL:")
        for class_name in ['normal', 'benign', 'malignant']:
            count = sum(1 for r in results if r['final_prediction'] == class_name)
            percentage = (count / len(results)) * 100
            print(f"   {class_name.upper()}: {count} gambar ({percentage:.1f}%)")

def check_model_availability():
    """Cek ketersediaan model"""
    print("\n🔍 CHECK KETERSEDIAAN MODEL:")
    
    nb_available = os.path.exists('models/pulmo_naive_bayes.pkl')
    cnn_available = os.path.exists('models/pulmo_cnn.pth')
    hybrid_available = nb_available and cnn_available
    
    print(f"   Naive Bayes: {'✅ Tersedia' if nb_available else '❌ Tidak tersedia'}")
    print(f"   CNN: {'✅ Tersedia' if cnn_available else '❌ Tidak tersedia'}")
    print(f"   Hybrid System: {'✅ Tersedia' if hybrid_available else '❌ Tidak tersedia'}")
    
    return nb_available, cnn_available, hybrid_available

if __name__ == "__main__":
    print("🚀 HYBRID PULMO CLASSIFIER - SISTEM TESTING")
    print("="*60)
    print("Sistem Hybrid CNN-Naive Bayes untuk Klasifikasi Kanker Paru-paru")
    print("="*60)
    
    # Cek ketersediaan model
    nb_available, cnn_available, hybrid_available = check_model_availability()
    
    if not nb_available and not cnn_available:
        print("\n❌ Tidak ada model yang tersedia. Jalankan training.py terlebih dahulu.")
        exit()
    
    while True:
        print("\nPilih opsi testing:")
        print("1. Test Naive Bayes dengan dataset")
        print("2. Test CNN dengan dataset") 
        print("3. Test Hybrid System dengan dataset")
        print("4. Bandingkan semua model")
        print("5. Prediksi single image (Hybrid System)")
        print("6. Prediksi single image (Individual Models)")
        print("7. Prediksi batch folder (Hybrid System)")
        print("8. Prediksi batch folder (Individual Models)")
        print("9. Cek ketersediaan model")
        print("10. Keluar")
        
        choice = input("\nMasukkan pilihan (1-10): ").strip()
        
        if choice == '1':
            if nb_available:
                test_naive_bayes_with_test_dataset()
            else:
                print("❌ Model Naive Bayes tidak tersedia")
        
        elif choice == '2':
            if cnn_available:
                test_cnn_with_test_dataset()
            else:
                print("❌ Model CNN tidak tersedia")
        
        elif choice == '3':
            if hybrid_available:
                test_hybrid_with_test_dataset()
            else:
                print("❌ Sistem Hybrid tidak tersedia")
        
        elif choice == '4':
            if nb_available or cnn_available:
                compare_all_models()
            else:
                print("❌ Tidak ada model yang tersedia untuk dibandingkan")
        
        elif choice == '5':
            if hybrid_available:
                image_path = input("Masukkan path gambar: ").strip()
                predict_single_image_hybrid(image_path)
            else:
                print("❌ Sistem Hybrid tidak tersedia")
        
        elif choice == '6':
            if nb_available or cnn_available:
                image_path = input("Masukkan path gambar: ").strip()
                predict_single_image_individual(image_path)
            else:
                print("❌ Tidak ada model yang tersedia")
        
        elif choice == '7':
            if hybrid_available:
                folder_path = input("Masukkan path folder: ").strip()
                batch_predict_hybrid(folder_path)
            else:
                print("❌ Sistem Hybrid tidak tersedia")
        
        elif choice == '8':
            if nb_available or cnn_available:
                folder_path = input("Masukkan path folder: ").strip()
                batch_predict_individual(folder_path)
            else:
                print("❌ Tidak ada model yang tersedia")
        
        elif choice == '9':
            check_model_availability()
        
        elif choice == '10':
            print("👋 Terima kasih menggunakan HybridPulmoClassifier!")
            break
        
        else:
            print("❌ Pilihan tidak valid")