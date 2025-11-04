import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import class hybrid dari training.py
from training import HybridPulmoClassifier, PulmoNaiveBayesClassifier, PulmoCNNClassifier

class TestingSystem:
    """Sistem testing untuk Hybrid Pulmo Classifier"""
    
    def __init__(self):
        self.hybrid_classifier = HybridPulmoClassifier()
        self.classes = ['normal', 'benign', 'malignant']
    
    def load_models(self):
        """Load semua model yang diperlukan"""
        try:
            # Load Naive Bayes
            if os.path.exists('models/pulmo_naive_bayes.pkl'):
                self.hybrid_classifier.nb_classifier.load_model('models/pulmo_naive_bayes.pkl')
                print("✅ Naive Bayes model loaded")
            else:
                print("❌ Naive Bayes model not found")
                return False
            
            # Load CNN
            if os.path.exists('models/pulmo_cnn.pth'):
                self.hybrid_classifier.cnn_classifier.load_model('models/pulmo_cnn.pth')
                print("✅ CNN model loaded")
            else:
                print("❌ CNN model not found")
                return False
            
            print("✅ Sistem Hybrid berhasil dimuat")
            return True
            
        except Exception as e:
            print(f"❌ Gagal memuat sistem hybrid: {e}")
            print("💡 Pastikan training sudah dijalankan dan file model ada")
            return False
    
    def evaluate_hybrid(self, test_dir):
        """Evaluasi sistem hybrid pada test set"""
        print("\n" + "="*60)
        print("📊 HYBRID PULMO CLASSIFIER - TESTING EVALUATION")
        print("="*60)
        
        if not os.path.exists(test_dir):
            print(f"❌ Folder test {test_dir} tidak ditemukan")
            return 0.0
        
        hybrid_predictions = []
        hybrid_true = []
        results_by_class = {'normal': [], 'benign': [], 'malignant': []}
        all_results = []
        
        for class_name in self.classes:
            class_dir = os.path.join(test_dir, class_name)
            if os.path.exists(class_dir):
                print(f"\n🔍 Processing {class_name} images...")
                count = 0
                
                for filename in os.listdir(class_dir):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        image_path = os.path.join(class_dir, filename)
                        
                        try:
                            # Prediksi dengan sistem hybrid
                            result = self.hybrid_classifier.predict_hybrid(image_path, verbose=False)
                            if result:
                                hybrid_predictions.append(result['prediction'])
                                hybrid_true.append(class_name)
                                results_by_class[class_name].append(result)
                                
                                # Simpan detail hasil
                                all_results.append({
                                    'file': filename,
                                    'actual': class_name,
                                    'predicted': result['prediction'],
                                    'confidence': result['confidence'],
                                    'correct': result['prediction'] == class_name
                                })
                                
                                count += 1
                                
                                if count % 20 == 0:
                                    print(f"  Processed {count} {class_name} images")
                        
                        except Exception as e:
                            print(f"  ❌ Error processing {filename}: {e}")
                
                print(f"✅ Completed: {count} {class_name} images processed")
            else:
                print(f"⚠️  Folder {class_name} tidak ditemukan")
        
        if not hybrid_true:
            print("❌ Tidak ada data test yang berhasil diproses")
            return 0.0
        
        # Tampilkan summary per kelas
        print("\n" + "="*60)
        print("🎯 HASIL EVALUASI SISTEM HYBRID PADA TEST SET")
        print("="*60)
        
        for class_name in self.classes:
            if results_by_class[class_name]:
                correct_predictions = sum(1 for r in results_by_class[class_name] 
                                       if r['prediction'] == class_name)
                total = len(results_by_class[class_name])
                accuracy = correct_predictions / total * 100
                avg_confidence = np.mean([r['confidence'] for r in results_by_class[class_name]]) * 100
                
                print(f"\n📊 {class_name.upper()}:")
                print(f"   ✅ Akurasi: {accuracy:.1f}% ({correct_predictions}/{total})")
                print(f"   📈 Confidence rata-rata: {avg_confidence:.1f}%")
        
        # Hitung akurasi keseluruhan
        hybrid_accuracy = accuracy_score(hybrid_true, hybrid_predictions)
        
        print(f"\n📈 AKURASI KESELURUHAN: {hybrid_accuracy * 100:.2f}%")
        print(f"📊 TOTAL SAMPLE TEST: {len(hybrid_true)}")
        print("="*60)
        
        # Tampilkan classification report
        print("\n📋 Laporan Klasifikasi:\n")
        print(classification_report(hybrid_true, hybrid_predictions, target_names=self.classes))
        
        # Plot confusion matrix untuk testing
        self.plot_test_confusion_matrix(hybrid_true, hybrid_predictions)
        
        # Simpan hasil detail ke CSV
        self.save_results_to_csv(all_results, hybrid_accuracy)
        
        # Tampilkan beberapa contoh prediksi
        self.display_sample_predictions(all_results)
        
        return hybrid_accuracy
    
    def plot_test_confusion_matrix(self, true_labels, pred_labels):
        """Plot confusion matrix untuk data testing"""
        cm = confusion_matrix(true_labels, pred_labels, labels=self.classes)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                   xticklabels=self.classes, 
                   yticklabels=self.classes,
                   annot_kws={"size": 16})
        plt.xlabel('Predicted', fontsize=14)
        plt.ylabel('Actual', fontsize=14)
        plt.title('HybridPulmoClassifier - Testing Confusion Matrix', fontsize=16)
        plt.tight_layout()
        plt.savefig('hybrid_pulmo_testing_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Testing confusion matrix disimpan: hybrid_pulmo_testing_confusion_matrix.png")
    
    def save_results_to_csv(self, results, overall_accuracy):
        """Simpan hasil testing ke file CSV"""
        if not results:
            return
        
        df = pd.DataFrame(results)
        
        # Simpan ke CSV
        df.to_csv('hybrid_testing_results.csv', index=False)
        
        # Buat file summary
        with open('hybrid_testing_summary.txt', 'w') as f:
            f.write("HYBRID PULMO CLASSIFIER - TESTING SUMMARY\n")
            f.write("=" * 50 + "\n")
            f.write(f"Overall Accuracy: {overall_accuracy * 100:.2f}%\n")
            f.write(f"Total Samples: {len(results)}\n")
            f.write(f"Correct Predictions: {sum(1 for r in results if r['correct'])}\n")
            f.write(f"Incorrect Predictions: {sum(1 for r in results if not r['correct'])}\n")
            f.write("\nPer-class Accuracy:\n")
            
            for class_name in self.classes:
                class_results = [r for r in results if r['actual'] == class_name]
                if class_results:
                    correct = sum(1 for r in class_results if r['correct'])
                    accuracy = correct / len(class_results) * 100
                    f.write(f"  {class_name.upper()}: {accuracy:.1f}% ({correct}/{len(class_results)})\n")
        
        print("✅ Detailed results saved: hybrid_testing_results.csv")
        print("✅ Summary saved: hybrid_testing_summary.txt")
    
    def display_sample_predictions(self, results, num_samples=5):
        """Tampilkan beberapa contoh prediksi"""
        print(f"\n🔍 CONTOH PREDIKSI (acak {num_samples} sample):")
        print("-" * 60)
        
        # Ambil sample acak
        if len(results) > num_samples:
            import random
            samples = random.sample(results, num_samples)
        else:
            samples = results
        
        for i, result in enumerate(samples, 1):
            status = "✅ BENAR" if result['correct'] else "❌ SALAH"
            print(f"{i}. File: {result['file']}")
            print(f"   Actual: {result['actual'].upper()}")
            print(f"   Predicted: {result['predicted'].upper()}")
            print(f"   Confidence: {result['confidence']*100:.1f}%")
            print(f"   Status: {status}")
            print()
    
    def predict_single_image(self, image_path):
        """Prediksi single image dengan sistem hybrid"""
        if not os.path.exists(image_path):
            print(f"❌ File {image_path} tidak ditemukan")
            return None
        
        print(f"\n🔍 PREDIKSI SINGLE IMAGE: {os.path.basename(image_path)}")
        print("=" * 50)
        
        try:
            result = self.hybrid_classifier.predict_hybrid(image_path, verbose=True)
            return result
        except Exception as e:
            print(f"❌ Error selama prediksi: {e}")
            return None
    
    def batch_predict(self, folder_path):
        """Prediksi semua gambar dalam folder dengan sistem hybrid"""
        if not os.path.exists(folder_path):
            print(f"❌ Folder {folder_path} tidak ditemukan")
            return
        
        print(f"\n🔍 BATCH PREDICTION: {folder_path}")
        print("=" * 50)
        
        results = []
        predictions_count = {'normal': 0, 'benign': 0, 'malignant': 0}
        supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        
        file_list = [f for f in os.listdir(folder_path) 
                    if f.lower().endswith(supported_formats)]
        
        if not file_list:
            print("❌ Tidak ada gambar yang ditemukan dalam folder")
            return
        
        print(f"📁 Ditemukan {len(file_list)} gambar")
        print("⏳ Memproses...")
        
        for i, filename in enumerate(file_list, 1):
            image_path = os.path.join(folder_path, filename)
            
            # Progress indicator
            if i % 10 == 0 or i == len(file_list):
                print(f"  Progress: {i}/{len(file_list)}")
            
            try:
                result = self.hybrid_classifier.predict_hybrid(image_path, verbose=False)
                
                if result:
                    results.append(result)
                    predictions_count[result['prediction']] += 1
                else:
                    print(f"   ❌ Gagal memproses: {filename}")
                    
            except Exception as e:
                print(f"   ❌ Error pada {filename}: {e}")
        
        # Tampilkan summary hasil
        if results:
            print(f"\n" + "="*60)
            print("📈 HYBRID PULMO CLASSIFIER - BATCH PREDICTION SUMMARY")
            print("="*60)
            print(f"📊 Total gambar diproses: {len(results)}")
            
            print(f"\n🎯 DISTRIBUSI HASIL:")
            for class_name in ['normal', 'benign', 'malignant']:
                count = predictions_count[class_name]
                percentage = (count / len(results)) * 100
                confidence_avg = np.mean([r['confidence'] for r in results if r['prediction'] == class_name]) * 100 if count > 0 else 0
                
                print(f"   {class_name.upper()}: {count} gambar ({percentage:.1f}%)")
                if count > 0:
                    print(f"     📊 Confidence rata-rata: {confidence_avg:.1f}%")
            
            # Simpan hasil batch
            batch_df = pd.DataFrame([{
                'file': r['file'],
                'prediction': r['prediction'],
                'confidence': r['confidence']
            } for r in results])
            
            batch_df.to_csv('batch_prediction_results.csv', index=False)
            print(f"\n✅ Hasil batch disimpan: batch_prediction_results.csv")
            
            return results
        else:
            print("❌ Tidak ada hasil prediksi yang berhasil")
            return []

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

def main():
    testing_system = TestingSystem()
    
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
        print("4. Keluar")
        
        choice = input("\nMasukkan pilihan (1-4): ").strip()
        
        if choice == '1':
            test_dir = "data/test"
            if testing_system.load_models():
                testing_system.evaluate_hybrid(test_dir)
        
        elif choice == '2':
            image_path = input("Masukkan path gambar: ").strip()
            if testing_system.load_models():
                testing_system.predict_single_image(image_path)
        
        elif choice == '3':
            folder_path = input("Masukkan path folder: ").strip()
            if testing_system.load_models():
                testing_system.batch_predict(folder_path)
        
        elif choice == '4':
            print("👋 Terima kasih menggunakan HybridPulmoClassifier!")
            break
        
        else:
            print("❌ Pilihan tidak valid")

if __name__ == "__main__":
    main()