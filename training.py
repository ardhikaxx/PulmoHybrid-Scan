import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
from PIL import Image
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision import datasets
import shutil
from scipy.stats import skew, kurtosis

print("Libraries imported successfully!")
print(f"OpenCV version: {cv2.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Scikit-learn version: {joblib.__version__}")
print(f"PyTorch version: {torch.__version__}")

class PulmoFeatureExtractor:
    """Class untuk ekstraksi fitur dari gambar CT scan paru-paru"""
    
    def __init__(self):
        self.feature_names = []
        
    def extract_features(self, image_path):
        """Ekstraksi fitur komprehensif dari gambar CT scan paru-paru"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Gagal membaca gambar: {image_path}")
                return None
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))
            
            features = []
            
            # 1. Fitur warna (RGB channels)
            for i in range(3):
                channel_data = image[:, :, i].flatten()
                features.extend([
                    np.mean(channel_data),
                    np.std(channel_data),
                    np.median(channel_data),
                    np.min(channel_data),
                    np.max(channel_data)
                ])
            
            # 2. Convert to grayscale untuk fitur tekstur
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 3. Fitur histogram grayscale
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist_features = [
                float(np.mean(hist)),
                float(np.std(hist)),
                float(np.median(hist)),
                float(np.min(hist)),
                float(np.max(hist))
            ]
            features.extend(hist_features)
            
            # 4. Fitur tekstur menggunakan gradient
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
            
            gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
            gradient_features = [
                np.mean(gradient_magnitude),
                np.std(gradient_magnitude),
                np.median(gradient_magnitude)
            ]
            features.extend(gradient_features)
            
            # 5. Fitur bentuk dan kontur
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                else:
                    circularity = 0
            else:
                area = perimeter = circularity = 0
                
            shape_features = [area, perimeter, circularity]
            features.extend(shape_features)
            
            # 6. Statistical texture features
            texture_features = [
                gray.mean(),
                gray.std(),
                np.median(gray),
                gray.var(),
                np.percentile(gray, 25),
                np.percentile(gray, 75),
            ]
            features.extend(texture_features)
            
            # 7. Fitur entropy dan statistik lanjutan
            hist_norm = hist / hist.sum()
            hist_norm = hist_norm[hist_norm > 0]
            entropy = -np.sum(hist_norm * np.log2(hist_norm))
            features.append(entropy)
            
            features.append(skew(gray.flatten()))
            features.append(kurtosis(gray.flatten()))
            
            return np.array(features, dtype=np.float64)
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None

class PulmoNaiveBayesClassifier:
    """Class untuk Naive Bayes Classifier"""
    
    def __init__(self):
        self.model = GaussianNB()
        self.scaler = StandardScaler()
        self.pca = None
        self.classes = ['normal', 'benign', 'malignant']
        self.feature_extractor = PulmoFeatureExtractor()
        
    def load_dataset(self, data_dir):
        """Load dataset dari folder"""
        X = []
        y = []
        
        print("Loading dataset untuk Naive Bayes...")
        
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Directory {class_dir} tidak ditemukan")
                continue
                
            print(f"Processing {class_name} images...")
            count = 0
            
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    image_path = os.path.join(class_dir, filename)
                    features = self.feature_extractor.extract_features(image_path)
                    
                    if features is not None:
                        X.append(features)
                        y.append(class_name)
                        count += 1
                        
                        if count % 50 == 0:
                            print(f"  Processed {count} {class_name} images")
            
            print(f"Completed: {count} {class_name} images loaded")
        
        if len(X) == 0:
            print("Tidak ada gambar yang berhasil diproses!")
            return np.array([]), np.array([])
            
        return np.array(X), np.array(y)
    
    def train(self, X_train, y_train):
        """Training model Naive Bayes"""
        print("Training Naive Bayes model...")
        
        # Standardisasi fitur
        print("Standardizing features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Tentukan jumlah komponen PCA
        n_components = min(30, X_train_scaled.shape[1], X_train_scaled.shape[0])
        self.pca = PCA(n_components=n_components, random_state=42)
        
        # PCA untuk reduksi dimensi
        print("Applying PCA...")
        X_train_pca = self.pca.fit_transform(X_train_scaled)
        
        print(f"Original features: {X_train.shape[1]}")
        print(f"Features after PCA: {X_train_pca.shape[1]}")
        print(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}")
        
        # Training Naive Bayes
        print("Training Naive Bayes classifier...")
        self.model.fit(X_train_pca, y_train)
        
        print("Naive Bayes training completed!")
        
    def predict(self, X):
        """Prediksi data baru"""
        if len(X) == 0:
            return np.array([])
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        return self.model.predict(X_pca)
    
    def predict_proba(self, X):
        """Prediksi probabilitas"""
        if len(X) == 0:
            return np.array([])
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        return self.model.predict_proba(X_pca)
    
    def save_model(self, filename='models/pulmo_naive_bayes.pkl'):
        """Save model Naive Bayes ke file"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'pca': self.pca,
            'classes': self.classes
        }
        joblib.dump(model_data, filename)
        print(f"✅ Model Naive Bayes saved as {filename}")
    
    def load_model(self, filename='models/pulmo_naive_bayes.pkl'):
        """Load model Naive Bayes dari file"""
        model_data = joblib.load(filename)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.pca = model_data['pca']
        self.classes = model_data['classes']
        print(f"✅ Model Naive Bayes loaded from {filename}")

# Definisikan dataset class
class PulmoDataset(Dataset):
    def __init__(self, data_dir, transform=None, classes=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = classes if classes else ['normal', 'benign', 'malignant']
        self.images = []
        self.labels = []
        
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(data_dir, class_name)
            if os.path.exists(class_dir):
                for filename in os.listdir(class_dir):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        self.images.append(os.path.join(class_dir, filename))
                        self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(image_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a dummy image if there's an error
            dummy_image = Image.new('RGB', (224, 224), color='white')
            if self.transform:
                dummy_image = self.transform(dummy_image)
            return dummy_image, label

class PulmoCNNClassifier:
    """Class untuk CNN Classifier dengan Transfer Learning"""
    
    def __init__(self, num_classes=3, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.num_classes = num_classes
        self.classes = ['normal', 'benign', 'malignant']
        self.model = None
        self.transform = None
        self.setup_transforms()
        self.setup_model()
        
    def setup_transforms(self):
        """Setup data transforms untuk training dan validation"""
        self.transform = {
            'train': transforms.Compose([
                transforms.RandomAdjustSharpness(2),
                transforms.RandomAutocontrast(),
                transforms.GaussianBlur(kernel_size=3),
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        }
    
    def setup_model(self):
        """Setup CNN model dengan transfer learning"""
        # Gunakan pre-trained ResNet50
        self.model = models.resnet50(pretrained=True)
        
        # Freeze early layers
        for name, param in self.model.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False
        
        # Replace the final layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),                  
            nn.Dropout(0.3),
            nn.Linear(512, self.num_classes)
        )
        
        self.model = self.model.to(self.device)
        print(f"✅ CNN Model setup completed on {self.device}")
    
    def create_dataset(self, data_dir, split='train'):
        """Create PyTorch dataset dari folder"""
        return PulmoDataset(data_dir, self.transform[split], self.classes)
    
    def train_model(self, train_dir, val_dir, epochs=35, batch_size=8):
        """Train CNN model"""
        print("🚀 Starting CNN Training with Transfer Learning...")
        
        # Create datasets
        train_dataset = self.create_dataset(train_dir, 'train')
        val_dataset = self.create_dataset(val_dir, 'val')
        
        if len(train_dataset) == 0:
            print("❌ No training data found!")
            return [], []
            
        if len(val_dataset) == 0:
            print("❌ No validation data found!")
            return [], []
        
        # Gunakan num_workers=0 untuk menghindari multiprocessing issues di Windows
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        # CLASS WEIGHTS (untuk imbalanced dataset)
        from sklearn.utils.class_weight import compute_class_weight

        # Ambil semua label training dari dataset
        y_train = np.array(train_dataset.labels)

        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )

        class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
        print("Class weights digunakan:", class_weights)
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        
        # Training history
        train_losses = []
        val_accuracies = []
        
        best_accuracy = 0.0
        
        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            print('-' * 50)
            
            # Training phase
            self.model.train()
            running_loss = 0.0
            running_corrects = 0
            
            # Manual progress bar
            print("Training...")
            total_batches = len(train_loader)
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                # MIXUP augmentation
                lam = np.random.beta(0.2, 0.2)
                index = torch.randperm(inputs.size(0)).to(self.device)

                mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
                labels_a, labels_b = labels, labels[index]

                outputs = self.model(mixed_inputs)

                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
                _, preds = torch.max(outputs, 1)

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

                # Print progress setiap 10 batch
                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                    print(f'  Batch {batch_idx + 1}/{total_batches}, Loss: {loss.item():.4f}')
            
            epoch_loss = running_loss / len(train_dataset)
            epoch_acc = running_corrects / len(train_dataset)
            train_losses.append(epoch_loss)
            
            print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Validation phase
            self.model.eval()
            val_running_corrects = 0
            
            print("Validation...")
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                    
                    val_running_corrects += torch.sum(preds == labels.data).item()
            
            val_accuracy = val_running_corrects / len(val_dataset)
            val_accuracies.append(val_accuracy)
            
            print(f'Val Accuracy: {val_accuracy:.4f}')
            
            # Save best model
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                self.save_model('models/pulmo_cnn.pth')
                print(f'✅ New best model saved with accuracy: {best_accuracy:.4f}')
            
            scheduler.step()
        
        print(f'\n🎉 CNN Training completed! Best accuracy: {best_accuracy:.4f}')
        return train_losses, val_accuracies
    
    def predict_single_image(self, image_path):
        """Predict single image"""
        if not os.path.exists(image_path):
            print(f"❌ File {image_path} tidak ditemukan")
            return None, None
        
        self.model.eval()
        
        # Load and preprocess image
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform['val'](image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probs = torch.softmax(outputs, 1)
                _, predicted = torch.max(outputs, 1)
            
            predicted_class = self.classes[predicted.item()]
            probabilities = probs.cpu().numpy()[0]
            
            return predicted_class, probabilities
        except Exception as e:
            print(f"❌ Error predicting image: {e}")
            return None, None
    
    def save_model(self, filename='models/pulmo_cnn.pth'):
        """Save CNN model"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'classes': self.classes,
            'num_classes': self.num_classes
        }, filename)
        print(f"✅ CNN Model saved as {filename}")
    
    def load_model(self, filename='models/pulmo_cnn.pth'):
        """Load CNN model"""
        try:
            checkpoint = torch.load(filename, map_location=self.device)
            self.setup_model()
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.classes = checkpoint['classes']
            self.num_classes = checkpoint['num_classes']
            print(f"✅ CNN Model loaded from {filename}")
        except Exception as e:
            print(f"❌ Error loading CNN model: {e}")

class HybridPulmoClassifier:
    """Sistem Hybrid CNN-Naive Bayes untuk Klasifikasi Citra CT-Scan Kanker Paru-paru"""
    
    def __init__(self):
        self.nb_classifier = PulmoNaiveBayesClassifier()
        self.cnn_classifier = PulmoCNNClassifier()
        self.classes = ['normal', 'benign', 'malignant']
        self.hybrid_weights = {'nb': 0.4, 'cnn': 0.6}  # Bobot untuk ensemble
        
    def train_hybrid(self, data_dir, train_dir, val_dir, epochs=35, batch_size=8):
        """Train kedua model untuk sistem hybrid"""
        print("🚀 HYBRID PULMO CLASSIFIER - TRAINING SYSTEM")
        print("=" * 70)
        
        # Train Naive Bayes Model
        print("\n" + "="*50)
        print("🧪 TRAINING NAIVE BAYES MODEL")
        print("="*50)
        
        X, y = self.nb_classifier.load_dataset(data_dir)
        
        if len(X) > 0:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            print(f"\nTraining set: {X_train.shape[0]} samples")
            print(f"Testing set: {X_test.shape[0]} samples")
            
            self.nb_classifier.train(X_train, y_train)
            self.nb_classifier.save_model('models/pulmo_naive_bayes.pkl')
        else:
            print("❌ Tidak ada data untuk training Naive Bayes")
        
        # Train CNN Model
        print("\n" + "="*50)
        print("🧠 TRAINING CNN WITH TRANSFER LEARNING")
        print("="*50)
        
        if os.path.exists(train_dir) and os.path.exists(val_dir):
            try:
                print("Memulai training CNN...")
                train_losses, val_accuracies = self.cnn_classifier.train_model(
                    train_dir, val_dir, epochs=epochs, batch_size=batch_size
                )
                
                # Plot training history
                if train_losses and val_accuracies:
                    self.plot_training_history(train_losses, val_accuracies)
                
            except Exception as e:
                print(f"❌ Error during CNN training: {e}")
        else:
            print("❌ Folder training atau validation tidak ditemukan")
    
    def plot_training_history(self, train_losses, val_accuracies):
        """Plot training history CNN"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses)
        plt.title('CNN Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(val_accuracies)
        plt.title('CNN Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('hybrid_pulmo_cnn_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_hybrid(self, image_path, verbose=True):
        """Prediksi hybrid menggunakan kedua model - Hanya tampilkan hasil akhir"""
        if not os.path.exists(image_path):
            print(f"❌ File {image_path} tidak ditemukan")
            return None
        
        # Prediksi dengan Naive Bayes
        features = self.nb_classifier.feature_extractor.extract_features(image_path)
        if features is not None:
            nb_proba = self.nb_classifier.predict_proba([features])[0]
        else:
            nb_proba = np.array([0.33, 0.33, 0.33])
        
        # Prediksi dengan CNN
        cnn_pred, cnn_proba = self.cnn_classifier.predict_single_image(image_path)
        if cnn_pred is None or cnn_proba is None:
            cnn_proba = np.array([0.33, 0.33, 0.33])
        
        # Gabungkan probabilitas dengan weighted average
        hybrid_proba = (self.hybrid_weights['nb'] * nb_proba + 
                       self.hybrid_weights['cnn'] * np.array(cnn_proba))
        
        hybrid_pred = self.classes[np.argmax(hybrid_proba)]
        confidence = max(hybrid_proba)
        
        # Hanya tampilkan hasil jika verbose=True
        if verbose:
            print("\n" + "="*60)
            print("🤖 HYBRID PULMO CLASSIFIER - PREDICTION RESULT")
            print("="*60)
            print(f"🎯 PREDIKSI: {hybrid_pred.upper()}")
            print(f"📊 CONFIDENCE: {confidence*100:.1f}%")
            print("\nProbabilitas Akhir:")
            for i, class_name in enumerate(self.classes):
                prob_percent = hybrid_proba[i] * 100
                print(f"  {class_name.upper()}: {prob_percent:.1f}%")
            print("="*60)
            
            # Tampilkan rekomendasi
            self.display_recommendation(hybrid_pred, confidence)
        
        return {
            'prediction': hybrid_pred,
            'confidence': confidence,
            'probabilities': dict(zip(self.classes, hybrid_proba)),
            'file': os.path.basename(image_path)
        }
    
    def display_recommendation(self, prediction, confidence):
        """Tampilkan rekomendasi berdasarkan prediksi"""
        print(f"\n💡 REKOMENDASI:")
        if prediction == 'normal':
            print("✅ STATUS: NORMAL - Paru-paru dalam kondisi sehat")
            print("   ✅ Tidak diperlukan tindakan khusus")
            print("   💡 Tetap jaga kesehatan dengan pola hidup sehat")
        elif prediction == 'benign':
            print("⚠️ STATUS: JINAK - Tumor jinak terdeteksi")
            print("   📋 Disarankan konsultasi rutin dengan dokter")
            print("   🔍 Monitoring berkala diperlukan")
            if confidence < 0.7:
                print("   ⚠️  Confidence rendah, disarankan pemeriksaan ulang")
        elif prediction == 'malignant':
            print("🚨 STATUS: GANAS - Kanker ganas terdeteksi")
            print("   🏥 Segera konsultasi dengan dokter spesialis onkologi")
            print("   ⚡ Penanganan medis segera diperlukan")
            if confidence < 0.8:
                print("   ⚠️  Confidence sedang, konfirmasi dengan pemeriksaan tambahan")
    
    def evaluate_hybrid(self, test_dir):
        """Evaluasi sistem hybrid pada test set"""
        print("\n" + "="*60)
        print("📊 HYBRID PULMO CLASSIFIER - EVALUATION")
        print("="*60)
        
        hybrid_predictions = []
        hybrid_true = []
        results_by_class = {'normal': [], 'benign': [], 'malignant': []}
        
        for class_name in self.classes:
            class_dir = os.path.join(test_dir, class_name)
            if os.path.exists(class_dir):
                print(f"\n🔍 Processing {class_name} images...")
                count = 0
                
                for filename in os.listdir(class_dir):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        image_path = os.path.join(class_dir, filename)
                        
                        # Gunakan verbose=False untuk menghindari output berulang
                        result = self.predict_hybrid(image_path, verbose=False)
                        if result:
                            hybrid_predictions.append(result['prediction'])
                            hybrid_true.append(class_name)
                            results_by_class[class_name].append(result)
                            count += 1
                
                print(f"✅ Processed {count} {class_name} images")
        
        if hybrid_true:
            # Tampilkan summary per kelas
            print("\n" + "="*60)
            print("🎯 HASIL EVALUASI SISTEM HYBRID")
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
            
            hybrid_accuracy = accuracy_score(hybrid_true, hybrid_predictions)
            print(f"\n📈 AKURASI KESELURUHAN: {hybrid_accuracy * 100:.2f}%")
            print(f"📊 TOTAL SAMPLE TEST: {len(hybrid_true)}")
            print("="*60)
            print("\nLaporan Klasifikasi:\n", classification_report(hybrid_true, hybrid_predictions))
            
            # Plot confusion matrix untuk hybrid
            cm_hybrid = confusion_matrix(hybrid_true, hybrid_predictions, labels=self.classes)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm_hybrid, annot=True, fmt='d', cmap='Greens',
                       xticklabels=self.classes, 
                       yticklabels=self.classes)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('HybridPulmoClassifier - Confusion Matrix')
            plt.tight_layout()
            plt.savefig('hybrid_pulmo_confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            return hybrid_accuracy
        else:
            print("❌ Tidak ada data untuk evaluasi hybrid")
            return 0.0
    
    def batch_predict_hybrid(self, folder_path):
        """Prediksi semua gambar dalam folder dengan sistem hybrid - Tampilkan summary saja"""
        if not os.path.exists(folder_path):
            print(f"❌ Folder {folder_path} tidak ditemukan")
            return
        
        print(f"\n🔍 Memproses gambar di folder: {folder_path}")
        
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
            
            # Progress indicator sederhana
            if i % 10 == 0 or i == len(file_list):
                print(f"  Progress: {i}/{len(file_list)}")
            
            try:
                # Gunakan verbose=False untuk menghindari output berulang
                result = self.predict_hybrid(image_path, verbose=False)
                
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
            
            # Tampilkan 3 contoh pertama dari setiap kelas
            print(f"\n🔍 CONTOH HASIL PREDIKSI:")
            displayed_examples = 0
            for class_name in ['normal', 'benign', 'malignant']:
                class_results = [r for r in results if r['prediction'] == class_name]
                if class_results:
                    # Ambil maksimal 2 contoh per kelas
                    examples = class_results[:2]
                    for example in examples:
                        print(f"\n   📄 File: {example['file']}")
                        print(f"   🎯 Prediksi: {example['prediction'].upper()}")
                        print(f"   📊 Confidence: {example['confidence']*100:.1f}%")
                        displayed_examples += 1
            
            print(f"\n💡 Total contoh ditampilkan: {displayed_examples} dari {len(results)} gambar")
            print("="*60)
            
            return results
        else:
            print("❌ Tidak ada hasil prediksi yang berhasil")
            return []
    
    def save_models(self):
        """Save semua model"""
        self.nb_classifier.save_model('models/pulmo_naive_bayes.pkl')
        self.cnn_classifier.save_model('models/pulmo_cnn.pth')
        print("✅ All models saved successfully!")
    
    def load_models(self):
        """Load semua model"""
        self.nb_classifier.load_model('models/pulmo_naive_bayes.pkl')
        self.cnn_classifier.load_model('models/pulmo_cnn.pth')
        print("✅ All models loaded successfully!")

def create_sample_dataset():
    """Buat sample dataset untuk testing jika belum ada data"""
    print("Membuat struktur folder dataset...")
    
    folders = [
        'data/train/normal',
        'data/train/benign', 
        'data/train/malignant',
        'data/val/normal',
        'data/val/benign',
        'data/val/malignant',
        'data/test/normal',
        'data/test/benign',
        'data/test/malignant',
        'models'
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"Created: {folder}")
    
    print("\nStruktur folder berhasil dibuat!")
    print("Silakan tambahkan gambar Anda ke folder:")
    print("- data/train/ (untuk training)")
    print("- data/val/ (untuk validation)") 
    print("- data/test/ (untuk testing)")

def copy_files_for_validation():
    """Copy files dari train ke val untuk membuat validation set"""
    print("Membuat validation set dari training data...")
    
    for class_name in ['normal', 'benign', 'malignant']:
        src_dir = f'data/train/{class_name}'
        val_dir = f'data/val/{class_name}'
        
        if os.path.exists(src_dir):
            # Buat folder val jika belum ada
            os.makedirs(val_dir, exist_ok=True)
            
            # Get list of images
            images = [f for f in os.listdir(src_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if images:
                # Ambil 20% untuk validation
                val_count = max(1, int(len(images) * 0.2))
                val_images = images[:val_count]
                
                print(f"Memindahkan {len(val_images)} gambar {class_name} ke validation set...")
                
                for img in val_images:
                    src_path = os.path.join(src_dir, img)
                    dst_path = os.path.join(val_dir, img)
                    
                    # Copy file jika belum ada di destination
                    if not os.path.exists(dst_path):
                        try:
                            shutil.copy2(src_path, dst_path)
                            print(f"  Copied: {img}")
                        except Exception as e:
                            print(f"  Error copying {img}: {e}")

def main():
    # Cek apakah folder data exists
    if not os.path.exists('data/train'):
        print("Folder data tidak ditemukan!")
        create_sample_dataset()
        return
    
    # Inisialisasi Hybrid Classifier
    hybrid_classifier = HybridPulmoClassifier()
    
    print("🚀 HYBRID PULMO CLASSIFIER - TRAINING SYSTEM")
    print("=" * 70)
    print("Sistem Hybrid CNN-Naive Bayes untuk Klasifikasi Kanker Paru-paru")
    print("=" * 70)
    
    # Check if validation folder exists and has data
    val_exists = os.path.exists('data/val') and any(
        os.path.exists(f'data/val/{cls}') and any(os.listdir(f'data/val/{cls}')) 
        for cls in ['normal', 'benign', 'malignant']
    )
    
    if not val_exists:
        print("Validation folder tidak ditemukan atau kosong, membuat validation set...")
        copy_files_for_validation()
    
    # Train hybrid system
    hybrid_classifier.train_hybrid(
        data_dir='data/train',
        train_dir='data/train'
        val_dir='data/val',
        epochs=35,
        batch_size=8
    )
    
    # Evaluate hybrid system jika test folder ada
    if os.path.exists('data/test'):
        print("\n" + "="*70)
        print("📊 EVALUATING HYBRID SYSTEM ON TEST DATA")
        print("="*70)
        hybrid_accuracy = hybrid_classifier.evaluate_hybrid('data/test')
    else:
        print("ℹ️  Folder test tidak ditemukan, skip hybrid evaluation")
    
    # Save all models
    hybrid_classifier.save_models()
    
    print("\n" + "="*70)
    print("🎉 HYBRID PULMO CLASSIFIER TRAINING COMPLETED!")
    print("="*70)
    print("✅ Naive Bayes Model: models/pulmo_naive_bayes.pkl")
    print("✅ CNN Model: models/pulmo_cnn.pth")
    print("📊 Training history saved: hybrid_pulmo_cnn_training_history.png")
    print("📊 Confusion matrix saved: hybrid_pulmo_confusion_matrix.png")
    print("\nModel siap digunakan untuk prediksi hybrid!")

if __name__ == "__main__":
    main()