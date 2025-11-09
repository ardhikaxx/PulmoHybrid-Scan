### PulmoHybrid-Scan: Sistem Hybrid CNN-Naive Bayes untuk Klasifikasi Citra CT-Scan Kanker Paru-paru

PulmoHybrid-Scan adalah sistem kecerdasan buatan canggih yang mengintegrasikan metode Convolutional Neural Network (CNN) dan Naive Bayes Classifier untuk klasifikasi citra CT-Scan kanker paru-paru. Sistem ini dirancang untuk mendeteksi tiga kategori utama: Normal, Benign (Jinak), dan Malignant (Ganas) dengan akurasi tinggi.

## Fitur Utama

1. Sistem Hybrid yang menggabungkan keunggulan CNN dan Naive Bayes
2. Akurasi Tinggi dengan weighted voting mechanism
3. Deteksi Tiga Kelas: Normal, Benign, Malignant
4. Visualisasi Komprehensif dengan confusion matrix dan analisis detail
5. Deteksi Area Abnormal pada citra CT-Scan
6. Model Persistence untuk penggunaan berulang
7. Training Monitoring dengan grafik loss dan accuracy

## Teknologi yang Digunakan
# Deep Learning & Computer Vision
- PyTorch & TorchVision - CNN dengan Transfer Learning (ResNet50)
- OpenCV - Image processing dan feature extraction
- PIL (Pillow) - Image manipulation

# Machine Learning & Statistics
- Scikit-learn - Naive Bayes Classifier, PCA, StandardScaler
- NumPy & SciPy - Komputasi numerik dan statistik
- Pandas - Data processing dan analisis

## Visualisasi
- Matplotlib & Seaborn - Plotting dan visualisasi data
- Confusion Matrix - Evaluasi performa model

## ⚠️Disclaimer
Sistem ini merupakan alat bantu diagnosis dan tidak menggantikan konsultasi medis profesional. Selalu konsultasikan dengan dokter spesialis untuk diagnosis dan penanganan medis.