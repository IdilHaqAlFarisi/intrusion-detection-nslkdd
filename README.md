# Intrusion Detection System (IDS) Menggunakan Decision Tree dengan NSL-KDD Dataset

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg)](https://scikit-learn.org/)

**Project - Kecerdasan Artifisial (Artificial Intelligence)**

Implementasi sistem deteksi intrusi jaringan (Intrusion Detection System/IDS) menggunakan algoritma Decision Tree dengan optimasi Pre-Pruning pada dataset NSL-KDD.



## 📋 Deskripsi Project

Project ini mengimplementasikan sistem deteksi intrusi jaringan menggunakan algoritma Machine Learning, khususnya Decision Tree. Sistem ini mampu mengklasifikasikan trafik jaringan menjadi kategori normal atau berbagai jenis serangan cyber dengan akurasi sangat tinggi (>99%).

### Fitur Utama:
- ✅ **Auto-Install Library**: Instalasi otomatis semua dependensi via requirements.txt
- ✅ **Pre-Pruning Optimization**: Optimasi Decision Tree dengan parameter `max_depth`, `min_samples_split`, dan `min_samples_leaf`
- ✅ **Comprehensive Analysis**: Evaluasi lengkap dengan confusion matrix, classification report, dan ROC-AUC
- ✅ **Visualization**: Visualisasi struktur Decision Tree dan analisis feature importance
- ✅ **Model Persistence**: Penyimpanan model terlatih untuk deployment



## 📊 Dataset: NSL-KDD

NSL-KDD adalah dataset standar untuk evaluasi sistem deteksi intrusi, merupakan perbaikan dari dataset KDD Cup 1999. Dataset ini berisi 41 fitur yang merepresentasikan berbagai karakteristik koneksi jaringan.

### File Dataset:
- `KDDTrain+.txt`: Data training (original)
- `KDDTest+.txt`: Data testing (original)
- `KDDTrain_processed.csv`: Data training (processed)
- `KDDTest_processed.csv`: Data testing (processed)
- `KDD_combined_processed.csv`: Kombinasi train & test (processed)

### Jenis Serangan:
Dataset mencakup berbagai jenis serangan seperti:
- **DoS (Denial of Service)**: neptune, smurf, pod, teardrop, land, back
- **Probe**: satan, ipsweep, portsweep, nmap
- **R2L (Remote to Local)**: warezclient, guess_passwd, warezmaster, imap, ftp_write, multihop, phf, spy
- **U2R (User to Root)**: buffer_overflow, rootkit, loadmodule, perl

---

## 🚀 Instalasi

### Prerequisites:
- Python 3.8 atau lebih tinggi
- pip (Python package manager)
- Jupyter Notebook atau JupyterLab

### Langkah Instalasi:

1. **Clone repository ini:**
```bash
git clone https://github.com/IdilHaqAlFarisi/intrusion-detection-nslkdd.git
cd intrusion-detection-nslkdd
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Jalankan Jupyter Notebook:**
```bash
jupyter notebook IDS_Offline.ipynb
```

---

## 💻 Cara Penggunaan

1. **Buka notebook**: `IDS_Offline.ipynb`
2. **Jalankan semua cell secara berurutan**: Cell → Run All
3. Notebook akan secara otomatis:
   - Install semua library yang dibutuhkan
   - Load dan preprocess data
   - Train model Decision Tree dengan optimasi Pre-Pruning
   - Evaluasi performa model
   - Visualisasi hasil dan struktur tree
   - Menyimpan model ke file `model_ids_dt.pkl`

---

## 📈 Hasil dan Performa

### Akurasi Model:
- **Training Accuracy**: 99.44%
- **Validation Accuracy**: 99.30%
- **Test Accuracy**: >99%
- **Gap (Overfitting Metric)**: 0.14% (sangat baik)

### Metrik Evaluasi:
- **ROC-AUC Score**: Sangat tinggi (mendekati 1.0)
- **Precision, Recall, F1-Score**: >99% untuk mayoritas kelas
- **Confusion Matrix**: Misklasifikasi sangat rendah

### Key Findings:
✅ Model memiliki **kemampuan generalisasi yang sangat baik**  
✅ **Bebas dari Overfitting** (gap <0.2%)  
✅ **Balanced Performance** pada semua metrik evaluasi  
✅ **Efisien secara komputasi** dengan Pre-Pruning

---

## 📁 Struktur File

```
intrusion-detection-nslkdd/
│
├── IDS_Offline.ipynb                # Main notebook - Full pipeline IDS
├── README.md                        # Dokumentasi project
├── requirements.txt                 # Python dependencies
│
├── KDDTrain+.txt                    # Dataset training (original)
├── KDDTest+.txt                     # Dataset testing (original)
├── KDDTrain_processed.csv           # Dataset training (processed)
├── KDDTest_processed.csv            # Dataset testing (processed)
├── KDD_combined_processed.csv       # Dataset gabungan (processed)
│
└── model_ids_dt.pkl                 # Trained Decision Tree model
```

---

## 🛠️ Teknologi yang Digunakan

- **Python 3.8+**: Bahasa pemrograman utama
- **Scikit-Learn**: Library machine learning untuk Decision Tree
- **Pandas**: Data manipulation dan analysis
- **NumPy**: Numerical computing
- **Matplotlib & Seaborn**: Data visualization
- **Jupyter Notebook**: Interactive development environment

---

## 📚 Metodologi

### 1. Data Preprocessing:
- Handling missing values
- Feature encoding (Label Encoding & One-Hot Encoding)
- Feature scaling (StandardScaler)
- Train-test-validation split

### 2. Model Development:
- **Algorithm**: Decision Tree Classifier
- **Optimization**: Pre-Pruning dengan hyperparameter tuning
- **Key Parameters**:
  - `max_depth`: Membatasi kedalaman tree
  - `min_samples_split`: Minimum samples untuk split node
  - `min_samples_leaf`: Minimum samples di leaf node
  - `criterion`: Gini impurity

### 3. Evaluation:
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)
- ROC-AUC Score
- Cross-validation
- Feature Importance Analysis
- Tree Visualization

---

## 🔍 Analisis Mendalam

Project ini mencakup analisis mendalam mengenai:
- **Struktur Decision Tree**: Visualisasi dan interpretasi pohon keputusan
- **Feature Importance**: Identifikasi fitur-fitur paling berpengaruh
- **Efek Pre-Pruning**: Perbandingan model dengan/tanpa pruning
- **Overfitting Prevention**: Strategi mencegah overfitting
- **Performance Optimization**: Teknik optimasi performa model

---

## 📝 Referensi

1. **NSL-KDD Dataset**: [https://www.unb.ca/cic/datasets/nsl.html](https://www.unb.ca/cic/datasets/nsl.html)
2. **Scikit-Learn Documentation**: [https://scikit-learn.org/](https://scikit-learn.org/)
3. **Decision Tree Algorithm**: Breiman, L., et al. (1984). Classification and Regression Trees
4. **Intrusion Detection Systems**: Various academic papers on IDS using Machine Learning

---

## 📧 Kontak

Untuk pertanyaan atau diskusi lebih lanjut, silakan hubungi melalui repository issues atau pull requests.

---

## 📄 Lisensi

Project ini dibuat untuk keperluan akademik - Final Project Kecerdasan Artifisial.

---

**⭐ Jika project ini bermanfaat, jangan lupa berikan star!**
