
# Laporan Proyek Machine Learning - Permata Ayu Rahmawati

## Project Overview

Proyek ini bertujuan untuk membangun model klasifikasi prediktif yang dapat mengidentifikasi risiko diabetes pada pasien berdasarkan data kesehatan yang dikumpulkan dari Pima Indians Diabetes Database. Proyek ini penting karena diabetes merupakan salah satu penyakit kronis yang berdampak besar pada kualitas hidup dan memerlukan deteksi dini untuk penanganan lebih cepat.

Penyakit diabetes telah menjadi perhatian global karena peningkatan prevalensinya. Menurut WHO, pada tahun 2021 diperkirakan lebih dari 537 juta orang di dunia hidup dengan diabetes. Deteksi dini melalui pendekatan prediktif dapat membantu dalam penanganan lebih cepat dan pencegahan komplikasi.  
Referensi: [World Health Organization - Diabetes](https://www.who.int/news-room/fact-sheets/detail/diabetes)
Referensi Dataset: [Pima Indians Diabetes Database - Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

## Business Understanding

### Problem Statements
- Bagaimana mengidentifikasi pasien yang berisiko tinggi terkena diabetes berdasarkan data medis?
- Bagaimana membangun model prediksi yang akurat untuk membantu proses skrining awal?

### Goals
- Mengklasifikasikan pasien ke dalam dua kelas: memiliki diabetes (1) atau tidak (0).
- Meningkatkan performa model prediktif melalui teknik tuning dan balancing.

### Solution Statements
- Menggunakan model XGBoost sebagai baseline dan model utama karena performanya yang unggul untuk data tabular dan kemampuan menangani fitur tidak terstandarisasi.
- Menerapkan teknik oversampling dengan SMOTE untuk menangani data imbalance karena metode ini menghasilkan sampel sintetis yang merepresentasikan distribusi minor class lebih baik.
- Melakukan hyperparameter tuning dengan GridSearchCV dan StratifiedKFold untuk meningkatkan akurasi dan generalisasi model.

## Data Understanding

Dataset yang digunakan berasal dari National Institute of Diabetes and Digestive and Kidney Diseases, dan tersedia secara publik melalui Kaggle: [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database/data)

Tujuan utama dataset ini adalah untuk memprediksi secara diagnostik apakah seorang pasien menderita diabetes berdasarkan sejumlah pengukuran diagnostik. Semua pasien dalam dataset ini adalah perempuan berusia minimal 21 tahun dari keturunan Pima Indian.

Dataset terdiri dari 768 sampel dan 8 fitur medis serta 1 label target (`Outcome`):
- `Pregnancies`: jumlah kehamilan
- `Glucose`: kadar glukosa darah
- `BloodPressure`: tekanan darah diastolik
- `SkinThickness`: ketebalan lipatan kulit trisep
- `Insulin`: kadar insulin serum 2 jam
- `BMI`: indeks massa tubuh
- `DiabetesPedigreeFunction`: fungsi silsilah diabetes (indikasi riwayat keluarga)
- `Age`: usia (dalam tahun)
- `Outcome`: label (0 = tidak diabetes, 1 = diabetes)

EDA menunjukkan:
- Data imbalance (Kelas 0 = 400, Kelas 1 = 214)
- Glucose, Age, dan BMI memiliki korelasi tertinggi terhadap `Outcome`
- Terdapat nilai 0 yang dianggap sebagai missing pada beberapa fitur medis

Dari visualisasi heatmap korelasi, terlihat bahwa fitur `Glucose` dan `BMI` memiliki korelasi positif yang cukup tinggi terhadap variabel target `Outcome`. Selain itu, distribusi nilai 0 pada kolom seperti `Insulin` dan `SkinThickness` menunjukkan kemungkinan adanya missing value, yang perlu ditangani pada tahap preprocessing.

## Data Preparation
*Pada tahap Data Preparation dilakukan beberapa tahapan penting:

- Penanganan Missing Value: Mengganti nilai 0 pada kolom medis (Glucose, BloodPressure, SkinThickness, Insulin, dan BMI) menggunakan nilai median untuk mempertahankan distribusi alami data.
- Pembagian Data: Dataset dibagi menjadi data train dan test menggunakan stratified split dengan rasio 80:20, untuk menjaga proporsi kelas.
- Handling Imbalance: Oversampling menggunakan SMOTE diterapkan pada data train untuk menyeimbangkan jumlah pasien diabetes dan non-diabetes.
- Normalisasi Data: Setelah SMOTE, semua fitur numerikal dinormalisasi menggunakan StandardScaler, untuk membuat distribusi fitur seragam, mempercepat konvergensi model, dan meningkatkan performa prediksi.

## Modeling

1. Model 1: XGBoost Classifier
Pembahasan Cara Kerja:
XGBoost (Extreme Gradient Boosting) adalah algoritma boosting berbasis pohon keputusan. XGBoost membangun model secara iteratif, di mana setiap pohon baru berusaha memperbaiki kesalahan dari pohon sebelumnya menggunakan konsep gradient descent pada fungsi loss.

- Proses utamanya meliputi:
Membuat pohon-pohon kecil berturut-turut.
Memberikan bobot lebih besar pada kesalahan prediksi sebelumnya.
Melakukan regularisasi untuk mencegah overfitting.
Optimasi dilakukan menggunakan teknik shrinkage (learning rate) dan subsampling.

- Pembahasan Parameter:
Default Parameters (sebelum tuning):
max_depth=3, learning_rate=0.3, n_estimators=100, subsample=1, gamma=0

2. Tuned Parameters (hasil tuning GridSearchCV):
max_depth=7, learning_rate=0.1, n_estimators=200, gamma=0, subsample=0.8

- Kelebihan:
Handal pada dataset tabular.
Mampu menangani missing value secara internal.
Mampu mengatasi overfitting dengan regularisasi.

- Kekurangan (opsional):
Membutuhkan tuning hyperparameter yang teliti untuk performa optimal.
Bisa boros memori untuk dataset besar.


## Evaluation

- Metrik Evaluasi yang Digunakan:

Accuracy: Proporsi prediksi benar terhadap semua prediksi.
Precision: Akurasi prediksi positif.
Recall: Kemampuan model menangkap semua kasus positif (sangat penting dalam kasus medis).
F1-Score: Harmonic mean antara precision dan recall, berguna untuk data imbalance.
ROC-AUC: Mengukur kemampuan model membedakan kelas 0 dan 1 di berbagai threshold, penting untuk data imbalance.

- Paparan Hasil Evaluasi:
Baseline Model (default XGBoost):

        Accuracy: 77%

        ROC-AUC: 0.814

        F1-Score kelas 1: 65%

    Tuned Model (hasil tuning):

        Accuracy: 75%

        ROC-AUC: 0.824

        F1-Score kelas 1: 66%

- Komparasi dan Penentuan Model Terbaik:
Meskipun akurasi sedikit menurun, tuned model menunjukkan peningkatan ROC-AUC dan F1-Score untuk kelas minoritas (penderita diabetes).
Karena fokus bisnis adalah mendeteksi pasien diabetes (bukan sekadar meningkatkan akurasi keseluruhan), Tuned Model dipilih sebagai model terbaik.

### Hubungan ke Business Understanding:

- Problem Statement Terjawab: Model dapat mengidentifikasi pasien berisiko tinggi diabetes.
- Goals Tercapai: Meningkatkan kemampuan prediksi dengan balancing data dan tuning hyperparameter.
- Solution Statement Berdampak: Implementasi XGBoost dengan SMOTE berhasil meningkatkan deteksi pasien diabetes tanpa mengorbankan banyak false positives, sesuai dengan tujuan bisnis deteksi dini.

### Hasil Evaluasi
- Tuned model menunjukkan peningkatan pada ROC-AUC dan F1 kelas 1
- Model menunjukkan performa cukup baik untuk prediksi risiko diabetes

## Sample Prediction

Contoh data pasien:
```
[6, 148, 72, 35, 0, 33.6, 0.627, 50]
```

Hasil:
- Prediksi: **Diabetes
- Probabilitas: 97.05%

Model berhasil memberikan prediksi yang masuk akal berdasarkan input medis dengan probabilitas tinggi.

## Acknowledgements

Dataset ini pertama kali digunakan oleh:  
Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988). *Using the ADAP learning algorithm to forecast the onset of diabetes mellitus*. Proceedings of the Symposium on Computer Applications and Medical Care (pp. 261–265). IEEE Computer Society Press.

## References

1. World Health Organization. (2021). *Diabetes Fact Sheet*. Retrieved from: [https://www.who.int/news-room/fact-sheets/detail/diabetes](https://www.who.int/news-room/fact-sheets/detail/diabetes)  
2. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.  
3. Kaggle. (2023). *Pima Indians Diabetes Database*. Retrieved from: [https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)  
4. Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988). *Using the ADAP learning algorithm to forecast the onset of diabetes mellitus*. Proceedings of the Symposium on Computer Applications and Medical Care (pp. 261–265). IEEE Computer Society Press.  
