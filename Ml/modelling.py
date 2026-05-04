import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from processing import Preprocessing


class DataPipeline:
    def __init__(self, base_path):
        self.preprocessor = Preprocessing(base_path)

    def muat_dan_proses(self, file_name, text_col='text', label_col='label_name'):
        df = self.preprocessor.muat_dataframe(file_name)
        print(f"  Memproses {len(df)} sampel dari {file_name}...")
        df['text_clean'] = df[text_col].apply(self.preprocessor.preprocess_full)
        X = df['text_clean']
        y = df[label_col]
        return X, y, df


def buat_tfidf_vectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9,
                          max_features=10000, sublinear_tf=True):
    return TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
    )


def buat_pipeline(model_type='logistic_regression', ngram_range=(1, 2),
                  min_df=3, max_df=0.9, max_features=10000):
    vectorizer = buat_tfidf_vectorizer(ngram_range, min_df, max_df, max_features)
    smote = SMOTE(random_state=42)
    if model_type == 'logistic_regression':
        model = SGDClassifier(
            loss='log_loss',          
            penalty='l2',
            alpha=1e-4,
            max_iter=1000,
            early_stopping=True,        
            validation_fraction=0.1,   
            n_iter_no_change=5,         
            tol=1e-3,
            random_state=42,
            verbose=1,               
        )
    elif model_type == 'linear_svm':
        model = SGDClassifier(
            loss='hinge',               
            penalty='l2',
            alpha=1e-4,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=5,
            tol=1e-3,
            random_state=42,
            verbose=1,
        )
    else:
        raise ValueError(
            f"model_type harus 'logistic_regression' atau 'linear_svm', "
            f"bukan '{model_type}'"
        )

    return ImbPipeline([
        ('tfidf', vectorizer),
        ('smote', smote),
        ('model', model),
    ])

def latih_model(pipeline, X_train, y_train):
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluasi_model(pipeline, X_test, y_test, label_names=None):
    y_pred = pipeline.predict(X_test)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    report = classification_report(y_test, y_pred, target_names=label_names)
    cm = confusion_matrix(y_test, y_pred, labels=label_names)
    print("=" * 70)
    print("HASIL EVALUASI MODEL")
    print("=" * 70)
    print(f"\n  Macro-F1 Score: {macro_f1:.4f}")
    print(f"\nClassification Report:\n{report}")
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
    ax.set_title('Confusion Matrix -- Analisis Misklasifikasi Antar Kelas',fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()
    return {
        'macro_f1': macro_f1,
        'classification_report': report,
        'confusion_matrix': cm,
    }

def visualisasi_top_kata_per_kelas(pipeline, top_n=15):
    vectorizer = pipeline.named_steps['tfidf']
    model = pipeline.named_steps['model']
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_
    classes = model.classes_
    num_classes = len(classes)
    n_cols = min(2, num_classes)
    n_rows = (num_classes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
    if num_classes == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = np.array(axes).reshape(n_rows, n_cols)
    axes_flat = axes.flatten()
    for idx, cls in enumerate(classes):
        ax = axes_flat[idx]
        top_indices = np.argsort(coefs[idx])[-top_n:]
        top_words = [feature_names[i] for i in top_indices]
        top_values = [coefs[idx][i] for i in top_indices]
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))
        ax.barh(range(top_n), top_values, color=colors)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_words)
        ax.set_xlabel('Bobot Koefisien (Prediktif)', fontsize=10)
        ax.set_title(f'Top {top_n} Kata Prediktif: {cls}',
        fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
    for idx in range(len(classes), len(axes_flat)):
        axes_flat[idx].axis('off')

    plt.suptitle(
        'Interpretasi Model: Kata dengan Bobot Prediktif Tertinggi per Kelas\n'
        '(Koefisien model linear ~ konsep lift/chi-square dari EDA)',
        fontsize=13, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    plt.show()

def simpan_model(pipeline, path):
    """Menyimpan pipeline model ke file menggunakan joblib."""
    joblib.dump(pipeline, path)
    print(f"  Model disimpan ke: {path}")

def muat_model(path):
    """Memuat pipeline model dari file."""
    pipeline = joblib.load(path)
    print(f"  Model dimuat dari: {path}")
    return pipeline



if __name__ == '__main__':
    # --- Konfigurasi ---
    BASE_PATH = r"D:\Semester 6\Tubes sg ai\Data"
    MODEL_TYPE = 'logistic_regression'  # 'logistic_regression' atau 'linear_svm'
    MODEL_SAVE_PATH = os.path.join(BASE_PATH, '..', 'Ml', 'model_pipeline.pkl')

    # --- 1. Muat & Proses Data ---
    print("=" * 70)
    print("STEP 1: Memuat dan memproses data...")
    print("=" * 70)

    data_pipeline = DataPipeline(BASE_PATH)
    X_train, y_train, df_train = data_pipeline.muat_dan_proses('train.json')
    X_test, y_test, df_test = data_pipeline.muat_dan_proses('test.json')

    label_names = sorted(y_train.unique())
    print(f"\n  Train: {len(X_train)} sampel")
    print(f"  Test : {len(X_test)} sampel")
    print(f"  Kelas: {label_names}")
    print(f"  Distribusi train:\n{y_train.value_counts().to_string()}")

    # --- 2. Buat & Latih Pipeline ---
    print("\n" + "=" * 70)
    print(f"STEP 2: Membuat dan melatih pipeline ({MODEL_TYPE})...")
    print("  TF-IDF (unigram+bigram) -> SMOTE -> SGDClassifier (early stopping)")
    print("=" * 70)

    pipeline = buat_pipeline(model_type=MODEL_TYPE)
    pipeline = latih_model(pipeline, X_train, y_train)
    print("\n  Model berhasil dilatih!")

    # --- 3. Evaluasi ---
    print("\n" + "=" * 70)
    print("STEP 3: Evaluasi model pada test set...")
    print("=" * 70)

    hasil = evaluasi_model(pipeline, X_test, y_test, label_names=label_names)

    # --- 4. Interpretasi ---
    print("\n" + "=" * 70)
    print("STEP 4: Interpretasi model -- Top kata prediktif per kelas...")
    print("=" * 70)

    visualisasi_top_kata_per_kelas(pipeline, top_n=15)

    # --- 5. Simpan Model ---
    print("\n" + "=" * 70)
    print("STEP 5: Menyimpan model...")
    print("=" * 70)

    simpan_model(pipeline, MODEL_SAVE_PATH)

    print("\n" + "=" * 70)
    print("Pipeline selesai!")
    print("=" * 70)
