# ml_model.py

# ----- Standard Libraries -----
import os
import time
import json
import pickle
import logging

# ----- Third-party Libraries -----
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ----- Local Modules -----
from db import get_db_connection

# ---------------- Konstanta ----------------
MODEL_DIR = "ml_models"
os.makedirs(MODEL_DIR, exist_ok=True)

RETRAIN_INTERVAL = 5        # retrain tiap N login baru
FEATURE_VECTOR_LEN = 30     # panjang tetap untuk setiap vektor (dwell, flight, latency, key_order)
FEATURE_LEN = 20            # panjang vektor standar untuk setiap sub-vektor (flatten helper)



def adjust_vector_length(vec, target_len=FEATURE_LEN):
    """
    Menyesuaikan panjang vektor menjadi target_len.
    Jika vektor lebih pendek, diinterpolasi; jika kosong, dikembalikan vektor nol.
    
    Args:
        vec (list/np.array): vektor input
        target_len (int): panjang vektor target
    
    Returns:
        np.array: vektor dengan panjang target_len
    """
    vec = np.asarray(vec, dtype=float).flatten()

    if len(vec) == 0:
        return np.zeros(target_len, dtype=float)
    if len(vec) == target_len:
        return vec

    # Interpolasi vektor ke panjang target
    x_old = np.linspace(0, 1, num=len(vec))
    x_new = np.linspace(0, 1, num=target_len)
    return np.interp(x_new, x_old, vec)



def extract_features_from_row(row, target_len=FEATURE_LEN):
    """
    Ekstrak semua fitur dari satu baris data user menjadi vektor fitur tetap panjangnya.

    Args:
        row (dict-like): baris data dari database
        target_len (int): panjang vektor untuk setiap sub-vektor
    
    Returns:
        np.array: vektor fitur gabungan
    """
    # Ambil dan sesuaikan panjang vektor dwell, flight, latency
    dwell = adjust_vector_length(json.loads(row.get('dwell') or "[]"), target_len)
    flight = adjust_vector_length(json.loads(row.get('flight') or "[]"), target_len)
    latency = adjust_vector_length(json.loads(row.get('latency_sequences') or "[]"), target_len)
    
    # Ambil key_order, ubah ke ASCII, dan sesuaikan panjangnya
    key_order_raw = json.loads(row.get('key_order') or "[]")
    key_order = [
        ord(k[0]) if isinstance(k, str) and len(k) > 0 else 0
        for k in key_order_raw
    ]
    key_order = adjust_vector_length(key_order, target_len)
    
    # Ambil typing speed sebagai fitur tunggal
    typing_speed = np.array([float(row.get('typing_speed') or 0.0)])
    
    # Gabungkan semua fitur menjadi satu vektor
    features = np.concatenate([dwell, flight, latency, key_order, typing_speed])
    return features



def get_user_features(user_id):
    """
    Ambil semua fitur dan label untuk satu user dari database.

    Args:
        user_id (int/str): ID user
    
    Returns:
        X (np.array): matriks fitur
        y (np.array): array label (1 = VALID, 0 = INVALID)
    """
    conn = get_db_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT dwell, flight, latency_sequences, typing_speed, key_order, status
        FROM keystroke_logs
        WHERE user_id=%s
    """, (user_id,))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    X, y = [], []
    for row in rows:
        try:
            feat = extract_features_from_row(row)
            X.append(feat)
            y.append(1 if row.get('status') == 'VALID' else 0)
        except Exception as e:
            logging.warning(f"Skip row untuk user {user_id} karena error: {e}")
            continue

    return np.array(X), np.array(y)



# ---------------- Load & Save Model ----------------

def get_model_path(user_id):
    """Return path file model untuk user tertentu."""
    return os.path.join(MODEL_DIR, f"user_{user_id}.pkl")


def get_meta_path(user_id):
    """Return path file metadata untuk user tertentu."""
    return os.path.join(MODEL_DIR, f"user_{user_id}_meta.json")


def load_model(user_id):
    """
    Load model dan metadata untuk user tertentu.

    Returns:
        model: objek model (RandomForestClassifier atau None)
        meta: dict metadata {'last_trained', 'total_samples', 'acc'}
    """
    model_path = get_model_path(user_id)
    meta_path = get_meta_path(user_id)

    model = None
    meta = {"last_trained": 0, "total_samples": 0, "acc": 0.0}

    if os.path.exists(model_path):
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
        except Exception as e:
            logging.warning(f"Gagal load model user {user_id}: {e}")

    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
        except Exception as e:
            logging.warning(f"Gagal load metadata user {user_id}: {e}")

    return model, meta


def save_model(user_id, model, meta):
    """
    Simpan model dan metadata untuk user tertentu.
    """
    try:
        with open(get_model_path(user_id), "wb") as f:
            pickle.dump(model, f)
        with open(get_meta_path(user_id), "w") as f:
            json.dump(meta, f)
        logging.info(f"Model dan metadata user {user_id} berhasil disimpan.")
    except Exception as e:
        logging.error(f"Gagal simpan model/metadata user {user_id}: {e}")



# ---------------- Train Model ----------------

def train_model(user_id):
    """
    Latih model RandomForest untuk user tertentu menggunakan data keystroke.
    
    Args:
        user_id (int): ID user.
    
    Returns:
        model: model terlatih (RandomForestClassifier) atau None jika data kurang
        acc: akurasi model pada validation set
        meta: dict metadata model
    """
    X, y = get_user_features(user_id)
    
    if len(X) < 20:
        logging.warning(f"User {user_id} belum cukup data untuk training ({len(X)} samples).")
        return None, 0.0, None

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Buat dan latih model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Prediksi untuk evaluasi
    y_pred = model.predict(X_val)

    # Hitung metrik
    acc       = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, zero_division=1)
    recall    = recall_score(y_val, y_pred, zero_division=1)
    f1        = f1_score(y_val, y_pred, zero_division=1)

    # Confusion matrix aman
    cm = confusion_matrix(y_val, y_pred)
    tn, fp, fn, tp = (cm.ravel() if cm.size == 4 else (0, 0, 0, 0))

    # Mean & std fitur + epsilon kecil supaya tidak 0
    mean_vector = (np.mean(X, axis=0) + 1e-6).tolist()
    std_vector  = (np.std(X, axis=0) + 1e-6).tolist()

    # Logging hasil evaluasi
    logging.info(f"[EVALUASI MODEL USER {user_id}]")
    logging.info(f"Accuracy : {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    logging.info(f"Confusion Matrix:\n{cm}")
    logging.info(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}")

    # Metadata model
    meta = {
        "last_trained": time.time(),
        "total_samples": len(X),
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm.tolist(),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "mean": mean_vector,
        "std": std_vector,
        "scores": []  # untuk simpan ml_scores jika perlu
    }

    # Simpan model dan metadata
    save_model(user_id, model, meta)
    return model, acc, meta



# ---------------- Predict ----------------
def predict_user(model, feature_vector):
    """
    Prediksi skor validitas login berdasarkan feature_vector dan model user.
    
    Args:
        model: model RandomForest terlatih
        feature_vector: numpy array atau list dari fitur user
    
    Returns:
        float: skor probabilitas user valid (0.0 - 1.0)
    """
    if model is None or len(feature_vector) == 0:
        return 0.0

    X = np.array(feature_vector, dtype=float)

    # Pastikan panjang fitur sesuai model
    n_features = getattr(model, "n_features_in_", len(X))
    if len(X) < n_features:
        X = np.pad(X, (0, n_features - len(X)), mode='constant')
    elif len(X) > n_features:
        X = X[:n_features]

    X = X.reshape(1, -1)

    try:
        if hasattr(model, "predict_proba"):
            score = model.predict_proba(X)[0][1]
        else:
            score = model.predict(X)[0]
    except Exception as e:
        logging.warning(f"[ML Predict Error] {e}")
        score = 0.0

    return float(score)



# ---------------- Auto Retrain ----------------
def auto_retrain_ml(user_id):
    """
    Periksa apakah ada cukup data baru untuk melakukan retrain model user.
    
    Args:
        user_id: ID user
    
    Returns:
        tuple: (model terlatih, metadata model)
    """
    model, meta = load_model(user_id)

    # Ambil semua data user
    X, y = get_user_features(user_id)
    total_samples = meta.get("total_samples", 0) if meta else 0

    # Cek apakah jumlah data baru cukup untuk retrain
    if len(X) - total_samples >= RETRAIN_INTERVAL:
        model, acc, meta = train_model(user_id)

    return model, meta



# ---------------- Key Conversion ----------------
def key_to_int(k):
    """
    Konversi tombol menjadi integer.
    
    Args:
        k (str): karakter atau string tombol
    
    Returns:
        int: nilai integer unik untuk tombol
    """
    if len(k) == 1:
        return ord(k)
    return hash(k) % 65536  # tombol panjang jadi nilai unik
