# ---------------- utils.py ----------------
import os
import time
import json
import logging
import joblib
import numpy as np
import mysql.connector
from db import get_db_connection
from ml_model import auto_retrain_ml

# ---------------- Konstanta ----------------
RETRAIN_INTERVAL = 10       # jumlah data baru sebelum retrain otomatis
ML_SCORE_THRESHOLD = 0.1    # ambang ML untuk valid

# ---------------- Setup logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)



# ---------------- Build initial template ----------------
def build_initial_template(feature_vector, history_limit=10):
    """
    Buat template awal dari fitur keystroke user dan tentukan threshold adaptif
    berdasarkan variasi ketikan pertama.
    """
    dwell = np.array(feature_vector.get("dwell", []), dtype=np.float32)
    flight = np.array(feature_vector.get("flight", []), dtype=np.float32)
    typing_speed = float(feature_vector.get("typing_speed", 0.0))

    # Hitung statistik dasar
    stats = compute_stats(dwell, flight, [typing_speed])

    # Variasi gabungan user
    combined = np.concatenate([
        dwell.flatten(),
        flight.flatten(),
        np.array([typing_speed], dtype=np.float32)
    ])
    overall_std = np.std(combined)
    overall_mean = np.mean(combined)

    # Threshold awal: empiris dari variasi user
    initial_threshold = max(overall_mean + 2.5 * overall_std, 0.8)

    # Inisialisasi template
    speed_history = [typing_speed] * min(history_limit, 1)
    template = {
        "mean": {
            "dwell": dwell.tolist(),
            "flight": flight.tolist(),
            "typing_speed": typing_speed,
            "key_order": feature_vector.get("key_order", [])
        },
        "stats": stats,
        "n": 1,
        "thresholds": {"distance": initial_threshold},
        "initial_threshold": initial_threshold,
        "speed_history": speed_history
    }

    logging.info(
        f"[Template Init] std={overall_std:.4f}, mean={overall_mean:.4f}, threshold={initial_threshold:.4f}"
    )
    return template



# ---------------- Normalization ----------------
def normalize_features(dwell, flight, typing_speed=None, template=None):
    """
    Normalisasi fitur dwell, flight, dan typing_speed
    dengan metode robust (median + IQR) dan speed_history adaptif.
    """
    def safe_normalize(arr):
        arr = np.array(arr, dtype=float)
        if len(arr) == 0:
            return arr
        median = np.median(arr)
        iqr = np.percentile(arr, 75) - np.percentile(arr, 25)
        iqr = iqr if iqr != 0 else 1.0
        return (arr - median) / iqr

    # Normalisasi dwell & flight
    dwell_norm = safe_normalize(dwell)
    flight_norm = safe_normalize(flight)
    typing_speed = 0.0 if typing_speed is None else typing_speed

    # Normalisasi typing_speed menggunakan speed_history jika tersedia
    if template and template.get("speed_history"):
        speed_array = np.array(template["speed_history"], dtype=np.float32)
        median_speed = np.median(speed_array)
        iqr_speed = np.percentile(speed_array, 75) - np.percentile(speed_array, 25)
        iqr_speed = iqr_speed if iqr_speed != 0 else 1.0
    else:
        median_speed = float(template.get("mean", {}).get("typing_speed", 0.0)) if template else 0.0
        iqr_speed = float(template.get("stats", {}).get("speed_std", 1.0)) if template else 1.0

    typing_speed_norm = (typing_speed - median_speed) / iqr_speed

    return {
        "dwell": dwell_norm.tolist(),
        "flight": flight_norm.tolist(),
        "typing_speed": typing_speed_norm
    }



# ---------------- Distance calculation (Mahalanobis) ----------------
def calculate_distance(dwell, flight, template=None, typing_speed=0,
                       normalize=True, key_order=None, penalty_weight=0.7,
                       min_std=0.4):
    """
    Hitung jarak Mahalanobis termasuk typing_speed dan penalti key_order.
    """
    dwell = np.array(dwell or [], dtype=np.float32)
    flight = np.array(flight or [], dtype=np.float32)

    # ---------------- Ambil stats dari template ----------------
    stats_dict = get_template_stats(template, dwell, flight, min_std)

    # ---------------- Sesuaikan panjang vektor ----------------
    dwell, dwell_mean = adjust_vector(dwell, stats_dict['dwell_mean'])
    flight, flight_mean = adjust_vector(flight, stats_dict['flight_mean'])

    # ---------------- Normalisasi terhadap template ----------------
    if normalize and template:
        dwell = (dwell - dwell_mean) / stats_dict['dwell_std']
        flight = (flight - flight_mean) / stats_dict['flight_std']
        dwell_mean = np.zeros_like(dwell_mean)
        flight_mean = np.zeros_like(flight_mean)

    # ---------------- Hitung Mahalanobis ----------------
    dist_maha = compute_mahalanobis(dwell, flight, dwell_mean, flight_mean,
                                    stats_dict['dwell_std'], stats_dict['flight_std'])

    # ---------------- Normalisasi typing speed ----------------
    dist_speed_norm = abs(typing_speed - stats_dict['speed_mean']) / stats_dict['speed_std']
    dist_total = dist_maha + dist_speed_norm

    # ---------------- Penalti key_order ----------------
    if key_order:
        dist_total += key_order_penalty(key_order, template, penalty_weight)

    return float(dist_total)


# ---------------- Helper functions ----------------

def get_template_stats(template, dwell, flight, min_std):
    """
    Ambil mean, std, speed dari template, fallback ke default jika tidak ada.
    """
    if template:
        mean_dict = template.get("mean", {})
        dwell_mean = np.array(mean_dict.get("dwell", []), dtype=np.float32)
        flight_mean = np.array(mean_dict.get("flight", []), dtype=np.float32)
        stats = template.get("stats", {})

        speed_array = np.array(template.get("speed_history", [mean_dict.get("typing_speed", 0.0)]),
                               dtype=np.float32)
        speed_mean = float(np.mean(speed_array))
        speed_std = max(float(np.std(speed_array)) if len(speed_array) > 1 else min_std, min_std)

        dwell_std = max(stats.get("dwell_std", 1.0), min_std)
        flight_std = max(stats.get("flight_std", 1.0), min_std)
    else:
        dwell_mean = np.zeros_like(dwell)
        flight_mean = np.zeros_like(flight)
        speed_mean = 0.0
        dwell_std = flight_std = speed_std = min_std

    return {
        "dwell_mean": dwell_mean,
        "flight_mean": flight_mean,
        "speed_mean": speed_mean,
        "dwell_std": dwell_std,
        "flight_std": flight_std,
        "speed_std": speed_std
    }


def adjust_vector(vec, vec_mean):
    """
    Sesuaikan panjang vector agar sama dengan mean vector.
    """
    target_len = min(len(vec), len(vec_mean)) if len(vec_mean) > 0 else len(vec)
    vec = adjust_vector_length(vec, target_len)
    vec_mean = adjust_vector_length(vec_mean, target_len)
    return vec, vec_mean


def compute_mahalanobis(dwell, flight, dwell_mean, flight_mean, dwell_std, flight_std):
    """
    Hitung jarak Mahalanobis untuk dwell+flight.
    """
    x = np.concatenate([dwell, flight])
    mean_vec = np.concatenate([dwell_mean, flight_mean])
    std_vec = np.concatenate([np.full_like(dwell, dwell_std), np.full_like(flight, flight_std)])
    cov = np.diag(std_vec**2)
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov)
    diff = x - mean_vec
    return float(np.sqrt(np.dot(np.dot(diff, cov_inv), diff.T)))


def key_order_penalty(key_order, template, penalty_weight=0.7):
    """
    Hitung penalti key_order dibanding template.
    """
    if template and "mean" in template and "key_order" in template["mean"]:
        template_order = template["mean"]["key_order"]
        mismatch = sum(
            1 for i in range(min(len(key_order), len(template_order)))
            if key_order[i] != template_order[i]
        )
        mismatch += abs(len(key_order) - len(template_order))
        return mismatch * penalty_weight
    return 0.0



# ---------------- Compute statistics ----------------
def compute_stats(dwell, flight, typing_speed_array=None):
    """
    Hitung mean dan std dari dwell, flight, dan typing_speed
    """
    dwell_arr = np.array(dwell, dtype=np.float32)
    flight_arr = np.array(flight, dtype=np.float32)
    typing_arr = (
        np.array(typing_speed_array, dtype=np.float32)
        if typing_speed_array is not None else np.array([0.0], dtype=np.float32)
    )

    return {
        "dwell_mean": float(np.mean(dwell_arr)) if dwell_arr.size else 0.0,
        "dwell_std": float(np.std(dwell_arr)) if len(dwell_arr) > 1 else 1.0,
        "flight_mean": float(np.mean(flight_arr)) if flight_arr.size else 0.0,
        "flight_std": float(np.std(flight_arr)) if len(flight_arr) > 1 else 1.0,
        "speed_mean": float(np.mean(typing_arr)),
        "speed_std": float(np.std(typing_arr)) if len(typing_arr) > 1 else 1.0
    }



# 🔹 Update template dengan history typing_speed
def update_template(template, fv, history_limit=10, alpha_fixed=0.4):
    """
    Update template dengan fitur baru, termasuk dwell, flight, typing_speed.
    Alpha tetap agar template lebih responsif.
    """
    n = template.get("n", 1)
    mean = template.get("mean", {})
    alpha = alpha_fixed  # tetap, bukan 1/(n+1)

    # ---------------- Update dwell & flight ----------------
    for key in ["dwell", "flight"]:
        old = np.array(mean.get(key, []), dtype=np.float32)
        new = np.array(fv.get(key, []), dtype=np.float32)
        max_len = max(len(old), len(new))
        old_p = np.pad(old, (0, max_len - len(old)), 'constant')
        new_p = np.pad(new, (0, max_len - len(new)), 'constant')
        mean[key] = ((1 - alpha) * old_p + alpha * new_p).tolist()

    # ---------------- Update typing_speed ----------------
    new_speed = float(fv.get("typing_speed", 0.0))
    speed_history = template.get("speed_history", [])
    speed_history.append(new_speed)
    if len(speed_history) > history_limit:
        speed_history = speed_history[-history_limit:]

    mean["typing_speed"] = np.mean(speed_history)
    template["speed_history"] = speed_history

    # ---------------- Update stats ----------------
    dwell_array = np.array(mean["dwell"], dtype=np.float32)
    flight_array = np.array(mean["flight"], dtype=np.float32)
    speed_array = np.array(speed_history, dtype=np.float32)
    stats = {
        "dwell_mean": float(np.mean(dwell_array)),
        "dwell_std": float(np.std(dwell_array)) if len(dwell_array) > 1 else 0.05,
        "flight_mean": float(np.mean(flight_array)),
        "flight_std": float(np.std(flight_array)) if len(flight_array) > 1 else 0.05,
        "speed_mean": float(np.mean(speed_array)),
        "speed_std": float(np.std(speed_array)) if len(speed_array) > 1 else 0.05
    }

    template.update({"n": n + 1, "mean": mean, "stats": stats})
    return template




# ---------------- Self-tuning Threshold (Hybrid) ----------------
def self_tuning_threshold(template, recent_distances, min_factor=1.5, max_factor=2.5,
                          alpha=0.8, min_threshold=1.0):
    """
    Threshold adaptif per user, menggabungkan pendekatan sederhana (berdasarkan distance awal)
    dengan smoothing dan batasan realistis.
    """
    if not recent_distances:
        # fallback ke threshold awal
        return float(template.get("thresholds", {}).get("distance", 3.0))

    recent_distances = np.array(recent_distances, dtype=np.float32)
    median_dist = float(np.median(recent_distances))
    std_dist = float(np.std(recent_distances))

    # ---------------- Tentukan adaptive threshold ----------------
    if len(recent_distances) < 3:
        # user baru: threshold = 2x median
        adaptive_threshold = median_dist * 2.0
    else:
        # user sudah sering login: pendekatan statistik ringan
        factor_dynamic = np.clip(1 + std_dist / (median_dist + 1e-6), min_factor, max_factor)
        adaptive_threshold = median_dist + factor_dynamic * std_dist

    # ---------------- Pastikan threshold minimal ----------------
    adaptive_threshold = max(adaptive_threshold, min_threshold)

    # ---------------- Smoothing dengan threshold lama ----------------
    old_threshold = float(template.get("thresholds", {}).get("distance", adaptive_threshold))
    initial_threshold = float(template.get("initial_threshold", adaptive_threshold))
    new_threshold = (1 - alpha) * old_threshold + alpha * adaptive_threshold

    # ---------------- Batasi agar tidak terlalu longgar/ketat ----------------
    min_allowed = max(min_threshold, initial_threshold * 0.8)
    max_allowed = max(initial_threshold * 2.0, min_allowed)
    new_threshold = np.clip(new_threshold, min_allowed, max_allowed)

    logging.info(
        f"[Self-Tune] median={median_dist:.4f}, std={std_dist:.4f}, "
        f"adaptive={adaptive_threshold:.4f}, final={new_threshold:.4f}"
    )

    return float(round(new_threshold, 4))



# ---------------- Simpan user baru ----------------
def save_user_data(username, password_hash, user_data, threshold_distance):
    """
    Simpan data user baru ke database, termasuk template dan threshold adaptif.
    """
    try:
        conn = get_db_connection()
        if conn is None:
            logging.error("Koneksi database gagal (None returned).")
            return False

        cur = conn.cursor()
        # Convert semua np.ndarray ke list agar JSON serializable
        json_safe = json.loads(json.dumps(
            user_data,
            default=lambda o: o.tolist() if isinstance(o, np.ndarray) else o
        ))

        cur.execute("""
            INSERT INTO users (username, password_hash, template, threshold, total_samples)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            username,
            password_hash,
            json.dumps(json_safe),
            threshold_distance,
            int(json_safe.get("n", 1))
        ))

        conn.commit()
        cur.close()
        conn.close()
        return True

    except mysql.connector.IntegrityError:
        logging.warning(f"Username '{username}' sudah terdaftar.")
        return False

    except mysql.connector.Error as err:
        logging.error(f"Gagal menyimpan user '{username}': {err}")
        return False

    except Exception as e:
        logging.error(f"Error tak terduga saat save_user_data: {e}")
        return False



# ---------------- Ambil data user dari DB ----------------
def get_user_by_username(username):
    """
    Ambil semua data user dari database berdasarkan username.
    """
    try:
        conn = get_db_connection()
        if conn is None:
            logging.error("Koneksi database gagal (None returned).")
            return None

        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT * FROM users WHERE username = %s", (username,))
        data = cur.fetchone()
        cur.close()
        conn.close()
        return data

    except mysql.connector.Error as err:
        logging.error(f"Gagal mengambil user '{username}': {err}")
        return None

    except Exception as e:
        logging.error(f"Error tak terduga saat get_user_by_username: {e}")
        return None



# ---------------- Adjust vector length ----------------
def adjust_vector_length(vec, target_len):
    """
    Menyesuaikan panjang vektor agar sama panjang menggunakan interpolasi linear.
    Lebih halus dibanding padding mean biasa.
    """
    vec = np.array(vec, dtype=np.float32).flatten()
    current_len = len(vec)

    if current_len == 0:
        return np.zeros(target_len, dtype=np.float32)
    if current_len == target_len:
        return vec

    # Interpolasi linear untuk menyamakan panjang
    x_old = np.linspace(0, 1, num=current_len)
    x_new = np.linspace(0, 1, num=target_len)
    vec_resampled = np.interp(x_new, x_old, vec).astype(np.float32)
    return vec_resampled



# ---------------- Brute-force / Rate Limiting ----------------
def record_failed_attempt(user_id, max_attempts=5, window_seconds=300):
    """
    Catat percobaan login gagal untuk user tertentu.
    Menggunakan rate-limiting berbasis jendela waktu.
    Return True jika account dianggap 'locked'.
    """
    conn = get_db_connection()
    cur = conn.cursor(dictionary=True)

    cur.execute("""
        SELECT failed_attempts, last_failed
        FROM login_attempts
        WHERE user_id = %s
    """, (user_id,))
    row = cur.fetchone()

    now = int(time.time())

    if row:
        failed_attempts = row["failed_attempts"]
        last_failed = row["last_failed"]

        # Jika sudah lewat window → reset counter
        if now - last_failed > window_seconds:
            failed_attempts = 1
        else:
            failed_attempts += 1

        cur.execute("""
            UPDATE login_attempts
            SET failed_attempts = %s, last_failed = %s
            WHERE user_id = %s
        """, (failed_attempts, now, user_id))

    else:
        # User belum punya record → buat baru
        failed_attempts = 1
        cur.execute("""
            INSERT INTO login_attempts (user_id, failed_attempts, last_failed)
            VALUES (%s, %s, %s)
        """, (user_id, failed_attempts, now))

    conn.commit()
    cur.close()
    conn.close()

    return failed_attempts >= max_attempts



# ---------------- Reset failed attempts ----------------
def reset_failed_attempts(user_id):
    """
    Reset counter gagal login jika user berhasil login.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM login_attempts WHERE user_id = %s", (user_id,))
    conn.commit()
    cur.close()
    conn.close()


# ---------------- Cek apakah user terkunci ----------------
def is_locked_out(user_id, max_attempts=5, window_seconds=300):
    """
    Cek apakah user sedang terkunci karena brute-force (login gagal terlalu banyak).
    """
    conn = get_db_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT failed_attempts, last_failed
        FROM login_attempts
        WHERE user_id = %s
    """, (user_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()

    now = int(time.time())

    if not row:
        return False

    # Reset jika sudah lewat window
    if now - row["last_failed"] > window_seconds:
        return False

    return row["failed_attempts"] >= max_attempts



# ---------------- Update template user di DB ----------------
def update_user_template(user_id, template):
    """
    Update template, threshold, dan total_samples user di database.
    """
    conn = get_db_connection()
    cur = conn.cursor()

    # Pastikan semua np.ndarray di-convert menjadi list agar JSON serializable
    json_safe = json.loads(json.dumps(
        template,
        default=lambda o: o.tolist() if isinstance(o, np.ndarray) else o
    ))

    cur.execute("""
        UPDATE users
        SET template = %s,
            threshold = %s,
            total_samples = %s
        WHERE id = %s
    """, (
        json.dumps(json_safe),
        float(json_safe["thresholds"]["distance"]),
        int(json_safe.get("n", 1)),
        int(user_id)
    ))

    conn.commit()
    cur.close()
    conn.close()



# ---------------- Safe list ----------------
def safe_list(l):
    """
    Pastikan input berupa list, bukan None.
    Jika None → return list kosong.
    """
    return l if l is not None else []


# ---------------- Jitter filter ----------------
def jitter_filter(seq, threshold=5.0):
    """
    Filter noise/jitter pada dwell atau flight times.
    
    Args:
        seq (list): urutan nilai dwell/flight
        threshold (float): maksimum perubahan yang dianggap jitter

    Returns:
        list: urutan nilai yang telah difilter
    """
    seq = safe_list(seq)
    if not seq:
        return []

    filtered = [seq[0]]
    for i in range(1, len(seq)):
        delta = seq[i] - seq[i - 1]
        if abs(delta) <= threshold:
            filtered.append(seq[i])
        else:
            # Jika delta terlalu besar → gunakan nilai sebelumnya
            filtered.append(seq[i - 1])

    return filtered



# ---------------- Login mode ML-only ----------------
def login_ml_only(user, feature_vector, dwell=None, flight=None, typing_speed=None, key_order=None):
    """
    Login mode ML-only (hanya ML, tidak update template).
    Login dikunci sampai ML model tersedia.

    Args:
        user (dict): hasil get_user_by_username()
        feature_vector (np.array): fitur hasil ekstraksi keystroke
        dwell, flight, typing_speed, key_order (optional): untuk simpan log ke DB

    Returns:
        status (str): "VALID", "INVALID", atau "HYBRID"
        ml_score (float): skor prediksi ML
    """
    username = user.get("username")
    user_id = user.get("id")

    # ---------------- Filter jitter ----------------
    dwell = jitter_filter(dwell)
    flight = jitter_filter(flight)

    # ---------------- Prediksi ML ----------------
    ml_score, ml_model = predict_ml_score(user_id, feature_vector, username)

    # ---------------- Tentukan status login ----------------
    status = determine_login_status(ml_model, ml_score, username)

    # ---------------- Simpan log ke DB ----------------
    save_keystroke_log(user_id, dwell, flight, typing_speed, key_order, feature_vector, status)

    # ---------------- Auto-retrain ML ----------------
    auto_retrain_user_ml(user_id, username, user.get("template"))

    return status, ml_score


# ---------------- Helper functions ----------------

def predict_ml_score(user_id, feature_vector, username):
    """
    Load ML model user dan prediksi skor.
    """
    ml_model = None
    ml_score = 0.0
    try:
        ml_model_path = f"ml_models/user_{user_id}_ml.pkl"
        if os.path.exists(ml_model_path):
            ml_model = joblib.load(ml_model_path)
        else:
            logging.warning(f"[ML-only] Model belum ada untuk {username}")
    except Exception as e:
        logging.error(f"[ML-only] Gagal load model {username}: {e}")

    # Prediksi
    try:
        if ml_model is not None:
            feature_vector = feature_vector.reshape(1, -1)
            if hasattr(ml_model, "predict_proba"):
                ml_score = float(ml_model.predict_proba(feature_vector)[0][1])
            else:
                ml_score = float(ml_model.predict(feature_vector)[0])
    except Exception as e:
        logging.error(f"[ML-only] Prediksi gagal untuk {username}: {e}")
        ml_score = 0.0
        ml_model = None

    return ml_score, ml_model


def determine_login_status(ml_model, ml_score, username):
    """
    Tentukan status login berdasarkan ketersediaan model dan skor.
    """
    if ml_model is None:
        logging.info(f"[ML-only] ML model belum ada untuk {username}, fallback ke hybrid")
        return "HYBRID"
    return "VALID" if ml_score >= ML_SCORE_THRESHOLD else "INVALID"


def save_keystroke_log(user_id, dwell, flight, typing_speed, key_order, feature_vector, status):
    """
    Simpan log keystroke user ke database.
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO keystroke_logs
            (user_id, dwell, flight, typing_speed, key_order, distance, status, latency_sequences, normalized_features)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            user_id,
            json.dumps(dwell) if dwell else None,
            json.dumps(flight) if flight else None,
            typing_speed,
            json.dumps(key_order) if key_order else None,
            None,  # distance tidak dipakai ml_only
            status,
            None,
            json.dumps(feature_vector.tolist())
        ))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logging.error(f"[ML-only] Gagal simpan log ke DB: {e}")


def auto_retrain_user_ml(user_id, username, template):
    """
    Jalankan auto-retrain ML jika diperlukan dan update DB.
    """
    template_for_db = template.copy() if template else {}
    try:
        ml_model_new, ml_meta = auto_retrain_ml(user_id)
        if ml_model_new:
            logging.info(f"[ML-only] Model user {username} berhasil diretrain. acc={ml_meta.get('acc')}")
            update_user_ml(
                user_id,
                ml_model_new,
                template=template_for_db,
                ml_meta=ml_meta
            )
            logging.info(f"[Patch] Metadata ML user_id={user_id} berhasil update di DB dan history.json")
    except Exception as e:
        logging.error(f"[ML-only] Auto-retrain gagal untuk {username}: {e}")



# 🔹 Update ML info user di DB dan simpan model
def update_user_ml(user_id, ml_model, template=None, ml_meta=None, score_history=None):
    """
    Simpan model ML user ke file dan update metadata ke database.
    Args:
        user_id (int): ID user.
        ml_model: model ML yang sudah dilatih (misal scikit-learn).
        template (dict, optional): template keystroke user.
        ml_meta (dict, optional): metadata ML (acc, f1_score, total_samples).
        score_history (list, optional): history skor ML (opsional)
    """
    template = template or {}
    stats = template.get("stats", {})

    ml_mean = float(stats.get("speed_mean", 0.05))
    ml_std = float(stats.get("speed_std", 0.05))

    acc = float(ml_meta.get("acc", 0.0)) if ml_meta else 0.0
    f1_score = float(ml_meta.get("f1_score", 0.0)) if ml_meta else 0.0
    total_samples = int(ml_meta.get("total_samples", 0)) if ml_meta else 0

    model_path = save_ml_model(user_id, ml_model)
    final_history = update_ml_history(user_id, acc, score_history)
    update_ml_metadata_db(user_id, model_path, template, ml_mean, ml_std, final_history, total_samples)


# ---------------- Helper functions ----------------

def save_ml_model(user_id, ml_model):
    """
    Simpan model ML ke file dan return path-nya.
    """
    model_dir = "ml_models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"user_{user_id}_ml.pkl")
    try:
        joblib.dump(ml_model, model_path)
        logging.info(f"[update_user_ml] Model user_id={user_id} berhasil disimpan di {model_path}")
    except Exception as e:
        logging.error(f"[update_user_ml] Gagal menyimpan model user_id={user_id}: {e}")
    return model_path


def update_ml_history(user_id, new_acc, score_history=None):
    """
    Update file history ML user dan kembalikan list skor final.
    """
    model_dir = "ml_models"
    history_path = os.path.join(model_dir, f"user_{user_id}_history.json")

    # Ambil history lama
    if os.path.exists(history_path):
        try:
            with open(history_path, "r") as f:
                score_history_old = json.load(f)
        except:
            score_history_old = []
    else:
        score_history_old = []

    # Gabungkan history baru
    if score_history:
        score_history_old.extend(score_history)
    if new_acc is not None:
        score_history_old.append(new_acc)

    # Hapus duplikat berurutan
    final_history = []
    for s in score_history_old:
        if not final_history or final_history[-1] != s:
            final_history.append(s)

    # Simpan kembali
    try:
        with open(history_path, "w") as f:
            json.dump(final_history, f, indent=2)
    except Exception as e:
        logging.error(f"[update_user_ml] Gagal menyimpan history user_id={user_id}: {e}")

    return final_history


def update_ml_metadata_db(user_id, model_path, template, ml_mean, ml_std, final_history, total_samples):
    """
    Update metadata ML di database.
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            UPDATE users
            SET ml_model_path=%s,
                ml_template=%s,
                ml_mean=%s,
                ml_std=%s,
                ml_scores=%s,
                total_samples=%s,
                updated_at=NOW()
            WHERE id=%s
        """, (
            model_path,
            json.dumps(template),
            ml_mean,
            ml_std,
            json.dumps(final_history),
            total_samples,
            user_id
        ))
        conn.commit()
        cur.close()
        conn.close()
        logging.info(f"[update_user_ml] Metadata ML user_id={user_id} berhasil diupdate di DB.")
    except Exception as e:
        logging.error(f"[update_user_ml] Gagal update metadata ML user_id={user_id}: {e}")
