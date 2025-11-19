# app.py
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash

import os
import json as js
import numpy as np
import logging

from db import get_db_connection
from utils import (
    calculate_distance,
    build_initial_template,
    is_locked_out,
    record_failed_attempt,
    reset_failed_attempts,
    update_template,
    save_user_data,
    get_user_by_username,
    update_user_template,
    self_tuning_threshold,
    normalize_features,
    login_ml_only,
    update_user_ml,
    ML_SCORE_THRESHOLD,
    RETRAIN_INTERVAL
)
from ml_model import FEATURE_LEN, get_user_features, load_model, predict_user, train_model


app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY")


if app.secret_key is None:
    raise RuntimeError("FLASK_SECRET_KEY tidak ditemukan! Set environment variable dulu.")

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)



# ---------------- Helper ----------------
def safe_json_loads(s, default=None):
    """Load JSON dengan fallback default jika error."""
    if default is None:
        default = []
    try:
        return js.loads(s)
    except (TypeError, js.JSONDecodeError):
        return default

# ---------------- Helper untuk ML ----------------
def auto_retrain_ml(user_id):
    """
    Cek apakah ada cukup data baru untuk retrain model user.
    """
    ml_model, ml_meta = load_model(user_id)
    
    # Ambil semua data user
    X, y = get_user_features(user_id)
    total_samples = ml_meta.get("total_samples", 0) if ml_meta else 0

    if len(X) - total_samples >= RETRAIN_INTERVAL:
        # Train model baru
        ml_model, acc, ml_meta = train_model(user_id)
        logging.info(f"[ML Retrain] user={user_id}, new_accuracy={acc:.3f}")
        return ml_model, ml_meta

    return ml_model, ml_meta


def key_to_int(k):
    """Konversi key menjadi integer unik."""
    if len(k) == 1:
        return ord(k)
    return hash(k) % 65536  # tombol panjang jadi nilai unik


# ---------------- Home Route ----------------
@app.route('/')
def index():
    return redirect(url_for('login'))


# ---------------- Register ----------------
@app.route('/register', methods=['GET', 'POST'])
def keystroke_register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        # Validasi dasar
        if not username or not password:
            flash("Username dan password wajib diisi.", "error")
            return redirect(url_for('keystroke_register'))

        # Ambil data keystroke
        dwell = safe_json_loads(request.form.get('dwell'))
        flight = safe_json_loads(request.form.get('flight'))
        key_order = safe_json_loads(request.form.get('key_order'))
        typing_speed = float(request.form.get('typing_speed') or 0.0)

        # Validasi input keystroke
        if len(dwell) == 0 or len(flight) == 0:
            flash("Silakan ketik password agar keystroke terekam.", "error")
            return redirect(url_for('keystroke_register'))

        # Bangun template awal
        feature_vector = {
            "dwell": dwell,
            "flight": flight,
            "typing_speed": typing_speed,
            "key_order": key_order
        }
        template = build_initial_template(feature_vector)

        # Hitung distance awal
        distance_initial = calculate_distance(
            dwell, flight, template, typing_speed,
            normalize=True, key_order=key_order, penalty_weight=0.5
        )

        # Jika distance terlalu kecil, gunakan variasi internal user
        if distance_initial < 0.1:
            dwell_std = np.std(dwell) if len(dwell) > 1 else 0.05
            flight_std = np.std(flight) if len(flight) > 1 else 0.05
            typing_var = np.std([typing_speed]) if typing_speed > 0 else 0.05
            distance_initial = max((dwell_std + flight_std + typing_var) * 5, 0.8)

        # Hitung threshold adaptif proporsional per user
        mean_val = float(np.mean(template.get("mean_vector", dwell)))
        std_val = float(np.mean(template.get("std_vector", np.array(flight))))

        normalized_distance = np.log1p(distance_initial)  # skala log
        adaptive_threshold = normalized_distance * 0.25  # faktor ketat

        # Batas dinamis tergantung konsistensi user
        min_th = max(0.8, normalized_distance * 0.2)
        max_th = max(2.0, normalized_distance * 0.5)
        adaptive_threshold = np.clip(adaptive_threshold, min_th, max_th)

        # Simpan threshold ke template
        template["thresholds"]["distance"] = adaptive_threshold
        template["initial_threshold"] = adaptive_threshold
        template["recent_distances"] = [distance_initial]

        logging.info(
            f"[Register] user={username}, "
            f"distance_initial={distance_initial:.4f}, mean={mean_val:.2f}, std={std_val:.2f}, "
            f"normalized={normalized_distance:.4f}, adaptive_threshold={adaptive_threshold:.4f}"
        )

        # Simpan user ke database
        password_hash = generate_password_hash(password)
        save_user_data(username, password_hash, template, adaptive_threshold)

        flash(
            "Registrasi berhasil! Threshold adaptif ditentukan berdasarkan pola ketikan kamu.",
            "success"
        )
        return redirect(url_for('login'))

    # Jika GET
    return render_template('register.html')



# ------------- Login Route ------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        user = get_user_by_username(username)
        if not user:
            flash("Login gagal! Username/password salah.", "error")
            return redirect(url_for('login'))

        user_id = user['id']

        # ---------------- Bruteforce prevention ----------------
        if is_locked_out(user_id):
            flash(
                "Akun terkunci sementara karena terlalu banyak percobaan login gagal. Coba lagi nanti.",
                "error"
            )
            return redirect(url_for('login'))

        # ---------------- Password check ----------------
        if not check_password_hash(user.get('password_hash', ''), password):
            record_failed_attempt(user_id)
            flash("Login gagal! Username/password salah.", "error")
            return redirect(url_for('login'))

        # ---------------- Ambil template user ----------------
        try:
            template = js.loads(user.get('template') or "{}")
        except:
            template = {}
        threshold = float(template.get("thresholds", {}).get("distance", user.get("threshold", 0.5)))

        # ---------------- Ambil keystroke input ----------------
        dwell = safe_json_loads(request.form.get('dwell'))
        flight = safe_json_loads(request.form.get('flight'))
        key_order = safe_json_loads(request.form.get('key_order'))
        typing_speed = float(request.form.get('typing_speed') or 0.0)

        # ---------------- Hitung Mahalanobis distance ----------------
        distance = calculate_distance(
            dwell, flight, template, typing_speed,
            normalize=True, key_order=key_order, penalty_weight=0.5
        )

        # ---------------- Normalisasi fitur → ML ----------------
        norm = normalize_features(dwell, flight, typing_speed, template)
        latency_sequence = [abs(flight[i] - flight[i-1]) for i in range(1, len(flight))]
        feature_vector = np.concatenate([
            norm["dwell"],
            norm["flight"],
            latency_sequence[:FEATURE_LEN],
            np.array([key_to_int(k) for k in key_order][:FEATURE_LEN]),
            np.array([typing_speed])
        ])

        # ---------------- Tentukan mode user ----------------
        user_mode = str(user.get("mode", "distance"))
        if user_mode not in ["distance", "hybrid", "ml_only"]:
            user_mode = "distance"

        ml_score = 0.0
        # ---------------- ML prediction ----------------
        if user_mode in ["hybrid", "ml_only"]:
            try:
                ml_model, ml_meta = load_model(user_id)
                if ml_model:
                    pred = predict_user(ml_model, feature_vector)
                    ml_score = float(pred[0]) if isinstance(pred, (list, np.ndarray)) else float(pred)
                else:
                    logging.warning(f"[ML Predict] Tidak ada model ML untuk user_id={user_id}")
            except Exception as e:
                logging.error(f"[ML Predict Error] user={username}: {e}")

        # ---------------- Tentukan login status ----------------
        if user_mode == "distance":
            login_status = "VALID" if distance <= threshold else "INVALID"
        elif user_mode == "hybrid":
            login_status = "VALID" if distance <= threshold and ml_score >= ML_SCORE_THRESHOLD else "INVALID"
        else:  # ml_only
            login_status = "VALID" if ml_score >= ML_SCORE_THRESHOLD else "INVALID"

        # ---------------- Simpan log ke DB ----------------
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO keystroke_logs
            (user_id, dwell, flight, typing_speed, key_order, distance, status,
             latency_sequences, normalized_features)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            user_id, js.dumps(dwell), js.dumps(flight), typing_speed,
            js.dumps(key_order), distance, login_status,
            js.dumps(latency_sequence), js.dumps(feature_vector.tolist())
        ))
        conn.commit()
        cur.close()
        conn.close()

        # ---------------- LOGIN VALID ----------------
        if login_status == "VALID":
            reset_failed_attempts(user_id)

            if user_mode == "ml_only":
                flash("Login berhasil (mode ML-only).", "success")
                logging.info(f"[Login SUCCESS] user={username}, ml_score={ml_score:.3f}, mode=ml_only")
                session["user"] = username
                return redirect(url_for("dashboard"))

            # Distance atau hybrid → update template & threshold
            db = get_db_connection()
            cursor = db.cursor()
            cursor.execute("""
                SELECT distance FROM keystroke_logs
                WHERE user_id=%s AND status='VALID'
                ORDER BY timestamp DESC LIMIT 10
            """, (user_id,))
            recent_distances = [row[0] for row in cursor.fetchall()]
            cursor.close()
            db.close()

            new_template = update_template(template, {
                "dwell": dwell,
                "flight": flight,
                "typing_speed": typing_speed,
                "key_order": key_order
            })

            new_threshold = self_tuning_threshold(new_template, recent_distances)
            new_template["thresholds"]["distance"] = new_threshold
            update_user_template(user_id, new_template)

            # ---------------- Auto retrain ML ----------------
            ml_model, ml_meta = auto_retrain_ml(user_id)
            if ml_model:
                try:
                    history_path = os.path.join("ml_models", f"user_{user_id}_history.json")
                    score_history = []
                    if os.path.exists(history_path):
                        with open(history_path, "r") as f:
                            score_history = js.load(f)

                    # Tambahkan skor retrain terbaru
                    new_score = ml_meta.get("acc")
                    if new_score is not None:
                        score_history.append(new_score)

                    # Update DB
                    update_user_ml(
                        user_id,
                        ml_model,
                        template=new_template,
                        ml_meta=ml_meta,
                        score_history=score_history
                    )

                    # Simpan kembali score_history ke file JSON
                    with open(history_path, "w") as f:
                        js.dump(score_history, f, indent=2)

                    logging.info(f"[Patch] Metadata ML user_id={user_id} berhasil update di DB dan history.json")

                except Exception as e:
                    logging.error(f"[Patch] Gagal update metadata ML user_id={user_id}: {e}")


            # ---------------- Upgrade mode otomatis ----------------
            if ml_model and user_mode == "distance":
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("UPDATE users SET mode='hybrid' WHERE id=%s", (user_id,))
                conn.commit()
                cursor.close()
                conn.close()
                user_mode = "hybrid"
                logging.info(f"[Mode Upgrade] user={username} distance → hybrid")

            if ml_model and ml_meta and user_mode == "hybrid":
                acc = ml_meta.get("acc", 0.0)
                f1 = ml_meta.get("f1_score", 0.0)
                total = ml_meta.get("total_samples", 0)
                history_path = os.path.join("ml_models", f"user_{user_id}_history.json")
                acc_history = []
                if os.path.exists(history_path):
                    with open(history_path, "r") as f:
                        acc_history = js.load(f)
                acc_history.append(acc)
                acc_history = acc_history[-3:]
                with open(history_path, "w") as f:
                    js.dump(acc_history, f)

                stable = len(acc_history) >= 3 and (max(acc_history) - min(acc_history)) <= 0.05
                if acc >= 0.93 and f1 >= 0.90 and total >= 100 and stable:
                    conn = get_db_connection()
                    cur = conn.cursor()
                    cur.execute("UPDATE users SET mode='ml_only' WHERE id=%s", (user_id,))
                    conn.commit()
                    cur.close()
                    conn.close()
                    user_mode = "ml_only"
                    logging.info(f"[Mode Upgrade] user={username} hybrid → ml_only (acc={acc:.3f}, f1={f1:.3f}, total={total}, stable={stable})")

            logging.info(f"[Login] user={username}, distance={distance:.3f}, threshold={new_threshold:.3f}, ml_score={ml_score}, mode={user_mode}")
            flash("Login berhasil. Template & threshold diperbarui adaptif.", "success")
            session['user'] = username
            return redirect(url_for('dashboard'))

        # ---------------- LOGIN INVALID ----------------
        else:
            record_failed_attempt(user_id)
            logging.warning(f"[Login FAILED] user={username}, distance={distance:.3f}, threshold={threshold:.3f}, ml_score={ml_score}, mode={user_mode}")
            flash("Login gagal. Pola ketikan tidak cocok.", "error")
            return redirect(url_for('login'))

    return render_template('login.html')




# ---------------- Dashboard ----------------
@app.route("/dashboard")
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))

    username = session['user']
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)

    # Ambil info user
    cursor.execute("SELECT id, threshold FROM users WHERE username=%s", (username,))
    user = cursor.fetchone()

    logs, threshold = [], None
    if user:
        threshold = user['threshold']
        # Ambil 10 log terakhir
        cursor.execute("""
            SELECT distance, status, timestamp
            FROM keystroke_logs
            WHERE user_id=%s
            ORDER BY timestamp DESC LIMIT 10
        """, (user['id'],))
        logs = cursor.fetchall()

    cursor.close()
    db.close()

    return render_template(
        "dashboard.html",
        username=username,
        logs=logs,
        threshold=threshold
    )



# ---------------- Logout ----------------
@app.route('/logout')
def logout():
    session.pop('user', None)
    flash("Logout berhasil.", "success")
    return redirect(url_for('login'))


if __name__ == '__main__':
    app.run(debug=True)
