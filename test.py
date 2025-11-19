import mysql.connector
import numpy as np
import json

# Konfigurasi koneksi database
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "keystroke_db"
}

user_id = "2"

try:
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

    query = "SELECT dwell, flight, typing_speed FROM keystroke_logs WHERE user_id = %s"
    cursor.execute(query, (user_id,))
    rows = cursor.fetchall()

    if not rows:
        print(f"Tidak ada data untuk user '{user_id}'")
    else:
        feature_vector = []
        for row in rows:
            for item in row:
                # Jika item berbentuk string JSON, parse dulu
                if isinstance(item, str):
                    parsed = json.loads(item)
                    feature_vector.extend(parsed)
                else:
                    feature_vector.append(float(item))

        feature_vector = np.array(feature_vector, dtype=float)
        print("Feature vector:", feature_vector)
        print("Mean:", np.mean(feature_vector))
        print("Std:", np.std(feature_vector))

except mysql.connector.Error as err:
    print("Error:", err)

finally:
    if 'cursor' in locals():
        cursor.close()
    if 'conn' in locals() and conn.is_connected():
        conn.close()
