-- Buat database
CREATE DATABASE IF NOT EXISTS keystroke_db;
USE keystroke_db;

-- Tabel users: menyimpan username dan password hash
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL
);

-- Tabel keystroke_logs: menyimpan pola ketikan tiap login
CREATE TABLE IF NOT EXISTS keystroke_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    dwell JSON NOT NULL,                   -- dwell time per key
    flight JSON NOT NULL,                  -- flight time per key
    latency_sequences JSON,                -- DD, FF, DF, dll
    typing_speed FLOAT,                    -- karakter per detik / total waktu mengetik
    error_correction JSON,                 -- jumlah backspace/delete per posisi
    key_order JSON,                        -- urutan tombol yang ditekan
    normalized_features JSON,              -- dwell/flight normalisasi per user
    distance FLOAT,                        -- jarak pola vs pola referensi
    status ENUM('VALID','INVALID') DEFAULT 'INVALID',
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
