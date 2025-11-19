#   db.py
import mysql.connector
import logging

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",       
    "database": "keystroke_db",
    "autocommit": False
}

def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="keystroke_db"
        )
        return conn
    except mysql.connector.Error as err:
        logging.error(f"Gagal koneksi ke database: {err}")
        return None

