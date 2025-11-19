-- cek jumlah data pada user
SELECT COUNT(*) FROM users;

-- cek jumlah data login pada keystroke_logs
SELECT COUNT(*) FROM keystroke_logs where user_id = 12;

-- cek jumlah login valid user id 6
SELECT COUNT(*) FROM keystroke_logs where user_id = 6 and status = 'valid';
-- cek jumlah login invalid user id 6
SELECT COUNT(*) FROM keystroke_logs where user_id = 6 and status = 'invalisd';

--cek valid dan invalid login user id 6
SELECT status, COUNT(*) FROM keystroke_logs where user_id = 12 GROUP BY status;


-- terakhir login user id 6
SELECT * FROM keystroke_logs where user_id = 9 ORDER BY timestamp DESC LIMIT 1;

-- cek 3 login terakhir user id 6
SELECT * FROM keystroke_logs where user_id = 9 ORDER BY timestamp DESC LIMIT 3;

-- cek total data pada table keystroke_logs
SELECT COUNT(*) FROM keystroke_logs;

-- cek berapa user yang pernah login
SELECT COUNT(DISTINCT user_id) FROM keystroke_logs;

-- hapus user id 7 beserta data keystroke_logsnya
DELETE FROM keystroke_logs WHERE user_id = 7;
DELETE FROM users WHERE id = 7;

--melihat id user dan username
SELECT id, username FROM users;

-- melihat kolom ml_score pada tabel keystroke_logs
SELECT ml_scores FROM users WHERE id = 12;