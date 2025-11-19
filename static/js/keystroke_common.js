// keystroke_common.js (Modular)

// ---------------------------
// Noise filtering
// ---------------------------
const JITTER_THRESHOLD = 5;        // ms diabaikan
const MAX_DWELL = 500;
const MAX_FLIGHT = 500;

function clamp(v, min, max) {
    return Math.max(min, Math.min(max, v));
}

/**
 * Inisialisasi keystroke recorder.
 * @param {string} passwordInputId - ID input password
 * @param {string} formId - ID form
 * @param {string} dwellInputId - ID hidden input dwell
 * @param {string} flightInputId - ID hidden input flight
 * @param {string} typingSpeedInputId - ID hidden input typing speed
 * @param {string} keyOrderInputId - ID hidden input key order
 */
function initKeystrokeRecorder(
    passwordInputId,
    formId,
    dwellInputId,
    flightInputId,
    typingSpeedInputId,
    keyOrderInputId
) {
    const passwordInput = document.getElementById(passwordInputId);
    const form = document.getElementById(formId);
    const dwellInput = document.getElementById(dwellInputId);
    const flightInput = document.getElementById(flightInputId);
    const typingSpeedInput = document.getElementById(typingSpeedInputId);

    if (!passwordInput || !form) {
        console.error("[Keystroke] Element tidak ditemukan.");
        return;
    }

    // Buffer untuk keystroke
    let dwellTimes = [];
    let flightTimes = [];
    let keyDownTimes = {};
    let lastKeyUpTime = null;
    let startTime = null;
    let keyOrder = [];

    // Reset buffer
    function resetKeystrokeData() {
        dwellTimes = [];
        flightTimes = [];
        keyDownTimes = {};
        lastKeyUpTime = null;
        startTime = null;
        keyOrder = [];
    }

    // Event Key Down
    passwordInput.addEventListener("keydown", (e) => {
        const now = performance.now();

        if (startTime === null) startTime = now; // mulai dari tombol pertama

        // Rekam urutan tombol
        keyOrder.push(e.key);

        // Simpan waktu keydown jika belum ada
        if (!keyDownTimes[e.code]) keyDownTimes[e.code] = now;

        // Hitung flight time
        if (lastKeyUpTime !== null) {
            let flight = now - lastKeyUpTime;
            if (flight > JITTER_THRESHOLD) {
                flight = clamp(flight, 0, MAX_FLIGHT);
                flightTimes.push(flight);
            }
        }
    });

    // Event Key Up
    passwordInput.addEventListener("keyup", (e) => {
        const now = performance.now();

        if (keyDownTimes[e.code]) {
            let dwell = now - keyDownTimes[e.code];
            if (dwell > JITTER_THRESHOLD) {
                dwell = clamp(dwell, 0, MAX_DWELL);
                dwellTimes.push(dwell);
            }
            delete keyDownTimes[e.code];
        }

        lastKeyUpTime = now;

        // Reset jika password kosong
        if (passwordInput.value.length === 0) resetKeystrokeData();
    });

    // Event Submit Form
    form.addEventListener("submit", (e) => {
        if (dwellTimes.length === 0 || flightTimes.length === 0) {
            alert("Silakan ketik password terlebih dahulu agar keystroke terekam!");
            e.preventDefault();
            return;
        }

        // Hitung typing speed
        const totalTime = performance.now() - startTime;
        const typingSpeed = passwordInput.value.length / (totalTime / 1000);
        typingSpeedInput.value = typingSpeed;

        // Set dwell & flight
        dwellInput.value = JSON.stringify(dwellTimes);
        flightInput.value = JSON.stringify(flightTimes);

        // Feature vector: dwell + flight + typing speed
        const featureVector = [...dwellTimes, ...flightTimes, typingSpeed];
        let fvInput = document.getElementById("feature_vector");
        if (!fvInput) {
            fvInput = document.createElement("input");
            fvInput.type = "hidden";
            fvInput.name = "feature_vector";
            fvInput.id = "feature_vector";
            form.appendChild(fvInput);
        }
        fvInput.value = JSON.stringify(featureVector);

        // Key order
        let keyOrderInput = document.getElementById(keyOrderInputId);
        if (!keyOrderInput) {
            keyOrderInput = document.createElement("input");
            keyOrderInput.type = "hidden";
            keyOrderInput.name = "key_order";
            keyOrderInput.id = keyOrderInputId;
            form.appendChild(keyOrderInput);
        }
        keyOrderInput.value = JSON.stringify(keyOrder);
    });
}

/**
 * Fungsi helper untuk memulai recorder di form manapun
 * Contoh pemakaian:
 *   initRecorderForForm("passwordInput", "registerForm", "dwell", "flight", "typing_speed", "key_order");
 */
function initRecorderForForm(passwordId, formId, dwellId, flightId, speedId, keyOrderId) {
    initKeystrokeRecorder(passwordId, formId, dwellId, flightId, speedId, keyOrderId);
}