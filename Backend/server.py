import serial
import time
import threading
from collections import deque
from flask import Flask, jsonify
from flask_cors import CORS

# ==============================
# CONFIG
# ==============================
SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 115200

READ_HZ = 200
API_HZ = 10

BASELINE_ALPHA = 0.002
CLENCH_ON = 12.0
CLENCH_OFF = 6.0

RMS_WINDOW = 25
AI_WINDOW = 256

# ==============================
# GLOBAL STATE
# ==============================
env_buffer = deque(maxlen=RMS_WINDOW)
ai_buffer = deque(maxlen=AI_WINDOW)

baseline = None
current_rms = 0.0
clenched = False

lock = threading.Lock()

# ==============================
# SERIAL THREAD
# ==============================
def serial_loop():
    global baseline, current_rms, clenched

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        print(f"✅ Connected to Arduino on {SERIAL_PORT}")
    except Exception as e:
        print("❌ Serial error:", e)
        return

    while True:
        try:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            try:
                val = int(line)
            except ValueError:
                continue

            with lock:
                # ================= AI BUFFER (NEW)
                ai_buffer.append(val)

                # ================= RMS PIPELINE (UNCHANGED)
                env_buffer.append(val)

                if len(env_buffer) < RMS_WINDOW:
                    continue

                rms = (
                    sum(v * v for v in env_buffer)
                    / len(env_buffer)
                ) ** 0.5

                if baseline is None:
                    baseline = rms
                    continue

                # adaptive baseline
                if not clenched:
                    baseline = (
                        (1 - BASELINE_ALPHA) * baseline
                        + BASELINE_ALPHA * rms
                    )

                delta = rms - baseline

                # hysteresis clench detection
                if not clenched and delta > CLENCH_ON:
                    clenched = True
                elif clenched and delta < CLENCH_OFF:
                    clenched = False

                current_rms = rms

        except Exception:
            continue

# ==============================
# FLASK APP
# ==============================
app = Flask(__name__)
CORS(app)

# ==============================
# EXISTING UI ENDPOINT (UNCHANGED)
# ==============================
@app.route("/prediction")
def prediction():
    with lock:
        return jsonify({
            "rms": round(current_rms, 2),
            "clenched": bool(clenched)
        })

# ==============================
# AI WINDOW ENDPOINT (NEW)
# ==============================
@app.route("/ai_window")
def ai_window():
    with lock:
        if len(ai_buffer) < AI_WINDOW:
            return jsonify({
                "ready": False,
                "window": []
            })

        return jsonify({
            "ready": True,
            "window": list(ai_buffer)
        })

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    print("""
=======================================================
🚀 EMG Hand State Server (STABLE + AI ANALYZER READY)
=======================================================
✋ OPEN vs ✊ CLENCHED → RMS Engine (UNCHANGED)
🧠 Adaptive Baseline + Hysteresis
🤖 Transformer AI Window Streaming ACTIVE
🌐 /prediction  → Main UI
🌐 /ai_window   → AI Analyzer
=======================================================
""")

    t = threading.Thread(
        target=serial_loop,
        daemon=True
    )
    t.start()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
