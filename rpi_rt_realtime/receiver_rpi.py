import socket
import struct
import cv2
import numpy as np
import easyocr
import re
from langdetect import detect
from collections import Counter
import time
import syslog

# ========= Configuration =========
PORT = 5001
HOST = ''
reader = easyocr.Reader(['en', 'hi'], verbose=False)
text_counter = Counter()
finalized_set = set()
target_words = {"HAND", "NOSE", "EYES", "HEAD"}
seen_targets = set()

frame_id = 0

# ========= Stats Tracking =========
wcet = 0.0
min_time = float('inf')
max_time = 0.0
total_time = 0.0
frame_exec_count = 0

# ========= Helper =========
def is_probably_english(text):
    return all(ord(c) < 128 for c in text) and len(text.split()) <= 3

# ========= Socket Setup =========
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((HOST, PORT))
sock.listen(1)
syslog.syslog(syslog.LOG_INFO, f"[INIT] Server listening on port {PORT}")
print(f"[INFO] Waiting for connection on port {PORT}...")
conn, addr = sock.accept()
syslog.syslog(syslog.LOG_INFO, f"[CONNECT] Connection from {addr}")
print(f"[INFO] Connection from {addr}")

data = b''
payload_size = struct.calcsize(">L")

try:
    while True:
        while len(data) < payload_size:
            packet = conn.recv(4096)
            if not packet:
                raise ConnectionError("Disconnected")
            data += packet

        packed_size = data[:payload_size]
        data = data[payload_size:]
        frame_size = struct.unpack(">L", packed_size)[0]

        while len(data) < frame_size:
            data += conn.recv(4096)

        frame_data = data[:frame_size]
        data = data[frame_size:]

        # ========= Start Timing =========
        start_time = time.perf_counter()

        frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
        frame_id += 1
        syslog.syslog(syslog.LOG_DEBUG, f"[FRAME] Received frame {frame_id} ({frame_size} bytes)")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = reader.readtext(rgb)
        combined_text = ""

        for bbox, text, conf in results:
            if conf > 0.5 and len(text.strip()) >= 3:
                clean_text = re.sub(r'[^\w\s]', '', text).strip()
                combined_text += clean_text + " "

                top_left = tuple(map(int, bbox[0]))
                bottom_right = tuple(map(int, bbox[2]))
                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
                cv2.putText(frame, f"{clean_text} ({conf:.2f})", (top_left[0], top_left[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        combined_text = combined_text.strip()
        if combined_text:
            text_counter[combined_text] += 1
            if text_counter[combined_text] == 2 and combined_text not in finalized_set:
                try:
                    lang = detect(combined_text)
                except:
                    lang = "unknown"
                log_msg = f"[OCR FINAL] Lang={lang.upper()} | Text='{combined_text}'"
                syslog.syslog(syslog.LOG_NOTICE, log_msg)
                finalized_set.add(combined_text)

                upper_text = combined_text.upper()
                for word in target_words:
                    if word in upper_text and word not in seen_targets:
                        seen_targets.add(word)
                        syslog.syslog(syslog.LOG_INFO, f"[MATCHED] {word}")

                if seen_targets == target_words:
                    syslog.syslog(syslog.LOG_NOTICE, "TEST PASSED: All target words detected.")
                    break

        # ========= End Timing =========
        end_time = time.perf_counter()
        exec_time = end_time - start_time
        wcet = max(wcet, exec_time)
        min_time = min(min_time, exec_time)
        max_time = max(max_time, exec_time)
        total_time += exec_time
        frame_exec_count += 1

        if frame_exec_count % 10 == 0:
            avg_time = total_time / frame_exec_count
            jitter = max_time - min_time
            stat_msg = (
                f"[STATS] Frames={frame_exec_count}, "
                f"WCET={wcet:.4f}s, MIN={min_time:.4f}s, MAX={max_time:.4f}s, "
                f"AVG={avg_time:.4f}s, JITTER={jitter:.4f}s"
            )
            syslog.syslog(syslog.LOG_INFO, stat_msg)

        cv2.imshow("OCR Stream", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    error_msg = f"[ERROR] {type(e).__name__}: {e}"
    syslog.syslog(syslog.LOG_ERR, error_msg)

finally:
    syslog.syslog(syslog.LOG_INFO, "[CLOSE] Closing sockets and display")
    conn.close()
    sock.close()
    cv2.destroyAllWindows()
