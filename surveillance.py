import cv2
import datetime
import os
import time
import winsound
import smtplib
import ssl
from email.message import EmailMessage

# ============ CONFIG ============
CONF_THRESHOLD = 0.5
OUTPUT_DIR = r"C:\Users\kk\Documents\python-programs\SmartSurveillance\captures"
LOG_FILE = "events.log"

# Gmail details
SENDER_EMAIL = "naman655362@gmail.com"
SENDER_PASSWORD = "yoid aiqs eabs vohc"  # App Password
RECEIVER_EMAIL = "abeerprofessional60@gmail.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465

WAIT_AFTER_PERSON = 15   # seconds to wait before stopping recording
SNAPSHOT_DELAY = 1.5     # delay before sending snapshot email
# ===============================

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Full absolute paths to model
prototxt = r"C:\Users\kk\Documents\python-programs\SmartSurveillance\MobileNetSSD_deploy.prototxt"
model = r"C:\Users\kk\Documents\python-programs\SmartSurveillance\MobileNetSSD_deploy.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt, model)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

# Logging function
def log_event(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")  # 12-hour format
    with open(LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] {message}\n")
    print(f"[LOG] {message}")

# Email sender
def send_email(frame):
    try:
        msg = EmailMessage()
        msg["From"] = SENDER_EMAIL
        msg["To"] = RECEIVER_EMAIL
        msg["Subject"] = "🚨 Person Detected!"
        msg.set_content("A person was detected by SmartSurveillance.")

        # Encode frame as JPEG in memory
        _, buffer = cv2.imencode(".jpg", frame)
        file_data = buffer.tobytes()
        file_name = "snapshot.jpg"
        msg.add_attachment(file_data, maintype="image", subtype="jpeg", filename=file_name)

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, context=context) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
        log_event("Email sent with snapshot")
    except Exception as e:
        log_event(f"Email sending failed: {e}")

# Camera setup
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = None
recording = False
last_seen = None
snapshot_taken = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # Night mode (optional)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = gray.mean()
    if brightness < 50:
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Prepare blob and detect
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    person_detected = False

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > CONF_THRESHOLD:
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] == "person":
                person_detected = True
                last_seen = datetime.datetime.now()
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                startX, startY, endX, endY = box.astype("int")
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                label = f"Person: {confidence*100:.1f}%"
                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Timestamp (12-hour format)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
    cv2.putText(frame, timestamp, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    now = datetime.datetime.now()

    if person_detected:
        if not recording:
            filename = os.path.join(OUTPUT_DIR, f"person_{now.strftime('%Y%m%d_%I%M%S%p')}.avi")
            out = cv2.VideoWriter(filename, fourcc, 20.0, (w, h))
            recording = True
            snapshot_taken = False
            log_event(f"Recording started: {filename}")

        # Write frame to video
        if out is not None:
            out.write(frame)

        # REC icon (permanent, no question mark)
        cv2.putText(frame, "REC", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.circle(frame, (75, 20), 8, (0, 0, 255), -1)

        # Take snapshot + play sound + send email (only once per detection)
        if not snapshot_taken:
            time.sleep(SNAPSHOT_DELAY)  # wait 1.5 seconds
            winsound.PlaySound("motion_detected.wav", winsound.SND_FILENAME)
            send_email(frame)
            snapshot_taken = True

    # Stop recording if no person for WAIT_AFTER_PERSON seconds
    elif recording and last_seen and (now - last_seen).seconds > WAIT_AFTER_PERSON:
        recording = False
        if out is not None:
            out.release()
            log_event("Recording stopped")
            out = None

    # Show live frame
    cv2.imshow("SmartSurveillance", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Cleanup
if recording and out is not None:
    out.release()
cap.release()
cv2.destroyAllWindows()
