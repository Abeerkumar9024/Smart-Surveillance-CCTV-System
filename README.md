# Smart Surveillance System (Phase-1)

Smart Surveillance System built using **Python, OpenCV, and Computer Vision**.  
The system monitors live camera feed, detects human presence in real time using **MobileNetSSD deep learning model**, records video automatically, captures snapshots, and sends email alerts with image attachment.

When a person is detected, recording starts automatically.  
If no person is detected for a few seconds, recording stops and the video is saved with a timestamp filename.  
If a person reappears within the waiting time, recording continues and empty frames are skipped to store only useful footage.

The system also supports night/low-brightness detection, sound alert, event logging, email notification, and automatic file management, making it suitable for offline smart surveillance applications.

---

## 🚀 Features

- Real-time human detection using OpenCV DNN  
- MobileNetSSD deep learning model  
- Automatic video recording on detection  
- Stop recording after inactivity  
- Smart trimming of empty frames  
- Timestamped video saving  
- Snapshot capture on detection  
- Email alert with image attachment  
- Sound alert on motion detection  
- Night / low brightness mode (OpenCV processing)  
- Event logging system  
- Automatic folder creation  
- Works offline (no cloud required)  

---

## 🧠 Technologies Used

- Python  
- OpenCV  
- Deep Learning (MobileNetSSD)  
- Computer Vision  
- SMTP Email (Gmail)  
- NumPy  
- Winsound  
- DateTime / OS / SSL / SMTP  

## 📂 Project Structure

```
SmartSurveillance/
│
├── surveillance.py
├── MobileNetSSD_deploy.prototxt
├── MobileNetSSD_deploy.caffemodel
├── motion_detected.wav
├── events.log
├── captures/
└── README.md
```

---

## ⚙️ How It Works

1. Camera starts live monitoring using OpenCV  
2. Frame is processed using DNN model  
3. If person detected:
   - Recording starts  
   - Snapshot captured  
   - Sound alert plays  
   - Email sent with image  
4. If no person detected for few seconds:
   - Recording stops  
   - Video saved with timestamp  
5. If person appears again within time:
   - Recording continues  
   - Empty frames skipped  
6. All events stored in log file  

---

## 📧 Email Alert Setup

Edit these values in code:

SENDER_EMAIL  
SENDER_PASSWORD  
RECEIVER_EMAIL  
SMTP_SERVER  
SMTP_PORT  

Use Gmail App Password instead of normal password.

---

## ▶️ How to Run

Install dependencies

pip install opencv-python  
pip install numpy  

Run program

python surveillance.py  

Press Q to exit.

---

## 📌 Phase-1

Phase-1 runs on Computer using OpenCV and Computer Vision.  
Next phase will run on ESP32 / Edge AI / IoT device.

---

## 👨‍💻 Author

Abeer Kumar  
B.Tech CSE (AI & ML)  
Computer Vision | IoT | AI | Embedded Systems
