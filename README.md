**BIMLearn** is an interactive, web-based educational platform designed to help users learn and practice **Malaysian Sign Language - Bahasa Isyarat Malaysia (BIM)**. The system utilizes a real-time, webcam-based AI inference engine to provide instant feedback on sign language execution directly in your browser. 

*Developed as a Final Year Project.*

---

## 🚀 Live Application
**https://bimlearn-app-89862233187.asia-southeast1.run.app**

---

## ✨ Core Features (Use Cases)

1. 📚 __Learning Modules - Modul Pembelajaran (UC-1)__ Browse the BIM vocabulary, view reference images/descriptions, and track your overall mastery percentage for each word.
2. **💪 Interactive Practice - Latihan Interaktif (UC-2)** Use your webcam to practice signs in real-time. The AI engine processes your hand landmarks and provides instant validation and confidence scores.
3. **📝 Evaluation Quiz - Kuiz Penilaian (UC-3)** Test your knowledge with a timed, interactive webcam-based quiz. Complete 5 random signs under pressure to score points.
4. **📊 Performance Statistics - Statistik Prestasi Pengguna (UC-4)** View your learning journey, including daily practice streaks, 14-day historical accuracy charts, and targeted feedback on "weak" words that require more practice.

---

## 🛠️ Tech Stack

* **Backend:** Python 3.11, Flask, Gunicorn
* **Machine Learning:** TensorFlow/Keras (EfficientNet-B0), Google MediaPipe (Hand Landmarks), OpenCV
* **Database:** PostgreSQL (hosted on Supabase), SQLAlchemy ORM
* **Deployment:** Docker, Google Cloud Build, Google Cloud Run
* **Frontend:** HTML5, CSS3, Vanilla JavaScript, Chart.js

---

## 📂 Project Structure

```text
bimlearn/
├── predictor.py              # AI inference core (MediaPipe + EfficientNet-B0)
├── models.py                 # SQLAlchemy DB models (Word, Progress, Quiz, Stats)
├── app.py                    # Flask app — handles routing and logic
├── wsgi.py                   # Gunicorn entry point for production
├── requirements.txt          # Python dependencies
├── Dockerfile                # Instructions for containerising the app
├── static/
│   ├── js/camera.js          # Shared webcam + API polling logic
│   └── signs/                # Reference images for learning
└── templates/                # HTML pages (Dashboard, Practice, Quiz, Stats)
