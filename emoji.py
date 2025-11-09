import streamlit as st
import cv2
from fer import FER
import numpy as np

st.set_page_config(page_title="Emotion Camera", page_icon="🎥", layout="centered")
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #1c1c1c, #2d3436);
        color: #fff;
        font-family: 'Poppins', sans-serif;
    }
    .stApp {
        background: linear-gradient(145deg, #111, #222);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 0 25px rgba(0,0,0,0.5);
    }
    .title {
        text-align: center;
        font-size: 3em;
        color: #f1c40f;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        color: #aaa;
        font-size: 1.1em;
        margin-bottom: 30px;
    }
    .emoji {
        text-align: center;
        font-size: 120px;
        margin-top: -30px;
        animation: bounce 1.5s infinite;
    }
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-15px); }
    }
    </style>
""", unsafe_allow_html=True)

# ---------- APP HEADER ----------
st.markdown("<div class='title'>😊 Emotion Camera</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Detects your facial expressions in real-time and displays matching emojis 🎭</div>", unsafe_allow_html=True)

# ---------- EMOTION DETECTOR ----------
detector = FER(mtcnn=True)

# Map emotions to emojis and theme colors
emotion_map = {
    "angry": {"emoji": "😠", "color": "#ff4c4c"},
    "disgust": {"emoji": "🤢", "color": "#9b59b6"},
    "fear": {"emoji": "😨", "color": "#5dade2"},
    "happy": {"emoji": "😄", "color": "#f1c40f"},
    "sad": {"emoji": "😢", "color": "#3498db"},
    "surprise": {"emoji": "😲", "color": "#e67e22"},
    "neutral": {"emoji": "😐", "color": "#95a5a6"},
}

# ---------- CAMERA SECTION ----------
start_camera = st.checkbox("📷 Start Camera")

if start_camera:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("🚫 Camera not accessible. Please allow webcam access.")
    else:
        stframe = st.empty()
        emoji_placeholder = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Unable to access webcam feed.")
                break

            frame = cv2.flip(frame, 1)
            results = detector.detect_emotions(frame)

            if results:
                emotion, score = detector.top_emotion(frame)
                if emotion:
                    e = emotion_map.get(emotion, emotion_map["neutral"])

                    # Draw face box and emotion text
                    for face in results:
                        (x, y, w, h) = face["box"]
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
                        cv2.putText(frame, f"{emotion} ({score:.2f})", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                    # Dynamic background color
                    st.markdown(f"""
                        <style>
                        .stApp {{
                            background: linear-gradient(145deg, {e['color']}, #000000);
                        }}
                        </style>
                    """, unsafe_allow_html=True)

                    # Display emoji
                    emoji_placeholder.markdown(
                        f"<div class='emoji'>{e['emoji']}</div>",
                        unsafe_allow_html=True
                    )
            else:
                emoji_placeholder.markdown("<div class='emoji'>🙂</div>", unsafe_allow_html=True)

            # Convert to RGB for Streamlit display
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(rgb, channels="RGB", use_container_width=True)

            if not start_camera:
                break

        cap.release()
        st.success("Camera stopped.")
else:
    st.info("✅ Click the checkbox above to start the emotion camera.")
