import streamlit as st
import cv2
from ultralytics import YOLO
from collections import defaultdict

def video_stream(model, confidence_threshold):
    # เปิดการเชื่อมต่อกับกล้อง (0 คือกล้องตัวหลักติด Laptop, 1 2... คือกล้องตัวที่เชื่อมกับ USB)
    cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 15)
    
    if not cap.isOpened():
        st.error("Cannot access the webcam. Please check your camera connection.")
        return

    frame_placeholder = st.empty()
    label_count_placeholder = st.empty()

    stop_button = st.sidebar.button("Stop Webcam")

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Cannot read frame from webcam.")
            break

        results = model(frame, conf=confidence_threshold)  # ใช้ค่า confidence จากผู้ใช้
        label_counts = defaultdict(int)

        for result in results:
            for box in result.boxes.data:
                x1, y1, x2, y2, conf, cls = box

                if conf >= confidence_threshold:
                    label = f"{model.names[int(cls)]}"
                    label_counts[label] += 1  # Count the label

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} ({conf:.2f})", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", width=640)

        label_count_placeholder.markdown("### Object Counts:")
        for label, count in label_counts.items():
            label_count_placeholder.write(f"- **{label}**: {count}")

        if stop_button:
            break
        
    cap.release()
    cv2.destroyAllWindows()
    
def main():
    st.set_page_config(page_title="YOLO Object detection", layout="wide")
    
    st.title("Web Camera  Snail  Detection")
    st.write("Please connect a Webcam to your computer")
    st.sidebar.header("งานวิจัยของศุภกร วงษ์เรืองพิบูล")
    model_path = "BestObjectDetect.pt"
    model = YOLO(model_path)
    st.sidebar.success("Model loaded successfully...")

    confidence_threshold = st.sidebar.slider("Set Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    start_button = st.sidebar.button("Start Webcam")
    
    if start_button:
        video_stream(model, confidence_threshold)

main()
