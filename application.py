import streamlit as st
import cv2
from ultralytics import YOLO
from collections import defaultdict

def video_stream(model, confidence_threshold):
    # เปิดการเชื่อมต่อกับกล้อง
    cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 15)
    
    if not cap.isOpened():
        st.error("Cannot access the webcam. Please check your camera connection.")
        return

    frame_placeholder = st.empty()
    label_count_placeholder = st.empty()

    try:
        # ใช้ st.session_state เป็นเงื่อนไขในการวนลูป
        while st.session_state.get("run_cam", False):
            ret, frame = cap.read()
            if not ret:
                st.error("Cannot read frame from webcam.")
                break

            results = model(frame, conf=confidence_threshold)
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

    finally:
        # วางคำสั่งคืนค่ากล้องไว้ใน finally 
        # เพื่อรับประกันว่ากล้องจะถูกปิดเสมอ แม้ Streamlit จะหยุดการรันกลางคัน
        cap.release()
        cv2.destroyAllWindows()
    
def main():
    st.set_page_config(page_title="YOLO Object detection", layout="wide")
    
    # กำหนดค่าเริ่มต้นให้กับ session_state
    if "run_cam" not in st.session_state:
        st.session_state["run_cam"] = False

    st.title("Web Camera  Snail Detection")
    st.write("Please connect a Webcam to your computer")
    st.sidebar.header("งานวิจัยของศุภกร วงษ์เรืองพิบูล")
    
    model_path = "BestObjectDetect.pt"
    model = YOLO(model_path)
    st.sidebar.success("Model loaded successfully...")

    confidence_threshold = st.sidebar.slider("Set Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    
    # ฟังก์ชัน Callback สำหรับปุ่ม Stop
    def stop_camera():
        st.session_state["run_cam"] = False

    # สร้างปุ่ม Start และ Stop โดยให้ Stop เรียกใช้ callback
    col1, col2 = st.sidebar.columns(2)
    start_button = col1.button("Start Webcam")
    stop_button = col2.button("Stop Webcam", on_click=stop_camera)
    
    # เมื่อกด Start ให้ตั้งสถานะเป็น True
    if start_button:
        st.session_state["run_cam"] = True

    # ตรวจสอบสถานะว่าควรเปิดกล้องหรือไม่
    if st.session_state["run_cam"]:
        video_stream(model, confidence_threshold)

if __name__ == "__main__":
    main()
