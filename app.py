import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import openai
import requests
import os
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    st.error("OpenAI API Key가 설정되지 않았습니다. .env 파일을 확인하세요.")
    st.stop()

# Mediapipe 및 YOLOv8 초기화
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
yolo_model = YOLO("best.pt")  # 강아지/고양이 얼굴 탐지 모델

st.set_page_config(page_title="AI 증명사진 생성기", layout="centered")
st.title(":camera_with_flash: AI 증명사진 생성기")
st.markdown("사람과 반려동물의 사진을 증명사진으로 자동 변환해보세요!")

st.sidebar.header("⚙️ 설정")
target = st.sidebar.selectbox("대상을 선택하세요", ["사람", "강아지", "고양이"])

if target == "사람":
    gender = st.sidebar.radio("성별을 선택하세요", ["여자", "남자"])

# 이미지 업로드
uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w, _ = image.shape
    bounding_boxes = []

    if target == "사람":
        face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.8)
        results = face_detection.process(image_rgb)
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                x_min = int(bboxC.xmin * w)
                y_min = int(bboxC.ymin * h)
                width = int(bboxC.width * w)
                height = int(bboxC.height * h)
                expand_ratio_height_top = 0.8
                expand_ratio_height_bottom = 0.8
                expand_ratio_width = 0.2
                new_y_min = max(0, y_min - int(height * expand_ratio_height_top))
                new_y_max = min(h, y_min + height + int(height * expand_ratio_height_bottom))
                new_x_min = max(0, x_min - int(width * expand_ratio_width))
                new_x_max = min(w, x_min + int(width * (1 + expand_ratio_width)))
                bounding_boxes.append((new_x_min, new_y_min, new_x_max - new_x_min, new_y_max - new_y_min))
    else:
        class_id = 1 if target == "강아지" else 0  # YOLO class ID
        results = yolo_model.predict(image_rgb, conf=0.8)
        for box, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls):
            if int(cls) == class_id:
                x1, y1, x2, y2 = map(int, box[:4])
                x1 = max(0, x1 - 10)
                y1 = max(0, y1 - 30)
                x2 = min(w, x2 + 10)
                y2 = min(h, y2 + 30)
                bounding_boxes.append((x1, y1, x2 - x1, y2 - y1))

    # GrabCut 배경 제거 함수
    def remove_background(image, bounding_box):
        mask = np.zeros(image.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        x, y, w, h = bounding_box
        rect = (x, y, w, h)
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, iterCount=5, mode=cv2.GC_INIT_WITH_RECT)
        mask_2d = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
        result_image = image * mask_2d[:, :, np.newaxis]
        return result_image
    
    if bounding_boxes:
        cols = st.columns(len(bounding_boxes))
        selected_box_index = 0
        with st.container():
            for i, col in enumerate(cols):
                if col.button(f"얼굴 {i+1}"):
                    selected_box_index = i
                selected_box = bounding_boxes[selected_box_index]
                cropped_face_with_bg_removed = remove_background(image_rgb.copy(), selected_box)
                st.image(cropped_face_with_bg_removed, caption="배경 제거된 얼굴", use_column_width=True)

        # 증명사진 생성 버튼 항상 보이게
        if st.button("증명사진 생성하기 :sparkles:"):
            try:
                input_image_path = "background_removed_face.png"
                cropped_face_pil = Image.fromarray(cropped_face_with_bg_removed)
                cropped_face_pil.save(input_image_path)

                if target == "사람":
                    if gender == "여자":
                        prompt_description = "A professional ID photo of a asian woman wearing a suit on a clean white background."
                    elif gender == "남자":
                        prompt_description = "A professional ID photo of a asian man wearing a suit on a clean white background."
                elif target == "강아지":
                    prompt_description = "A cute professional portrait of a dog wearing a bow tie on a studio background."
                elif target == "고양이":
                    prompt_description = "A charming ID photo of a cat in a tuxedo on a clean white background."

                response = openai.Image.create(
                    prompt=prompt_description,
                    n=1,
                    size="1024x1024"
                )

                generated_image_url = response["data"][0]["url"]
                st.success("증명사진이 성공적으로 생성되었습니다!")

                response_img_data = requests.get(generated_image_url)
                output_image_path = "generated_id_photo.png"
                with open(output_image_path, "wb") as output_file:
                    output_file.write(response_img_data.content)

                final_image_pil = Image.open(output_image_path)
                resized_image_pil = final_image_pil.resize((413, 531))
                resized_output_path = "resized_id_photo.png"
                resized_image_pil.save(resized_output_path)

                st.image(resized_image_pil, caption="Resized ID Photo (413x531)", use_column_width=True)
                with open(resized_output_path, "rb") as file:
                    st.download_button(
                        label=":inbox_tray: 증명사진 다운로드",
                        data=file,
                        file_name="id_photo.png",
                        mime="image/png",
                    )
            except Exception as e:
                st.error(f"증명사진 생성 중 오류가 발생했습니다: {e}")
        else:
            st.warning("먼저 얼굴을 인식한 후 증명사진을 생성하세요.")
    else:
        st.warning("얼굴을 찾을 수 없습니다. 다른 이미지를 시도해보세요.")
else:
    st.info("좌측 사이드바에서 설정 후 이미지를 업로드해 주세요.")
