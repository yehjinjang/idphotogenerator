import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import cv2
from io import BytesIO
from ultralytics import YOLO
import os

# from dotenv import load_dotenv
# load_dotenv() # .env 파일에서 환경변수 불러오기

# -----------------------------
# 설정
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

st.set_page_config(page_title="AI 증명사진 생성기", layout="centered")
st.title("📸 AI 증명사진 생성기")
st.markdown("사망과 반려동물의 사진을 증명사진으로 자동 변화해보세요!")

# -----------------------------
# 사이드바 설정
# -----------------------------
st.sidebar.header("⚙️ 설정")

st.sidebar.subheader("📦 모델 선택")
model_choice = st.sidebar.selectbox("모델 선택", ["사람 (기본)", "dog-cat-detector"])


target = st.sidebar.selectbox("대상을 선택하세요", ["선택하세요", "사람", "강아지/고양이"], index=0)
if target == "선택하세요":
    target = None

# [수정됨] 얼굴 감지 방법 선택 제거됨 (YOLO만 사용)
bg_color = st.sidebar.selectbox("배경 색상 선택", ["흰색", "파란색", "민트"])
st.sidebar.markdown("## 📄 추가 정보")
name = st.sidebar.text_input("이름 (선택)")
breed = st.sidebar.text_input("종 / 출생일 등 (선택)")

# -----------------------------
# 파일 업로더
# -----------------------------
uploaded_file = None
if target:
    uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])
else:
    st.info("📌 먼저 **대상을 선택**하면 이미지를 업로드할 수 있어요!")



# -----------------------------
# 모델 로드
# -----------------------------
@st.cache_resource
def load_model(model_choice):
    if model_choice == "dog-cat-detector":
        model_path = os.path.join(MODEL_DIR, "dog-cat-detector.pt")
    else:
        st.error("사람 모델은 아직 지원되지 않아요.")
        return None

    if not os.path.exists(model_path):
        st.error(f"모델 파일이 존재하지 않아요: {model_path}")
        return None

    return YOLO(model_path)  # ultralytics에서 YOLO 모델 로딩

# -----------------------------
# IOU 중복 박스 제거
# -----------------------------
# def non_max_suppression(boxes, iou_threshold=0.5):
#     if len(boxes) == 0:
#         return []

#     boxes = sorted(boxes, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)
#     selected_boxes = []

#     while boxes:
#         chosen_box = boxes.pop(0)
#         selected_boxes.append(chosen_box)

#         def iou(box1, box2):
#             x1, y1, x2, y2 = box1
#             x1_p, y1_p, x2_p, y2_p = box2
#             xi1, yi1 = max(x1, x1_p), max(y1, y1_p)
#             xi2, yi2 = min(x2, x2_p), min(y2, y2_p)
#             intersection = max(0, xi2 - xi1) * max(0, yi2 - yi1)
#             area1 = (x2 - x1) * (y2 - y1)
#             area2 = (x2_p - x1_p) * (y2_p - y1_p)
#             union = area1 + area2 - intersection
#             return intersection / union if union > 0 else 0

#         boxes = [box for box in boxes if iou(chosen_box, box) < iou_threshold]

#     return selected_boxes

# -----------------------------
# 강아지/고양이 얼굴 감지 및 증명사진 생성
# -----------------------------
def process_pet(image, model_choice):

    model = load_model(model_choice)
    if model is None:
        st.warning("모델을 불러올 수 없습니다.")
        return


    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    with st.spinner("반려동물 얼굴 인식 중... ⏳"):
        results = model.predict(source=img_bgr, save=False, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

    #boxes = non_max_suppression(boxes, iou_threshold=0.5)

    if len(boxes) == 0:
        st.warning("반려동물 얼굴을 찾지 못했어요 😢")
        return

    st.markdown("### 🟩 인식된 반려동물 얼굴 (선택 가능)")
    cropped_faces = []
    boxed_image = img_np.copy()

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        box_width = x2 - x1
        desired_aspect = 3 / 4
        desired_crop_height = int(box_width / desired_aspect)

        new_y1 = y1
        new_y2 = min(y1 + desired_crop_height, img_np.shape[0])

        if new_y2 - new_y1 < 50:
            continue

        cv2.rectangle(boxed_image, (x1, new_y1), (x2, new_y2), (0, 255, 0), 2)
        cropped_faces.append(img_np[new_y1:new_y2, x1:x2])

    st.image(boxed_image, caption="감지된 얼굴", use_container_width=True)

    if len(cropped_faces) == 0:
        st.warning("얼굴 중심 영역이 너무 작아서 사용할 수 없어요.")
        return

    selected_face = st.selectbox("변환할 얼굴을 선택하세요", list(range(1, len(cropped_faces) + 1)))
    selected = cropped_faces[selected_face - 1]

    st.markdown("#### ✅ 선택한 얼굴 미리보기")
    st.image(selected, caption=f"선택한 얼굴 #{selected_face}", use_container_width=False)

    if selected.shape[0] < 50 or selected.shape[1] < 50:
        st.warning("감지된 얼굴이 너무 작아 품질이 떨어질 수 있어요 😅")

    face_image = Image.fromarray(selected)

    if st.button("📸 증명사진 생성하기 ✨"):
        display_result_section(image, face_image, target="반려동물")



# -----------------------------
# 테두리 추가
# -----------------------------
def add_border(image, border_size, border_color=(0, 0, 0)):
    if isinstance(image, Image.Image):
        image = np.array(image)
    bordered_image = cv2.copyMakeBorder(
        image, border_size, border_size, border_size, border_size,
        cv2.BORDER_CONSTANT, value=border_color
    )
    return Image.fromarray(bordered_image)

# -----------------------------
# 결과 표시
# -----------------------------
def display_result_section(original_img, result_img, target="대상"):
    result_img_with_border = add_border(result_img, border_size=1)
    st.success(f"{target}의 증명사진 생성 완료!")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### ⬅️ Before")
        st.image(original_img, use_column_width=True)
    with col2:
        st.markdown("#### ➡️ After")
        st.image(result_img_with_border, use_column_width=True)

    buf = BytesIO()
    result_img_with_border.save(buf, format="PNG")
    st.download_button("📥 이미지 다운로드", data=buf.getvalue(), file_name="id_photo.png", mime="image/png")

# -----------------------------
# 사람 처리
# -----------------------------
def process_human(image):
    st.info("사람 얼굴 인식 기능은 곧 추가될 예정입니다!")

# -----------------------------
# 메인 실행
# -----------------------------
if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.markdown("### 🖼️ 원본 이미지")
        st.image(image, caption="업로드된 이미지", use_column_width=True)

        if target == "사람":
            process_human(image)
        elif target == "강아지/고양이":
            process_pet(image, model_choice=model_choice)

    except Exception as e:
        st.error(f"이미지 로딩 중 오류 발생: {e}")
