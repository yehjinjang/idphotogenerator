# app.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from rembg import remove
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from io import BytesIO

st.set_page_config(layout="wide")
st.title("증명사진 스타일 변환 서비스")
st.write("한 명 이상의 얼굴이 감지되면, 원하는 얼굴을 선택해서 증명사진으로 변환하세요.")

uploaded_file = st.file_uploader("사진을 업로드하세요", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    st.image(image, caption="업로드된 사진", use_column_width=True)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        st.error("얼굴이 감지되지 않았습니다.")
    else:
        st.subheader("얼굴을 선택하세요")
        selected_index = st.radio("얼굴 선택", list(range(len(faces))), horizontal=True)

        x, y, w, h = faces[selected_index]
        cropped_face = image_np[y - int(h * 0.3): y + int(h * 1.2), x - int(w * 0.1): x + int(w * 1.1)]
        st.image(cropped_face, caption="선택된 얼굴", use_column_width=True)

        cropped_face_pil = Image.fromarray(cropped_face)
        no_bg_face = remove(cropped_face_pil)
        st.image(no_bg_face, caption="배경 제거된 얼굴", use_column_width=True)

        st.subheader("증명사진 스타일로 변환 중...")

        model_id = "runwayml/stable-diffusion-v1-5"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        with st.spinner("Stable Diffusion 모델 로딩 중..."):
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            )
            pipe = pipe.to(device)

        init_image = no_bg_face.resize((384, 384))
        prompt = (
            "a professional passport photo of a person, white background, studio lighting, "
            "formal attire, highly detailed, centered face"
        )

        with st.spinner("이미지 생성 중..."):
            result = pipe(prompt=prompt, image=init_image, strength=0.8, guidance_scale=7.5)
            result_image = result.images[0]
            st.image(result_image, caption="증명사진 결과", use_column_width=True)

            # 다운로드 버튼
            buf = BytesIO()
            result_image.save(buf, format="JPEG")
            byte_im = buf.getvalue()

            st.download_button(
                label="증명사진 다운로드",
                data=byte_im,
                file_name="passport_photo.jpg",
                mime="image/jpeg"
            )