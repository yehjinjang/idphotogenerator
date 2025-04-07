import streamlit as st
from io import BytesIO
from modules.idphoto_human import generate_human_id_photo
from modules.idphoto_pet import generate_pet_id_photo
from PIL import Image
import requests

# 페이지 설정
st.set_page_config(page_title="AI 사진관", layout="centered")
st.title("📸 AI 증명사진 생성기")
st.sidebar.header("⚙️ 설정")

# 대상 선택
target = st.sidebar.selectbox("대상 선택", ["사람", "강아지/고양이"], index=0)
uploaded_file = st.sidebar.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

# 👤 사람용 옵션
if target == "사람":
    gender = st.sidebar.radio("성별 선택", ["여자", "남자"], index=0)
    background_color = st.sidebar.radio("배경 색상", ["흰색", "하늘색", "회색"], index=0)
    face_ratio = st.sidebar.slider("얼굴 크기 비율", 0.6, 0.9, 0.75, 0.01)
    upscale = st.sidebar.radio("업스케일 배율", ["원본", "2배", "4배"], index=0)

    if uploaded_file and st.button("✨ 증명사진 생성"):
        pil_image = Image.open(uploaded_file).convert("RGB")

        with st.spinner("생성 중..."):
            result = generate_human_id_photo(uploaded_file, gender, background_color, face_ratio, upscale)
            st.success("✅ 증명사진 생성 완료!")

            # 📸 이미지 나란히
            col1, col2 = st.columns(2)
            with col1:
                st.image(uploaded_file, caption="📷 원본 이미지", use_column_width=True)
            with col2:
                st.image(result, caption="✨ AI 증명사진", use_column_width=True)

            # 다운로드
            buf = BytesIO()
            result.save(buf, format="PNG")
            st.download_button("📥 이미지 다운로드", data=buf.getvalue(), file_name="human_photo.png", mime="image/png")

# 🐶🐱 강아지/고양이용 옵션
elif target == "강아지/고양이":
    background_color = st.sidebar.radio("배경 색상 선택", ["핑크", "민트", "하늘색", "흰색", "검정"], index=3)
    frame_style = st.sidebar.selectbox("액자 스타일", ["심플", "꽃무늬", "빈티지", "고급 금테"])

    if uploaded_file and st.button("✨ AI 증명사진 생성"):
        with st.spinner("생성 중..."):
            result = generate_pet_id_photo(uploaded_file, frame_style, background_color)
            st.success("✅ 증명사진 생성 완료!")

            # 📸 이미지 나란히
            col1, col2 = st.columns(2)
            with col1:
                st.image(uploaded_file, caption="📷 원본 이미지", use_column_width=True)
            with col2:
                st.image(result, caption="✨ AI 증명사진", use_column_width=True)

            # 다운로드
            buf = BytesIO()
            result.save(buf, format="PNG")
            st.download_button("📥 이미지 다운로드", data=buf.getvalue(), file_name="pet_photo.png", mime="image/png")

st.caption("AI 증명사진 생성기")
