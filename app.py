import streamlit as st
from io import BytesIO
from modules.idphoto_human import generate_human_id_photo
from modules.idphoto_pet import generate_pet_id_photo
from PIL import Image

# 초기 상태 세팅
if "step" not in st.session_state:
    st.session_state.step = "upload"

# 페이지 기본 설정
st.set_page_config(page_title="AI 증명사진 생성기", layout="wide")
st.markdown("<h1 style='text-align: center; color: #4A90E2;'>📸 AI 사진관</h1>", unsafe_allow_html=True)

# Step 1: 업로드 & 옵션 선택
if st.session_state.step == "upload":
    st.markdown("### 1️⃣ 대상을 선택해주세요")
    target = st.radio("사람 또는 강아지/고양이 중 하나를 선택하세요!", ["사람", "강아지/고양이"], horizontal=True)

    st.markdown("---")
    st.markdown("### 2️⃣ 이미지를 업로드하세요")
    uploaded_file = st.file_uploader("이미지 파일을 업로드하세요 (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        pil_image = Image.open(uploaded_file).convert("RGB")
        st.image(uploaded_file, caption="업로드한 이미지", use_column_width=True)

        st.markdown("---")
        st.markdown("### 3️⃣ 옵션을 선택하세요")

        # 사람 옵션
        if target == "사람":
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                gender = st.radio("성별", ["여자", "남자"], horizontal=True)
            with col2:
                background_color = st.radio("배경 색상", ["흰색", "하늘색", "회색"], index=0, horizontal=True)
            with col3:
                upscale = st.radio("업스케일", ["원본", "2배", "4배"], index=0, horizontal=True)
            with col4:
                face_ratio = st.slider("얼굴 비율", 0.6, 0.9, 0.75, 0.01)

            st.markdown("### 4️⃣ 생성하기")
            if st.button("🚀 AI 증명사진 생성하기"):
                with st.spinner("AI가 사진을 만드는 중입니다...✨"):
                    result = generate_human_id_photo(uploaded_file, gender, background_color, face_ratio, upscale)
                    st.session_state.result_image = result
                    st.session_state.original_image = uploaded_file
                    st.session_state.target = "사람"
                    st.session_state.step = "result"
                    st.experimental_rerun()

        # 강아지/고양이 옵션
        elif target == "강아지/고양이":
            col1, col2 = st.columns(2)
            with col1:
                background_color = st.radio("배경 색상", ["핑크", "민트", "하늘색", "흰색", "검정"], index=3)
            with col2:
                frame_style = st.selectbox("액자 스타일", ["심플", "꽃무늬", "빈티지", "고급 금테"])

            st.markdown("### 4️⃣ 생성하기")
            if st.button("🐾 AI 증명사진 생성하기"):
                with st.spinner("AI가 사진을 만드는 중입니다... 🐶🐱"):
                    result = generate_pet_id_photo(uploaded_file, frame_style, background_color)
                    st.session_state.result_image = result
                    st.session_state.original_image = uploaded_file
                    st.session_state.target = "강아지/고양이"
                    st.session_state.step = "result"
                    st.experimental_rerun()
    else:
        st.info("이미지를 업로드하면 다음 단계로 넘어갈 수 있어요! 😊")

# Step 2: 결과 화면
elif st.session_state.step == "result":
    st.success("✅ 증명사진 생성 완료!")
    st.balloons()

    col1, col2, col3 = st.columns([4, 1, 4])
    with col1:
        st.image(st.session_state.original_image, caption="원본 이미지", use_column_width=True)
    with col2:
        st.markdown("<h3 style='text-align: center;'>➡️</h3>", unsafe_allow_html=True)
    with col3:
        st.image(st.session_state.result_image, caption="AI 증명사진", use_column_width=True)

    # 다운로드
    buf = BytesIO()
    st.session_state.result_image.save(buf, format="PNG")
    file_name = "human_id_photo.png" if st.session_state.target == "사람" else "pet_id_photo.png"
    st.download_button("📥 이미지 다운로드", data=buf.getvalue(), file_name=file_name, mime="image/png")

    # 다시 만들기 버튼
    if st.button("🔄 다시 만들기"):
        st.session_state.step = "upload"
        st.experimental_rerun()

# 푸터
st.markdown("---")
st.caption("AI 증명사진 생성기 • 만든 사람: 장예진 정수인 정재욱 🤖")