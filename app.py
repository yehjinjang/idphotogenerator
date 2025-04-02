import streamlit as st
from PIL import Image

st.set_page_config(page_title="AI 증명사진 생성기", layout="centered")

st.title("📸 AI 증명사진 생성기")
st.markdown("사람과 반려동물의 사진을 증명사진으로 자동 변환해보세요!")

# 사이드바 설정
st.sidebar.header("⚙️ 설정")
target = st.sidebar.selectbox("대상을 선택하세요", ["사람", "강아지", "고양이"])
bg_color = st.sidebar.selectbox("배경 색상 선택", ["흰색", "파란색", "민트"])

# 텍스트 입력 (선택사항)
st.sidebar.markdown("## 📄 추가 정보")
name = st.sidebar.text_input("이름 (선택)")
breed = st.sidebar.text_input("견종 / 출생일 등 (선택)")

# 이미지 업로드
uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    st.markdown("### 🖼️ 원본 이미지")
    st.image(image, caption="업로드된 이미지", use_column_width=True)

    if st.button("증명사진 생성하기 ✨"):
        processed_image = image  # 실제로는 여기서 모델 결과를 받아야 함

        st.success(f"{target}의 증명사진 생성 완료!")

        # 가로로 Before / After 이미지 출력
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### ⬅️ Before")
            st.image(image, use_column_width=True)

        with col2:
            st.markdown("#### ➡️ After")
            st.image(processed_image, use_column_width=True)

        st.download_button("📥 이미지 다운로드", data=processed_image.tobytes(), file_name="id_photo.png")
else:
    st.info("좌측 사이드바에서 설정 후 이미지를 업로드해 주세요.")
