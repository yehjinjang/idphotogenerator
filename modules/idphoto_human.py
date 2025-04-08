import torch
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
from diffusers import StableDiffusionInpaintPipeline
from safetensors.torch import load_file

# 모델 & LoRA 로드
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16,
    use_safetensors=False
).to("cuda")


lora_state_dict = load_file("./models/lora/identification-photo_v3.0.safetensors")
pipe.load_lora_weights(lora_state_dict) 
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()
pipe.safety_checker = None
def detect_face_and_mask(uploaded_image):
    # UploadedFile → PIL 이미지
    pil_image = Image.open(uploaded_image).convert('RGB')

    # PIL → NumPy (OpenCV가 이해할 수 있게 BGR 포맷으로 변환)
    image = np.array(pil_image)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Face Mesh 모델 준비
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

    # 얼굴 인식 수행
    results = face_mesh.process(image_bgr)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 얼굴 랜드마크 좌표 수집
            all_points = [
                (int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0]))
                for landmark in face_landmarks.landmark
            ]
            all_points_np = np.array([all_points], dtype=np.int32)

            # 마스크 생성
            cv2.fillPoly(mask, all_points_np, 255)

            # 마스크 확장 (디테일 보완)
            kernel = np.ones((40, 40), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)

            # 얼굴 중심 및 박스 영역 계산
            x_coords, y_coords = zip(*all_points)
            center_x, center_y = int(np.mean(x_coords)), int(np.mean(y_coords))
            box_size = int(max(max(x_coords) - min(x_coords), max(y_coords) - min(y_coords)) * 2.5)

            x_min = max(center_x - box_size // 2, 0)
            x_max = min(center_x + box_size // 2, image.shape[1])
            y_min = max(center_y - box_size // 2, 0)
            y_max = min(center_y + box_size // 2, image.shape[0])

            # 이미지 및 마스크 크롭
            cropped_image = image[y_min:y_max, x_min:x_max]  # 여기는 BGR 유지
            cropped_mask = mask[y_min:y_max, x_min:x_max]
            inverse_mask = cv2.bitwise_not(cropped_mask)

            return cropped_image, inverse_mask

    # 얼굴 인식 실패 시
    raise ValueError("얼굴을 찾지 못했습니다.")



def generate_human_id_photo(uploaded_image, gender, background_color, face_ratio, upscale):

    cropped_image, inverse_mask = detect_face_and_mask(uploaded_image)

    # NumPy(BGR) → PIL(RGB) 변환
    # original_pil = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    original_pil = Image.fromarray(cropped_image)
    
    mask_rgb = cv2.cvtColor(inverse_mask, cv2.COLOR_GRAY2RGB)
    mask_pil = Image.fromarray(mask_rgb)
    

    # 배경 색상 프롬프트 설정
    background_prompt = {"흰색": "clean white background", "하늘색": "light blue background", "회색": "light gray background"}.get(background_color, "clean white background")

    # 성별에 따른 의상 및 헤어스타일 설정
    if gender == "여자":
        clothes = "woman wearing black blazer and white blouse, formal attire, visible collar, neat appearance"
        hair = "neatly woman hair, professional hairstyle, tidy and smooth hair"
    else:
        clothes = "man wearing black blazer, white shirt, and black tie, formal attire, visible collar, neat appearance"
        hair = "short neat man hair, professional hairstyle"

    prompt = (
        f"professional id photo, passport photo, {background_prompt}, "
        f"{clothes}, visible shoulders, upper body emphasized, {hair}, centered face, "
        f"neutral expression, clean studio lighting, high quality, sharp focus, "
        f"formal attire, professional look, corporate style, portrait studio photo, clean details, well-dressed"
    )

    # 모델 실행
    result = pipe(
        prompt=prompt,
        image=original_pil,
        mask_image=mask_pil,
        guidance_scale=10.5,
        num_inference_steps=70,
        generator=torch.manual_seed(42)
    ).images[0].convert("RGBA")

    # 흰 배경을 투명으로 처리
    datas = result.getdata()
    newData = [(255, 255, 255, 0) if item[:3] == (255, 255, 255) else item for item in datas]
    result.putdata(newData)

    # 업스케일 처리
    scale_factor = {"원본": 1, "2배": 2, "4배": 4}.get(upscale, 1)
    if scale_factor > 1:
        new_size = (result.width * scale_factor, result.height * scale_factor)
        result = result.resize(new_size, Image.Resampling.LANCZOS)

    output_path = "./final_result.png"
    result.save(output_path, dpi=(300, 300))

    return result
