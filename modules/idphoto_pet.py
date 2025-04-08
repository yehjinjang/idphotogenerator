import openai
from PIL import Image
from io import BytesIO
import base64
import requests
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def get_dalle_prompt_from_image(image: Image.Image, frame_style: str, bg_color: str):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    base64_image = base64.b64encode(img_bytes).decode("utf-8")
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional photo studio designer and prompt engineer who specializes in pet portraits. "
                        "Your task is to write a highly detailed, vivid, and photorealistic DALL·E 3 prompt based on the image provided by the user. "
                        "The goal is to create a studio-quality ID photo that preserves the pet's original face while enhancing the background and adding a specific frame style."
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"Generate a highly detailed and photorealistic studio portrait of the pet in the uploaded image. "
                                f"The result should be indistinguishable from a real studio photograph taken with professional lighting and equipment. "
                                f"Use a solid '{bg_color}' background that resembles a clean studio backdrop, and frame the portrait with a clearly visible and realistic '{frame_style}' frame with depth, shadow, and texture.\n\n"

                                f"Accurately describe the pet's appearance using pixel-level detail:\n"
                                f"- Fur color, texture, and length (short, medium, long, or fluffy), matching the photo exactly\n"
                                f"- Fur pattern (e.g., stripes, spots, tuxedo, or solid)\n"
                                f"- Facial features and expression\n"
                                f"- Eye color and the emotion it conveys\n"
                                f"- Ear shape and position (e.g., upright, floppy, alert)\n"
                                f"- Nose color and texture (e.g., black, pink, shiny, moist)\n"
                                f"- Mouth shape or visible tongue (if present)\n"
                                f"- Overall body shape or build (e.g., slim, muscular, chubby)\n"
                                f"- Body position or pose\n"
                                f"- Paws visibility and placement\n"
                                f"- Gaze direction (e.g., looking directly at the camera or slightly to the side)\n"
                                f"- Optional accessories (like a bow, collar, or bandana), only if they are visible\n\n"

                                f"Preserve the pet’s original head and body orientation from the uploaded photo — do not reorient the pet to face forward or stylize the pose. "
                                f"The pet’s face, fur color, and nose color must remain exactly as shown — do not reinterpret or invent these features. "
                                f"Use pixel-level accuracy when describing texture and color.\n\n"

                                f"The final image should look like an enhanced, cinematic version of the original photo — not a reimagining. "
                                f"It must be photo-realistic, emotionally expressive, and professionally composed. "
                                f"Avoid any cartoonish, painted, stylized, or illustrated appearances.\n\n"

                                f"To ensure maximum likeness, replicate the pet's face structure, eye spacing, muzzle size, and overall facial proportions exactly as shown in the image. Maintain the original expression, including any visible smile or tongue position. The portrait should closely match the pet's facial geometry, not just general features.\n"
                                f"Also preserve the camera angle from the photo, including the vertical perspective — do not rotate, zoom in, or change the viewpoint. The pet's face must appear in the same relative orientation, angle, and distance as in the original photo.\n"

                                f"Ensure the pet is fully contained within the frame — no part of its body should extend beyond it. "
                                f"The pet should occupy approximately 80% of the frame space, leaving clear, balanced margins around it. "
                                f"The frame must visibly and cleanly enclose the portrait like a physical photo frame."
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=600
        )
        return response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"프롬프트 생성 오류: {e}")

def generate_image_from_prompt(prompt: str):
    try:
        response = openai.Image.create(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            response_format="url",
            n=1
        )
        image_url = response['data'][0]['url']
        response = requests.get(image_url)
        return Image.open(BytesIO(response.content))
    except Exception as e:
        raise RuntimeError(f"DALL·E 생성 오류: {e}")

def generate_pet_id_photo(uploaded_file, frame_style, background_color):
    img = Image.open(uploaded_file)
    dalle_prompt = get_dalle_prompt_from_image(img, frame_style, background_color)
    final_image = generate_image_from_prompt(dalle_prompt)
    return final_image

