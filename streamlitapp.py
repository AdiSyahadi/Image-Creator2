import streamlit as st
import numpy as np
from diffusers import DiffusionPipeline, LCMScheduler
from PIL import Image
import time
from googletrans import Translator

translator = Translator()

pipe = DiffusionPipeline.from_pretrained("Lykon/dreamshaper-7")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")

def disabled_safety_checker(images, clip_input):
    if len(images.shape) == 4:
        num_images = images.shape[0]
        return images, [False]*num_images
    else:
        return images, False
pipe.safety_checker = disabled_safety_checker

st.title("Streamlit Diffusion Demo")

# Input prompt
user_prompt_id = st.text_input("Masukkan prompt untuk inferensi (atau ketik 'exit' untuk keluar): ")

# Input jumlah langkah inferensi
num_inference_steps = st.slider("Masukkan jumlah langkah inferensi (num_inference_steps): ", 1, 10, 5)

# Tombol untuk memulai inferensi
if st.button("Proses Inferensi"):
    if user_prompt_id.lower() == 'exit':
        st.write("Inferensi dihentikan.")
    else:
        try:
            user_prompt_en = translator.translate(user_prompt_id, src='id', dest='en').text
        except Exception as e:
            st.error("Terjadi kesalahan saat menerjemahkan. Silakan coba lagi.")
            user_prompt_en = ""  # Memberikan nilai default jika terjemahan gagal
        st.write("Terjemahan: ", user_prompt_en)

        start_time = time.time()

        results = pipe(
            prompt=user_prompt_en,
            num_inference_steps=num_inference_steps,
            guidance_scale=0.3,
            nsfw=False
        )

        end_time = time.time()

        latency_seconds = end_time - start_time
        latency_minutes = int(latency_seconds // 60)
        remaining_seconds = latency_seconds % 60

        st.write(f"Latensi: {latency_minutes} menit {remaining_seconds:.2f} detik")

        # Check if results is not None before attempting to display the image
        if results is not None and results.images is not None and len(results.images) > 0:
            # Convert Image to NumPy array
            image_array = np.array(results.images[0])
            st.image(image_array, caption='Inferensi Result', use_column_width=True)
        else:
            st.write("Hasil inferensi tidak tersedia.")
