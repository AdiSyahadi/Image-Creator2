import streamlit as st
from diffusers import DiffusionPipeline, LCMScheduler
from PIL import Image
import time
from googletrans import Translator

translator = Translator()

# Inisialisasi pipeline dan scheduler
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

# Menambahkan judul aplikasi
st.title("Streamlit Diffusion Demo")

# Membuat tombol untuk memproses data
if st.button("Proses"):
    # Menerima input prompt dalam bahasa Indonesia
    user_prompt_id = st.text_input("Masukkan prompt untuk inferensi (atau ketik 'exit' untuk keluar): ")

    # Keluar dari loop jika pengguna memasukkan 'exit'
    if user_prompt_id.lower() == 'exit':
        st.stop()

    # Terjemahkan prompt ke bahasa Inggris
    try:
        user_prompt_en = translator.translate(user_prompt_id, src='id', dest='en').text
        st.write("Terjemahan: ", user_prompt_en)
    except Exception as e:
        st.error(f"Terjadi kesalahan saat menerjemahkan: {str(e)}")
        st.stop()

    num_inference_steps = st.slider("Masukkan jumlah langkah inferensi (num_inference_steps): ", 1, 10, 5)

    start_time = time.time()

    # Menjalankan inferensi dengan prompt yang sudah diterjemahkan
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
        st.image(results.images[0].numpy(), caption='Inferensi Result', use_column_width=True)
    else:
        st.write("Hasil inferensi tidak tersedia.")
