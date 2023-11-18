from diffusers import DiffusionPipeline, LCMScheduler
from PIL import Image
import time

pipe = DiffusionPipeline.from_pretrained("Lykon/dreamshaper-7")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")  # ya, ini LoRA biasa

while True:
    # Meminta pengguna untuk memasukkan prompt
    print("")
    user_prompt = input("Masukkan prompt untuk inferensi (atau ketik 'exit' untuk keluar): ")
    
    # Keluar dari loop jika pengguna memasukkan 'exit'
    if user_prompt.lower() == 'exit':
        break

    # Mengukur waktu awal
    start_time = time.time()

    # Menjalankan inferensi dengan prompt yang dimasukkan pengguna
    results = pipe(
        prompt=user_prompt,
        num_inference_steps=4,
        guidance_scale=0.3,
        nsfw=False
    )

    # Mengukur waktu akhir
    end_time = time.time()

    # Menghitung latensi dalam detik
    latency_seconds = end_time - start_time

    # Konversi latensi ke menit dan detik
    latency_minutes = int(latency_seconds // 60)
    remaining_seconds = latency_seconds % 60

    print(f"Latensi: {latency_minutes} menit {remaining_seconds:.2f} detik")

    # Menampilkan gambar di jendela baru
    results.images[0].show()
