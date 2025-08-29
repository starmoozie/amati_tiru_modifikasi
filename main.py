import os
import re
import requests
from datetime import datetime
from urllib.parse import urlparse, parse_qs

import whisper
from langdetect import detect
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip
from moviepy.video.VideoClip import TextClip
import textwrap
import yt_dlp
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Inisialisasi Groq
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# =============== CONFIG ===============
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
TEMP_DIR = "temp"
OUTPUT_DIR = "output"
FONT_PATH = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============== 1. Ekstrak Video ID dari URL ===============
def get_video_id(url):
    # Bisa ekstrak ID dari berbagai platform
    if "tiktok.com" in url:
        # Ekstrak dari URL TikTok
        return re.findall(r'/video/(\d+)', url)[0]
    elif "youtube.com" in url:
        return parse_qs(urlparse(url).query).get("v", [None])[0]
    else:
        return "unknown"


# =============== 2. Download Video dari YouTube ===============
def download_yt_video(url):
    print(f"⏬ Downloading from: {url}")
    
    # Ekstrak ID untuk nama file
    video_id = get_video_id(url)
    filename = os.path.join(TEMP_DIR, f"video_{video_id}.mp4")

    ydl_opts = {
        'format': 'best',
        'outtmpl': filename,
        'noplaylist': True,
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15'
        },
        'retries': 3,
        'fragment_retries': 5,
        'socket_timeout': 15,
        'quiet': False,
        'no_warnings': False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([url])
            return filename
        except Exception as e:
            raise Exception(f"Download failed ({url}): {str(e)}")


# =============== 3. Transkrip Audio pakai Whisper ===============
def transcribe_audio(video_file):
    print("🎙️ Transcribing audio...")
    model = whisper.load_model("base")  # gunakan "small" untuk akurasi lebih
    result = model.transcribe(video_file)
    return result["text"]


# =============== 4. Deteksi Bahasa Otomatis ===============
def detect_language(text):
    try:
        lang = detect(text)
        return "id" if lang == "id" else "en"
    except:
        return "en"  # fallback ke English


# =============== 5. Generate Narasi Baru (dengan Groq + Llama 3.1) ===============
def generate_script(transcript, lang):
    print(f"✍️ Generating script in {'Indonesian' if lang == 'id' else 'English'}...")
    
    prompts = {
        "id": """
        Buat narasi berita viral gaya TikTok/YouTube Shorts, durasi 30 detik.
        Struktur:
        1. Hook: kalimat mengejutkan
        2. Fakta: siapa, kapan, di mana
        3. Detail: bagaimana kejadiannya
        4. Penutup: pesan emosional
        Gunakan bahasa Indonesia yang natural, maksimal 150 kata. Jangan gunakan markdown.
        """,
        "en": """
        Create a viral news script for TikTok/YouTube Shorts, 30 seconds.
        Structure:
        1. Hook: shocking opening
        2. Facts: who, when, where
        3. Details: what happened
        4. Closing: emotional message
        Use natural English, max 150 words. No markdown.
        """
    }

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # Model terbaik di Groq
        messages=[
            {"role": "system", "content": prompts[lang]},
            {"role": "user", "content": transcript}
        ],
        max_tokens=300,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()


# =============== 6. Voice-over (ElevenLabs) ===============
def text_to_speech(text, lang, output_file):
    print("🔊 Generating voice-over...")
    # Pilih suara berdasarkan bahasa
    voice_id = "zYcjlYFOd3taleS0gkk3"  # English - "Bella"
    if lang == "id":
        voice_id = "3mAVBNEqop5UbHtD8oxQ"  # Indonesian - "Galih"

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    payload = {
        "text": text,
        "voice_settings": {"stability": 0.7, "similarity_boost": 0.8},
        "model_id": "eleven_multilingual_v2"
    }
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code != 200:
        raise Exception(f"ElevenLabs error: {response.text}")
    with open(output_file, "wb") as f:
        f.write(response.content)
    return output_file


# =============== 7. Cari Video dari Pexels ===============
def get_pexels_video(keywords, orientation="portrait", clip_duration=4, max_clips=4):
    print(f"🎥 Searching Pexels for multiple clips: '{keywords}'")
    url = "https://api.pexels.com/videos/search"
    headers = {"Authorization": PEXELS_API_KEY}
    params = {
        "query": f"{keywords} dramatic",
        "per_page": 10,
        "min_width": 1080,
        "min_height": 1920,
        "orientation": "portrait"
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            print(f"❌ Pexels API error: {response.status_code}")
            return None

        videos = response.json().get("videos", [])
        if not videos:
            return None

        clips = []
        for i, vid in enumerate(videos[:max_clips]):
            try:
                # Cari video 1080x1920
                video_url = None
                for file in vid["video_files"]:
                    if file["width"] == 1080 and file["height"] == 1920:
                        video_url = file["link"]
                        break
                if not video_url:
                    continue

                # Download
                video_path = os.path.join(TEMP_DIR, f"clip_{i}.mp4")
                print(f"📥 Downloading clip {i+1}...")
                with open(video_path, "wb") as f:
                    f.write(requests.get(video_url).content)

                # Potong & tambahkan ke klip
                clip = VideoFileClip(video_path).subclip(0, clip_duration).resize((1080, 1920))
                clips.append(clip)

            except Exception as e:
                print(f"⚠️ Failed to process video {i}: {str(e)}")
                continue

        return clips if clips else None

    except Exception as e:
        print(f"❌ Pexels request failed: {str(e)}")
        return None


# =============== 8. Ekstrak Kata Kunci dari Narasi ===============
def extract_keywords(script):
    print("🔍 Extracting keywords with Groq LLM...")
    prompt = (
        "Extract only 5-10 SINGLE English words representing visual elements from this news script. "
        "No explanation, no numbers, no bullet points. "
        "Only return words separated by commas. "
        "Avoid violent or restricted terms."
        "DO NOT USE INDONESIAN LANGUAGE, ONLY USING ENGLISH LANGUAGE. DO NOT USE NUMBERS OR PUNCTUATION."
        "Example: smoke, fire, night"
    )

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": script}
            ],
            max_tokens=100
        )
        raw = response.choices[0].message.content.strip()
        print(f"🔧 Raw LLM output: {raw}")

        # ✅ Bersihkan: hapus angka, titik, prompt
        cleaned = re.sub(r'\d+[\.\)\]\s]*', '', raw)  # hapus "1 ", "2)", dll
        cleaned = re.sub(r'[^a-zA-Z\s,]', '', cleaned)  # hapus simbol
        words = [w.strip() for w in re.split(r'[,\s]+', cleaned) if w.strip()]
        words = [w.lower() for w in words if len(w) > 3]  # minimal 4 huruf

        # Batasi 3 kata
        final_keywords = ", ".join(set(words[:5]))
        print(f"✅ Cleaned keywords: {final_keywords}")
        return final_keywords
    except Exception as e:
        print(f"⚠️ Groq keyword extraction failed: {str(e)}. Using fallback.")
        # Fallback ke regex jika gagal
        words = re.findall(r'\b\w+\b', script.lower())
        ignore = {'dan', 'di', 'ke', 'dari', 'yang', 'untuk', 'ada', 'ini', 'itu', 'ia', 'dia', 'akan', 'dengan', 
                  'pada', 'dalam', 'the', 'and', 'in', 'to', 'for', 'a', 'of', 'on', 'is', 'are'}
        filtered = [w for w in words if len(w) > 5 and w not in ignore]
        return "breaking news" if not filtered else " ".join(set(filtered[:3]))


def create_caption_clips(script, duration):
    """
    Buat animasi teks besar (caption) dari narasi
    """
    print("📝 Creating captions...")
    
    # Pisah teks jadi baris pendek
    wrapped = textwrap.fill(script, width=30)  # max 30 karakter per baris
    lines = wrapped.split('\n')
    
    # Total durasi teks
    total_duration = duration
    line_duration = total_duration / len(lines) if lines else 1

    clips = []
    for i, line in enumerate(lines):
        txt_clip = TextClip(
            line.upper(),  # teks besar
            fontsize=60,
            font=FONT_PATH,
            color="yellow",
            stroke_color="black",
            stroke_width=3,
            size=(1080, 1920),
            method='caption',
            align='center'
        ).set_position(('center', 1400)) \
         .set_duration(line_duration) \
         .set_start(i * line_duration)

        clips.append(txt_clip)

    return clips

# =============== 9. Gabung Jadi Video Final ===============
def create_final_video(voiceover_file, bg_video_file, output_file, script):
    print("🎬 Rendering final video...")
    audio = AudioFileClip(voiceover_file)
    duration = audio.duration

    # Cek apakah input adalah list (montase) atau file
    if isinstance(bg_video_file, list):
        print("🧩 Creating montage from multiple clips...")
        final_clip = concatenate_videoclips(bg_video_file, method="compose")
        if final_clip.duration >= duration:
            bg = final_clip.subclip(0, duration)
        else:
            bg = final_clip.loop(duration=duration)
    elif bg_video_file and os.path.exists(bg_video_file):
        bg = VideoFileClip(bg_video_file)
        if bg.duration >= duration:
            bg = bg.subclip(0, duration)
        else:
            bg = bg.loop(duration=duration)
    else:
        print("⚠️ Using fallback video")
        fallback = "stock_news_vertical.mp4"
        if not os.path.exists(fallback):
            raise Exception("Fallback video 'stock_news_vertical.mp4' not found")
        bg = VideoFileClip(fallback)
        bg = bg.loop(duration=duration) if bg.duration < duration else bg.subclip(0, duration)

    # === Caption ===
    caption_clips = create_caption_clips(script, duration)

    final = CompositeVideoClip([bg] + caption_clips)
    final = final.set_audio(audio)

    final.write_videofile(
        output_file,
        fps=24,
        codec="libx264",
        audio_codec="aac",
        threads=4,
        preset="ultrafast"
    )
    print(f"✅ Video selesai: {output_file}")


# =============== 10. MAIN PROCESS ===============
def process_video(url):
    try:
        video_id = get_video_id(url)
        print(f"\n🚀 Processing video: {video_id}")

        # 1. Download
        video_file = download_yt_video(url)

        # 2. Transkrip
        transcript = transcribe_audio(video_file)

        # 3. Deteksi bahasa
        lang = detect_language(transcript)
        print(f"🌐 Detected language: {'Indonesia' if lang == 'id' else 'English'}")

        # 4. Generate narasi baru
        new_script = generate_script(transcript, lang)
        print("📜 Generated script:", new_script[:100] + "...")

        # 5. Voice-over
        voiceover_file = os.path.join(TEMP_DIR, "voiceover.mp3")
        text_to_speech(new_script, lang, voiceover_file)

        # 6. Cari video latar
        keywords = extract_keywords(new_script)
        bg_video = get_pexels_video(keywords, max_clips=4)
        if not bg_video:
            print("⚠️ Pexels video not found, using fallback")
            bg_video = "stock_news_vertical.mp4"  # pastikan file ini ada

        # 7. Render video
        output_name = f"viral_{video_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_name)
        create_final_video(voiceover_file, bg_video, output_path, new_script)

        # 8. Cleanup (opsional)
        os.remove(video_file); os.remove(voiceover_file)

    except Exception as e:
        print(f"❌ Error processing {url}: {str(e)}")


# =============== 11. Baca dari File ===============
if __name__ == "__main__":
    if not os.path.exists("input_urls.txt"):
        exit()

    with open("input_urls.txt", "r") as f:
        urls = [line.strip() for line in f if line.strip() and line.strip().startswith('http')]

    print(f"🔍 Ditemukan {len(urls)} URL. Memulai proses...")

    for url in urls:
        process_video(url)

    print("🎉 SEMUA VIDEO SELESAI!")