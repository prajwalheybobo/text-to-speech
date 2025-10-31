from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
from boson_multimodal.data_types import ChatMLSample, Message
import torch
import torchaudio

with open("examples/voice_prompts/belinda.txt", "r") as f:
    belinda_text = f.read()

# ✅ Local paths
MODEL_PATH = "/mnt/data/home/shared/models/higgs-audio-v2-3B/model"
AUDIO_TOKENIZER_PATH = "/mnt/data/home/shared/models/higgs-audio-v2-3B/audio_tokenizer"

# ✅ Optional: customize system prompt
system_prompt = (
    "Generate high-quality speech audio from text.\n"
    "<|scene_desc_start|>\n"
    "The audio should sound clear and natural, recorded in a quiet studio.\n"
    "<|scene_desc_end|>"
)

# ✅ Text input
messages = [
    Message(role="system", content=system_prompt),
    Message(role="user", content=belinda_text),
]


device = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ Initialize inference engine
serve_engine = HiggsAudioServeEngine(MODEL_PATH, AUDIO_TOKENIZER_PATH,
    device=device
)

# ✅ Generate output audio
output: HiggsAudioResponse = serve_engine.generate(
    chat_ml_sample=ChatMLSample(messages=messages),
    max_new_tokens=2048,
    temperature=0.3,
    top_p=0.95,
    top_k=50,
    stop_strings=["<|end_of_text|>", "<|eot_id|>"]
)

# ✅ Save to WAV
torchaudio.save("output2.wav", torch.from_numpy(output.audio)[None, :], output.sampling_rate)
print(f"✅ Audio saved: output2.wav ({output.sampling_rate} Hz)")

# ✅ (Optional) Convert to MP3 if pydub + ffmpeg installed
try:
    from pydub import AudioSegment
    AudioSegment.from_wav("output2.wav").export("output2.mp3", format="mp3")
    print("🎧 MP3 saved as output2.mp3")
except Exception:
    print("⚠️ MP3 conversion skipped (install `pydub` + `ffmpeg` to enable)")