import base64
import requests
import io
import soundfile as sf

# Read audio file and encode base64
audio_data, sr = sf.read("./cat.wav")
buffer = io.BytesIO()
sf.write(buffer, audio_data, sr, format="WAV")
audio_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

payload = {"audio_data": audio_b64}
response = requests.post("http://localhost:8000/inference", json=payload)
print(response.json())

