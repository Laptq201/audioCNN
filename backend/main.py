import base64
import io 
import numpy as np
import torch 
import torch.nn as nn
import torchaudio.transforms as T
import librosa
import soundfile as sf

from fastapi import FastAPI, HTTPException 
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import audioCNN

app = FastAPI(title = "Audio CNN inference API",)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Hoặc ["http://localhost:3000"] nếu muốn giới hạn
    allow_credentials=True,
    allow_methods=["*"],  # Cho phép OPTIONS, POST, GET, v.v.
    allow_headers=["*"],  # Cho phép Content-Type, Authorization,...
)

class inferenceRequest(BaseModel):
    audio_data: str #base64 encoded WAV 


class audioProcessor:
    def __init__(self):
        self.transform = nn.Sequential(
            T.MelSpectrogram(
                sample_rate=22050,
                n_fft=1024,
                hop_length=512,
                n_mels=128,
                f_min=0.0,
                f_max=11025,
            ),
            T.AmplitudeToDB()
        )
    
    def process_audio_chunk(self, audio_data):
        waveform = torch.from_numpy(audio_data).float()
        waveform = waveform.unsqueeze(0)  # Add batch dimension [1, time]
        spectrogram = self.transform(waveform)
        return spectrogram.unsqueeze(0)  # Add channel dimension [1, 1, freq, time]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_path = "./models/best_model.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)
classes = checkpoint['classes']

model = audioCNN(num_classes=len(classes))
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

audio_processor = audioProcessor()


@app.post("/inference")
def inference(request: inferenceRequest):
    try:
        audio_bytes = base64.b64decode(request.audio_data)
        audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))

        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        if sample_rate != 44100:
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=44100)

        spectrogram = audio_processor.process_audio_chunk(audio_data)
        spectrogram = spectrogram.to(device)

        with torch.no_grad():
            output, feature_maps = model(spectrogram, return_feature_map=True)
            output = torch.nan_to_num(output)
            probabilities = torch.softmax(output, dim=1)
            top3_probs, top3_indices = torch.topk(probabilities[0], k=3)
        
        predictions = [
            {
                "class": classes[idx.item()],
                "probability": prob.item()
            } for idx, prob in zip(top3_indices, top3_probs)
        ]

        visualizations_data = {}
        for name, tensor in feature_maps.items():
            if tensor.dim() == 4:
                aggregated_tensor = torch.mean(tensor, dim=1)  # Aggregate over channels
                squeezed_tensor = aggregated_tensor.squeeze(0)
                numpy_array = squeezed_tensor.cpu().numpy()
                clean_array = np.nan_to_num(numpy_array)  # Clean NaN values
                visualizations_data[name] = {
                    "shape": list(clean_array.shape),
                    "values": clean_array.tolist()
                }
        
        spectrogram_np = spectrogram.squeeze(0).squeeze(0).cpu().numpy()
        clean_spectrogram = np.nan_to_num(spectrogram_np)

        max_samples = 8000
        waveform_sample_rate = 44100
        if len(audio_data) > max_samples:
            step = len(audio_data) // max_samples
            waveform_data = audio_data[::step]
        else:
            waveform_data = audio_data
        
        return {
            "predictions": predictions,
            "visualization": visualizations_data,
            "spectrogram": {
                "shape": list(clean_spectrogram.shape),
                "values": clean_spectrogram.tolist()
            },
            "waveform": {
                "values": waveform_data.tolist(),
                "sample_rate": waveform_sample_rate,
                "duration": len(audio_data) / waveform_sample_rate
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
