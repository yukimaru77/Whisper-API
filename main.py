from fastapi import FastAPI, File, UploadFile
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import os

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=32,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

def pad_number(number,pad):
    return str(number).zfill(pad)

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    # 音声ファイルを一時ファイルに保存
    with open(file.filename, "wb") as buffer:
        buffer.write(await file.read())

    # 音声ファイルをWhisperのpipelineに渡して文字起こしを行う
    result = pipe(file.filename, return_timestamps=False,generate_kwargs={"language": "Japanese", "task": "transcribe"})
    # 一時ファイルを削除
    os.remove(file.filename)

    return {"transcription": result["text"]}


