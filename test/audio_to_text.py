import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import os
import sys
import time

project_path = r"D:\JetBrains\pyproject\ai_rag_backup\MOSS-Transcribe-Diarize"
sys.path.insert(0, project_path)

print("========== 1. 开始导入 ==========")

from moss_transcribe_diarize import parse_transcript
from moss_transcribe_diarize.inference_utils import (
    build_transcription_messages,
    generate_transcription,
    resolve_device,
)

print("========== 2. 导入完成 ==========")

model_id = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
audio_path = r"D:\JetBrains\pyproject\ai_rag_backup\test\b.wav"

print("========== 3. 开始检测设备 ==========")

device = resolve_device("auto")
dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

print(f"device = {device}")
print(f"dtype = {dtype}")

print("========== 4. 开始加载 Model ==========")
start = time.time()

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    dtype="auto",
    attn_implementation="sdpa",
)

print(f"Model 加载完成，耗时 {time.time() - start:.2f}s")

print("========== 5. 开始移动 Model ==========")

model = model.to(dtype=dtype).to(device).eval()

print("========== 6. Model 已经放到设备 ==========")

print("========== 7. 开始加载 Processor ==========")

processor = AutoProcessor.from_pretrained(
    model_id,
    trust_remote_code=True
)

print("========== 8. Processor 加载完成 ==========")

print("========== 9. 开始构造 messages ==========")

messages = build_transcription_messages(audio_path)

print("messages = ")
print(messages)

print("========== 10. 开始推理 ==========")
start = time.time()

result = generate_transcription(
    model,
    processor,
    messages,
    max_new_tokens=2048,
    do_sample=False,
    device=device,
    dtype=dtype,
)

print(f"========== 11. 推理完成，耗时 {time.time() - start:.2f}s ==========")

print(result["text"])

print("========== 12. 开始解析 ==========")

for segment in parse_transcript(result["text"]):
    print(
        segment.start,
        segment.end,
        segment.speaker,
        segment.text
    )

print("========== 13. 全部完成 ==========")