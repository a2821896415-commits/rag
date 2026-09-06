from kokoro import KPipeline
from IPython.display import display, Audio
import soundfile as sf
import torch
import numpy as np
# from pydub import AudioSegment
pipeline = KPipeline(lang_code='z')
text = open('../source/doc/三体简介.md', 'r', encoding='utf-8')
# text='''
# 当年明月以通俗讲史写作受到广泛关注，其代表作《明朝那些事儿》把历史人物和制度变化写得更具故事性与可读性，语言接近普通读者，叙事推进较快。
# '''
generator = pipeline(text, voice='zf_xiaobei')

all_audio = []
for i, (gs, ps, audio) in enumerate(generator):
    print(i, gs, ps)
    all_audio.append(audio)
    # display(Audio(data=audio, rate=24000, autoplay=i==0))
    # sf.write(f'a.wav', audio,24000)

full_audio = np.concatenate(all_audio)
sf.write('aa.flac', full_audio, 24000, format='FLAC')

# audio = AudioSegment.from_wav("temp.wav")
# audio.export("b.mp3", format="mp3", bitrate="64k")

