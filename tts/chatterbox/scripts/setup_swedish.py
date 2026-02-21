import torchaudio as ta
import torch
from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")


svensk_text = "Hej! Jag heter Chatterbox, och jag kan tala svenska med en naturlig och tydlig röst."

# Use an audio file as a reference for how you want your voice to sound.

wav_svensk = multilingual_model.generate(svensk_text,
                                         language_id="sv"
                                         ,exaggeration = 0.5
                                         ,temperature=0.8
                                         ,cfg_weight=0.5)
ta.save("test-svensk.wav", wav_svensk, multilingual_model.sr)



# English example
model = ChatterboxTTS.from_pretrained(device="cuda")

text = "Hej! Jag heter Chatterbox, och jag kan tala svenska med en naturlig och tydlig röst."
wav = model.generate(text)
ta.save("test-english.wav", wav, model.sr)
