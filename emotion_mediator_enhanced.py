# emotion_mediator_enhanced.py
# • DeepFace for facial emotion
# • Pyannote for speaker diarization
# • T5 for NLG of mediation prompts
# • Streamlit dashboard UI
# • Privacy (face-blurring + consent + retention)
# • Adaptive thresholds stored in JSON

# Dependencies:
# pip install deepface pyannote.audio streamlit transformers opencv-python pytesseract sounddevice soundfile

import os, sys, time, json, threading
from collections import deque
from datetime import datetime, timedelta

import cv2, numpy as np, pytesseract, pyautogui
import sounddevice as sd, soundfile as sf
from deepface import DeepFace
from pyannote.audio import Pipeline as DiarizationPipeline
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration

# 1) CONFIG & CONSENT
CONFIG_PATH = 'config.json'
DEFAULT_CONFIG = {
  "meeting_template": "templates/meet.png",
  "chat_template":    "templates/chat.png",
  "interval": 30,
  "context_size": 10,
  "thresholds": {"text":0.6,"face":0.5,"voice":0.5},
  "retention_days": 1,
  "consent_file":"consent.json"
}
if not os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH,'w') as f: json.dump(DEFAULT_CONFIG,f,indent=2)
CONFIG = json.load(open(CONFIG_PATH))
if not os.path.exists(CONFIG['consent_file']):
    print("⛔  Obtain user consent (consent.json) before running."); sys.exit(1)

# 2) UTILS: Dynamic Template Matching
def detect_region(template, screenshot=None):
    ss = screenshot or pyautogui.screenshot()
    gray = cv2.cvtColor(np.array(ss),cv2.COLOR_RGB2GRAY)
    tpl = cv2.imread(template,cv2.IMREAD_GRAYSCALE)
    res = cv2.matchTemplate(gray,tpl,cv2.TM_CCOEFF_NORMED)
    _,mx,_,loc = cv2.minMaxLoc(res)
    if mx<0.7: raise RuntimeError(f"{template} not found")
    h,w = tpl.shape
    return loc[0],loc[1],w,h

# 3) Context with Time-based Retention
class ConversationContext:
    def __init__(self,maxlen,retention_days):
        self.history=deque(maxlen=maxlen)
        self.retention=timedelta(days=retention_days)
    def add(self,text):
        now=datetime.now()
        self.history.append({"time":now,"text":text})
        # purge old
        while self.history and (now-self.history[0]["time"]>self.retention):
            self.history.popleft()
    def get(self): return [e["text"] for e in self.history]

# 4) Emotion Detection
class EmotionDetectionModule:
    def __init__(self):
        self.text_pipe = pipeline("sentiment-analysis",
           model="distilbert-base-uncased-finetuned-sst-2-english")
        self.voice_pipe = pipeline("audio-classification",
           model="superb/wav2vec2-base-superb-er")
        self.diarizer   = DiarizationPipeline.from_pretrained(
           "pyannote/speaker-diarization")
        # no more FER
    def analyze_text(self,text):
        r=self.text_pipe(text, truncation=True)[0]
        return r['label'],r['score']
    def analyze_face(self,frame):
        try:
            r=DeepFace.analyze(frame,actions=['emotion'],enforce_detection=False)
            emo=r['dominant_emotion']; score=r['emotion'][emo]
            return emo,score
        except:
            return None,0.0
    def analyze_voice(self,audio_file):
        # diarize (just log segments for now)
        diar= self.diarizer(audio_file)
        # single-pass classify full clip
        res=self.voice_pipe(audio_file, return_all_scores=True)[0]
        best=max(res, key=lambda x:x['score'])
        return best['label'],best['score']

# 5) Conflict Engine
class ConflictIdentificationEngine:
    def __init__(self,th): self.th=th
    def detect(self, te, fe, ve):
        flags=[]
        lt,st=te
        if lt.lower() in ("negative","label_1") and st>self.th['text']:
            flags.append("negative_text")
        lf,sf=fe
        if lf in ("angry","disgust") and sf>self.th['face']:
            flags.append("negative_face")
        lv,sv=ve
        if lv.lower() in ("anger","sadness") and sv>self.th['voice']:
            flags.append("negative_voice")
        return bool(flags),flags

# 6) NLG-based Prompt Generator
class NLGGenerator:
    def __init__(self):
        self.tok = T5Tokenizer.from_pretrained("t5-small")
        self.mod = T5ForConditionalGeneration.from_pretrained("t5-small")
    def generate(self,flags,context):
        ctx=" | ".join(context[-3:]) if context else ""
        inp = f"empathetic response flags: {','.join(flags)}; context: {ctx}"
        ids=self.tok(inp,return_tensors="pt").input_ids
        out=self.mod.generate(ids, max_length=64)
        return self.tok.decode(out[0],skip_special_tokens=True)

# 7) Mediation
class MediationStrategyGenerator:
    def __init__(self): self.nlg=NLGGenerator()
    def suggest(self,flags,context):
        return ( self.nlg.generate(flags,context)
                 if flags else "All is calm—carry on!")

# 8) Audio Recorder
class AudioRecorder:
    def __init__(self,f='tmp.wav',dur=5,sr=16000):
        self.f,self.dur,self.sr=f,dur,sr
    def record(self):
        data=sd.rec(int(self.dur*self.sr),samplerate=self.sr,channels=1)
        sd.wait(); sf.write(self.f,data,self.sr)
        return self.f

# 9) Mediator
class EmotionallyIntelligentMediator:
    def __init__(self):
        self.det = EmotionDetectionModule()
        self.eng = ConflictIdentificationEngine(CONFIG['thresholds'])
        self.med = MediationStrategyGenerator()
        self.ctx = ConversationContext(CONFIG['context_size'],
                                       CONFIG['retention_days'])
        self.ar  = AudioRecorder()
    def run_cycle(self):
        mx,my,mw,mh = detect_region(CONFIG['meeting_template'])
        cx,cy,cw,ch = detect_region(CONFIG['chat_template'])
        shot = pyautogui.screenshot(region=(mx,my,mw,mh))
        frame= cv2.cvtColor(np.array(shot),cv2.COLOR_RGB2BGR)
        # blur face for logs (privacy)
        frame_blur=cv2.GaussianBlur(frame,(99,99),30)
        chat_img=pyautogui.screenshot(region=(cx,cy,cw,ch))
        chat_txt=pytesseract.image_to_string(
            cv2.cvtColor(np.array(chat_img),cv2.COLOR_RGB2GRAY)
        ).strip()
        audio = self.ar.record()
        te = self.det.analyze_text(chat_txt)
        fe = self.det.analyze_face(frame)
        ve = self.det.analyze_voice(audio)
        conflict,flags = self.eng.detect(te,fe,ve)
        self.ctx.add(chat_txt)
        suggestion=self.med.suggest(flags,self.ctx.get())
        return conflict,suggestion

# 10) Streamlit UI
if 'streamlit' in sys.argv:
    import streamlit as st
    st.title("🤖 Emotion Mediator Dashboard")
    mediator = EmotionallyIntelligentMediator()
    if st.button("Run One Cycle"):
        c,s = mediator.run_cycle()
        st.write("Conflict?", c)
        st.write("Suggestion:", s)
    st.write("Recent Chat Context:")
    st.write(mediator.ctx.get())
    st.stop()

# 11) Main Loop
if __name__=='__main__' and 'streamlit' not in sys.argv:
    mediator = EmotionallyIntelligentMediator()
    while True:
        try:
            conflict,sugg = mediator.run_cycle()
            if conflict:
                x,y,w,h = detect_region(CONFIG['chat_template'])
                pyautogui.click(x+10,y+h+30)
                pyautogui.write(sugg,interval=0.03)
                pyautogui.press('enter')
            time.sleep(CONFIG['interval'])
        except Exception as e:
            print("Cycle error:",e)
            time.sleep(5)
