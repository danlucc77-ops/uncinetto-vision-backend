from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from PIL import Image
import google.genai as genai
import os

app = FastAPI(title="Uncinetto Vision API", version="1.0")

# Configurazione CORS per permettere le chiamate da Flutter Web e dispositivi
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permette le richieste da qualsiasi origine
    allow_credentials=True,
    allow_methods=["*"],  # Permette tutti i metodi (GET, POST, ecc.)
    allow_headers=["*"],  # Permette tutti gli header
)

# Inizializza il client GenAI usando la variabile d'ambiente di Render
client = genai.Client()

@app.get("/")
def home():
    return {"status": "online", "message": "Uncinetto Vision API è attiva!"}

@app.post("/generate-pattern")
async def generate_pattern(file: UploadFile = File(...)):
    try:
        # Legge i byte dell'immagine inviata dall'app Flutter
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Immagine non valida o illeggibile.")

        # Pre-elaborazione: ridimensionamento per ottimizzare l'invio all'AI
        max_dimension = 1024
        h, w = img.shape[:2]
        if max(h, w) > max_dimension:
            scaling_factor = max_dimension / float(max(h, w))
            img = cv2.resize(img, (int(w * scaling_factor), int(h * scaling_factor)))
        
        # Conversione in formato PIL RGB
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)

        # Prompt di sistema per la maestra d'uncinetto digitale
        prompt = """
        Agisci come una designer esperta di uncinetto e una maestra artigiana. 
        Analizza attentamente l'immagine fornita che mostra un capo o un dettaglio realizzato all'uncinetto.
        Il tuo compito è generare un pattern (schema) preciso e riproducibile per realizzarlo.
        
        Restituisci l'output strutturato rigorosamente in queste sezioni:
        1. **Nome del punto o della trama identificata**
        2. **Materiali stimati Consigliati** (Tipo di filato e misura dell'uncinetto)
        3. **Glossario dei punti utilizzati** (Abbreviazioni e spiegazione)
        4. **Costruzione e Istruzioni passo-passo** (Divise in righe o giri)
        5. **Consigli speciali**
        """

        # Chiamata al modello multimodale
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[pil_img, prompt]
        )

        return {
            "status": "success",
            "pattern": response.text
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
