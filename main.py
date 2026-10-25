from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from PIL import Image
import google.genai as genai
import json
from typing import List

app = FastAPI(title="Uncinetto Vision API SVG", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client()

@app.get("/")
def home():
    return {"status": "online", "message": "Uncinetto Vision API SVG è attiva!"}

@app.post("/generate-pattern")
async def generate_pattern(
    files: List[UploadFile] = File(...),
    garment_type: str = Form(default="Maglione"),
    work_direction: str = Form(default="In righe (avanti e indietro)"),
    yarn_weight: str = Form(default="DK (Medio)"),
    hook_size: str = Form(default="4.0 mm")
):
    try:
        pil_images = []
        for file in files:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is not None:
                max_dimension = 1024
                h, w = img.shape[:2]
                if max(h, w) > max_dimension:
                    scaling_factor = max_dimension / float(max(h, w))
                    img = cv2.resize(img, (int(w * scaling_factor), int(h * scaling_factor)))
                
                image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_images.append(Image.fromarray(image_rgb))

        if not pil_images:
            raise HTTPException(status_code=400, detail="Nessuna immagine valida caricata.")

        # Prompt avanzato per istruzioni sartoriali e codice SVG dello schema grafico
        prompt = f"""
        Agisci come una maestra d'uncinetto e una redattrice di riviste di moda d'eccellenza.
        Parametri tecnici:
        - Tipo di capo: '{garment_type}'
        - Direzione di lavorazione: '{work_direction}'
        - Spessore filato: '{yarn_weight}'
        - Misura uncinetto: '{hook_size}'
        
        Analizza le immagini e genera:
        1. Una guida sartoriale dettagliata (Introduzione, Misure in cm, Materiali, Multiplo, Istruzioni riga per riga, Assemblaggio, Rifiniture) formattata in Markdown.
        2. Un codice SVG valido e completo (inizia con <svg ...> e finisce con </svg>) che rappresenti lo schema grafico vettoriale del punto visibile nelle foto (con griglia di sfondo, simboli di catenelle 'o', maglie alte '|' e ventagli 'V', disposti su righe numerate R1, R2, ecc.).
        
        Restituisci un oggetto JSON valido (e nessun altro testo fuori dal JSON) con questa struttura esatta:
        {{
          "nome_punto": "Nome tecnico della trama",
          "istruzioni_testuali": "GUIDA EDITORIALE COMPLETA IN MARKDOWN...",
          "schema_svg": "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 800 600' width='100%' height='100%'><rect width='100%' height='100%' fill='white'/><!-- Inserisci qui tracciati, cerchi, linee e testi per lo schema grafico uncinetto --></svg>"
        }}
        """

        contents_payload = pil_images + [prompt]
        response = client.models.generate_content(
            model='gemini-3.8-flash',
            contents=contents_payload
        )

        raw_text = response.text.strip()
        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_text:
            raw_text = raw_text.split("```")[1].split("```")[0].strip()

        data_json = json.loads(raw_text)

        return {
            "status": "success",
            "nome_punto": data_json.get("nome_punto", "Schema Uncinetto"),
            "pattern": data_json["istruzioni_testuali"],
            "schema_svg": data_json.get("schema_svg", "")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
