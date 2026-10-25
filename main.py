from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from PIL import Image
import google.genai as genai
import json
import io
import base64
from typing import List

app = FastAPI(title="Uncinetto Vision AI Native", version="5.0")

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
    return {"status": "online", "message": "Uncinetto Vision API AI-Native è attiva!"}

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

        # Prompt progettato per richiedere un'analisi sartoriale profonda e la descrizione/struttura dello schema grafico
        prompt = f"""
        Agisci come una maestra d'uncinetto e una redattrice di riviste di moda d'eccellenza.
        Parametri tecnici obbligatori:
        - Tipo di capo: '{garment_type}'
        - Direzione di lavorazione: '{work_direction}'
        - Spessore filato: '{yarn_weight}'
        - Misura uncinetto: '{hook_size}'
        
        Analizza le immagini con la massima attenzione e restituisci un oggetto JSON valido (e nessun altro testo fuori dal JSON) con questa struttura esatta:
        {{
          "nome_punto": "Nome tecnico della trama (es. Punto Ananas e Ventagli)",
          "istruzioni_testuali": "GUIDA EDITORIALE COMPLETA IN MARKDOWN (Introduzione, Misure in cm, Materiali, Multiplo, Istruzioni riga per riga, Assemblaggio, Rifiniture)...",
          "descrizione_schema_grafico": "Descrizione dettagliata riga per riga dello schema grafico tradizionale (simboli, catenelle, archetti, ventagli) da mostrare all'utente."
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

        # Generazione dell'immagine dello schema grafico tramite le capacità visive/creative dell'IA
        chart_prompt = f"""
        Crea un'immagine grafica professionale e pulita di uno schema all'uncinetto in stile rivista (tipo ManiFantasia), 
        rappresentando il modulo del punto per il capo '{garment_type}' con i simboli tradizionali dell'uncinetto 
        (catenelle 'o', maglie alte '|', ventagli e archetti) disposti su una griglia chiara a sfondo bianco con righe numerate R1, R2, R3...
        Il disegno deve essere ordinato, tecnico, nitido e altamente leggibile, senza scritte confuse.
        """

        image_response = client.models.generate_images(
            model='imagen-3.0-generate-002',
            prompt=chart_prompt,
            config=dict(
                number_of_images=1,
                output_mime_type="image/png",
                aspect_ratio="4:3"
            )
        )

        chart_base64 = ""
        for generated_image in image_response.generated_images:
            chart_base64 = base64.b64encode(generated_image.image.image_bytes).decode('utf-8')
            break

        return {
            "status": "success",
            "nome_punto": data_json.get("nome_punto", "Schema Uncinetto"),
            "pattern": data_json["istruzioni_testuali"],
            "schema_image": chart_base64
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
