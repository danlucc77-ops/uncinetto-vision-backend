from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from PIL import Image
import google.genai as genai
import json
from typing import List

app = FastAPI(title="Uncinetto Vision API", version="2.0")

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
    return {"status": "online", "message": "Uncinetto Vision API Pro è attiva!"}

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

        # Prompt editoriale e sartoriale di livello professionale
        prompt = f"""
        Agisci come una maestra d'uncinetto e una redattrice di riviste di moda e artigianato d'eccellenza.
        Ti sono state fornite immagini dettagliate del capo e dei suoi schemi.
        Parametri tecnici obbligatori:
        - Tipo di capo: '{garment_type}'
        - Direzione di lavorazione: '{work_direction}'
        - Spessore filato: '{yarn_weight}'
        - Misura uncinetto: '{hook_size}'
        
        Analizza le immagini con la massima attenzione e genera un pattern estremamente dettagliato, strutturato rigorosamente con queste sezioni editoriali (stile tutorial professionale):

        1. **Introduzione al Modello**: Descrizione della forma (es. struttura a T, linearità, vestibilità).
        2. **Misure e Taglia di Riferimento**: Dimensioni indicative in centimetri (es. larghezza, lunghezza totale, altezza scalfo, lunghezza maniche).
        3. **Materiale Occorrente**: Tipo di filato consigliato, grammatura stimata e misura dell'uncinetto.
        4. **Multiplo e Campione**: La regola matematica per avviare la catenella iniziale in base al motivo visibile nelle foto.
        5. **Istruzioni Dettagliate dei Motivi**: Spiegazione passo-passo, riga per riga o giro per giro, di ogni schema di punto presente nelle immagini.
        6. **Assemblaggio e Cuciture**: Come unire i pannelli (dritto contro dritto, cucitura delle spalle, scollo, maniche).
        7. **Rifiniture**: Bordi finali.

        Restituisci un oggetto JSON valido (e nessun altro testo fuori dal JSON) con questa struttura esatta:
        {{
          "nome_punto": "Nome tecnico della trama principale",
          "istruzioni_testuali": "IL TESTO EDITORIALE COMPLETO E DETTAGLIATO CON TUTTE LE SEZIONI SOPRA ELENCATE, FORMATTATO IN MARKDOWN..."
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
            "pattern": data_json["istruzioni_testuali"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
