from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import google.genai as genai
import json
import os

app = FastAPI(title="Uncinetto Vision API", version="1.0")

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
    return {"status": "online", "message": "Uncinetto Vision API è attiva!"}

@app.post("/generate-pattern")
async def generate_pattern(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Immagine non valida.")

        max_dimension = 1024
        h, w = img.shape[:2]
        if max(h, w) > max_dimension:
            scaling_factor = max_dimension / float(max(h, w))
            img = cv2.resize(img, (int(w * scaling_factor), int(h * scaling_factor)))
        
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)

        # Prompt avanzato per ottenere istruzioni e una rappresentazione a griglia
        prompt = """
        Agisci come una maestra d'uncinetto e un software di pattern design.
        Analizza l'immagine e restituisci un oggetto JSON valido (e nessun altro testo fuori dal JSON) con questa struttura esatta:
        {
          "nome_punto": "Nome della trama",
          "istruzioni_testuali": "Istruzioni passo-passo riga per riga...",
          "righe_schema": [
            {"riga": 1, "simboli": ["cat", "cat", "cat", "m_alta", "m_alta"]},
            {"riga": 2, "simboli": ["cat", "m_alta", "ventaglio", "m_alta"]}
          ]
        }
        Dove i simboli ammessi sono: 'cat' (catenella), 'm_alta' (maglia alta), 'm_bassa' (maglia bassa), 'ventaglio'.
        """

        response = client.models.generate_content(
            model='gemini-3.8-flash',
            contents=[pil_img, prompt]
        )

        raw_text = response.text.strip()
        
        # Pulizia per estrarre il JSON puro
        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_text:
            raw_text = raw_text.split("```")[1].split("```")[0].strip()

        data_json = json.loads(raw_text)

        # --- GENERAZIONE GRAFICA AUTOMATICA DELLO SCHEMA ---
        # Creiamo un'immagine bianca per disegnare i simboli dello schema
        chart_img = Image.new("RGB", (600, 400), color="white")
        draw = ImageDraw.Draw(chart_img)
        
        y_offset = 30
        for riga_info in data_json.get("righe_schema", []):
            x_offset = 30
            draw.text((10, y_offset), f"R{riga_info['riga']}:", fill="black")
            for simbolo in riga_info["simboli"]:
                # Disegniamo simboli visivi geometrici in base al tipo di punto
                if simbolo == "cat":
                    # Un cerchietto per la catenella
                    draw.ellipse([x_offset, y_offset, x_offset + 12, y_offset + 12], outline="blue", width=2)
                elif simbolo == "m_alta":
                    # Una croce o lineetta per la maglia alta
                    draw.line([x_offset + 6, y_offset, x_offset + 6, y_offset + 15], fill="red", width=3)
                else:
                    # Un rettangolino per altri punti
                    draw.rectangle([x_offset, y_offset, x_offset + 12, y_offset + 12], outline="green", width=2)
                
                x_offset += 20
            y_offset += 30

        # Salviamo l'immagine generata in memoria o la convertiamo in base64 da mandare all'app
        import io
        buffered = io.BytesIO()
        chart_img.save(buffered, format="PNG")
        import base64
        chart_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {
            "status": "success",
            "pattern": data_json["istruzioni_testuali"],
            "schema_image": chart_base64
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
