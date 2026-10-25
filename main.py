from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import google.genai as genai
import json
import io
import base64
from typing import List

app = FastAPI(title="Uncinetto Vision AI Native Pro", version="9.0")

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
    return {"status": "online", "message": "Uncinetto Vision API AI-Native Pro è attiva!"}

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

        # Chiediamo a Gemini di analizzare la trama e restituire la struttura millimetrica esatta dei simboli
        prompt = f"""
        Agisci come una maestra d'uncinetto e una redattrice di riviste di moda d'eccellenza.
        Parametri tecnici obbligatori:
        - Tipo di capo: '{garment_type}'
        - Direzione di lavorazione: '{work_direction}'
        - Spessore filato: '{yarn_weight}'
        - Misura uncinetto: '{hook_size}'
        
        Analizza le immagini con la massima attenzione e restituisci un oggetto JSON valido (e nessun altro testo fuori dal JSON) con questa struttura esatta:
        {{
          "nome_punto": "Nome tecnico della trama",
          "istruzioni_testuali": "GUIDA EDITORIALE COMPLETA IN MARKDOWN (Introduzione, Misure in cm, Materiali, Multiplo, Istruzioni riga per riga, Assemblaggio, Rifiniture)...",
          "schema_griglia": [
            {{"riga": 1, "simboli": ["cat", "cat", "cat", "m_alta", "m_alta", "m_alta", "cat", "cat"]}},
            {{"riga": 2, "simboli": ["cat", "m_alta", "archetto", "ventaglio", "archetto", "m_alta"]}},
            {{"riga": 3, "simboli": ["cat", "ventaglio", "m_alta", "ventaglio", "cat"]}},
            {{"riga": 4, "simboli": ["cat", "m_alta", "m_alta", "m_alta", "m_alta", "cat"]}}
          ]
        }}
        I simboli ammessi nell'array 'simboli' sono rigorosamente: 'cat' (catenella), 'm_alta' (maglia alta), 'm_bassa' (maglia bassa), 'ventaglio', 'archetto'.
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

        # Renderizziamo graficamente l'output basandoci esattamente sulla mappa strutturata dall'IA
        img_width, img_height = 1100, 700
        chart_img = Image.new("RGB", (img_width, img_height), color="#FFFFFF")
        draw = ImageDraw.Draw(chart_img)
        
        # Sfondo griglia millimetrata stile rivista tecnica
        grid_step = 30
        for x in range(0, img_width, grid_step):
            draw.line([(x, 0), (x, img_height)], fill="#F0F0F0", width=1)
        for y in range(0, img_height, grid_step):
            draw.line([(0, y), (img_width, y)], fill="#F0F0F0", width=1)

        y_offset = 60
        for riga_info in data_json.get("schema_griglia", []):
            r_num = riga_info.get("riga", 1)
            draw.text((30, y_offset + 5), f"R {r_num}", fill="#333333")
            
            x_offset = 100
            for simbolo in riga_info.get("simboli", []):
                if simbolo == "cat":
                    # Catenella (ovale blu pulito)
                    draw.ellipse([x_offset, y_offset + 5, x_offset + 16, y_offset + 22], outline="#1D3557", width=2)
                elif simbolo == "m_alta":
                    # Maglia alta (linea verticale con trattino orizzontale rosso/nero)
                    draw.line([x_offset + 8, y_offset, x_offset + 8, y_offset + 30], fill="#E63946", width=3)
                    draw.line([x_offset + 3, y_offset + 14, x_offset + 13, y_offset + 14], fill="#E63946", width=2)
                elif simbolo == "m_bassa":
                    # Maglia bassa (crocetta)
                    draw.line([x_offset + 3, y_offset + 8, x_offset + 13, y_offset + 22], fill="#2A9D8F", width=3)
                    draw.line([x_offset + 13, y_offset + 8, x_offset + 3, y_offset + 22], fill="#2A9D8F", width=3)
                elif simbolo == "archetto":
                    # Archetto di catenelle
                    draw.arc([x_offset, y_offset - 5, x_offset + 32, y_offset + 25], start=180, end=360, fill="#457B9D", width=2)
                elif simbolo == "ventaglio":
                    # Simbolo stilizzato del ventaglio a raggiera
                    center_x = x_offset + 12
                    center_y = y_offset + 28
                    draw.arc([center_x - 18, center_y - 20, center_x + 18, center_y], start=180, end=360, fill="#6D597A", width=3)
                    draw.line([center_x, center_y, center_x, center_y - 12], fill="#6D597A", width=2)
                
                x_offset += 42
            
            y_offset += 50
            if y_offset > img_height - 60:
                break

        buffered = io.BytesIO()
        chart_img.save(buffered, format="PNG")
        chart_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {
            "status": "success",
            "nome_punto": data_json.get("nome_punto", "Schema Uncinetto Professionale"),
            "pattern": data_json["istruzioni_testuali"],
            "schema_image": chart_base64
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
