from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from PIL import Image, ImageDraw
import google.genai as genai
import json
import io
import base64

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
async def generate_pattern(
    file: UploadFile = File(...),
    garment_type: str = Form(default="Maglione"),
    work_direction: str = Form(default="In righe (avanti e indietro)"),
    yarn_weight: str = Form(default="DK (Medio)"),
    hook_size: str = Form(default="4.0 mm")
):
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

        # Prompt avanzato che include la gestione di collo, maniche e sagomature per capi complessi
        prompt = f"""
        Agisci come una maestra d'uncinetto e un designer di moda esperto.
        Parametri tecnici:
        - Tipo di capo: '{garment_type}'
        - Direzione di lavorazione: '{work_direction}'
        - Spessore filato: '{yarn_weight}'
        - Misura uncinetto: '{hook_size}'
        
        Analizza l'immagine e restituisci un oggetto JSON valido (e nessun altro testo fuori dal JSON) con questa struttura:
        {{
          "nome_punto": "Nome della trama",
          "istruzioni_testuali": "Istruzioni dettagliate divise per sezioni (es. Pannello Corpo, Giromanica, Scollo, Maniche) con indicazione esatta di aumenti e diminuzioni...",
          "righe_schema": [
            {{"riga": 1, "sezione": "Corpo - Base", "simboli": ["cat", "cat", "cat", "m_alta", "m_alta"]}},
            {{"riga": 2, "sezione": "Corpo - Lavorazione", "simboli": ["cat", "m_alta", "ventaglio", "m_alta"]}},
            {{"riga": 10, "sezione": "Giromanica / Scollo", "simboli": ["diminuzione", "m_alta", "m_alta", "diminuzione"]}}
          ]
        }}
        I simboli ammessi nell'array devono essere scelti tra: 'cat', 'm_alta', 'm_bassa', 'ventaglio', 'diminuzione', 'aumento'.
        """

        response = client.models.generate_content(
            model='gemini-3.8-flash',
            contents=[pil_img, prompt]
        )

        raw_text = response.text.strip()
        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_text:
            raw_text = raw_text.split("```")[1].split("```")[0].strip()

        data_json = json.loads(raw_text)

        # Generazione grafica dello schema con indicazione delle sezioni (Corpo, Scollo, Maniche)
        chart_img = Image.new("RGB", (700, 500), color="white")
        draw = ImageDraw.Draw(chart_img)
        
        y_offset = 30
        current_section = ""
        
        for riga_info in data_json.get("righe_schema", []):
            sezione = riga_info.get("sezione", "Generale")
            
            # Se cambia la sezione (es. passiamo al collo o alle maniche), stampiamo un'intestazione visiva nel grafico
            if sezione != current_section:
                current_section = sezione
                draw.text((15, y_offset), f"--- [{current_section.upper()}]:", fill="purple")
                y_offset += 25

            x_offset = 30
            draw.text((10, y_offset), f"R{riga_info['riga']}:", fill="black")
            
            for simbolo in riga_info["simboli"]:
                if simbolo == "cat":
                    draw.ellipse([x_offset, y_offset, x_offset + 12, y_offset + 12], outline="blue", width=2)
                elif simbolo == "m_alta":
                    draw.line([x_offset + 6, y_offset, x_offset + 6, y_offset + 15], fill="red", width=3)
                elif simbolo == "diminuzione":
                    draw.line([x_offset, y_offset, x_offset + 12, y_offset + 15], fill="orange", width=3)
                    draw.line([x_offset + 12, y_offset, x_offset, y_offset + 15], fill="orange", width=3)
                else:
                    draw.rectangle([x_offset, y_offset, x_offset + 12, y_offset + 12], outline="green", width=2)
                x_offset += 20
            y_offset += 30

        buffered = io.BytesIO()
        chart_img.save(buffered, format="PNG")
        chart_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {
            "status": "success",
            "pattern": data_json["istruzioni_testuali"],
            "schema_image": chart_base64
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
