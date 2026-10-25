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
    garment_type: str = Form(default=""),
    work_direction: str = Form(default=""),
    yarn_weight: str = Form(default=""),
    hook_size: str = Form(default="")
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

        # Prompt ultra-dettagliato con la scheda tecnica del progetto
        prompt = f"""
        Agisci come una maestra d'uncinetto e un software di pattern design di livello professionale.
        Scheda tecnica del progetto fornita dall'utente:
        - Tipo di capo: '{garment_type if garment_type else "Non specificato"}'
        - Direzione di lavorazione: '{work_direction if work_direction else "Non specificata"}'
        - Spessore filato: '{yarn_weight if yarn_weight else "Non specificato"}'
        - Misura uncinetto: '{hook_size if hook_size else "Non specificata"}'
        
        Analizza attentamente l'immagine e restituisci un oggetto JSON valido (e nessun altro testo fuori dal JSON) con questa struttura esatta:
        {{
          "nome_punto": "Nome della trama o del punto",
          "istruzioni_testuali": "Istruzioni dettagliate passo-passo riga per riga...",
          "righe_schema": [
            {{"riga": 1, "simboli": ["cat", "cat", "cat", "m_alta", "m_alta"]}},
            {{"riga": 2, "simboli": ["cat", "m_alta", "ventaglio", "m_alta"]}}
          ]
        }}
        I simboli ammessi nell'array devono essere scelti tra: 'cat' (catenella), 'm_alta' (maglia alta), 'm_bassa' (maglia bassa), 'ventaglio'.
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

        # Generazione grafica dello schema
        chart_img = Image.new("RGB", (600, 400), color="white")
        draw = ImageDraw.Draw(chart_img)
        
        y_offset = 30
        for riga_info in data_json.get("righe_schema", []):
            x_offset = 30
            draw.text((10, y_offset), f"R{riga_info['riga']}:", fill="black")
            for simbolo in riga_info["simboli"]:
                if simbolo == "cat":
                    draw.ellipse([x_offset, y_offset, x_offset + 12, y_offset + 12], outline="blue", width=2)
                elif simbolo == "m_alta":
                    draw.line([x_offset + 6, y_offset, x_offset + 6, y_offset + 15], fill="red", width=3)
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
