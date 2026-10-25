from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from PIL import Image, ImageDraw
import google.genai as genai
import json
import io
import base64
from typing import List

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

        # Prompt dettagliato che sfrutta tutte le foto (panoramica + dettagli ravvicinati)
        prompt = f"""
        Agisci come una maestra d'uncinetto e un designer di moda esperto.
        Ti sono state fornite più immagini dello stesso progetto (foto d'insieme e dettagli ravvicinati del punto).
        Parametri tecnici obbligatori:
        - Tipo di capo: '{garment_type}'
        - Direzione di lavorazione: '{work_direction}'
        - Spessore filato: '{yarn_weight}'
        - Misura uncinetto: '{hook_size}'
        
        Analizza le immagini e restituisci un oggetto JSON valido (e nessun altro testo fuori dal JSON) con questa struttura:
        {{
          "nome_punto": "Nome della trama",
          "istruzioni_testuali": "Istruzioni dettagliate passo-passo divise per sezioni (es. Corpo, Scollo, Maniche) adatte per un {garment_type}...",
          "righe_schema": [
            {{"riga": 1, "sezione": "Base / Avvio", "simboli": ["cat", "cat", "cat", "m_alta", "m_alta"]}},
            {{"riga": 2, "sezione": "Corpo - Motivo", "simboli": ["cat", "m_alta", "ventaglio", "m_alta"]}},
            {{"riga": 3, "sezione": "Giromanica / Scollo", "simboli": ["diminuzione", "m_alta", "m_alta"]}}
          ]
        }}
        I simboli ammessi nell'array devono essere scelti esclusivamente tra: 'cat' (catenella), 'm_alta' (maglia alta), 'm_bassa' (maglia bassa), 'ventaglio', 'diminuzione', 'aumento'.
        """

        # Inviamo tutte le immagini insieme al prompt a Gemini
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

        # --- GENERAZIONE GRAFICO GRANDE, PULITO E LEGGIBILE (Stile Carta a Quadretti) ---
        img_width, img_height = 900, 700
        chart_img = Image.new("RGB", (img_width, img_height), color="white")
        draw = ImageDraw.Draw(chart_img)
        
        # Disegniamo una griglia leggera stile carta a quadretti
        grid_size = 30
        for x in range(0, img_width, grid_size):
            draw.line([(x, 0), (x, img_height)], fill="#F0F0F0", width=1)
        for y in range(0, img_height, grid_size):
            draw.line([(0, y), (img_width, y)], fill="#F0F0F0", width=1)

        y_offset = 40
        current_section = ""
        
        for riga_info in data_json.get("righe_schema", []):
            sezione = riga_info.get("sezione", "Lavorazione")
            
            if sezione != current_section:
                current_section = sezione
                y_offset += 10
                draw.text((30, y_offset), f"--- [ {current_section.upper()} ] ---", fill="purple")
                y_offset += 35

            # Etichetta Riga
            draw.text((30, y_offset), f"R {riga_info['riga']}:", fill="black")
            
            x_offset = 100
            for simbolo in riga_info["simboli"]:
                # Disegniamo simboli grandi, ben distanziati e chiaramente leggibili
                if simbolo == "cat":
                    # Cerchietto blu chiaro/scuro per la catenella
                    draw.ellipse([x_offset, y_offset - 2, x_offset + 18, y_offset + 16], outline="blue", width=3)
                elif simbolo == "m_alta":
                    # Lineetta verticale rossa con trattino orizzontale per la maglia alta
                    draw.line([x_offset + 9, y_offset - 5, x_offset + 9, y_offset + 20], fill="red", width=3)
                    draw.line([x_offset + 4, y_offset + 7, x_offset + 14, y_offset + 7], fill="red", width=2)
                elif simbolo == "m_bassa":
                    # Crocetta o X verde per la maglia bassa
                    draw.line([x_offset + 3, y_offset + 2, x_offset + 15, y_offset + 14], fill="green", width=3)
                    draw.line([x_offset + 15, y_offset + 2, x_offset + 3, y_offset + 14], fill="green", width=3)
                elif simbolo == "diminuzione":
                    # Simbolo di diminuzione arancione
                    draw.line([x_offset, y_offset, x_offset + 18, y_offset + 16], fill="orange", width=3)
                    draw.line([x_offset + 18, y_offset, x_offset, y_offset + 16], fill="orange", width=3)
                else:
                    # Rettangolo generico per ventagli o altri punti
                    draw.rectangle([x_offset, y_offset + 2, x_offset + 16, y_offset + 14], outline="darkgreen", width=3)
                
                x_offset += 35  # Spaziatura ampia per evitare sovrapposizioni
            
            y_offset += 45
            if y_offset > img_height - 50:
                break # Evita di uscire dall'immagine se ci sono molte righe

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
