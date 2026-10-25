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
        5. **Istruzioni Dettagliate dei Motivi**: Spiegazione passo-passo, riga per riga o giro per giro, di ogni schema di punto presente nelle immagini (es. Motivo 1, Motivo 2, ecc.).
        6. **Assemblaggio e Cuciture**: Come unire i pannelli (dritto contro dritto, cucitura delle spalle, lasciare l'apertura per lo scollo a barchetta e per le maniche).
        7. **Rifiniture**: Bordi finali (es. giro di maglie basse sullo scollo e sulle maniche).

        Restituisci un oggetto JSON valido (e nessun άλλο testo fuori dal JSON) con questa struttura:
        {{
          "nome_punto": "Nome tecnico della trama principale",
          "istruzioni_testuali": "IL TESTO EDITORIALE COMPLETO E DETTAGLIATO CON TUTTE LE SEZIONI SOPRA ELENCATE, FORMATTATO IN MARKDOWN...",
          "righe_schema": [
            {{"riga": 1, "sezione": "Base / Avvio", "simboli": ["cat", "cat", "cat", "m_alta", "m_alta"]}},
            {{"riga": 2, "sezione": "Motivo Principale", "simboli": ["cat", "m_alta", "ventaglio", "m_alta"]}},
            {{"riga": 3, "sezione": "Rifinitura", "simboli": ["m_bassa", "m_bassa"]}}
          ]
        }}
        I simboli ammessi nell'array devono essere scelti tra: 'cat', 'm_alta', 'm_bassa', 'ventaglio', 'diminuzione', 'aumento'.
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

        # Generazione grafico stile carta a quadretti
        img_width, img_height = 900, 700
        chart_img = Image.new("RGB", (img_width, img_height), color="white")
        draw = ImageDraw.Draw(chart_img)
        
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

            draw.text((30, y_offset), f"R {riga_info['riga']}:", fill="black")
            
            x_offset = 100
            for simbolo in riga_info["simboli"]:
                if simbolo == "cat":
                    draw.ellipse([x_offset, y_offset - 2, x_offset + 18, y_offset + 16], outline="blue", width=3)
                elif simbolo == "m_alta":
                    draw.line([x_offset + 9, y_offset - 5, x_offset + 9, y_offset + 20], fill="red", width=3)
                    draw.line([x_offset + 4, y_offset + 7, x_offset + 14, y_offset + 7], fill="red", width=2)
                elif simbolo == "m_bassa":
                    draw.line([x_offset + 3, y_offset + 2, x_offset + 15, y_offset + 14], fill="green", width=3)
                    draw.line([x_offset + 15, y_offset + 2, x_offset + 3, y_offset + 14], fill="green", width=3)
                elif simbolo == "diminuzione":
                    draw.line([x_offset, y_offset, x_offset + 18, y_offset + 16], fill="orange", width=3)
                    draw.line([x_offset + 18, y_offset, x_offset, y_offset + 16], fill="orange", width=3)
                else:
                    draw.rectangle([x_offset, y_offset + 2, x_offset + 16, y_offset + 14], outline="darkgreen", width=3)
                
                x_offset += 35
            
            y_offset += 45
            if y_offset > img_height - 50:
                break

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
