import asyncio
import base64
import io
import logging
import os
import warnings
from functools import lru_cache
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google import genai
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field, ValidationError
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("uncinetto-vision")

TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-3.8-flash")
IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-nano-banana-2.1")
IMAGE_ASPECT_RATIO = os.getenv("IMAGE_ASPECT_RATIO", "4:3")
IMAGE_SIZE = os.getenv("IMAGE_SIZE", "2K")
AI_TIMEOUT_SECONDS = int(os.getenv("AI_TIMEOUT_SECONDS", "180"))
MAX_FILES = int(os.getenv("MAX_FILES", "6"))
MAX_FILE_BYTES = int(os.getenv("MAX_FILE_BYTES", str(8 * 1024 * 1024)))
MAX_TOTAL_BYTES = int(os.getenv("MAX_TOTAL_BYTES", str(20 * 1024 * 1024)))
MAX_IMAGE_PIXELS = 25_000_000

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
warnings.simplefilter("error", Image.DecompressionBombWarning)

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Uncinetto Vision AI", version="10.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

ai_semaphore = asyncio.Semaphore(int(os.getenv("MAX_CONCURRENT_AI_REQUESTS", "2")))


class CrochetAnalysis(BaseModel):
    nome_punto: str = Field(description="Nome tecnico del punto o della lavorazione.")
    istruzioni_testuali: str = Field(
        description="Guida tecnica italiana completa e prudente, con titoli in testo semplice."
    )
    descrizione_visiva: str = Field(
        description="Descrizione precisa della struttura, dei simboli e della ripetizione da disegnare."
    )


@lru_cache
def get_client() -> genai.Client:
    if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("Configura GEMINI_API_KEY nelle variabili d'ambiente del backend.")
    return genai.Client()


def _clean_form_value(value: str, fallback: str, field_name: str) -> str:
    cleaned = value.strip() or fallback
    if len(cleaned) > 100:
        raise HTTPException(status_code=422, detail=f"{field_name} è troppo lungo.")
    return cleaned


def _prepare_image(contents: bytes) -> bytes:
    try:
        with Image.open(io.BytesIO(contents)) as source:
            source.verify()
        with Image.open(io.BytesIO(contents)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=90, optimize=True)
            return output.getvalue()
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise HTTPException(
            status_code=400,
            detail="Una foto non è un'immagine valida oppure supera i limiti consentiti.",
        ) from exc


async def _read_images(files: List[UploadFile]) -> List[bytes]:
    if not files:
        raise HTTPException(status_code=400, detail="Carica almeno una fotografia.")
    if len(files) > MAX_FILES:
        raise HTTPException(
            status_code=413,
            detail=f"Puoi caricare al massimo {MAX_FILES} fotografie alla volta.",
        )

    images: List[bytes] = []
    total_bytes = 0
    allowed_types = {"image/jpeg", "image/png", "image/webp", "application/octet-stream", ""}

    for file in files:
        content_type = (file.content_type or "").lower()
        if content_type not in allowed_types:
            raise HTTPException(
                status_code=415,
                detail="Formato non supportato. Usa fotografie JPG, PNG o WebP.",
            )

        contents = await file.read(MAX_FILE_BYTES + 1)
        await file.close()
        if len(contents) > MAX_FILE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Ogni fotografia può pesare al massimo {MAX_FILE_BYTES // (1024 * 1024)} MB.",
            )
        total_bytes += len(contents)
        if total_bytes > MAX_TOTAL_BYTES:
            raise HTTPException(status_code=413, detail="Le fotografie superano il limite totale consentito.")
        images.append(_prepare_image(contents))

    return images


def _image_inputs(images: List[bytes]) -> List[dict]:
    return [
        {
            "type": "image",
            "data": base64.b64encode(image).decode("ascii"),
            "mime_type": "image/jpeg",
        }
        for image in images
    ]


def _analyze_sync(
    images: List[bytes],
    garment_type: str,
    work_direction: str,
    yarn_weight: str,
    hook_size: str,
) -> CrochetAnalysis:
    prompt = f"""
Agisci come una maestra d'uncinetto e redattrice tecnica. Analizza soltanto ciò che è
visibile nelle fotografie; non inventare misure del corpo, taglie o una costruzione
completa del capo se non sono ricavabili dalle immagini.

Parametri forniti dall'utente:
- Tipo di progetto: {garment_type}
- Direzione: {work_direction}
- Filato: {yarn_weight}
- Uncinetto: {hook_size}

Produci una guida in italiano che includa, quando ricavabili: nome del punto, materiali,
multiplo, sequenza delle righe o dei giri, legenda dei simboli e rifiniture. Distingui
chiaramente le osservazioni dalle ipotesi. Ricorda che occorre realizzare e misurare un
campione per verificare tensione, multipli e misure. La descrizione_visiva servirà a un
secondo modello per disegnare uno schema tecnico coerente con la guida.
"""
    interaction = get_client().interactions.create(
        model=TEXT_MODEL,
        input=[{"type": "text", "text": prompt}] + _image_inputs(images),
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": CrochetAnalysis.model_json_schema(),
        },
    )
    if not interaction.output_text:
        raise ValueError("Il modello non ha restituito l'analisi testuale.")
    analysis = CrochetAnalysis.model_validate_json(interaction.output_text)
    if len(analysis.istruzioni_testuali.strip()) < 80 or len(analysis.descrizione_visiva.strip()) < 30:
        raise ValueError("L'analisi ricevuta è incompleta.")
    return analysis


def _generate_chart_sync(
    images: List[bytes],
    analysis: CrochetAnalysis,
    work_direction: str,
) -> tuple[bytes, str]:
    prompt = f"""
Crea direttamente un'IMMAGINE di uno schema all'uncinetto professionale, realistico
nello stile di una rivista tecnica editoriale, usando le fotografie come riferimento
visivo e l'analisi seguente come specifica obbligatoria.

ANALISI TECNICA:
{analysis.model_dump_json()}

Direzione di lavorazione: {work_direction}

Requisiti grafici:
- schema tecnico all'uncinetto, NON una fotografia del capo e NON un'illustrazione decorativa;
- fondo bianco, griglia chiara, simboli internazionali nitidi e uniformi;
- righe o giri numerati, direzione di lavorazione e ripetizione del multiplo ben visibili;
- legenda essenziale e titolo leggibile in italiano;
- struttura coerente con l'analisi, senza aggiungere punti o misure inventate;
- composizione ordinata, margini ampi, niente simboli tagliati, niente testo piccolo;
- niente mani, persone, gomitoli, uncinetti o elementi fotografici di contorno.

Titolo dello schema: {analysis.nome_punto}
Descrizione da seguire: {analysis.descrizione_visiva}
"""
    interaction = get_client().interactions.create(
        model=IMAGE_MODEL,
        input=[{"type": "text", "text": prompt}] + _image_inputs(images),
        response_format={
            "type": "image",
            "mime_type": "image/png",
            "aspect_ratio": IMAGE_ASPECT_RATIO,
            "image_size": IMAGE_SIZE,
        },
    )
    generated = interaction.output_image
    if generated is None or not generated.data:
        raise ValueError("Il modello non ha restituito alcuna immagine.")

    encoded = generated.data
    if encoded.startswith("data:"):
        encoded = encoded.split(",", 1)[1]
    image_bytes = base64.b64decode(encoded, validate=True)
    with Image.open(io.BytesIO(image_bytes)) as check:
        check.verify()
    return image_bytes, generated.mime_type or "image/png"


@app.get("/")
def home():
    return {"status": "online", "message": "Uncinetto Vision AI è attiva."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate-pattern")
@limiter.limit(os.getenv("RATE_LIMIT", "5/minute"))
async def generate_pattern(
    request: Request,
    files: List[UploadFile] = File(...),
    garment_type: str = Form(default="Maglione"),
    work_direction: str = Form(default="In righe (avanti e indietro)"),
    yarn_weight: str = Form(default="DK (Medio - 3)"),
    hook_size: str = Form(default="4.0 mm"),
):
    del request  # Richiesto dal limiter; la validazione dell'IP è gestita da SlowAPI.
    garment_type = _clean_form_value(garment_type, "Maglione", "Il tipo di capo")
    work_direction = _clean_form_value(work_direction, "In righe", "La direzione")
    yarn_weight = _clean_form_value(yarn_weight, "Non specificato", "Il filato")
    hook_size = _clean_form_value(hook_size, "Non specificata", "La misura dell'uncinetto")
    images = await _read_images(files)

    try:
        async with ai_semaphore:
            analysis = await asyncio.wait_for(
                asyncio.to_thread(
                    _analyze_sync,
                    images,
                    garment_type,
                    work_direction,
                    yarn_weight,
                    hook_size,
                ),
                timeout=AI_TIMEOUT_SECONDS,
            )
            chart_bytes, chart_mime = await asyncio.wait_for(
                asyncio.to_thread(_generate_chart_sync, images, analysis, work_direction),
                timeout=AI_TIMEOUT_SECONDS,
            )

        return JSONResponse(
            content={
                "status": "success",
                "nome_punto": analysis.nome_punto,
                "pattern": analysis.istruzioni_testuali,
                "schema_image": base64.b64encode(chart_bytes).decode("ascii"),
                "schema_mime_type": chart_mime,
                "image_provider": IMAGE_MODEL,
                "notice": "Bozza generata dall'AI: verifica simboli, multipli e misure con un campione.",
            },
            headers={"Cache-Control": "no-store"},
        )
    except HTTPException:
        raise
    except asyncio.TimeoutError as exc:
        logger.warning("Timeout durante la generazione AI", exc_info=True)
        raise HTTPException(
            status_code=504,
            detail="La generazione sta impiegando troppo tempo. Riprova tra poco.",
        ) from exc
    except (ValidationError, ValueError) as exc:
        logger.warning("Risposta AI non valida: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="L'AI ha restituito un risultato incompleto. Riprova con foto più nitide.",
        ) from exc
    except Exception as exc:
        logger.exception("Errore durante la generazione AI")
        raise HTTPException(
            status_code=502,
            detail="Il servizio AI non è disponibile in questo momento. Riprova più tardi.",
        ) from exc
