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
app = FastAPI(title="Uncinetto Vision AI", version="12.1")
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


class CrochetMotif(BaseModel):
    nome: str = Field(description="Nome del singolo motivo, senza numero di sezione.")
    multiplo: str = Field(description="Multiplo o rapporto di ripetizione, indicando se è ipotetico.")
    descrizione_visiva: str = Field(
        description="Struttura precisa del motivo, sequenza dei simboli, righe e ripetizione da disegnare."
    )


class CrochetAnalysis(BaseModel):
    nome_punto: str = Field(description="Titolo complessivo del progetto o della lavorazione.")
    istruzioni_testuali: str = Field(
        description="Scheda tutorial italiana completa, con materiali, motivi, montaggio e rifiniture."
    )
    motivi: List[CrochetMotif] = Field(
        min_length=1,
        max_length=3,
        description="Da uno a tre motivi realmente distinti osservati nelle fotografie.",
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

Produci una scheda tutorial in italiano, ordinata come una pagina tecnica di rivista:
panoramica del progetto, osservazioni e ipotesi, materiali, campione obbligatorio,
legenda e abbreviazioni, poi una sezione per ciascun motivo con multiplo e istruzioni
riga per riga. Se sono visibili più fasce, indica il loro ordine dal basso verso l'alto.
Concludi con montaggio, eventuali maniche e rifiniture soltanto quando ricavabili o
chiaramente proposti come ipotesi. Non inventare taglie, misure o diminuzioni.

Nel campo motivi inserisci da uno a tre motivi realmente distinti: se il progetto usa un
solo punto, restituisci un solo elemento; non creare varianti artificiali per arrivare a
tre. Per ogni motivo fornisci una descrizione_visiva sufficientemente precisa da guidare
un secondo modello nella creazione di un singolo schema bianco e nero. Ogni schema dovrà
essere separato dagli altri e coerente con le istruzioni testuali. Ricorda di verificare
tensione, multipli e misure con un campione.
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
    if len(analysis.istruzioni_testuali.strip()) < 80 or not analysis.motivi:
        raise ValueError("L'analisi ricevuta è incompleta.")
    for motif in analysis.motivi:
        if len(motif.descrizione_visiva.strip()) < 30 or not motif.multiplo.strip():
            raise ValueError("Uno dei motivi ricevuti è incompleto.")
    return analysis


def _generate_chart_sync(
    images: List[bytes],
    motif: CrochetMotif,
    motif_number: int,
    work_direction: str,
) -> tuple[bytes, str]:
    prompt = f"""
Crea direttamente l'IMMAGINE di UN SOLO schema all'uncinetto, con lo stile essenziale
degli schemi tecnici in bianco e nero pubblicati nelle riviste e nei tutorial cartacei.
Usa le fotografie come riferimento visivo e la specifica seguente come vincolo tecnico.

MOTIVO {motif_number}: {motif.nome}
MULTIPLO O RIPETIZIONE: {motif.multiplo}
DIREZIONE DI LAVORAZIONE: {work_direction}
STRUTTURA DA DISEGNARE: {motif.descrizione_visiva}

Requisiti grafici obbligatori:
- rappresenta esclusivamente questo motivo, senza unirlo ad altre fasce o sezioni;
- simboli internazionali neri su fondo bianco, sottili, nitidi e ben distanziati;
- aspetto pulito di un diagramma tecnico, non un rendering fotografico e non un disegno decorativo;
- numerazione delle righe o dei giri ai lati, seguendo la direzione di lavorazione;
- evidenzia con una parentesi o un riquadro leggero il rapporto di ripetizione;
- mostra almeno due ripetizioni orizzontali quando il motivo lo consente, senza deformare i simboli;
- titolo piccolo e leggibile, con il solo nome del motivo e il multiplo;
- una sola legenda minima, soltanto se necessaria, senza duplicarla;
- niente griglia colorata, intestazioni enormi, paragrafi, fotografie, mani, persone o gomitoli;
- ampi margini bianchi e nessun simbolo tagliato;
- non aggiungere punti, righe, misure o dettagli non presenti nella specifica.
"""
    interaction = get_client().interactions.create(
        model=IMAGE_MODEL,
        input=[{"type": "text", "text": prompt}] + _image_inputs(images),
        response_format={
            "type": "image",
            "mime_type": "image/jpeg",
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
    return image_bytes, generated.mime_type or "image/jpeg"


def _refine_chart_sync(
    images: List[bytes],
    current_chart: bytes,
    motif_title: str,
    multiple: str,
    work_direction: str,
    feedback: str,
) -> tuple[bytes, str]:
    prompt = f"""
Perfeziona lo schema all'uncinetto fornito come ULTIMA immagine. Le immagini precedenti
sono le fotografie del progetto originale. Non ripartire da zero: conserva ciò che è
corretto nello schema attuale e applica la correzione richiesta dall'utente.

Titolo: {motif_title}
Multiplo attuale: {multiple}
Direzione: {work_direction}
Correzione prioritaria dell'utente: {feedback}

Mantieni lo stile di un diagramma tecnico bianco e nero: simboli internazionali nitidi,
righe numerate, rapporto di ripetizione evidenziato, margini ampi e una sola legenda
minima. Migliora anche spaziatura e leggibilità senza modificare altri aspetti tecnici.
Non aggiungere altri motivi, fotografie, decorazioni, paragrafi o misure non richieste.
Se la correzione riguarda il multiplo o il numero delle righe, applica quel vincolo in
modo coerente in tutto il diagramma invece di cambiare soltanto il testo del titolo.
"""
    interaction = get_client().interactions.create(
        model=IMAGE_MODEL,
        input=[{"type": "text", "text": prompt}]
        + _image_inputs(images)
        + _image_inputs([current_chart]),
        response_format={
            "type": "image",
            "mime_type": "image/jpeg",
            "aspect_ratio": IMAGE_ASPECT_RATIO,
            "image_size": IMAGE_SIZE,
        },
    )
    generated = interaction.output_image
    if generated is None or not generated.data:
        raise ValueError("Il modello non ha restituito lo schema perfezionato.")
    encoded = generated.data
    if encoded.startswith("data:"):
        encoded = encoded.split(",", 1)[1]
    image_bytes = base64.b64decode(encoded, validate=True)
    with Image.open(io.BytesIO(image_bytes)) as check:
        check.verify()
    return image_bytes, generated.mime_type or "image/jpeg"


def _generate_realistic_sync(
    source_images: List[bytes],
    source_kind: str,
    garment_type: str,
    yarn_weight: str,
    color: str,
    notes: str,
    pattern_text: str = "",
) -> tuple[bytes, str]:
    source = (
        "Lo SCHEMA TESTUALE completo riportato sotto descrive il capo da visualizzare."
        if source_kind == "pattern"
        else "Il DIAGRAMMA GRAFICO allegato descrive il motivo e la struttura del capo da visualizzare."
    )
    prompt = f"""
Crea direttamente una sola immagine fotorealistica di un capo lavorato all'uncinetto.
{source}
Interpreta il riferimento come una proposta visiva, senza sostenere che sia una riproduzione
tecnicamente esatta. Rispetta il tipo di capo, i motivi, i trafori, la silhouette e le
finiture descritti. Non aggiungere accessori o decorazioni non richiesti.

TIPO DI CAPO: {garment_type}
FILATO: {yarn_weight}
COLORE PREFERITO: {color}
INDICAZIONI AGGIUNTIVE: {notes}
SCHEMA TESTUALE: {pattern_text}

Presenta il capo intero, ben visibile e centrato, su manichino sartoriale neutro oppure
disteso su una superficie chiara. Texture del filato credibile e punti leggibili, luce
naturale morbida, fotografia artigianale editoriale. Nessuna persona, nessuna scritta,
nessun watermark, nessuno schema grafico sovrapposto, nessun collage.
"""
    interaction = get_client().interactions.create(
        model=IMAGE_MODEL,
        input=[{"type": "text", "text": prompt}] + _image_inputs(source_images),
        response_format={
            "type": "image",
            "mime_type": "image/jpeg",
            "aspect_ratio": IMAGE_ASPECT_RATIO,
            "image_size": IMAGE_SIZE,
        },
    )
    generated = interaction.output_image
    if generated is None or not generated.data:
        raise ValueError("Il modello non ha restituito l'immagine realistica.")
    encoded = generated.data
    if encoded.startswith("data:"):
        encoded = encoded.split(",", 1)[1]
    image_bytes = base64.b64decode(encoded, validate=True)
    with Image.open(io.BytesIO(image_bytes)) as check:
        check.verify()
    return image_bytes, generated.mime_type or "image/jpeg"


async def _realistic_response(
    source_images: List[bytes], source_kind: str, garment_type: str,
    yarn_weight: str, color: str, notes: str, pattern_text: str = "",
) -> JSONResponse:
    garment_type = _clean_form_value(garment_type, "Maglione", "Il tipo di capo")
    yarn_weight = _clean_form_value(yarn_weight, "Non specificato", "Il filato")
    color = _clean_form_value(color, "Ecru", "Il colore")
    notes = notes.strip()
    if len(notes) > 1500:
        raise HTTPException(status_code=422, detail="Le indicazioni aggiuntive sono troppo lunghe.")
    try:
        async with ai_semaphore:
            image_bytes, mime_type = await asyncio.wait_for(
                asyncio.to_thread(
                    _generate_realistic_sync, source_images, source_kind, garment_type,
                    yarn_weight, color, notes, pattern_text,
                ), timeout=AI_TIMEOUT_SECONDS,
            )
        return JSONResponse(
            content={
                "status": "success",
                "realistic_image": base64.b64encode(image_bytes).decode("ascii"),
                "realistic_mime_type": mime_type,
                "notice": "Interpretazione visiva AI: la resa reale dipende da filato, mano, misure e campione.",
            }, headers={"Cache-Control": "no-store"},
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail="La generazione sta impiegando troppo tempo. Riprova tra poco.") from exc
    except (ValidationError, ValueError) as exc:
        logger.warning("Immagine realistica non valida: %s", exc)
        raise HTTPException(status_code=502, detail="L'AI non ha restituito un'immagine valida. Riprova con indicazioni più precise.") from exc
    except Exception as exc:
        logger.exception("Errore durante la generazione realistica")
        raise HTTPException(status_code=502, detail="Il servizio AI non è disponibile in questo momento. Riprova più tardi.") from exc


@app.post("/generate-realistic-from-pattern")
@limiter.limit(os.getenv("REALISTIC_RATE_LIMIT", "5/minute"))
async def generate_realistic_from_pattern(
    request: Request,
    pattern: str = Form(...),
    garment_type: str = Form(default="Maglione"),
    yarn_weight: str = Form(default="Non specificato"),
    color: str = Form(default="Ecru"),
    notes: str = Form(default=""),
):
    del request
    pattern = pattern.strip()
    if len(pattern) < 30 or len(pattern) > 20000:
        raise HTTPException(status_code=422, detail="Inserisci uno schema del capo da 30 a 20.000 caratteri.")
    return await _realistic_response([], "pattern", garment_type, yarn_weight, color, notes, pattern)


@app.post("/generate-realistic-from-chart")
@limiter.limit(os.getenv("REALISTIC_RATE_LIMIT", "5/minute"))
async def generate_realistic_from_chart(
    request: Request,
    chart: UploadFile = File(...),
    garment_type: str = Form(default="Maglione"),
    yarn_weight: str = Form(default="Non specificato"),
    color: str = Form(default="Ecru"),
    notes: str = Form(default=""),
):
    del request
    contents = await chart.read(MAX_FILE_BYTES + 1)
    await chart.close()
    if len(contents) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="Lo schema grafico è troppo grande.")
    return await _realistic_response([_prepare_image(contents)], "chart", garment_type, yarn_weight, color, notes)


def _crochet_advice_sync(answers: dict[str, str]) -> str:
    labels = {
        "experience": "Livello", "project": "Progetto", "material": "Materiale del filato",
        "weight": "Spessore del filato", "hook": "Uncinetto", "direction": "Lavorazione",
        "difficulty": "Difficoltà principale", "goal": "Obiettivo",
    }
    profile = "\n".join(f"- {labels[key]}: {value}" for key, value in answers.items())
    prompt = f"""
Agisci come una maestra d'uncinetto esperta, concreta e incoraggiante. Rispondi in italiano
a questo profilo scelto dall'utente tramite un questionario:
{profile}

Crea una scheda personale intitolata CONSIGLI, TRUCCHI E SEGRETI DEL MESTIERE, adatta
esattamente al livello, al progetto, ai materiali e alla difficoltà indicati. Organizzala con:
1. Il consiglio più importante per questo profilo.
2. Cinque consigli pratici numerati, specifici e immediatamente applicabili.
3. Due trucchi del mestiere che aiutano davvero, senza promesse miracolose.
4. Tre errori frequenti da evitare e come riconoscerli.
5. Un esercizio breve sul campione, con un criterio semplice per valutare il risultato.
6. Una checklist finale di cinque voci.

Non inventare multipli, misure o numero di maglie non forniti. Se parli di campione, tensione,
filato o uncinetto, ricorda che servono prove sul proprio progetto. Testo chiaro, tono caldo,
nessuna introduzione generica, nessuna domanda finale, massimo 900 parole.
"""
    interaction = get_client().interactions.create(
        model=TEXT_MODEL,
        input=[{"type": "text", "text": prompt}],
    )
    advice = (interaction.output_text or "").strip()
    if len(advice) < 150:
        raise ValueError("I consigli ricevuti sono incompleti.")
    return advice


@app.post("/crochet-advice")
@limiter.limit(os.getenv("ADVICE_RATE_LIMIT", "8/minute"))
async def crochet_advice(
    request: Request,
    experience: str = Form(...), project: str = Form(...), material: str = Form(...),
    weight: str = Form(...), hook: str = Form(...), direction: str = Form(...),
    difficulty: str = Form(...), goal: str = Form(...),
):
    del request
    raw = {
        "experience": experience, "project": project, "material": material,
        "weight": weight, "hook": hook, "direction": direction,
        "difficulty": difficulty, "goal": goal,
    }
    answers = {key: _clean_form_value(value, "Non specificato", key) for key, value in raw.items()}
    try:
        async with ai_semaphore:
            advice = await asyncio.wait_for(
                asyncio.to_thread(_crochet_advice_sync, answers), timeout=AI_TIMEOUT_SECONDS,
            )
        return JSONResponse(
            content={"status": "success", "advice": advice},
            headers={"Cache-Control": "no-store"},
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail="La preparazione dei consigli sta impiegando troppo tempo. Riprova tra poco.") from exc
    except (ValidationError, ValueError) as exc:
        logger.warning("Consigli AI non validi: %s", exc)
        raise HTTPException(status_code=502, detail="L'AI non ha restituito consigli completi. Riprova.") from exc
    except Exception as exc:
        logger.exception("Errore durante la generazione dei consigli")
        raise HTTPException(status_code=502, detail="Il servizio AI non è disponibile in questo momento. Riprova più tardi.") from exc


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
            generated_charts = []
            for index, motif in enumerate(analysis.motivi, start=1):
                chart_bytes, chart_mime = await asyncio.wait_for(
                    asyncio.to_thread(
                        _generate_chart_sync,
                        images,
                        motif,
                        index,
                        work_direction,
                    ),
                    timeout=AI_TIMEOUT_SECONDS,
                )
                generated_charts.append(
                    {
                        "titolo": motif.nome,
                        "multiplo": motif.multiplo,
                        "immagine": base64.b64encode(chart_bytes).decode("ascii"),
                        "mime_type": chart_mime,
                    }
                )

        return JSONResponse(
            content={
                "status": "success",
                "nome_punto": analysis.nome_punto,
                "pattern": analysis.istruzioni_testuali,
                "schemi": generated_charts,
                "schema_image": generated_charts[0]["immagine"],
                "schema_mime_type": generated_charts[0]["mime_type"],
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


@app.post("/refine-chart")
@limiter.limit(os.getenv("REFINE_RATE_LIMIT", "8/minute"))
async def refine_chart(
    request: Request,
    chart: UploadFile = File(...),
    files: List[UploadFile] = File(...),
    motif_title: str = Form(default="Motivo"),
    multiple: str = Form(default="Non specificato"),
    work_direction: str = Form(default="In righe"),
    feedback: str = Form(...),
):
    del request
    motif_title = _clean_form_value(motif_title, "Motivo", "Il titolo")
    multiple = _clean_form_value(multiple, "Non specificato", "Il multiplo")
    work_direction = _clean_form_value(work_direction, "In righe", "La direzione")
    feedback = feedback.strip()
    if len(feedback) < 3 or len(feedback) > 1000:
        raise HTTPException(
            status_code=422,
            detail="Scrivi una correzione da 3 a 1000 caratteri.",
        )

    chart_contents = await chart.read(MAX_FILE_BYTES + 1)
    await chart.close()
    if len(chart_contents) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="Lo schema da perfezionare è troppo grande.")
    current_chart = _prepare_image(chart_contents)
    images = await _read_images(files)

    try:
        async with ai_semaphore:
            chart_bytes, chart_mime = await asyncio.wait_for(
                asyncio.to_thread(
                    _refine_chart_sync,
                    images,
                    current_chart,
                    motif_title,
                    multiple,
                    work_direction,
                    feedback,
                ),
                timeout=AI_TIMEOUT_SECONDS,
            )
        return JSONResponse(
            content={
                "status": "success",
                "schema_image": base64.b64encode(chart_bytes).decode("ascii"),
                "schema_mime_type": chart_mime,
                "notice": "Schema perfezionato: verifica nuovamente simboli, multiplo e righe prima dell'uso.",
            },
            headers={"Cache-Control": "no-store"},
        )
    except asyncio.TimeoutError as exc:
        logger.warning("Timeout durante il perfezionamento AI", exc_info=True)
        raise HTTPException(
            status_code=504,
            detail="Il perfezionamento sta impiegando troppo tempo. Riprova tra poco.",
        ) from exc
    except (ValidationError, ValueError) as exc:
        logger.warning("Risposta di perfezionamento non valida: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="L'AI non ha restituito uno schema valido. Riprova con una correzione più precisa.",
        ) from exc
    except Exception as exc:
        logger.exception("Errore durante il perfezionamento AI")
        raise HTTPException(
            status_code=502,
            detail="Il servizio AI non è disponibile in questo momento. Riprova più tardi.",
        ) from exc
