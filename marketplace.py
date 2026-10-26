"""Vetrina multiutente: account, prodotti, ordini, acquisti e vendite.

I dati condivisi vivono in PostgreSQL su Render. SQLite è disponibile soltanto
per prove locali. Nessun pagamento viene elaborato da questo modulo.
"""

import base64
import hashlib
import io
import os
import re
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, LargeBinary, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./uncinetto_marketplace.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg://", 1)
elif DATABASE_URL.startswith("postgresql://") and "+psycopg" not in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
router = APIRouter(prefix="/market", tags=["Vetrina"])
bearer = HTTPBearer(auto_error=False)
market_limiter = Limiter(key_func=get_remote_address)
SESSION_DAYS = int(os.getenv("MARKET_SESSION_DAYS", "30"))
MAX_MARKET_IMAGE_BYTES = int(os.getenv("MAX_MARKET_IMAGE_BYTES", str(5 * 1024 * 1024)))
EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


class Base(DeclarativeBase):
    pass


class MarketUser(Base):
    __tablename__ = "market_users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(100))
    password_hash: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class MarketSession(Base):
    __tablename__ = "market_sessions"

    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("market_users.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class Product(Base):
    __tablename__ = "market_products"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    seller_id: Mapped[str] = mapped_column(ForeignKey("market_users.id"), index=True)
    title: Mapped[str] = mapped_column(String(100))
    description: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(50), index=True)
    price_cents: Mapped[int] = mapped_column(Integer)
    quantity: Mapped[int] = mapped_column(Integer)
    shipping_info: Mapped[str] = mapped_column(String(500))
    photo: Mapped[bytes] = mapped_column(LargeBinary)
    active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class Order(Base):
    __tablename__ = "market_orders"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    product_id: Mapped[str] = mapped_column(ForeignKey("market_products.id"), index=True)
    seller_id: Mapped[str] = mapped_column(ForeignKey("market_users.id"), index=True)
    buyer_id: Mapped[str] = mapped_column(ForeignKey("market_users.id"), index=True)
    product_title: Mapped[str] = mapped_column(String(100))
    quantity: Mapped[int] = mapped_column(Integer)
    unit_price_cents: Mapped[int] = mapped_column(Integer)
    total_cents: Mapped[int] = mapped_column(Integer)
    buyer_name: Mapped[str] = mapped_column(String(100))
    buyer_email: Mapped[str] = mapped_column(String(320))
    shipping_address: Mapped[str] = mapped_column(Text)
    notes: Mapped[str] = mapped_column(Text)
    payment_method: Mapped[str] = mapped_column(String(40), default="Da concordare")
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    stock_restored: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


Base.metadata.create_all(bind=engine)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def user_dict(user: MarketUser) -> dict:
    return {"id": user.id, "email": user.email, "display_name": user.display_name}


def password_hash(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, dklen=64)
    return f"{base64.b64encode(salt).decode()}${base64.b64encode(digest).decode()}"


def password_matches(password: str, stored: str) -> bool:
    try:
        salt_text, digest_text = stored.split("$", 1)
        salt = base64.b64decode(salt_text, validate=True)
        expected = base64.b64decode(digest_text, validate=True)
        actual = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, dklen=len(expected))
        return secrets.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


def token_digest(token: str) -> str:
    return hashlib.sha256(token.encode("ascii")).hexdigest()


def create_session(db: Session, user: MarketUser) -> str:
    token = secrets.token_urlsafe(32)
    now = utcnow()
    db.add(MarketSession(token_hash=token_digest(token), user_id=user.id, created_at=now, expires_at=now + timedelta(days=SESSION_DAYS)))
    db.commit()
    return token


def current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer),
    db: Session = Depends(db_session),
) -> MarketUser:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Accedi alla vetrina per continuare.")
    session = db.get(MarketSession, token_digest(credentials.credentials))
    if session is None:
        raise HTTPException(status_code=401, detail="Sessione non valida. Accedi di nuovo.")
    expires = session.expires_at if session.expires_at.tzinfo else session.expires_at.replace(tzinfo=timezone.utc)
    if expires <= utcnow():
        db.delete(session)
        db.commit()
        raise HTTPException(status_code=401, detail="Sessione scaduta. Accedi di nuovo.")
    user = db.get(MarketUser, session.user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="Account non disponibile.")
    return user


def clean_text(value: str, label: str, minimum: int, maximum: int) -> str:
    cleaned = value.strip()
    if not minimum <= len(cleaned) <= maximum:
        raise HTTPException(status_code=422, detail=f"{label}: inserisci da {minimum} a {maximum} caratteri.")
    return cleaned


def money(cents: int) -> str:
    return f"{Decimal(cents) / 100:.2f}"


def product_dict(product: Product, seller_name: str) -> dict:
    return {
        "id": product.id,
        "seller_id": product.seller_id,
        "seller_name": seller_name,
        "title": product.title,
        "description": product.description,
        "category": product.category,
        "price_cents": product.price_cents,
        "price": money(product.price_cents),
        "quantity": product.quantity,
        "shipping_info": product.shipping_info,
        "photo": base64.b64encode(product.photo).decode("ascii"),
        "photo_mime_type": "image/jpeg",
        "active": product.active,
        "created_at": product.created_at.isoformat(),
        "updated_at": product.updated_at.isoformat(),
    }


def order_dict(order: Order) -> dict:
    return {
        "id": order.id,
        "product_id": order.product_id,
        "product_title": order.product_title,
        "seller_id": order.seller_id,
        "buyer_id": order.buyer_id,
        "quantity": order.quantity,
        "unit_price_cents": order.unit_price_cents,
        "total_cents": order.total_cents,
        "total": money(order.total_cents),
        "buyer_name": order.buyer_name,
        "buyer_email": order.buyer_email,
        "shipping_address": order.shipping_address,
        "notes": order.notes,
        "payment_method": order.payment_method,
        "status": order.status,
        "created_at": order.created_at.isoformat(),
        "updated_at": order.updated_at.isoformat(),
    }


def prepare_product_photo(contents: bytes) -> bytes:
    if not contents:
        raise HTTPException(status_code=400, detail="Carica una fotografia del prodotto.")
    if len(contents) > MAX_MARKET_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="La foto del prodotto supera 5 MB.")
    try:
        with Image.open(io.BytesIO(contents)) as source:
            source.verify()
        with Image.open(io.BytesIO(contents)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((1400, 1400), Image.Resampling.LANCZOS)
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=85, optimize=True)
            return output.getvalue()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise HTTPException(status_code=400, detail="La foto del prodotto non è valida.") from exc


class RegisterInput(BaseModel):
    email: str = Field(min_length=5, max_length=320)
    password: str = Field(min_length=8, max_length=128)
    display_name: str = Field(min_length=2, max_length=100)


class LoginInput(BaseModel):
    email: str = Field(min_length=5, max_length=320)
    password: str = Field(min_length=1, max_length=128)


class ProductUpdate(BaseModel):
    title: str = Field(min_length=3, max_length=100)
    description: str = Field(min_length=10, max_length=2000)
    category: str = Field(min_length=2, max_length=50)
    price_cents: int = Field(gt=0, le=10_000_000)
    quantity: int = Field(ge=0, le=999)
    shipping_info: str = Field(min_length=3, max_length=500)
    active: bool = True


class OrderInput(BaseModel):
    product_id: str
    quantity: int = Field(ge=1, le=99)
    buyer_name: str = Field(min_length=2, max_length=100)
    shipping_address: str = Field(min_length=5, max_length=500)
    notes: str = Field(default="", max_length=1000)


class StatusInput(BaseModel):
    status: str


@router.post("/auth/register")
@market_limiter.limit("6/minute")
def register(request: Request, data: RegisterInput, db: Session = Depends(db_session)):
    del request
    email = data.email.strip().lower()
    if not EMAIL_RE.match(email):
        raise HTTPException(status_code=422, detail="Inserisci un indirizzo email valido.")
    name = clean_text(data.display_name, "Nome pubblico", 2, 100)
    if db.scalar(select(MarketUser).where(MarketUser.email == email)):
        raise HTTPException(status_code=409, detail="Questa email è già registrata. Prova ad accedere.")
    user = MarketUser(id=str(uuid.uuid4()), email=email, display_name=name, password_hash=password_hash(data.password), created_at=utcnow())
    db.add(user)
    db.commit()
    return {"token": create_session(db, user), "user": user_dict(user)}


@router.post("/auth/login")
@market_limiter.limit("10/minute")
def login(request: Request, data: LoginInput, db: Session = Depends(db_session)):
    del request
    user = db.scalar(select(MarketUser).where(MarketUser.email == data.email.strip().lower()))
    if user is None or not password_matches(data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Email o password non corretti.")
    return {"token": create_session(db, user), "user": user_dict(user)}


@router.get("/auth/me")
def me(user: MarketUser = Depends(current_user)):
    return {"user": user_dict(user)}


@router.post("/auth/logout")
def logout(credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer), db: Session = Depends(db_session)):
    if credentials is not None:
        session = db.get(MarketSession, token_digest(credentials.credentials))
        if session is not None:
            db.delete(session)
            db.commit()
    return {"status": "ok"}


@router.get("/products")
def public_products(category: str = "", db: Session = Depends(db_session)):
    query = select(Product, MarketUser.display_name).join(MarketUser, MarketUser.id == Product.seller_id).where(Product.active.is_(True))
    if category.strip():
        query = query.where(Product.category == category.strip())
    rows = db.execute(query.order_by(Product.created_at.desc()).limit(100)).all()
    return {"products": [product_dict(product, seller_name) for product, seller_name in rows]}


@router.get("/my-products")
def my_products(user: MarketUser = Depends(current_user), db: Session = Depends(db_session)):
    rows = db.execute(select(Product).where(Product.seller_id == user.id).order_by(Product.created_at.desc())).scalars().all()
    return {"products": [product_dict(product, user.display_name) for product in rows]}


@router.post("/products", status_code=201)
@market_limiter.limit("12/minute")
async def create_product(
    request: Request,
    title: str = Form(...),
    description: str = Form(...),
    category: str = Form(...),
    price_cents: int = Form(...),
    quantity: int = Form(...),
    shipping_info: str = Form(...),
    photo: UploadFile = File(...),
    user: MarketUser = Depends(current_user),
    db: Session = Depends(db_session),
):
    del request
    if not 0 < price_cents <= 10_000_000:
        raise HTTPException(status_code=422, detail="Il prezzo deve essere compreso fra 0,01 e 100.000 euro.")
    if not 0 <= quantity <= 999:
        raise HTTPException(status_code=422, detail="La quantità deve essere compresa fra 0 e 999.")
    contents = await photo.read(MAX_MARKET_IMAGE_BYTES + 1)
    await photo.close()
    now = utcnow()
    product = Product(
        id=str(uuid.uuid4()), seller_id=user.id,
        title=clean_text(title, "Titolo", 3, 100),
        description=clean_text(description, "Descrizione", 10, 2000),
        category=clean_text(category, "Categoria", 2, 50),
        price_cents=price_cents, quantity=quantity,
        shipping_info=clean_text(shipping_info, "Spedizione", 3, 500),
        photo=prepare_product_photo(contents), active=True, created_at=now, updated_at=now,
    )
    db.add(product)
    db.commit()
    return {"product": product_dict(product, user.display_name)}


@router.put("/products/{product_id}")
def update_product(product_id: str, data: ProductUpdate, user: MarketUser = Depends(current_user), db: Session = Depends(db_session)):
    product = db.get(Product, product_id)
    if product is None or product.seller_id != user.id:
        raise HTTPException(status_code=404, detail="Inserzione non trovata.")
    product.title = clean_text(data.title, "Titolo", 3, 100)
    product.description = clean_text(data.description, "Descrizione", 10, 2000)
    product.category = clean_text(data.category, "Categoria", 2, 50)
    product.price_cents = data.price_cents
    product.quantity = data.quantity
    product.shipping_info = clean_text(data.shipping_info, "Spedizione", 3, 500)
    product.active = data.active
    product.updated_at = utcnow()
    db.commit()
    return {"product": product_dict(product, user.display_name)}


@router.delete("/products/{product_id}")
def archive_product(product_id: str, user: MarketUser = Depends(current_user), db: Session = Depends(db_session)):
    product = db.get(Product, product_id)
    if product is None or product.seller_id != user.id:
        raise HTTPException(status_code=404, detail="Inserzione non trovata.")
    product.active = False
    product.updated_at = utcnow()
    db.commit()
    return {"status": "ok"}


@router.post("/orders", status_code=201)
@market_limiter.limit("12/minute")
def create_order(request: Request, data: OrderInput, user: MarketUser = Depends(current_user), db: Session = Depends(db_session)):
    del request
    product = db.scalar(select(Product).where(Product.id == data.product_id).with_for_update())
    if product is None or not product.active:
        raise HTTPException(status_code=404, detail="Questo prodotto non è più disponibile.")
    if product.seller_id == user.id:
        raise HTTPException(status_code=400, detail="Non puoi acquistare una tua inserzione.")
    if data.quantity > product.quantity:
        raise HTTPException(status_code=409, detail="Quantità non disponibile. Aggiorna la vetrina e riprova.")
    now = utcnow()
    order = Order(
        id=str(uuid.uuid4()), product_id=product.id, seller_id=product.seller_id, buyer_id=user.id,
        product_title=product.title, quantity=data.quantity, unit_price_cents=product.price_cents,
        total_cents=product.price_cents * data.quantity,
        buyer_name=clean_text(data.buyer_name, "Nome acquirente", 2, 100),
        buyer_email=user.email,
        shipping_address=clean_text(data.shipping_address, "Indirizzo", 5, 500),
        notes=data.notes.strip(),
        payment_method="Da concordare", status="pending", stock_restored=False, created_at=now, updated_at=now,
    )
    product.quantity -= data.quantity
    product.updated_at = now
    db.add(order)
    db.commit()
    seller = db.get(MarketUser, order.seller_id)
    return {
        "order": {
            **order_dict(order),
            "seller_name": seller.display_name if seller else "Venditore",
            "seller_email": seller.email if seller else "",
        }
    }


@router.get("/purchases")
def purchases(user: MarketUser = Depends(current_user), db: Session = Depends(db_session)):
    rows = db.execute(select(Order, MarketUser.display_name, MarketUser.email).join(MarketUser, MarketUser.id == Order.seller_id).where(Order.buyer_id == user.id).order_by(Order.created_at.desc())).all()
    return {"orders": [{**order_dict(order), "seller_name": name, "seller_email": email} for order, name, email in rows]}


@router.get("/sales")
def sales(user: MarketUser = Depends(current_user), db: Session = Depends(db_session)):
    rows = db.execute(select(Order).where(Order.seller_id == user.id).order_by(Order.created_at.desc())).scalars().all()
    return {"orders": [order_dict(order) for order in rows]}


@router.patch("/orders/{order_id}/status")
def change_order_status(order_id: str, data: StatusInput, user: MarketUser = Depends(current_user), db: Session = Depends(db_session)):
    order = db.scalar(select(Order).where(Order.id == order_id).with_for_update())
    if order is None:
        raise HTTPException(status_code=404, detail="Ordine non trovato.")
    new_status = data.status.strip().lower()
    is_seller = order.seller_id == user.id
    is_buyer = order.buyer_id == user.id
    allowed = False
    if is_seller:
        allowed = (order.status == "pending" and new_status in {"confirmed", "cancelled"}) or (order.status == "confirmed" and new_status in {"shipped", "cancelled"})
    if is_buyer:
        allowed = (order.status == "pending" and new_status == "cancelled") or (order.status == "shipped" and new_status == "completed")
    if not allowed:
        raise HTTPException(status_code=409, detail="Passaggio di stato non consentito per questo ordine.")
    if new_status == "cancelled" and not order.stock_restored:
        product = db.scalar(select(Product).where(Product.id == order.product_id).with_for_update())
        if product is not None:
            product.quantity += order.quantity
            product.updated_at = utcnow()
        order.stock_restored = True
    order.status = new_status
    order.updated_at = utcnow()
    db.commit()
    return {"order": order_dict(order)}
