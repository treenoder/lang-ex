"""FastAPI web interface for inspecting and chatting with trained models."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from tiny_llm.catalog import ModelCatalog
from tiny_llm.inference import GenerationConfig, generate

WEB_DIR = Path(__file__).parent
catalog = ModelCatalog()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # The cache holds only one model to keep this laptop application's RAM predictable.
    app.state.loaded = None
    yield
    app.state.loaded = None


app = FastAPI(title="Tiny LLM Lab", version="0.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")
templates = Jinja2Templates(directory=WEB_DIR / "templates")


class ChatRequest(BaseModel):
    model: str
    prompt: str = Field(min_length=1, max_length=8_000)
    max_new_tokens: int = Field(100, ge=1, le=512)
    temperature: float = Field(0.8, ge=0, le=2)
    top_k: int = Field(40, ge=1, le=500)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/api/models")
def models():
    return [
        {
            "name": item.name,
            "architecture": item.architecture,
            "size": item.size,
            "parameters": item.parameters,
            "formatted_parameters": item.formatted_parameters,
            "updated_at": item.updated_at,
        }
        for item in catalog.list()
    ]


@app.post("/api/chat")
def chat(body: ChatRequest, request: Request):
    known = {item.name for item in catalog.list()}
    if body.model not in known:
        raise HTTPException(status_code=404, detail="Model not found")
    loaded = request.app.state.loaded
    if loaded is None or loaded[0] != body.model:
        try:
            model, tokenizer, _ = catalog.load(body.model, device="cpu")
        except (OSError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=500, detail=f"Could not load model: {exc}") from exc
        request.app.state.loaded = (body.model, model, tokenizer)
    _, model, tokenizer = request.app.state.loaded
    result = generate(
        model,
        tokenizer,
        body.prompt,
        GenerationConfig(body.max_new_tokens, body.temperature, body.top_k),
    )
    return {"text": result, "model": body.model}
