#from fastapi import FastAPI
import os
import json
import logging
from typing import List, Optional, Literal, Dict, Any
from difflib import SequenceMatcher

from fastapi import FastAPI, HTTPException, Body, Query
from pydantic import BaseModel, Field, conlist

# Optional: Providers
try:
    import google.generativeai as genai
    HAS_GENAI = True
except Exception:
    HAS_GENAI = False

try:
    from openai import AzureOpenAI  # pip install openai
    HAS_AZURE_OPENAI = True
except Exception:
    HAS_AZURE_OPENAI = False


# -----------------------------
# Models
# -----------------------------
class FieldModel(BaseModel):
    name: str = Field(..., description="Column name")
    type: Optional[str] = Field(None, description="Column type, e.g., VARCHAR, INT")

class TableModel(BaseModel):
    schema: str = Field(..., description="Schema name")
    name: str = Field(..., description="Table name")
    fields: conlist(FieldModel, min_items=1) = Field(..., description="List of fields")

class MappingEntryModel(BaseModel):
    sourceFieldName: str
    targetFieldName: str
    transformRule: Optional[str] = Field(None, description="SQL fragment or 'Direct'")
    confidence: float = Field(..., ge=0.0, le=1.0)

class PredictRequest(BaseModel):
    sourceTable: TableModel
    targetTable: TableModel

class PredictResponse(BaseModel):
    mappings: List[MappingEntryModel]


# -----------------------------
# Provider Interface
# -----------------------------
class MappingsProvider:
    def predict(self, source: TableModel, target: TableModel) -> List[MappingEntryModel]:
        raise NotImplementedError


class GeminiProvider(MappingsProvider):
    def __init__(self, api_key: str, model_name: str = "gemini-3-flash-preview"):
        if not HAS_GENAI:
            raise RuntimeError("google-generativeai not available. `pip install google-generativeai`")
        genai.configure(api_key=api_key)
        self.model_name = model_name

    def predict(self, source: TableModel, target: TableModel) -> List[MappingEntryModel]:
        prompt = f"""
Task: Map source table fields to target table fields for an ETL process.

Rules:
1. Identify semantic matches (e.g., 'F_NAME' maps to 'first_name').
2. Suggest transformation rules if names or types differ (e.g., casting, concatenation, or case conversion).
3. Provide a confidence score between 0.0 and 1.0.

Source Table: {source.schema}.{source.name}
Fields: {json.dumps([f.dict() for f in source.fields])}

Target Table: {target.schema}.{target.name}
Fields: {json.dumps([f.dict() for f in target.fields])}

Return a JSON array of mapping objects strictly following:
[
  {{
    "sourceFieldName": "string",
    "targetFieldName": "string",
    "transformRule": "SQL fragment or 'Direct'",
    "confidence": 0.0-1.0
  }}
]
Only return valid JSON. No commentary.
        """.strip()

        model = genai.GenerativeModel(model_name=self.model_name)
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json", "temperature": 0}
        )

        raw_text = getattr(response, "text", None) or ""
        if not raw_text and hasattr(response, "candidates") and response.candidates:
            parts = response.candidates[0].content.parts
            if parts and hasattr(parts[0], "text"):
                raw_text = parts[0].text

        if not raw_text:
            return []

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            return []

        result: List[MappingEntryModel] = []
        if isinstance(data, list):
            for item in data:
                s = item.get("sourceFieldName")
                t = item.get("targetFieldName")
                c = item.get("confidence")
                tr = item.get("transformRule")
                if s and t and isinstance(c, (int, float)):
                    result.append(
                        MappingEntryModel(
                            sourceFieldName=s,
                            targetFieldName=t,
                            transformRule=tr,
                            confidence=float(c),
                        )
                    )
        return result


class AzureOpenAIProvider(MappingsProvider):
    """
    Optional Azure OpenAI provider. Requires:
      - AZURE_OPENAI_API_KEY
      - AZURE_OPENAI_ENDPOINT
      - AZURE_OPENAI_DEPLOYMENT (model deployment name)
    """
    def __init__(self, api_key: str, endpoint: str, deployment: str):
        if not HAS_AZURE_OPENAI:
            raise RuntimeError("Azure OpenAI client not available. `pip install openai`")
        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-01-preview"),
        )
        self.deployment = deployment

    def predict(self, source: TableModel, target: TableModel) -> List[MappingEntryModel]:
        system = (
            "You are an ETL mapping assistant. Return ONLY valid JSON array of mapping objects. "
            "Follow the requested schema strictly."
        )
        user = f"""
Task: Map source table fields to target table fields for an ETL process.

Rules:
1. Identify semantic matches (e.g., 'F_NAME' maps to 'first_name').
2. Suggest transformation rules if names or types differ (e.g., casting, concatenation, or case conversion).
3. Provide a confidence score between 0.0 and 1.0.

Source Table: {source.schema}.{source.name}
Fields: {json.dumps([f.dict() for f in source.fields])}

Target Table: {target.schema}.{target.name}
Fields: {json.dumps([f.dict() for f in target.fields])}

Return a JSON array of mapping objects strictly following:
[
  {{
    "sourceFieldName": "string",
    "targetFieldName": "string",
    "transformRule": "SQL fragment or 'Direct'",
    "confidence": 0.0-1.0
  }}
]
Only return valid JSON. No commentary.
        """.strip()

        resp = self.client.chat.completions.create(
            model=self.deployment,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0,
            response_format={"type": "json_object"},  # ensures JSON object; we'll accept array inside
        )

        text = resp.choices[0].message.content if resp.choices else ""
        if not text:
            return []

        try:
            data = json.loads(text)
            # If the model wrapped the array in an object, try common keys
            if isinstance(data, dict):
                for key in ("mappings", "data", "result"):
                    if key in data and isinstance(data[key], list):
                        data = data[key]
                        break
        except json.JSONDecodeError:
            return []

        result: List[MappingEntryModel] = []
        if isinstance(data, list):
            for item in data:
                s = item.get("sourceFieldName")
                t = item.get("targetFieldName")
                c = item.get("confidence")
                tr = item.get("transformRule")
                if s and t and isinstance(c, (int, float)):
                    result.append(MappingEntryModel(
                        sourceFieldName=s,
                        targetFieldName=t,
                        transformRule=tr,
                        confidence=float(c)
                    ))
        return result


# -----------------------------
# Heuristic Fallback
# -----------------------------
def normalize_name(s: str) -> str:
    return s.strip().lower().replace("-", "_")

def tokenized(s: str) -> List[str]:
    return [t for t in normalize_name(s).replace("__", "_").split("_") if t]

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_name(a), normalize_name(b)).ratio()

def suggest_transform(src_type: Optional[str], tgt_type: Optional[str], src_name: str, tgt_name: str) -> str:
    if src_type and tgt_type and normalize_name(src_type) != normalize_name(tgt_type):
        return f"CAST({src_name} AS {tgt_type})"
    # simple case conversion suggestion
    if "name" in normalize_name(tgt_name):
        return "INITCAP" if any(k in normalize_name(src_name) for k in ["f", "l", "name"]) else "Direct"
    return "Direct"

def heuristic_predict(source: TableModel, target: TableModel) -> List[MappingEntryModel]:
    mappings: List[MappingEntryModel] = []
    tgt_fields = target.fields

    for sf in source.fields:
        candidates = []
        for tf in tgt_fields:
            # token overlap + similarity
            overlap = len(set(tokenized(sf.name)).intersection(set(tokenized(tf.name))))
            sim = similarity(sf.name, tf.name)
            score = min(1.0, 0.6 * sim + 0.4 * (overlap / max(1, len(tokenized(tf.name)))))
            candidates.append((tf, score))

        if not candidates:
            continue

        tf_best, conf = sorted(candidates, key=lambda x: x[1], reverse=True)[0]
        tr = suggest_transform(sf.type, tf_best.type, sf.name, tf_best.name)
        mappings.append(MappingEntryModel(
            sourceFieldName=sf.name,
            targetFieldName=tf_best.name,
            transformRule=tr,
            confidence=round(conf, 3)
        ))
    return mappings


# -----------------------------
# App & Config
# -----------------------------
app = FastAPI(title="ETL Mapping Service", version="1.0.0")
logger = logging.getLogger("uvicorn.error")

PROVIDER: Literal["gemini", "azure"] = os.getenv("PROVIDER", "gemini").lower()  # default gemini

def get_provider() -> MappingsProvider:
    if PROVIDER == "azure":
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        if not all([api_key, endpoint, deployment]):
            raise RuntimeError("Missing Azure OpenAI env vars: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT")
        return AzureOpenAIProvider(api_key=api_key, endpoint=endpoint, deployment=deployment)

    # default: gemini
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY env var.")
    return GeminiProvider(api_key=api_key, model_name=os.getenv("GEMINI_MODEL", "gemini-3-flash-preview"))

@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "provider": PROVIDER}

@app.post("/etl/mappings/predict", response_model=PredictResponse)
def predict_mappings_endpoint(
    req: PredictRequest = Body(...),
    use_fallback: bool = Query(True, description="Use heuristic fallback when LLM yields nothing."),
) -> PredictResponse:
    """
    Predict field mappings between source and target tables for an ETL process.
    Uses LLM provider first; falls back to heuristic if enabled and necessary.
    """
    try:
        provider = get_provider()
    except Exception as e:
        logger.error(f"Provider configuration error: {e}")
        if use_fallback:
            # Proceed with heuristic only
            mappings = heuristic_predict(req.sourceTable, req.targetTable)
            return PredictResponse(mappings=mappings)
        raise HTTPException(status_code=500, detail=str(e))

    try:
        # First attempt: LLM
        mappings = provider.predict(req.sourceTable, req.targetTable)
        if (not mappings) and use_fallback:
            mappings = heuristic_predict(req.sourceTable, req.targetTable)
        return PredictResponse(mappings=mappings)
    except Exception as e:
        logger.exception(f"Mapping prediction error: {e}")
        if use_fallback:
            mappings = heuristic_predict(req.sourceTable, req.targetTable)
            return PredictResponse(mappings=mappings)
        raise HTTPException(status_code=500, detail="Mapping prediction failed without fallback.")
``


"""
app = FastAPI()

@app.get("/hello")
def hello_world():
    return {"message": "Hello Ram!!"}

"""
