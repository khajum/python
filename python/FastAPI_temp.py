
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Any
import os
import json

# ---- Google GenAI (newer SDK) ----
try:
    from google.genai import Client
    from google.genai.types import Schema, Type
    GENAI_SDK = "google-genai"
except ImportError:
    GENAI_SDK = None

# ---- Fallback to older library if needed ----
if GENAI_SDK is None:
    try:
        import google.generativeai as genai
        GENAI_SDK = "google-generativeai"
    except ImportError:
        GENAI_SDK = None

app = FastAPI(title="ETL Mappings Predictor", version="1.0.0")


# ----------------------------
# Pydantic models (request/response)
# ----------------------------
class FieldSchema(BaseModel):
    name: str
    dataType: Optional[str] = None
    description: Optional[str] = None
    sampleValues: Optional[List[Any]] = None


class TableSchema(BaseModel):
    name: str
    description: Optional[str] = None
    fields: List[FieldSchema]


class MappingPrediction(BaseModel):
    targetFieldName: str
    sourceFieldName: Optional[str] = None  # can be null if no match
    confidence: int = Field(..., ge=0, le=100)
    reasoning: str
    transformationSuggestion: Optional[str] = None


class PredictRequest(BaseModel):
    sourceTable: TableSchema
    targetTable: TableSchema


# ----------------------------
# JSON schema for structured output
# ----------------------------
def mapping_json_schema_dict() -> dict:
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "targetFieldName": {"type": "string"},
                "sourceFieldName": {"type": ["string", "null"]},
                "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
                "reasoning": {"type": "string"},
                "transformationSuggestion": {"type": ["string", "null"]},
            },
            "required": ["targetFieldName", "sourceFieldName", "confidence", "reasoning"],
        },
    }


def mapping_schema_obj() -> Optional[Schema]:
    # Only build Schema object if using google-genai
    if GENAI_SDK == "google-genai":
        return Schema(
            type=Type.ARRAY,
            items=Schema(
                type=Type.OBJECT,
                properties={
                    "targetFieldName": Schema(type=Type.STRING),
                    "sourceFieldName": Schema(type=Type.STRING, nullable=True),
                    "confidence": Schema(type=Type.INTEGER),
                    "reasoning": Schema(type=Type.STRING),
                    "transformationSuggestion": Schema(type=Type.STRING, nullable=True),
                },
                required=["targetFieldName", "sourceFieldName", "confidence", "reasoning"],
            ),
        )
    return None


# ----------------------------
# Prompt builder
# ----------------------------
def build_prompt(source: TableSchema, target: TableSchema) -> str:
    return f"""
You are an expert Data Engineer specializing in ETL (Extract, Transform, Load) processes.

Your task is to map fields from a **Source Table** to a **Target Table**.

### Source Table: {source.name}
Description: {source.description or ""}
Fields:
{json.dumps([f.dict() for f in source.fields], ensure_ascii=False, indent=2)}

### Target Table: {target.name}
Description: {target.description or ""}
Fields:
{json.dumps([f.dict() for f in target.fields], ensure_ascii=False, indent=2)}

### Instructions:
1. For EACH field in the Target Table, find the best corresponding field in the Source Table.
2. Consider field names, data types, descriptions, and sample values.
3. If multiple source fields are needed (e.g., First Name + Last Name -> Full Name), specify the primary source field in 'sourceFieldName' and explain the combination in 'transformationSuggestion'.
4. If no suitable match exists, set 'sourceFieldName' to null.
5. Provide a confidence score (0-100) and brief reasoning.

Return the output strictly as a JSON array adhering to the schema.
""".strip()


# ----------------------------
# Endpoint
# ----------------------------
@app.post("/etl/mappings/predict", response_model=List[MappingPrediction])
async def predict_mappings(req: PredictRequest):
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="API_KEY is missing from environment variables.")

    prompt = build_prompt(req.sourceTable, req.targetTable)

    # Prefer newer google-genai SDK if available
    if GENAI_SDK == "google-genai":
        try:
            client = Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "temperature": 0.1,
                    "response_schema": mapping_schema_obj(),  # strict schema
                },
            )

            text = getattr(response, "text", None) or getattr(response, "output_text", None)
            if not text:
                raise HTTPException(status_code=502, detail="Empty response from Gemini")

            try:
                raw = json.loads(text)
            except json.JSONDecodeError:
                raise HTTPException(status_code=502, detail="Invalid JSON returned by model")

            # Validate against our response model
            try:
                return [MappingPrediction(**item) for item in raw]
            except ValidationError as ve:
                raise HTTPException(status_code=502, detail=f"Response validation failed: {ve}")

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error predicting mappings: {str(e)}")

    # Fallback: older google.generativeai library
    elif GENAI_SDK == "google-generativeai":
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.5-flash")  # adjust if not available
            response = model.generate_content(
                prompt,
                generation_config={
                    "response_mime_type": "application/json",
                    "temperature": 0.1,
                },
            )
            # Older SDK uses .text
            text = getattr(response, "text", None)
            if not text:
                raise HTTPException(status_code=502, detail="Empty response from Gemini")

            try:
                raw = json.loads(text)
            except json.JSONDecodeError:
                raise HTTPException(status_code=502, detail="Invalid JSON returned by model")

            try:
                return [MappingPrediction(**item) for item in raw]
            except ValidationError as ve:
                raise HTTPException(status_code=502, detail=f"Response validation failed: {ve}")

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error predicting mappings: {str(e)}")

    else:
        raise HTTPException(
            status_code=500,
            detail=(
                "No Google GenAI SDK found. Please install either 'google-genai' "
                "or 'google-generativeai'."
            ),
        )
