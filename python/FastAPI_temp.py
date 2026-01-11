
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


#-------------------------------------------------------------------------
##---- Modulerize code
#-------------------------------------------------------------------------
Add __init__.py in app/, app/routers/, app/models/, and app/schemas/ 
if you prefer traditional Python packages:
touch app/__init__.py app/routers/__init__.py app/models/__init__.py app/schemas/__init__.py


# app/main.py
#---------------------------------------------------------------
from fastapi import FastAPI
from app.routers.etl_mappings import router as etl_router

app = FastAPI(title="ETL Mappings Predictor", version="1.0.0")

# Health/Root endpoint (optional)
@app.get("/")
def root():
    return {"status": "ok", "service": "ETL Mappings Predictor"}

# Include the ETL router with base prefix
app.include_router(etl_router, prefix="/etl/mappings")

app.include_router(script_router, prefix="/etl/scripts")

# app/routers/etl_mappings.py
#---------------------------------------------------------------

from fastapi import APIRouter, HTTPException
from typing import List
import json
from pydantic import ValidationError

from app.schemas.mapping import PredictRequest, MappingPrediction
from app.models.prompt_builder import build_prompt
from app.models.genai_client import predict_with_gemini
from app.schemas.response_schema import mapping_json_schema_dict, mapping_schema_obj

router = APIRouter(tags=["ETL Mappings"])

@router.post("/predict", response_model=List[MappingPrediction])
async def predict_mappings(req: PredictRequest):
    # Build prompt from request
    prompt = build_prompt(req.sourceTable, req.targetTable)

    # Build both SDK-specific and generic JSON schemas
    strict_schema_obj = mapping_schema_obj()
    strict_schema_dict = mapping_json_schema_dict()

    try:
        # Ask Gemini for a structured JSON response
        text = predict_with_gemini(
            prompt=prompt,
            response_schema_obj=strict_schema_obj,
            temperature=0.1,
            model_name="gemini-2.5-flash"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting mappings: {str(e)}")

    if not text:
        raise HTTPException(status_code=502, detail="Empty response from Gemini")

    # Parse and validate against Pydantic models
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail="Invalid JSON returned by model")

    try:
        return [MappingPrediction(**item) for item in raw]
    except ValidationError as ve:
        raise HTTPException(status_code=502, detail=f"Response validation failed: {ve}")

# app/routers/etl_scripts.py
#---------------------------------------------------------------

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

from app.schemas.script import GenerateScriptRequest
from app.models.etl_prompt_builder import build_script_prompt
from app.models.genai_client import generate_text_with_gemini

router = APIRouter(tags=["ETL Scripts"])

@router.post("/generate", response_class=PlainTextResponse)
async def generate_etl_script(req: GenerateScriptRequest):
    """
    Returns ONLY the generated script as text (no JSON wrapper).
    """
    prompt = build_script_prompt(
        source=req.sourceTable,
        target=req.targetTable,
        mappings=req.mappings,
        dialect=req.dialect
    )

    try:
        text = generate_text_with_gemini(
            prompt=prompt,
            model_name="gemini-2.5-flash",
            temperature=0.2,
            response_mime_type="text/plain"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating script: {str(e)}")

    if not text:
        # Mirrors your original fallback
        return "-- No script generated."

    return text

# app/models/genai_client.py
#----------------------------------------------------------------------------------

import os

# Prefer newer google-genai SDK if available
try:
    from google.genai import Client
    GENAI_SDK = "google-genai"
except ImportError:
    GENAI_SDK = None

# Fallback to older google-generativeai library
if GENAI_SDK is None:
    try:
        import google.generativeai as genai
        GENAI_SDK = "google-generativeai"
    except ImportError:
        GENAI_SDK = None

def predict_with_gemini(
    prompt: str,
    response_schema_obj=None,
    temperature: float = 0.1,
    model_name: str = "gemini-2.5-flash",
) -> str:
    """
    Calls Gemini to generate structured JSON text based on the given prompt.
    Returns raw text output (JSON string). Raises exceptions for any errors.
    """
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError("API_KEY is missing from environment variables.")

    if GENAI_SDK == "google-genai":
        # Newer SDK with structured output support
        client = Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "temperature": temperature,
                "response_schema": response_schema_obj,  # may be None
            },
        )
        # For google-genai SDK, text may be under .text or .output_text
        return getattr(response, "text", None) or getattr(response, "output_text", None)

    elif GENAI_SDK == "google-generativeai":
        # Older SDK; schema enforcement done by our post-parse validation
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            prompt,
            generation_config={
                "response_mime_type": "application/json",
                "temperature": temperature,
            },
        )
        return getattr(response, "text", None)

    else:
        raise RuntimeError(
            "No Google GenAI SDK found. Install 'google-genai' or 'google-generativeai'."
        )


def generate_text_with_gemini(
    prompt: str,
    model_name: str = "gemini-2.5-flash",
    temperature: float = 0.2,
    response_mime_type: str = "text/plain"
) -> str:
    """
    Generates plain text (e.g., SQL script) with Gemini.
    Returns raw text output. Raises exceptions for any errors.
    """
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError("API_KEY is missing from environment variables.")

    if GENAI_SDK == "google-genai":
        client = Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={
                "response_mime_type": response_mime_type,
                "temperature": temperature,
            },
        )
        return getattr(response, "text", None) or getattr(response, "output_text", None)

    elif GENAI_SDK == "google-generativeai":
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            prompt,
            generation_config={
                "response_mime_type": response_mime_type,
                "temperature": temperature,
            },
        )
        return getattr(response, "text", None)

    else:
        raise RuntimeError("No Google GenAI SDK found. Install 'google-genai' or 'google-generativeai'.")
``
# app/models/prompt_builder.py
#--------------------------------------------------------------------------------

import json
from app.schemas.mapping import TableSchema

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

# app/models/prompt_builder.py
#--------------------------------------------------------------------------------

import json
from typing import List
from app.schemas.mapping import TableSchema, MappingPrediction

def build_script_prompt(
    source: TableSchema,
    target: TableSchema,
    mappings: List[MappingPrediction],
    dialect: str
) -> str:
    """
    Builds a prompt that instructs Gemini to produce a production-grade ETL script
    for the specified SQL dialect, returning ONLY the code (no fences).
    """
    mapping_json = json.dumps([m.dict() for m in mappings], ensure_ascii=False, indent=2)

    return f"""
Act as a Senior Data Engineer.
Generate a production-grade ETL script to load data from the Source Table to the Target Table.

### Context
Dialect: {dialect}
Source Table: {source.name}
Target Table: {target.name}

### Schema Mapping
{mapping_json}

### Requirements
1. Write a complete, valid script for the specified dialect ({dialect}).
2. Use a MERGE statement (upsert) or INSERT SELECT statement as appropriate for the dialect.
3. If 'sourceFieldName' is null in the mapping, handle it gracefully (e.g., insert NULL or a default value).
4. Apply any logic found in 'transformationSuggestion' from the mapping.
5. Include helpful comments explaining the logic.
6. Return ONLY the code. Do not wrap it in markdown code blocks.
""".strip()

# app/schemas/mapping.py
#----------------------------------------------------------------------------------

from pydantic import BaseModel, Field
from typing import List, Optional, Any

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
    sourceFieldName: Optional[str] = None  # null if no match
    confidence: int = Field(..., ge=0, le=100)
    reasoning: str
    transformationSuggestion: Optional[str] = None

class PredictRequest(BaseModel):
    sourceTable: TableSchema
    targetTable: TableSchema

# app/schemas/response_schema.py
#-----------------------------------------------------------------------------------

def mapping_json_schema_dict() -> dict:
    """Generic JSON schema dictionary for our mapping predictions."""
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

def mapping_schema_obj():
    """
    Builds a google-genai 'Schema' object if the SDK is available.
    Returns None when the SDK is not present; the caller should still
    request JSON output and validate post-parse using Pydantic.
    """
    try:
        from google.genai.types import Schema, Type
    except Exception:
        return None

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

#app/schemas/script.py
#----------------------------------------------------------------------------------
from pydantic import BaseModel
from typing import List
from app.schemas.mapping import TableSchema, MappingPrediction

class GenerateScriptRequest(BaseModel):
    sourceTable: TableSchema
    targetTable: TableSchema
    mappings: List[MappingPrediction]
    dialect: str   # e.g., 'postgres', 'snowflake', 'bigquery', 'sqlserver', 'mysql'

# start_up.py
#----------------------------------------------------------------------------------

# From the project root (fastapi-hello-world/)
python -m venv venv
source venv/bin/activate     # Linux/macOS
# .\venv\Scripts\Activate    # Windows PowerShell

pip install -r requirements.txt

export API_KEY="<YOUR_GOOGLE_GENAI_API_KEY>"  # macOS/Linux
# set API_KEY=<YOUR_GOOGLE_GENAI_API_KEY>     # Windows

uvicorn app.main:app --reload --port 8000

# API Test
#--------------------------------------------------------------------------

curl -X POST "http://localhost:8000/etl/mappings/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "sourceTable": {
      "name": "crm_users",
      "description": "CRM export of user details",
      "fields": [
        {"name": "first_name", "dataType": "string"},
        {"name": "last_name", "dataType": "string"},
        {"name": "email", "dataType": "string"},
        {"name": "dob", "dataType": "date"}
      ]
    },
    "targetTable": {
      "name": "warehouse_customers",
      "description": "Normalized customer dimension",
      "fields": [
        {"name": "full_name", "dataType": "string"},
        {"name": "email_address", "dataType": "string"},
        {"name": "birth_date", "dataType": "date"}
      ]
    }
  }'


curl -X POST "http://localhost:8000/etl/scripts/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "sourceTable": {
      "name": "crm_users",
      "description": "CRM export of user details",
      "fields": [
        {"name": "first_name", "dataType": "string"},
        {"name": "last_name", "dataType": "string"},
        {"name": "email", "dataType": "string"},
        {"name": "dob", "dataType": "date"}
      ]
    },
    "targetTable": {
      "name": "warehouse_customers",
      "description": "Normalized customer dimension",
      "fields": [
        {"name": "customer_name", "dataType": "string"},
        {"name": "email_address", "dataType": "string"},
        {"name": "birth_date", "dataType": "date"}
      ]
    },
    "mappings": [
      {
        "targetFieldName": "customer_name",
        "sourceFieldName": "first_name",
        "confidence": 92,
        "reasoning": "Combine first and last for full name",
        "transformationSuggestion": "CONCAT(first_name, ' ', last_name)"
      },
      {
        "targetFieldName": "email_address",
        "sourceFieldName": "email",
        "confidence": 98,
        "reasoning": "Exact match",
        "transformationSuggestion": null
      },
      {
        "targetFieldName": "birth_date",
        "sourceFieldName": "dob",
        "confidence": 95,
        "reasoning": "Alias mapping",
        "transformationSuggestion": null
      }
    ],
    "dialect": "postgres"
  }'
``
