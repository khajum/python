# Project Setup Guide

---

## 1️⃣ Prerequisites

Make sure you have:

- Python 3.8+
- pip available

Check versions:

```bash
python --version
pip --version
```

## 2️⃣ Create a project folder
```bash
mkdir project-name
cd project-name
```

(Optional but recommended)

```bash
python -m venv venv
venv\Scripts\activate   # for Windows
#source venv/bin/activate  # for macOS/Linux
```

## 3️⃣ Install FastAPI and Uvicorn
```bash
pip install fastapi uvicorn
```
## 4️⃣ Create main.py
```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def hello_world():
    return {"message": "Hello World"}
```

## 5️⃣ Run the application
```bash
uvicorn main:app --reload
```

You should see output like:

Uvicorn running on http://127.0.0.1:8000

## 6️⃣ Test it

Open browser: 👉 http://127.0.0.1:8000/

Response:

{"message":"Hello World"}

## 7️⃣ Built-in API documentation (FastAPI magic ✨)

FastAPI automatically generates docs:

Swagger UI → http://127.0.0.1:8000/docs

ReDoc → http://127.0.0.1:8000/redoc

This is very useful for QA, API testing, and automation.

## 8️⃣ Recommended project structure (next step)
```
fastapi-hello-world/
│── app/
│   ├── main.py
│   ├── routers/
│   ├── models/
│   └── schemas/
│── venv/
│── requirements.txt
```

Generate requirements.txt:

pip freeze > requirements.txt
