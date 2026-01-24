# python

<details>
 <summary> <h2>Python Virtual Environment (venv)</h2></summary>

### What is a Virtual Environment (`venv`)?
A **virtual environment** is an **isolated Python runtime** with its own Python
interpreter and installed packages, completely separate from the system Python.

Each project gets its **own dependency environment**.

---

### Why do we need `venv`?

#### 1. Dependency Isolation
Different projects may require different versions of the same library.

Example:
- Project A → `django==3.2`
- Project B → `django==5.0`

Using `venv` avoids version conflicts by isolating dependencies per project.

---

#### 2. Prevents “Works on My Machine” Issues
- Ensures consistent dependency versions
- Improves reproducibility across developers, QA, and CI/CD pipelines

---

#### 3. Protects System Python
Installing packages globally can:
- Break OS-level tools
- Require admin permissions

`venv` keeps the system Python clean and stable.

---

#### 4. Industry Standard Practice
Virtual environments are a best practice for:
- Team-based development
- CI/CD pipelines
- Cloud and production deployments

---

### How does `venv` work?

#### 1. Create a Virtual Environment
```bash
python -m venv venv
 ```
### 2. Activate the Environment

**Windows (PowerShell)**
```bash
.\venv\Scripts\activate
```
### 3. Install Dependencies
```bash
pip install fastapi uvicorn

```
Packages are installed <strong>only inside the virtual environment.</strong>

### 4. Freeze Dependencies
```bash
pip freeze > requirements.txt
```

Used to reproduce the same environment across machines and pipelines.

### 5. Deactivate the Environment
```bash
deactivate
```
</details>
