# Multimodal Medical AI Assistant

A multimodal, bidirectional medical assistant system that integrates specialized medical AI with real-time verification protocols. Built using the Google Agent Development Kit, this system enables accurate evaluation of medical assets such as X-rays, prescriptions, and reports through a distributed multi-agent architecture.

---

## Features

* Real-time bidirectional communication (text, voice, video)
* Multimodal input support (images, prescriptions, reports)
* Medical reasoning using MedGemma
* Verification layer using web-based validation
* Distributed multi-agent system
* Artifact handling via Model Context Protocol (MCP)

---

## Architecture

### 1. Interaction Layer — Gemini Live

* Handles user interaction (text, voice, video)
* Accepts uploads (X-rays, prescriptions, reports)

### 2. Reasoning Layer — MedGemma Agent

* Performs medical inference
* Processes uploaded artifacts using MCP
* Runs on Google Colab (T4 GPU ~8GB VRAM)

### 3. Verification Layer — Search Agent (A2A)

* Performs Agent-to-Agent communication
* Scrapes trusted medical sources
* Cross-checks model outputs

---

## Workflow

1. User interacts via Gemini Live
2. Uploads medical data (image/report/prescription)
3. MedGemma processes and generates response
4. Search Agent verifies using trusted sources
5. Final validated response is returned

---

## Tech Stack

* Frontend: Bun (JavaScript runtime)
* Backend: FastAPI (Python)
* AI Model: MedGemma
* Agent Framework: Google Agent Development Kit
* Protocols: MCP, A2A
* Compute: Google Colab (GPU) / Local

---

## Setup Instructions

### 1. Clone the Repository

```bash
https://github.com/Sadhana-up/CuraNova.git
```

---

### 2. Backend Setup (FastAPI)

```bash
cd backend
python -m venv venv
venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

Run FastAPI server:

```bash
uvicorn main:app --reload
```

Server will run at:

```
http://127.0.0.1:8000
```

---

### 3. Frontend Setup (Bun)

Make sure Bun is installed:

```bash
bun --version
```

Install dependencies:

```bash
cd frontend
bun install
```

Run dev server:

```bash
bun run dev
```

---

### 4. Jupyter / Model Setup (Optional)

```bash
pip install notebook ipykernel
python -m ipykernel install --user --name=medical-ai
jupyter notebook
```

---

### 5. MedGemma Setup (Colab)

1. Open Google Colab
2. Enable GPU: Runtime → Change runtime → T4 GPU
3. Upload model and inference scripts
4. Expose endpoint (optional via ngrok or API bridge)

---

### 6. Environment Variables

Create `.env` file in backend:

```env
GEMINI_API_KEY=your_key_here
SEARCH_API_KEY=your_key_here
MODEL_ENDPOINT=your_colab_or_local_endpoint
```

---

## API Integration Flow

Frontend (Bun) → FastAPI → MedGemma (Colab) → Search Agent → FastAPI → Frontend

---

## Agent Communication

### MCP (Model Context Protocol)

* Handles structured artifacts (images, reports)

### A2A (Agent-to-Agent)

* Enables:

  * MedGemma ↔ Search Agent communication
  * Real-time verification

---

## Limitations

* Not a replacement for professional medical advice
* Requires GPU for optimal performance
* Dependent on external verification sources

---

## Safety

* Responses are verification-backed
* Designed to reduce hallucinations
* Intended for decision support only

---

## Future Improvements

* Local GPU deployment (no Colab dependency)
* Fine-tuned medical datasets
* Offline verification system
* Mobile app integration

---

## Contributing

Feel free to fork and submit pull requests.

---

## License

MIT License
