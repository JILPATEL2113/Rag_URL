# URL RAG: kisi bhi page se questions pucho

User URL dalta hai, page load hota hai, phir us page ke baare me questions puch sakta hai.
Answer ke saath wo passages bhi dikhte hain jinse answer bana.

**Stack:** FastAPI + simple HTML (no build step) + TF-IDF retrieval + Google Gemini API (answer generation)

## Folder structure

```
url-rag/
├── app/
│   ├── __init__.py
│   └── main.py          # backend (load, retrieve, ask)
├── static/
│   └── index.html       # poora frontend
├── requirements.txt
├── render.yaml
└── .env.example
```

## Ye kaise kaam karta hai

1. `POST /api/load`: URL fetch -> text nikala -> ~180 words ke chunks -> TF-IDF index (memory me)
2. `POST /api/ask`: question se top 5 chunks -> Gemini ko context ke saath bheja -> answer + sources
3. Answer me `[1]`, `[2]` citations aate hain, frontend unhe clickable chips bana deta hai

Retrieval ke liye TF-IDF isliye liya hai ki Render free plan (512 MB RAM) par heavy embedding model nahi chalta.

## Local par chalao

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
export GEMINI_API_KEY=your-key   # Windows: set GEMINI_API_KEY=your-key
uvicorn app.main:app --reload
```

Browser me kholo: http://localhost:8000

## Render par live karo

1. Ye folder GitHub repo me push karo.
2. Render dashboard -> **New** -> **Blueprint** -> apna repo chuno (`render.yaml` apne aap detect hoga).
   Ya **New Web Service** me manually ye daalo:
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
3. **Environment** me `GEMINI_API_KEY` add karo (https://aistudio.google.com/apikey se key milegi).
4. Deploy hone ke baad Render ka URL kholo. Done.

## Dhyan rakhne wali baatein

- **Free plan sleep hota hai:** 15 min idle ke baad server so jaata hai, pehla request ~30-60 sec leta hai.
- **Sessions memory me hain:** server restart/sleep hone par loaded page chala jaata hai. UI user ko bol deta hai "page dobara load karo". Permanent store chahiye to Redis ya database add karna padega.
- **JavaScript se bane pages** (React/SPA sites) ka content plain fetch me nahi aata, "No readable text" error aayega.
- **Sirf HTML/text pages** supported hain. PDF support alag se add karna padega (`pypdf`).
- **Security:** private/internal IPs block hain (SSRF protection) aur har IP par rate limit hai (load 10/min, ask 15/min). Public app hai to API key ka kharcha dhyan me rakho.
- **Model badalna ho** to `GEMINI_MODEL` env variable change karo (default: `gemini-3.6-flash`). Agar model name ka error aaye to AI Studio me current Flash model ka naam dekhkar yahan daal do.

## Aage kya add kar sakte ho

- Semantic search (embeddings + vector store) taaki meaning ke basis par match ho, sirf words par nahi
- Ek se zyada URLs ek saath
- Streaming answers
- PDF upload
