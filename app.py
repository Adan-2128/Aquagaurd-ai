from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import os
from dotenv import load_dotenv
import asyncio
# Machine Learning
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# LangChain + RAG
from langchain_ibm import WatsonxLLM
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.documents import Document

# ---------------- APP SETUP ----------------
load_dotenv()
app = FastAPI()

if not os.path.exists("static"): os.makedirs("static")
if not os.path.exists("templates"): os.makedirs("templates")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- DATA MANAGEMENT ----------------
DATA_FILE = Path("water_quality_history.json")
CSV_SOURCE = Path("water_dataX.csv")
CHROMA_DIR = "./chroma_water_quality"
FEATURES = ["pH", "Turbidity", "DO", "BOD", "Temperature"]
PRED_STEPS = 12

def load_historical_data():
    if DATA_FILE.exists():
        try:
            df = pd.read_json(DATA_FILE)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df
        except: pass

    if CSV_SOURCE.exists():
        try:
            df = pd.read_csv(CSV_SOURCE, encoding="latin1")
            df = df.rename(columns={"PH": "pH", "Temp": "Temperature", "D.O. (mg/l)": "DO", "B.O.D. (mg/l)": "BOD"})
            if "Turbidity" not in df.columns: df["Turbidity"] = 3.0
            df["timestamp"] = pd.to_datetime(df["year"].astype(str) + "-01-01") if "year" in df.columns else pd.date_range(end=datetime.now(), periods=len(df), freq="H")
            for col in FEATURES:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median() if not df[col].empty else 0)
            return df[FEATURES + ["timestamp"]].tail(100)
        except: pass

    return pd.DataFrame({
        "timestamp": pd.date_range(start=datetime.now()-timedelta(hours=47), periods=48, freq="H"),
        "pH": np.random.uniform(6.5, 8.5, 48), "Turbidity": np.random.uniform(1, 5, 48),
        "DO": np.random.uniform(5.0, 8.0, 48), "BOD": np.random.uniform(1, 3, 48),
        "Temperature": np.random.uniform(20, 30, 48)
    })

historical_data = load_historical_data()

# ---------------- RAG & LLM ----------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(collection_name="water_quality_knowledge", embedding_function=embeddings, persist_directory=CHROMA_DIR)

if vectorstore._collection.count() == 0:
    vectorstore.add_documents([
        Document(page_content="WHO guideline: Dissolved Oxygen (DO) > 5 mg/L is healthy. Below 4 mg/L is critical."),
        Document(page_content="Safe pH range is 6.5 to 8.5. Values below 6.0 are acidic and harmful."),
        Document(page_content="Turbidity should be below 5 NTU. Higher levels indicate suspended solids/pollution."),
        Document(page_content="BOD above 5 mg/L suggests organic pollution; above 10 mg/L is highly contaminated.")
    ])

llm = WatsonxLLM(
    model_id="ibm/granite-4-h-small",
    url=os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com"),
    apikey=os.getenv("WATSONX_APIKEY"),
    project_id=os.getenv("WATSONX_PROJECT_ID"),
    params={
        "max_new_tokens": 200,   # 🔥 increase this (VERY IMPORTANT)
        "temperature": 0.3,      # lower = more stable answers
        "top_p": 0.9
    }
)
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

# ---------------- ROUTES ----------------
class PredictRequest(BaseModel):
    pH: float; Turbidity: float; DO: float; BOD: float; Temperature: float

class ChatRequest(BaseModel):
    message: str

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(data: PredictRequest):
    global historical_data
    try:
        # 1. Update History
        new_row = pd.DataFrame([{"timestamp": datetime.now(), "pH": data.pH, "Turbidity": data.Turbidity, "DO": data.DO, "BOD": data.BOD, "Temperature": data.Temperature}])
        historical_data = pd.concat([historical_data, new_row], ignore_index=True)
        
        # 2. Alert Logic (Check Current Values Immediately)
        alerts = []
        if data.pH < 6.5 or data.pH > 8.5: alerts.append(f"Invalid pH: {data.pH:.2f}")
        if data.DO < 5.0: alerts.append(f"Critical Low Oxygen: {data.DO:.2f} mg/L")
        if data.BOD > 5.0: alerts.append(f"High Organic Pollution (BOD): {data.BOD:.2f} mg/L")
        if data.Turbidity > 5.0: alerts.append(f"High Turbidity: {data.Turbidity:.2f} NTU")

        # 3. Forecast Logic
        X = historical_data[FEATURES].values
        t = np.arange(len(X)).reshape(-1, 1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        n_comp = min(len(X), 3)
        pca = PCA(n_components=n_comp)
        X_pca = pca.fit_transform(X_scaled)

        forecast_steps = np.arange(len(X), len(X) + PRED_STEPS).reshape(-1, 1)
        pca_forecast = []
        for i in range(n_comp):
            model = Ridge(alpha=0.5).fit(t, X_pca[:, i])
            pca_forecast.append(model.predict(forecast_steps))

        pca_forecast = np.array(pca_forecast).T
        X_forecast = scaler.inverse_transform(pca.inverse_transform(pca_forecast))
        forecast_df = pd.DataFrame(X_forecast, columns=FEATURES)

        # 4. Check Forecast for Future Risks
        if forecast_df["DO"].iloc[-1] < 4.0 and data.DO >= 4.0:
            alerts.append("Warning: DO levels predicted to drop to critical levels soon.")

        return {
            "current": historical_data.iloc[-1].to_dict(),
            "forecast": forecast_df.to_dict(orient="records"),
            "forecast_times": [(datetime.now() + timedelta(hours=i+1)).strftime("%H:%M") for i in range(PRED_STEPS)],
            "alerts": alerts if alerts else ["✅ Quality Stable"]
        }
    except Exception as e:
        return {"error": str(e)}
@app.post("/chat")
async def chat(query: ChatRequest):
    try:
        print("Incoming:", query.message)

        # Run QA chain safely (non-blocking)
        response = await asyncio.to_thread(
            qa_chain.invoke,
            {"query": query.message}
        )

        print("Raw response:", response)

        # Extract reply safely
        reply = (
            response.get("result")
            or response.get("answer")
            or response.get("output_text")
        )

        # 🔥 If empty → fallback to LLM with structured prompt
        if not reply or reply.strip() == "":
            print("⚠️ Empty response from QA chain. Falling back to LLM...")

            llm_response = await asyncio.to_thread(
                llm.invoke,
                f"""
You are a water quality expert.

Answer the question clearly.

Guidelines:
- Drinking water pH safe range: 6.5–8.5 (WHO)
- Skin is slightly acidic (~5.5 pH)
- High pH (>8.5) may cause dryness or irritation

Question: {query.message}

Give:
1. Short direct answer
2. Explanation (2-3 lines max)

Answer:
"""
            )

            reply = str(llm_response)

        # ✅ Clean output (removes weird spacing / garbage text)
        if reply:
            reply = " ".join(reply.split())

        # Final safety fallback
        if not reply or reply.strip() == "":
            reply = "Sorry, I couldn't generate a response. Please try again."

        return {"reply": reply}

    except Exception as e:
        print("ERROR:", str(e))
        return {"reply": f"AI Error: {str(e)}"}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)