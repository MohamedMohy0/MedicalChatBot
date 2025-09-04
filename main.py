import os
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_ollama import OllamaLLM
from fastapi.middleware.cors import CORSMiddleware
# ----------------------------
# 1) إعداد FastAPI
# ----------------------------
app = FastAPI(title="Medical Chatbot API", description="🩺 Medical Chatbot powered by Ollama + RAG", version="1.0")
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
DATA_PATH = "medical_chatbot_data_extended.xlsx"
MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.2")

# ----------------------------
# 2) تحميل البيانات
# ----------------------------
def load_data_and_prepare():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"لم يتم العثور على ملف البيانات: {DATA_PATH}")

    df = pd.read_excel(DATA_PATH)

    def preprocess_data(df):
        texts = []
        for _, row in df.iterrows():
            text = (
                f"العرض: {row['العرض']}\n"
                f"الوصف: {row['وصف إضافي']}\n"
                f"احتمالات التشخيص: {row['احتمالات التشخيص']}\n"
                f"نصائح أولية: {row['نصائح أولية']}"
            )
            texts.append(text)
        return texts

    texts = preprocess_data(df)

    embed_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=True).astype("float32")

    return df, texts, embed_model, embeddings

df, texts, embed_model, embeddings = load_data_and_prepare()

# ----------------------------
# 3) تهيئة نموذج LLM عبر Ollama
# ----------------------------
llm = OllamaLLM(model=MODEL_NAME)

# ----------------------------
# 4) دالة الرد
# ----------------------------
def ask_bot(query: str, top_k: int = 3, threshold: float = 0.4):
    query_emb = embed_model.encode(query, convert_to_numpy=True).astype("float32").reshape(1, -1)
    sims = cosine_similarity(query_emb, embeddings)[0]

    # أفضل top_k نتائج
    top_indices = np.argsort(sims)[-top_k:][::-1]
    retrieved_texts = [texts[i] for i in top_indices if float(sims[i]) > threshold]

    if not retrieved_texts:
        return "❌ المعلومة غير متوفرة في قاعدة البيانات الطبية."

    combined_info = "\n\n---\n\n".join(retrieved_texts)

    prompt = f"""
    أنت مساعد طبي افتراضي.
    مهمتك تقديم معلومات أولية عامة فقط (ليست تشخيصاً نهائياً).
    اعتمد فقط على البيانات التالية للإجابة:

    {combined_info}

    السؤال: {query}

    المطلوب:
    - لخص الاحتمالات الطبية المذكورة.
    - أعطِ نصائح أولية عملية.
    - أختم بتنبيه صريح لمراجعة طبيب مختص.
    """

    try:
        response = llm.invoke(prompt)
    except Exception as ex:
        response = f"⚠️ حدث خطأ أثناء استدعاء النموذج: {ex}"
    return response

# ----------------------------
# 5) تعريف الـ API Endpoints
# ----------------------------
class Query(BaseModel):
    question: str
    top_k: int = 3
    threshold: float = 0.4

@app.get("/")
def home():
    return {"message": "🩺 Medical Chatbot API is running!"}

@app.post("/chat")
def chat(query: Query):
    answer = ask_bot(query.question, top_k=query.top_k, threshold=query.threshold)
    return {"question": query.question, "answer": answer}
