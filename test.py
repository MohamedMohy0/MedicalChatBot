import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from langchain_ollama import OllamaLLM
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# 1. تحميل البيانات
# ----------------------------
data_path = "medical_chatbot_data_extended.xlsx"
df = pd.read_excel(data_path)

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

# ----------------------------
# 2. Embeddings + FAISS
# ----------------------------
# أفضل للمحتوى العربي
embed_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

embeddings = np.array([embed_model.encode(text) for text in texts], dtype='float32')

d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings)

# ----------------------------
# 3. الموديل
# ----------------------------
llm = OllamaLLM(model="llama3.2")

# ----------------------------
# 4. دالة الرد
# ----------------------------
def ask_bot(query, top_k=3, threshold=0.4):
    query_emb = embed_model.encode(query).astype('float32').reshape(1, -1)
    sims = cosine_similarity(query_emb, embeddings)[0]  

    # جلب أفضل top_k نتائج
    top_indices = np.argsort(sims)[-top_k:][::-1]  
    retrieved_texts = [texts[i] for i in top_indices if sims[i] > threshold]

    if not retrieved_texts:
        return "❌ المعلومة غير متوفرة في قاعدة البيانات الطبية."

    combined_info = "\n\n---\n\n".join(retrieved_texts)

    prompt = f"""
    أنت مساعد طبي افتراضي.
    مهمتك هي تقديم معلومات أولية عامة فقط (ليست تشخيصاً نهائياً).
    اعتمد فقط على البيانات التالية للإجابة:

    {combined_info}

    السؤال: {query}

    المطلوب: 
    - لخص الاحتمالات الطبية المذكورة.
    - أعطِ نصائح أولية عملية.
    - أختم بتنبيه أن المستخدم يجب أن يراجع طبيب مختص.
    """

    response = llm.invoke(prompt)
    return response

# ----------------------------
# 5. شات كونسول
# ----------------------------
print("🤖 شات بوت طبي (اكتب 'خروج' للإنهاء)\n")
while True:
    q = input("🧑‍⚕️ سؤالك: ")
    if q.strip().lower() == "خروج":
        print("تم إنهاء الجلسة ✅")
        break
    answer = ask_bot(q)
    print("🤖 البوت:", answer, "\n")
