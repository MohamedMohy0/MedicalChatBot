#  Medical Chatbot API

A **Medical Chatbot** powered by **Ollama LLM** + **RAG (Retrieval-Augmented Generation)**.  
This project allows you to query Arabic medical data and receive concise, helpful information and initial advice.  

> ⚠️ **Note:** The dataset included is minimal and in Arabic. It is intended for demonstration purposes only. For production, you should replace it with a more comprehensive dataset.


## Features
- Query medical data in Arabic.
- Combines **retrieval-based search** and **generative AI** responses.
- Returns concise summaries of symptoms, possible diagnoses, and initial advice.
- Built with **FastAPI** for easy API deployment.
- Uses **SentenceTransformers** for embeddings and **Ollama LLM** for generation.
- Lightweight and easy to host locally.

## Demo
Run the FastAPI server and interact with the bot:

```bash
uvicorn main:app --reload
```

## Installation
1.Clone the repository:
```bash
git clone https://github.com/MohamedMohy0/MedicalChatBot.git
cd MedicalChatBot
```
2.Install dependencies:
```bash
pip install -r requirements.txt
```

3.Ensure you have Ollama installed and running locally.

# Dataset

File: medical_chatbot_data_extended.xlsx
Content: Minimal Arabic medical dataset with columns:

العرض (Symptoms)

وصف إضافي (Additional Description)

احتمالات التشخيص (Possible Diagnoses)

نصائح أولية (Initial Advice)

⚠️ The dataset is small and only for demonstration. You can extend it with your own data in Arabic.
