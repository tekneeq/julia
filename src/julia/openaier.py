from openai import OpenAI
from dotenv import load_dotenv

import chromadb
import os

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)
chroma = chromadb.Client()
collection = chroma.create_collection(name="sentences")

# Feed in your sentences
sentences = [
    "Helm charts are templates for Kubernetes resources.",
    "Jenkins automates CI/CD pipelines.",
    "Nutanix Prism Central manages virtual clusters.",
]

# Embed and store
for i, s in enumerate(sentences):
    emb = (
        client.embeddings.create(model="text-embedding-3-small", input=s)
        .data[0]
        .embedding
    )

    collection.add(
        ids=[f"sent_{i}"],  # 👈 REQUIRED: unique ID
        documents=[s],
        embeddings=[emb],
    )

# Ask a question
question = "Which sentences mention Kubernetes?"
q_emb = (
    client.embeddings.create(model="text-embedding-3-small", input=question)
    .data[0]
    .embedding
)

# Retrieve top matches
results = collection.query(query_embeddings=[q_emb], n_results=3)
context = " ".join(results["documents"][0])

# Ask GPT-5 using retrieved context
answer = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {
            "role": "system",
            "content": "You are an assistant that answers based on context.",
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        },
    ],
)

print(answer.choices[0].message.content)
