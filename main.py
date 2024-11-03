from fastapi import FastAPI
from handler import retrieve_context
from handler import SYSTEM_PROMPT, llm_groq
from typing import Dict, Any

app=FastAPI(
    title="Jasmine RAG API",
    description="APIs to get ontology nodes from Neo4J",
    version="1.0.0"  
)

@app.get("/")
def read_root():
    return {"message": "oke"}

@app.get("/biomedical-response/", response_model=Dict[str, Any])
def generateBiomedicalResponse(question: str):
    """
    Endpoint for answer, context and context graph.
    """
    question = 'Are there any latest drugs used for weight management in patients with Bardet-Biedl syndrome?'
    
    context, df  = retrieve_context(question)

    enriched_prompt = "Context: " + context + "\n" + "Question: " + question
    messages = [
        ( SYSTEM_PROMPT ),
        ( enriched_prompt ),
    ]
    answer = llm_groq.invoke(messages)

    return {
        'answer': answer.content,
        'graph': df.to_dict(orient="records")
    }
