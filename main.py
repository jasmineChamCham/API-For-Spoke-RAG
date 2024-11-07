from fastapi import FastAPI
from handler import retrieve_context, retriever_context
from typing import Dict, Any
from services.chatgroq_service import *
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank

compressor = FlashrankRerank()
def create_compression_retriever(retriever, compressor=compressor):
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    return compression_retriever

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
    # question = 'Are there any latest drugs used for weight management in patients with Bardet-Biedl syndrome?'
    
    context, graph  = retrieve_context(question)
    prompt_answer = """
    You are an expert biomedical researcher. 
    For answering the Question at the end with brevity, you need to first read the Context provided. 
    If the Context says UNKNOWN DISEASE, please say UNKNOWN DISEASE and don't try to answer.
    Otherwise, give your final answer briefly, by citing the Provenance information from the context. 
    You can find Provenance from the Context statement 'Provenance of this association is <Provenance>'. 
    Context: {context}
    Question: {question}
    Do not forget to cite the Provenance information.
    Note that, if Provenance is 'GWAS' report it as 'GWAS Catalog'. If Provenance is 'DISEASES' report it as 'DISEASES database - https://diseases.jensenlab.org'. Additionally, when providing drug or medication suggestions, give maximum information available and then advise the user to seek guidance from a healthcare professional as a precautionary measure.
    Answer the question and provide additional helpful information,
    based on the pieces of information, if applicable. Be succinct.
    """

    compression_retriever_disease = create_compression_retriever(retriever_context)
    chat_bot = create_chat_bot(prompt_template=prompt_answer, retriever=compression_retriever_disease, verbose=False)
    adjusted_question  = "Context: " + context + "\n" + "Question: " + question
    answer = chat_bot.invoke(adjusted_question)

    return  {
        'answer': answer['result'],
        'graph': graph
    }