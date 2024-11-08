from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

groq_api_key = "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"
llm_groq = ChatGroq(temperature=0,
               api_key = groq_api_key,
               model_name="llama3-70b-8192"
               )

api_key_gpt = "key"
llmGPT = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=api_key_gpt,
)

DISEASE_ENTITY_EXTRACTION_prompt = """
    You are an expert disease entity extractor from a sentence and report it as JSON in the following format:
    Diseases: <List of extracted entities>

    Context: {context}
    Question: {question}

    Please report only Diseases. Do not report any other entities like Genes, Proteins, Enzymes etc.
"""

def get_prompt(prompt_template, input_variables=[]):
    prompt = PromptTemplate(
        template=prompt_template, input_variables=input_variables
    )
    return prompt

def create_chat_bot(llm=llmGPT, retriever=[], prompt_template=DISEASE_ENTITY_EXTRACTION_prompt, verbose=False):
    prompt = get_prompt(prompt_template, ["context", "question"])
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": prompt,
            "verbose": verbose,
            "document_variable_name": "context",  
        },
    )
    return qa