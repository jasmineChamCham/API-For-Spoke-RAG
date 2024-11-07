from services import qdrant_service
from services.chatgroq_service import *
import json
from sklearn.metrics.pairwise import cosine_similarity
from get_env_variables import getEnvVariables
from services.spoke_service import get_context_using_spoke_api, get_true_disease_name
import numpy as np
from services.load_sentence_transformer import load_sentence_transformer
from services.cache_service import *
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import pandas as pd
from transformers import pipeline

config_data, system_prompts = getEnvVariables()

SYSTEM_PROMPT = system_prompts["KG_RAG_BASED_TEXT_GENERATION"]
CONTEXT_VOLUME = int(config_data["CONTEXT_VOLUME"])
QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD = float(config_data["QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD"])
QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY = float(config_data["QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY"])
VECTOR_DB_PATH = config_data["VECTOR_DB_PATH"]
NODE_CONTEXT_PATH = config_data["NODE_CONTEXT_PATH"]
KG_RAG_FLAG = True # we set false only when RAG without KG
EDGE_EVIDENCE_FLAG = True # Used only when KG_RAG_FLAG=True
SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL = config_data["SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL"]
SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL = config_data["SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL"]

embedding_model = SentenceTransformerEmbeddings(model_name=SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL)
qdrant_client = qdrant_service.load_qdrant(collection_name = 'SPOKE_DISEASES', vector_size=384)
retriever = qdrant_service.vector_data_qdrant(qdrant_client, embedding_model)
embedding_function_for_context_retrieval = load_sentence_transformer(SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL)
qdrant_context = qdrant_service.load_qdrant(collection_name = 'CACHED_CONTEXT', vector_size=768)
retriever_context = qdrant_service.vector_data_qdrant(qdrant_context, embedding_function_for_context_retrieval, collection_name="CACHED_CONTEXT")

qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

def disease_entity_extractor(question):
    chat_bot = create_chat_bot(llm_groq, retriever) # chatbot llm_groq
    resp_text = chat_bot.invoke({"query": question})["result"]
    print(f'resp_text = {resp_text}')

    try:
        json_part = resp_text[resp_text.rfind('{') : (resp_text.rfind('}')+1)]
        entity_dict = json.loads(json_part)
        print(f'entity_dict = {entity_dict}')

        diseases = entity_dict.get("Diseases", [])
        print(f'diseases = {diseases}')

        return diseases
    except  Exception as e:
        return []

def get_relevance_score(question, context):
    result = qa_pipeline(question=question, context=context)
    print(f'result = {result}')
    return result['score']  

def retrieve_context(question, 
                     context_volume = CONTEXT_VOLUME, 
                     edge_evidence=EDGE_EVIDENCE_FLAG):
    entities = disease_entity_extractor(question)
    list_diseases = []
    df = pd.DataFrame()

    if len(entities) > 0:
        max_number_of_relevant_contexts_per_node = int(context_volume/len(entities))
        for entity in entities:
            list_true_disease_names = get_true_disease_name(entity)
            list_diseases.extend(list_true_disease_names)

        node_context_extracted = ""
        df_all = pd.DataFrame()

        for node_name in list_diseases:
            # Search in cache 
            cache_context = search_context_qdrant(qdrant=qdrant_client, embeddings_func=embedding_function_for_context_retrieval, disease_name=node_name)
            cache_graph = search_neighbors_neo4j(node_name)

            if cache_context and cache_graph:
                return "", cache_graph # rag retriever has this info already, no need to return anything

            # Get the context when answer is not stored in cache
            node_context, df = get_context_using_spoke_api(node_name) # node_context here is context gained from all neighboring nodes.
            if df.empty:
                return "UNKNOWN DISEASE", []

            # compare similarity between question and each item of context => just get the ones with highest similarity score
            node_context_list = node_context.split('. ')
            highest_relevant_ctx = []
            highest_relevant_ctx_scores = []

            # Calculate relevance scores for each context
            for ctx in node_context_list:
                score = get_relevance_score(question=question, context=ctx)
                highest_relevant_ctx_scores.append(score)

            percentile_threshold = np.percentile(highest_relevant_ctx_scores, 75)

            high_relevant_indices = [i for i, score in enumerate(highest_relevant_ctx_scores) if score > percentile_threshold and score > 0.7]

            if len(high_relevant_indices) > max_number_of_relevant_contexts_per_node:
                high_relevant_indices = high_relevant_indices[:max_number_of_relevant_contexts_per_node]
            highest_relevant_ctx = [node_context_list[i] for i in high_relevant_indices]

            if edge_evidence:
                highest_relevant_ctx = list(map(lambda x: x + '.', highest_relevant_ctx))
                df = df[df['context'].isin(highest_relevant_ctx)]
                df.loc[:, 'context_with_edge'] = df['source_type'] + " " + df['source_name'] + " " + df['predicate'].str.lower() + " " + df['target_type'] + " " + df['target_name'] + ' and Provenance of this association is ' + df['provenance'] + " and attributes associated with this association is in the following JSON format:\n " + df['evidence'].astype('str') + "\n\n"
                node_context_extracted += df['context'].str.cat(sep=' ')
            else:
                node_context_extracted += ". ".join(highest_relevant_ctx)
                node_context_extracted += ". "

            df.to_csv(f'./data/graph/graph_{node_name}.csv')
            save_df_neo4j(df)

            if df_all.empty:
                df_all = pd.concat([df_all, df.iloc[1:]], ignore_index=True)

            if cache_context == False:
                save_context_qdrant(qdrant=qdrant_client, embeddings_func=embedding_function_for_context_retrieval, disease_name=node_name, context=node_context_extracted)
        
        return node_context_extracted, df_all.to_dict(orient="records")
    else:
        return "UNKNOWN DISEASE", []

