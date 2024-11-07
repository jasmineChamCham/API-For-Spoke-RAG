from neo4j import GraphDatabase
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import json
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

def generate_unique_id(disease_name, context):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f'{disease_name}_{context}'))

# DB Neo4j Connection
uri = "bolt://localhost:7687" 
username = "neo4j"
password = "Ngoctram123"
driver = GraphDatabase.driver(uri, auth=(username, password))

# MongoDB Connection
# uri = "mongodb+srv://jasminebkdn:Ngoctram123@kgmongocontextcache.sihqh.mongodb.net/?retryWrites=true&w=majority&appName=KGMongoContextCache"
# qdrant_client = MongoClient(uri, server_api=ServerApi('1'))
# try:
#     qdrant_client.admin.command('ping')
#     db_mongo = qdrant_client["KG_MongoDB"]  
#     collection_mongo = db_mongo["Cache_Context_MongoDB"] 
# except Exception as e:
#     print(e)

def create_nodes_and_relationships(tx, row):
    source_type = row['source_type']
    target_type = row['target_type']

    tx.run(f"""
        MERGE (source:{source_type} {{id: $source_id, name: $source_name, type: $source_type}})
    """, source_id=row['source'], source_name=row['source_name'], source_type=source_type)
    
    # Node Target
    tx.run(f"""
        MERGE (target:{target_type} {{id: $target_id, name: $target_name, type: $target_type}})
    """, target_id=row['target'], target_name=row['target_name'], target_type=target_type)

    # Serialize non-primitive types like dictionaries into strings
    evidence = json.dumps(row['evidence']) if isinstance(row['evidence'], dict) else row['evidence']
    context_with_edge = json.dumps(row['context_with_edge']) if isinstance(row['context_with_edge'], dict) else row['context_with_edge']

    # Create relationship between source and target with properties
    tx.run("""
        MATCH (source {id: $source_id})
        MATCH (target {id: $target_id})
        MERGE (source)-[r:RELATIONSHIP_TYPE {type: $edge_type}]->(target)
        SET r.provenance = $provenance,
            r.evidence = $evidence,
            r.predicate = $predicate,
            r.context = $context,
            r.context_with_edge = $context_with_edge
    """, source_id=row['source'], target_id=row['target'], edge_type=row['edge_type'],
         provenance=row['provenance'], evidence=evidence, predicate=row['predicate'],
         context=row['context'], context_with_edge=context_with_edge)
    
def save_df_neo4j(df):
    with driver.session() as session:
        for _, row in df.iterrows():
            session.write_transaction(create_nodes_and_relationships, row)

def search_neighbors_neo4j(node_name):
    query = """
    MATCH (d:Disease {name: $node_name})-[r:RELATIONSHIP_TYPE]->(neighbor)
    RETURN d, neighbor, r
    """

    with driver.session() as session:
        result = session.run(query, node_name=node_name)

        if not result.peek():
            print(f'No results cache neo4j found for Disease {node_name}')
            return None
            
        cached_graph = []
        for record in result:
            source = dict(record['d'].items())
            target = dict(record['neighbor'].items())
            relationship = dict(record['r'].items())
            
            # Rename source properties
            source_renamed = {
                'source': source['id'],
                'source_name': source['name'],
                'source_type': source['type']
            }
            
            # Rename target properties
            target_renamed = {
                'target': target['id'],
                'target_name': target['name'],
                'target_type': target['type']
            }
            
            data = {
                **source_renamed, 
                **target_renamed, 
                **relationship
            }
            cached_graph.append(data)

        return cached_graph
    
def split_context(context):
    if (len(context) > 0):
        documents = text_splitter.create_documents([context])
        return list(map(lambda x: x.page_content, documents))

def save_context_qdrant(qdrant, embeddings_func, disease_name, context="", collection_name="CACHED_CONTEXT"):
    list_chunk_contexts = split_context(context)
    list_vector_data = []

    for chunk_context in list_chunk_contexts:
        unique_id = generate_unique_id(disease_name, chunk_context)
        data_embedding = embeddings_func.embed_documents([chunk_context])[0]

        vector_data = {
            "id": unique_id,
            "vector": data_embedding,
            "payload": {
                "disease_name": disease_name,
                "context": chunk_context,
            }
        }
        list_vector_data.append(vector_data)
    try:
        qdrant.upsert(collection_name=collection_name, points=list_vector_data)
    except Exception as e:
        print(f"Error upserting document in qdrant: {e}")
        
def search_context_qdrant(qdrant, embeddings_func, disease_name, context="", collection_name="CACHED_CONTEXT"):
    data_embedding = embeddings_func.embed_documents([context])[0]

    results = qdrant.search(
        collection_name=collection_name,
        query_vector=data_embedding,
        limit=3  
    )

    context_qdrant_exists = any(
        result.payload.get("disease_name", {}) == disease_name 
        for result in results
    )

    if context_qdrant_exists:
        return True
    return False 

def save_context_mongodb(node_name, node_context_extracted): # node_name = disease name
    data = {
        'Disease Name': node_name,
        'Context': node_context_extracted
    }
    collection_mongo.insert_one(data)

def search_mongodb(disease_name):
    query = {"Disease Name": disease_name}
    result = collection_mongo.find_one(query)
    return result