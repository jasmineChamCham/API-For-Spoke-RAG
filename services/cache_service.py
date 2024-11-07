from neo4j import GraphDatabase
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import json

# DB Neo4j Connection
uri = "bolt://localhost:7687" 
username = "neo4j"
password = "Ngoctram123"
driver = GraphDatabase.driver(uri, auth=(username, password))

# MongoDB Connection
uri = "mongodb+srv://jasminebkdn:Ngoctram123@kgmongocontextcache.sihqh.mongodb.net/?retryWrites=true&w=majority&appName=KGMongoContextCache"
mongo_client = MongoClient(uri, server_api=ServerApi('1'))
try:
    mongo_client.admin.command('ping')
    print("Pinged your deployed MongoDB. You successfully connected to KG Cache Context MongoDB!")
    db_mongo = mongo_client["KG_MongoDB"]  
    collection_mongo = db_mongo["Cache_Context_MongoDB"] 
except Exception as e:
    print(e)

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
            print(json.dumps(data, indent=5))
            cached_graph.append(data)

        return cached_graph


def save_context_mongodb(node_name, node_context_extracted): # node_name = disease name
    data = {
        'Disease Name': node_name,
        'Context': node_context_extracted
    }
    collection_mongo.insert_one(data)
    print("Inserted context document of the disease:", data)

def search_mongodb(disease_name):
    query = {"Disease Name": disease_name}
    result = collection_mongo.find_one(query)

    print(f'result from search cached context from mongodb: {result}')
    return result