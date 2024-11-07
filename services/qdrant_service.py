from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from langchain_qdrant import QdrantVectorStore

QDRANT_API_KEY = '4x-j-E2jkFlMLn7dasU7rh3yzy5V5vY3Ds9ctZuxd7O4-PiCqBtscQ'
api_key="qdrant"
url_finnish_qdrant = "http://192.168.0.16:16333"
qdrant_T = "https://db01a1a9-113c-4d41-8252-a545fee8a32f.europe-west3-0.gcp.cloud.qdrant.io"


def load_qdrant(collection_name, vector_size = 384):
  client = QdrantClient(api_key=QDRANT_API_KEY, url=qdrant_T)
  distance_metric = rest.Distance.COSINE

  if not client.collection_exists(collection_name):
      client.create_collection(
          collection_name=collection_name,
          vectors_config=rest.VectorParams(
              size=vector_size,
              distance=distance_metric
          )
      )

  return client

def vector_data_qdrant(client, embeddings_func, collection_name='SPOKE_DISEASES'):
    qdrant = QdrantVectorStore(
                client=client,
                collection_name = collection_name,
                embedding=embeddings_func
                )
    retriever = qdrant.as_retriever(search_kwargs={"k": 5})
    return retriever