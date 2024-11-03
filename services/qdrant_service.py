from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from langchain_qdrant import QdrantVectorStore

QDRANT_API_KEY = '4x-j-E2jkFlMLn7dasU7rh3yzy5V5vY3Ds9ctZuxd7O4-PiCqBtscQ'
api_key="qdrant"
collection_name = 'SPOKE_DISEASES'
url_finnish_qdrant = "http://192.168.0.16:16333"
qdrant_T = "https://db01a1a9-113c-4d41-8252-a545fee8a32f.europe-west3-0.gcp.cloud.qdrant.io"

def load_qdrant(sentence_embedding_model):
  embedding_model = SentenceTransformerEmbeddings(model_name=sentence_embedding_model)

  client = QdrantClient(api_key=QDRANT_API_KEY, url=qdrant_T)

  # Define vector configuration (adjust vector_size to match the sentence embedding size)
  vector_size = 384 # that is the vector size of the model sentence-transformers/all-MiniLM-L6-v2
  distance_metric = rest.Distance.COSINE

  if not client.collection_exists(collection_name):
      client.create_collection(
          collection_name=collection_name,
          vectors_config=rest.VectorParams(
              size=vector_size,
              distance=distance_metric
          )
      )

  return client, embedding_model

def vector_data_qdrant(client, embeddings_func, collection_name='SPOKE_DISEASES'):
    qdrant = QdrantVectorStore(
                client=client,
                collection_name = collection_name,
                embedding=embeddings_func
                )
    retriever = qdrant.as_retriever(search_kwargs={"k": 5})
    return retriever