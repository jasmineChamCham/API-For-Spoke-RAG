from pydantic import BaseModel
from typing import List, Dict, Any

class NodeListQueryResponse(BaseModel):
    data: List[Dict[str, Any]]