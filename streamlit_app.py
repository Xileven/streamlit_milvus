import streamlit as st

import dotenv
dotenv.load_dotenv()


# set Milvus API key and URI
# the env vars are set in the github action
import os
os.environ["ZILLIZ_URI"] = os.getenv("ZILLIZ_URI")
os.environ["MILVUS_API_KEY"] = os.getenv("ZILLIZ_API_KEY")
MILVUS_URI = os.getenv('ZILLIZ_URI')
MILVUS_API_KEY = os.getenv('ZILLIZ_API_KEY')


# Tavily API key
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')


from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import VectorStoreIndex
from tavily import TavilyClient
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.agent.openai import OpenAIAgent


def web_search(query: str) -> str:
    if not TAVILY_API_KEY:
        return "Error: Tavily API key is not set. Please set the TAVILY_API_KEY environment variable."
    
    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)
        result = client.search(
            query=query,
            search_depth="advanced",
            # topic="news", # news will make it irrelevant, dont use it
            time_range="y",
            include_answer="advanced",
            max_results=5,
        )
        
        # Extract the answer and search results
        answer = result.get('answer', '')
        search_results = result.get('results', [])
        
        # Combine the information
        combined_info = [answer]
        for res in search_results:
            combined_info.append(f"- {res.get('title')}: {res.get('content')}")
        
        return "\n".join(combined_info)
    except Exception as e:
        return f"Error: {str(e)}"

def hybrid_search(query):
    # Get RAG results
    rag_response = recursive_query_engine.query(query)
    
    # Get web search results
    web_response = web_search(query)
    
    # Create tools for the final agent
    rag_tool = QueryEngineTool(
        query_engine=recursive_query_engine,
        metadata=ToolMetadata(
            name="rag_knowledge",
            description="Provides information from the local knowledge base"
        )
    )
    
    # Create the final agent to combine results
    final_agent = OpenAIAgent.from_tools(
        [rag_tool],
        verbose=True
    )
    
    # Combine the results
    combined_prompt = f"""
    Please provide a comprehensive answer based on both local knowledge and web search results:
    
    Local Knowledge: {rag_response}
    Web Search Results: {web_response}
    
    Synthesize both sources to provide the most up-to-date and accurate information.
    If the information from different sources conflicts, prefer more recent sources and explain the discrepancy.
    """
    
    final_response = final_agent.chat(combined_prompt)
    return final_response

vector_store = MilvusVectorStore(
    uri=MILVUS_URI,
    token=MILVUS_API_KEY,
    collection_name="bama_llm_demo__EMBED_text_embedding_ada_002__LLM_gpt_3P5_turbo_0125",
    dim=1536,  # 1536 is default dim for OpenAI

)

recursive_index = VectorStoreIndex.from_vector_store(
            vector_store = vector_store,
            # storage_context = storage_context,
            show_progress=True
            )


from llama_index.postprocessor.flag_embedding_reranker import (
    # pruning away irrelevant nodes from the context
    FlagEmbeddingReranker,
)
reranker = FlagEmbeddingReranker(
                                model="BAAI/bge-reranker-large",
                                top_n=5,
)

recursive_query_engine = recursive_index.as_query_engine(
                                        similarity_top_k=10, 
                                        node_postprocessors=[reranker], 
                                        verbose=True,
                                        synthesize=True
)


query = "Summarize how Scharle Schwab Bank doing in 2024"
rag_response = recursive_query_engine.query(query)
print("====================================RAG Response====================================")
print(rag_response)

print("====================================Hybrid Response====================================")
hybrid_response = hybrid_search(query)
print(hybrid_response)