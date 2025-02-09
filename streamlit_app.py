#%%
import streamlit as st

import dotenv
# dotenv.load_dotenv('/Users/jinwenliu/github/.env/.env')  # local test
dotenv.load_dotenv()    # streamlit production

# Create new event loop for Milvus async client
import asyncio
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

# set Milvus API key and URI
# the env vars are set in the github action
import os
os.environ["ZILLIZ_URI"] = os.getenv("ZILLIZ_URI")
os.environ["MILVUS_API_KEY"] = os.getenv("ZILLIZ_API_KEY")
MILVUS_URI = os.getenv('ZILLIZ_URI')
MILVUS_API_KEY = os.getenv('ZILLIZ_API_KEY')

#%%
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
            time_range="y",
            include_answer="advanced",
            max_results=5,
        )
        
        # Extract the answer and search results
        answer = result.get('answer', '')
        search_results = result.get('results', [])
        
        # Format the information in a more structured way
        formatted_info = ["### Summary From Web\n" + answer + "\n\n## Sources"]
        
        for i, res in enumerate(search_results, 1):
            title = res.get('title', 'Untitled')
            content = res.get('content', 'No content available')
            url = res.get('url', '#')
            published_date = res.get('published_date', '')
            
            # Make content more concise by limiting length and removing redundant spaces
            content = ' '.join(content.split())  # Remove extra whitespace
            if len(content) > 200:  # Limit to first 200 characters
                content = content[:197] + "..."
            
            # Add date if available
            date_str = f" ({published_date})" if published_date else ""
            
            # Format each source with a number, title as a link, and concise content
            formatted_info.append(f"{i}. **[{title}]({url})**{date_str}\n   {content}\n")
        
        return "\n".join(formatted_info)
    except Exception as e:
        return f"Error: {str(e)}"

def hybrid_search(query, enable_web=True):
    # Get RAG results
    rag_response = recursive_query_engine.query(query)
    
    if not enable_web:
        return rag_response
    
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
    use_async=False,  # Force synchronous mode
    overwrite=False
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


# ================= Streamlit UI =================
st.set_page_config(page_title="Hybrid Search Chatbot", layout="wide")
st.title("Financial Research Assistant")

# Add web search toggle in sidebar
with st.sidebar:
    st.header("Search Settings")
    enable_web = st.toggle("Enable Web Browsing", value=True)
    if enable_web and not TAVILY_API_KEY:
        st.warning("Web browsing requires a Tavily API key in .env")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("rag_response"):
            with st.expander("📚 Knowledge Base Analysis"):
                st.markdown(message["rag_response"])
        if message.get("web_response"):
            with st.expander("🌐 Web Search Results"):
                st.markdown(message["web_response"])

# Process user input
if prompt := st.chat_input("Ask a financial research question:"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get responses
    with st.spinner("Thinking..."):
        rag_result = recursive_query_engine.query(prompt)
        web_only_result = web_search(prompt) if enable_web else None
        hybrid_result = hybrid_search(prompt, enable_web=enable_web)

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(str(hybrid_result))
        
        # Show RAG details
        with st.expander("📚 Knowledge Base Analysis"):
            st.markdown(str(rag_result))
        
        # Show Web-only details
        if enable_web and web_only_result:
            with st.expander("🌐 Web Search Results"):
                st.markdown(web_only_result)

    # Add responses to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": str(hybrid_result),
        "rag_response": rag_result,
        "web_response": web_only_result if enable_web else None
    })