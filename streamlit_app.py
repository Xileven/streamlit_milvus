#%%
import streamlit as st
import dotenv
import time
from contextlib import contextmanager

dotenv.load_dotenv('/Users/jinwenliu/github/.env/.env')  # local test
# dotenv.load_dotenv()    # streamlit production

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
        formatted_info = ["#### Summary From Web\n" + answer + "\n\n#### Sources"]
        
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

def hybrid_search(query, enable_web=True, rag_response=None):
    # Get RAG results
    if rag_response is None:
        rag_response = recursive_query_engine.query(query)
    
    # Extract file names from source nodes and format citation
    source_files = []
    if hasattr(rag_response, 'source_nodes'):
        for node in rag_response.source_nodes:
            if hasattr(node, 'metadata') and 'file_name' in node.metadata:
                source_files.append(node.metadata['file_name'])
    
    # Format RAG response with citations for Knowledge Base Analysis section
    rag_response_with_citations = str(rag_response)
    if source_files:
        source_files = list(set(source_files))  # Remove duplicates
        citations = "\n\n📄 Sources: " + ", ".join(f"`{file}`" for file in source_files)
        rag_response_with_citations += citations
    
    if not enable_web:
        return str(rag_response), rag_response_with_citations
    
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
    
    Local Knowledge: {str(rag_response)}
    Web Search Results: {web_response}
    
    Present your response in the following format:
    1. First show any relevant data or markdowntables from the local knowledge, show "From Documents:". 
    2. Then show "Web Search Results:" 
    3. End with "Overall:" summarizing the key points from both sources

    If the information from different sources conflicts, prefer Local Knowledge sources and explain the discrepancy.
    If there is any table in Local Knowledge, keep it as is, and do not modify it.
    Do not summarize Web Search Result as table, unless there is a original table from web search.


    """

    final_response = final_agent.chat(combined_prompt)
    return str(final_response), rag_response_with_citations





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
st.set_page_config(page_title="Domain Knowledge Augmented LLM Chatbot", layout="wide")
st.title("Domain Knowledge Augmented LLM Chatbot")
st.subheader("BAMA, Feb 2025")

st.write("""
##### This demo is a POC of:
    1. LLM interacts with user specified documents.
    2. 3 files are ingested at the same time (Pre-Processed)
         - PDF, Schwab 2024 Q10
         - Word, Schwab 2023 Q10 (Converted from PDF to Word)
         - Excel, A table extracted from Schwab 2022 K10 (page 26)
* [SEC 10K 2022](https://content.schwab.com/web/retail/public/about-schwab/SEC_Form10k_2022.pdf)
* [SEC 10Q 2023](https://content.schwab.com/web/retail/public/about-schwab/SEC_Form10-Q_093023.pdf)
* [SEC 10Q 2024](https://content.schwab.com/web/retail/public/about-schwab/SEC_Form10Q_093024.pdf)


##### Due to limitation of hardware (memory, storage, GPU, API), demo is restricted from
    1. Reasoning(Ambiguous questions)
    2. Sematic questioning (follow up questions)
    3. Large model (accuracy, tradeoff between speed and accuracy)
    4. Fine-tune (optimization)
    5. Unstable output
         


      
""")

# Display sample questions
st.markdown("""
**Sample Questions:**
```
- [PDF] How Integration of Ameritrade impact client metrics from 2023 to 2024?
- [Excel] Where is the headquarters of schwab and what is its size, including leased and owned
- [PDF & Word] Compare Client Metrics of Three Month Ended from 2022, to 2023, to 2024, in numbers, and printout in table
- [PDF & Word] Compare Total Net Revenue from 2022, to 2023, to 2024 and printout in table
- [Summary] based on Client Metrics of Three Month Ended from 2022, to 2023, to 2024, analyze the business
- Compare Total Net Revenue from 2022, to 2023, to 2024 and printout in table
```
""")

# Add web search toggle in sidebar
with st.sidebar:
    st.markdown("Domain Knowledge Augmented LLM")
    st.markdown("BAMA, Feb 2025")

    # Add a visual separator
    st.markdown("---")

    st.header("Settings")
    enable_web = st.toggle("Enable Web Browsing", value=False)
    if enable_web and not TAVILY_API_KEY:
        st.warning("Web browsing requires a Tavily API key in .env")
    
    # Add a visual separator
    st.markdown("---")
    st.markdown("upload demo")
    # Move Document Upload to bottom
    st.header("Document Upload")
    uploaded_files = st.file_uploader(
        "Upload your documents",
        type=["pdf", "docx", "txt", "csv", "xlsx", "xls"],
        accept_multiple_files=True,
        help="Supported formats: PDF, Word, Text, CSV, Excel"
    )
    
    if uploaded_files:
        st.write(f"📄 {len(uploaded_files)} files uploaded")
        for file in uploaded_files:
            st.write(f"- {file.name}")

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
if prompt := st.chat_input("Type your question:"):
    # Clear any existing containers
    st.empty()
    
    # Create a container for the new response
    response_container = st.container()
    
    with response_container:
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Add to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Create containers for spinner and status
        with st.spinner("Thinking..."):
            # Create a container for the status updates
            status_container = st.empty()

            def update_status(text):
                status_container.markdown(f"*{text}*")

            if enable_web:
                update_status("🔍 Searching through documents...")
                rag_response = recursive_query_engine.query(prompt)
                time.sleep(1)  # Ensure minimum visibility
                
                update_status("🌐 Searching web for relevant information...")
                web_only_result = web_search(prompt)
                time.sleep(1)  # Ensure minimum visibility
                
                update_status("🤔 Analyzing and combining information...")
                final_response, rag_with_citations = hybrid_search(prompt, enable_web=True, rag_response=rag_response)
                time.sleep(1)  # Ensure minimum visibility
            else:
                # When web search is off, just use RAG result
                update_status("🔍 Searching through documents...")
                final_response, rag_with_citations = hybrid_search(prompt, enable_web=False)
                time.sleep(1)  # Ensure minimum visibility
                web_only_result = None

            # Clear the status
            status_container.empty()

        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(final_response)
            
            # Show RAG details with citations
            with st.expander("📚 Knowledge Base Analysis"):
                st.markdown(rag_with_citations)
            
            # Show Web-only details
            if enable_web and web_only_result:
                with st.expander("🌐 Web Search Results"):
                    st.markdown(web_only_result)

        # Add responses to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": final_response,
            "rag_response": rag_with_citations,
            "web_response": web_only_result if enable_web else None
        })