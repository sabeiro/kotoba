import os
from streamlit_interface import st
import chatbot_utils as c_t
from langchain_core.chat_history import BaseChatMessageHistory

modL = ["gpt-4o@openai","gpt-4-turbo@openai","gpt-3.5-turbo@openai","mixtral-8x7b-instruct-v0.1@aws-bedrock","llama-2-70b-chat@aws-bedrock","codellama-34b-instruct@together-ai","gemma-7b-it@fireworks-ai","claude-3-haiku@anthropic","claude-3-opus@anthropic","claude-3-sonnet@anthropic","mistral-7b-instruct-v0.1@fireworks-ai","mistral-7b-instruct-v0.2@fireworks-ai"]
dynamic_provider = ["lowest-input-cost", "lowest-output-cost", "lowest-itl", "lowest-ttft", "highest-tks-per-sec"]
model_reset_dict = {"slider_model_temperature": "model_temperature"}
splitter_reset_dict = {"slider_chunk_size": "chunk_size","slider_chunk_overlap": "chunk_overlap"}
retriever_reset_dict = {"slider_k": "k","slider_fetch_k": "fetch_k","slider_lambda_mult": "lambda_mult","slider_score_threshold": "score_threshold"}
model_max_context_limit = {"mixtral-8x7b-instruct-v0.1": 32000,"llama-2-70b-chat": 4096,"llama-2-13b-chat": 4096,"mistral-7b-instruct-v0.2": 8192,"llama-2-7b-chat": 4096,"codellama-34b-instruct": 4096,"gemma-7b-it": 8192,"mistral-7b-instruct-v0.1": 512,"mixtral-8x22b-instruct-v0.1": 65536,"codellama-13b-instruct": 4096,"codellama-7b-instruct": 4096,"yi-34b-chat": 4096,"llama-3-8b-chat": 8192,"llama-3-70b-chat": 8192,"pplx-7b-chat": 4096,"mistral-medium": 32000,"gpt-4o": 32000,"gpt-4": 32000,"pplx-70b-chat": 4096,"gpt-3.5-turbo": 16000,"deepseek-coder-33b-instruct": 16000,"gemma-2b-it": 8192,"gpt-4-turbo": 128000,"mistral-small": 32000,"mistral-large": 32000,"claude-3-haiku": 200000,"claude-3-opus": 200000,"claude-3-sonnet": 200000}
baseDir = os.environ['HOME'] + '/lav/dauvi/portfolio/audit/'


#---------------------------------------------------UI--------------------------------------------------
        
def clear_history():
    """Clears the history stored in the session state."""
    if "store" in st.session_state:
        st.session_state.store = {}
    if "messages" in st.session_state:
        st.session_state.messages = []

def output_chunks(chain, query):
    """Generates answers for the given query and a chain.

    Args:
        chain: The chain given by the user selection.
        query: The query to generate answers for.

    Yields:
        str: The generated answer.
    """
    for chunk in chain.stream(
            {"input": query},
            config={"configurable": {"session_id": "abc123"}}
    ):
        if "answer" in chunk.keys():
            yield chunk["answer"]

def get_history(session_id: str):
  """
        Retrieves the chat history for a given session.
        Parameters:
        session_id (str): The ID of the session.
        Returns:
        BaseChatMessageHistory: The chat history for the provided session ID.
  """
  if session_id not in st.session_state.store:
    st.session_state.store[session_id] = c_t.get_chat_message()
  return st.session_state.store[session_id]

def field_callback(field):
    """Displays a toast message when a field is updated."""
    st.toast(f"{field} Updated Successfully!", icon="🎉")


    
def process_inputs():
    """Processes the user inputs and performs vector storage."""
    
    if not st.session_state.unify_api_key or not st.session_state.endpoint or not st.session_state.pdf_docs:
        st.warning("Please enter the missing fields and upload your pdf document(s)")
    else:
        with st.status("Processing Document(s)"):
            st.write("Extracting Text")
            docL = c_t.pdf_page(st.session_state.pdf_docs,chunk_size=st.session_state.chunk_size,chunk_overlap=st.session_state.chunk_overlap)
            st.write("Splitting Text")
            st.write("Performing Vector Storage")
            if st.session_state.vector_selection == "FAISS":
                st.session_state.vector_store = c_t.faiss_vector_storage(docL,collN="web",baseDir=baseDir)
            elif st.session_state.vector_selection == "Pinecone":
                st.session_state.vector_store = c_t.pinecone_vector_storage(docL)

            st.session_state.processed_input = True
            st.success('File(s) Submitted successfully!')

def reset_slider_value(reset_dict):
    '''Resets the value of sliders in the session state.'''
    for key, value in reset_dict.items():
        del st.session_state[value]
        init_keys()
        st.session_state[key] = st.session_state[value]

def get_retriever():
    """ Creates a retriever using the vector store in the session state and the selected search parameters."""
    if st.session_state.search_type == "similarity":
        st.session_state.search_kwargs = {"k": st.session_state.k}
    elif st.session_state.search_type == "similarity_score_threshold":
        st.session_state.search_kwargs = {
            "k": st.session_state.k,
            "score_threshold": st.session_state.score_threshold
        }
    elif st.session_state.search_type == "mmr":
        st.session_state.search_kwargs = {
            "k": st.session_state.k,
            "fetch_k": st.session_state.fetch_k,
            "lambda_mult": st.session_state.lambda_mult
        }
    retriever = st.session_state.vector_store.as_retriever(
        search_type=st.session_state.search_type,
        search_kwargs=st.session_state.search_kwargs
    )
    return retriever



def chat_bot():
    """ Takes user queries and generates responses. It writes the user query and the response to the chat window."""
    if query := st.chat_input("Ask your document anything...", key="query"):
        if "processed_input" not in st.session_state:
            st.warning("Please input your details in the sidebar first")
            return

        st.chat_message("human").write(query)
        if "vector_store" not in st.session_state:
          process_inputs()

        retriever = get_retriever()
        model = c_t.get_llm()
        if not st.session_state.history_unaware:
          rag_engine = c_t.create_conversational_rag_chain(model, retriever, get_history)
        else:
          rag_engine = c_t.create_qa_chain(model, retriever)
          
        response = st.chat_message("assistant").write_stream(output_chunks(rag_engine, query))
        if not st.session_state.history_unaware:
          st.session_state.messages.append((query, response))
      
