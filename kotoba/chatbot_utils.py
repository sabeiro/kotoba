import os, re, sys, json
import kotoba.chatbot_prompt as c_p
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import MarkdownTextSplitter, MarkdownHeaderTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.chat_models import ChatOpenAI
#from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
#from langchain_core.documents import Document # with page_content
from llama_index.core import Document
from llama_index.core.node_parser import SimpleFileNodeParser
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_parse import LlamaParse
from unify import Unify
#from llama_parse import LlamaParse  # pip install llama-parse
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
# from langchain.chains import create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_unify.chat_models import ChatUnify
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.chat_models import ChatOpenAI

#--------------------------------------parse-pdf--------------------------------------------------

def pdf2tree(pdf_doc):
    """Extracts text from PDF.
    Args:
        pdf_docs: A PDF document.
    Returns:
        str: The extracted text from the PDF documents.
    """
    from llmsherpa.readers import LayoutPDFReader
    llmsherpa_api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
    pdf_reader = LayoutPDFReader(llmsherpa_api_url)
    doc = pdf_reader.read_pdf(pdf_doc)
    docL = []
    for s in doc.sections():
        sectS = ''
        for p in s.children:
            sectS += p.to_text()
        if sectS == '':
            sectS = '-'
        docL.append(Document(text=sectS,metadata={"sect":s.to_context_text(),"lev":s.level}))
    for t in doc.tables():
        docL.append(Document(text=t.to_text(),metadata={"table":s.block_idx,"lev":t.level}))
    return docL

def pdf2md(pdf_doc):
    """Extracts text from PDF.
    Args:
        pdf_docs: A PDF document.
    Returns:
        str: The extracted text from the PDF documents.
    """
    #from langchain_community.document_loaders import PyMuPDFLoader
    import pymupdf4llm
    import pymupdf
    md_text = pymupdf4llm.to_markdown(pdf_doc,pages=[0,1])
    md_text = pymupdf4llm.to_markdown(pdf_doc)
    # parser = LlamaParse(api_key="...",result_type="markdown")
    # documents = parser.load_data("./my_file.pdf") 
    #single_sentences_list = re.split(r'(?<=[.?!])\s+', essay)
    headers_split = [("#", "Chapter"),("##", "Section"),('###','Subsection')]
    splitter = MarkdownHeaderTextSplitter(headers_split)#,strip_headers=True,return_each_line=False,)
    docL = splitter.split_text(md_text)
    #splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=200)
    #splitter = SentenceSplitter(chunk_size=200,chunk_overlap=15)
    #elements = partition_pdf(filename=pdf_doc,strategy="hi_res",infer_table_structure=True,model_name="yolox")
    return docL

def pdf_llama(pdf_doc,collN):
    os.environ["LLAMA_CLOUD_API_KEY"] = "llx-"
    llm = get_llm()
    parsing_instructions = '''The document describes IT security policies for audit. It contains many tables. Answer questions using the information in this article and be precise.'''
    documents = LlamaParse(result_type="markdown", parsing_instructions=parsing_instructions).load_data(pdf_doc)
    print(documents[0].text[:1000])
    node_parser = MarkdownElementNodeParser(llm=llm, num_workers=8).from_defaults()
    nodes = node_parser.get_nodes_from_documents(documents)
    base_nodes, objects = node_parser.get_nodes_and_objects(nodes)
    


def pdf_page(pdf_docs,chunk_size=100,chunk_overlap=15):
    """Extracts text from PDF documents.
    Args:
        pdf_docs: A list of PDF documents.

    Returns:
        str: The extracted text from the PDF documents.
    """
    from PyPDF2 import PdfReader
    text = ""
    docL = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for i, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            docL.append(Document(text=text,metadata={"page":i}))
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    # text_chunks = text_splitter.split_text(textL)
    return docL

#--------------------------------------llm-opeerations--------------------------------------------------

def create_summary(textL):
    chain = ({"doc": lambda x: x}
             | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")
             | ChatOpenAI(max_retries=0)
             | StrOutputParser())
    summL = chain.batch(textL, {"max_concurrency": 5})
    return summL

def ask_openai(q,retL):
    chain = ({"doc": lambda x: x}
             | ChatPromptTemplate.from_template("The following document answers "+q+":\n\n{doc} \n\n Answer your confidence")
             | ChatOpenAI(max_retries=0)
             | StrOutputParser())
    summaries = chain.batch(retL, {"max_concurrency": 5})
    return summaries

def rank_openai(resL):
    doc = ".".join([str(i) + ") " + x for i,x in enumerate(resL)])    
    chain = ({"doc": lambda x: x}
             | ChatPromptTemplate.from_template("What answer is the most confident in the following series:\n\n{doc}")
             | ChatOpenAI(max_retries=0)
             | StrOutputParser())
    summaries = chain.batch([doc], {"max_concurrency": 1})
    return summaries

def ask_question(q,retL,unify):
    #prompt = f"Read this text and answer the question: {q}:\n{doc}"
    prompt = "The following document answers "+q+":\n\n{doc} \n\n Answer your confidence"
    ansL = []
    for doc in retL:
        ans = unify.generate(prompt)
        ansL.append(ans)
    return ansL

def rank_question(resL,unify):
    doc = ".".join([str(i) + ") " + x for i,x in enumerate(resL)])
    prompt = f"Which of the following answer is confident in the following document?:\n\n{doc}\n\n please mention numbers only"
    ans = unify.generate(prompt)
    return ans

def ask_unify(query="Two plus two is even? Respond with true/false"):
    unify = Unify("mistral-7b-instruct-v0.2@fireworks-ai")
    response = unify.generate(query)
    return response

def get_unify(modelN="mistral-7b-instruct-v0.2@fireworks-ai",temp=.3):
    model = Unify(modelN)
    #model = ChatUnify(model=modelN,unify_api_key=os.environ['UNIFY_KEY'],temperature=temp)
    return model

def get_llm():
    llm = ChatOpenAI()
    return llm



def get_chat_history(retriever):
    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    llm = ChatOpenAI()
    chain = create_history_aware_retriever(llm, retriever, rephrase_prompt)
    #chain.invoke({"input": "...", "chat_history": })
    return chain

def get_chat_message() -> BaseChatMessageHistory:
    return ChatMessageHistory()

#--------------------------------------vector-storage--------------------------------------------------

from langchain.vectorstores import Chroma
import chromadb
from chromadb.utils import embedding_functions
from llama_index.core import SimpleDirectoryReader, load_index_from_storage, VectorStoreIndex, StorageContext
# from llama_index.vector_stores.faiss import FaissVectorStore
#import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

#from langchain_pinecone import PineconeVectorStore

def create_collection(docL,collN,baseDir):
    """create two collections from a pdf, chapter wise and their summaries.
    Args:
        pdf_doc: A PDF document.
    Returns:
        collT, collS: collection of texts and theirs summaries
    """
    try:
        textL = [x.text for x in docL]
    except:
        textL = [x.page_content for x in docL]        
    metaL = [x.metadata for x in docL]
    idL = ["%06d" % x for x in range(len(textL))]
    summL = create_summary(textL)
    client = chromadb.PersistentClient(path=baseDir + "/chroma")
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(model_name="text-embedding-ada-002",api_key=os.environ['OPENAI_API_KEY'])
    collN = re.sub(".pdf","",pdf_doc).split("/")[-1]
    # vectorstore = Chroma(collection_name="summaries",embedding_function=openai_af)
    # store = InMemoryByteStore()
    # retriever = MultiVectorRetriever(vectorstore=vectorstore,byte_store=store,id_key=collN,)
    # summary_docs = [Document(page_content=s, metadata={id_key: idL[i]}) for i, s in enumerate(summL) ]
    # retriever.vectorstore.add_documents(summL)
    # retriever.docstore.mset(list(zip(doc_ids, docL)))
    # for i, doc in enumerate(docL):
    #     doc.metadata[id_key] = docL[i]
    # retriever.vectorstore.add_documents(docL)
    try: 
        client.delete_collection(name=collN+"_text")
        client.delete_collection(name=collN+"_summaries")
    except:
        pass
    collT = client.create_collection(name=collN+"_text",metadata={"hnsw:space":"cosine"},embedding_function=openai_ef)
    collS = client.create_collection(name=collN+"_summaries",metadata={"hnsw:space":"cosine"},embedding_function=openai_ef)
    collT.add(embeddings=embdL,documents=textL,metadatas=metaL,ids=idL)
    collS.add(embeddings=embsL,documents=summL,metadatas=metaL,ids=idL)
    return collT, collS

def faiss_vector_storage(docL,collN,baseDir):
    """Creates a FAISS vector store from the given text chunks.
    Args:
        text_chunks: A list of text chunks to be vectorized.
    Returns:
        FAISS: A FAISS vector store.
    """
    try:
        textL = [x.text for x in docL]
    except:
        textL = [x.page_content for x in docL]        
    metaL = [x.metadata for x in docL]
    faiss_index = faiss.IndexFlatL2(1536) # dimensions of text-ada-embedding-002
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(textL, embedding=embeddings)
    #vector_store = FaissVectorStore(faiss_index=faiss_index)
    #storage_context = StorageContext.from_defaults(vector_store=vector_store)
    #index = VectorStoreIndex.from_documents(docL, storage_context=storage_context)
    #index.storage_context.persist(persist_dir=baseDir+"./faiss")    
    #return index
    return vector_store

def elastic_vector_storage(docL,collN,baseDir):
    """Creates a elasticsearch vector store from the given text chunks.
    Args:
        text_chunks: A list of text chunks to be vectorized.
    Returns:
        elastic search vector store.
    """
    from llama_index.vector_stores.elasticsearch import ElasticsearchStore, AsyncDenseVectorStrategy
    from llama_index.core import StorageContext, VectorStoreIndex
    vector_store = ElasticsearchStore(index_name=collN,es_url="http://localhost:9200",retrieval_strategy=AsyncDenseVectorStrategy())
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(docL, storage_context=storage_context)
    # retriever = index.as_retriever()
    # results = retriever.retrieve(query)
    # query_engine = index.as_query_engine()
    # response = query_engine.query(query)
    return index

def load_faiss(collN,baseDir):
    vector_store = FaissVectorStore.from_persist_dir(baseDir+"./faiss")
    storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=baseDir+"./faiss")
    index = load_index_from_storage(storage_context=storage_context)
    return index

def load_chroma(collN,baseDir):
    client = chromadb.PersistentClient(path=baseDir + "/chroma")
    collT = client.get_or_create_collection(name=collN+"_text",metadata={"hnsw:space":"cosine","hnsw:M": 32})
    collS = client.get_or_create_collection(name=collN+"_summaries",metadata={"hnsw:space":"cosine","hnsw:M": 32})
    return collT, collS

def pinecone_vector_storage(pdf_doc,baseDir):
    """Creates a Pinecone vector store from the given text chunks.
    Args:
        text_chunks: A list of text chunks to be vectorized.
    Returns:
        PineconeVectorStore: A Pinecone vector store.
    """
    vector_store = None
    os.environ['PINECONE_API_KEY'] = st.session_state.pinecone_api_key
    if st.session_state.embedding_model == "HuggingFaceEmbeddings":
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        try:
            # Clear existing index data if there's any
            PineconeVectorStore.from_existing_index(
                index_name=st.session_state.pinecone_index,
                embedding=embeddings
            ).delete(delete_all=True)
        except Exception as e:
            print("The index is empty")
        finally:
            vector_store = PineconeVectorStore.from_texts(
                text_chunks,
                embedding=embeddings,
                index_name=st.session_state.pinecone_index
            )
    return vector_store
    
#--------------------------------------chains--------------------------------------------------

def format_docL(docs):
    """Formats the given documents into a list."""
    return [doc for doc in docs]

def format_docs(docs):
  return "\n\n".join(doc.page_content for doc in docs)

def get_vectorstore(baseDir,collN):
  openai_ef = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
  # vectorstore = Chroma.from_documents(documents, openai)
  client = chromadb.PersistentClient(path=baseDir + "/chroma")
  db = Chroma(client=client,embedding_function=openai_ef,collection_name=collN+"_text",collection_metadata={"hnsw:space":"cosine"})
  #con = db.similarity_search_with_relevance_scores(q)
  return db

def get_retrieval_qa(baseDir,collN):
    db = c_t.get_vectorstore(baseDir,collN)
    qa = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0),chain_type="stuff",retriever=db.as_retriever(),return_source_documents=True,)
    return qa

def get_chain_confidence(llm,baseDir,collN):
  prompt = PromptTemplate(input_variables=["question","context"], template=c_p.promptConf)
  db = get_vectorstore(baseDir,collN)
  chain = ({'context': db.as_retriever(search_kwargs={'k':5}) | format_docs, "question": RunnablePassthrough()} | prompt | llm | c_p.parserS)
  # chain = ({'context': db.as_retriever(search_kwargs={'k':3}) | format_docs, "question": RunnablePassthrough()} | prompt | llm)
  return chain

def format_confidence(res):
    try:
        res['answer'] = bool(c_p.yesRe.match(res['answer']))
        res['confidence'] = float(res['confidence'])
    except:
        pass
    return res

def create_conversational_rag_chain(model, retriever, get_history):
    """
    Creates a conversational RAG chain. This is a question-answering (QA) system with the ability to consider historical context.
    Parameters:
    model: The model selected by the user.
    retriever: The retriever to use for fetching relevant documents.
    Returns:
    RunnableWithMessageHistory: The conversational chain that generates the answer to the query.
    """
    system_prompt = ("You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}")

    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages([("system", contextualize_q_system_prompt),MessagesPlaceholder("chat_history"),("human", "{input}"),])
    #prompt = ChatPromptTemplate.from_messages([("system", system_prompt),("human", "{input}"),])
    history_aware_retriever = create_history_aware_retriever(model,retriever |format_docs, contextualize_q_prompt)
    system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    {context}"""
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt),MessagesPlaceholder("chat_history"),("human", "{input}"),])
    question_answer_chain = create_stuff_documents_chain(model, prompt)
    # rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    conversational_rag_chain = RunnableWithMessageHistory(rag_chain,get_history,input_messages_key="input",history_messages_key="chat_history",output_messages_key="answer",)
    return conversational_rag_chain

def create_qa_chain(model, retriever):
    """
    Creates a question-answering (QA) chain for a chatbot without considering historical context.
    Parameters:
    model: The model selected by the user.
    retriever: The retriever to use for fetching relevant documents.
    Returns:
    chain: it takes a user's query as input and produces a chatbot's response as output.
    """
    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    {context}"""
    qa_prompt_no_memory = ChatPromptTemplate.from_messages([("system", qa_system_prompt),("human", "{input}"),])
    question_answer_chain = create_stuff_documents_chain(model, qa_prompt_no_memory)
    chain = create_retrieval_chain(retriever, question_answer_chain)
    return chain


if False:
    url = "https://api.unify.ai/v0/chat/completions"
    headers = {"Authorization": "Bearer " + os.environ['UNIFY_KEY'],'Content-Type':'application/json'}
    payload = {"model": "mistral-7b-instruct-v0.2@fireworks-ai","messages": [{"role": "user",
                                                                              "content": "two plus two is?"}],"stream": True}
    response = requests.post(url, json=payload, headers=headers, stream=True)
    print(response.status_code)
    if response.status_code == 200:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                print(chunk.decode("utf-8"))
                resp = json.loads(chunk.decode("utf-8")[6:])
    else:
        print(response.text)

