from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
import chromadb
from langchain_openai import AzureChatOpenAI
import openai
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_community.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.memory import ConversationSummaryMemory
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
from unstructured.cleaners.core import clean
import pytesseract 
from unidecode import unidecode
from pdf2image import convert_from_bytes
import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from chroma_store import ChromaStore
from langchain.storage._lc_store import create_kv_docstore
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import BM25Retriever
from langchain.schema import Document


load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

@st.cache_data
def extract_text_from_pdf(uploaded_pdf):
    images = convert_from_bytes(uploaded_pdf.getvalue())
    ocr_text_list = []
    for i, image in enumerate(images):
        page_content = pytesseract.image_to_string(image)
        page_content = '***PDF Page {}***\n'.format(i+1) + page_content
        ocr_text_list.append(page_content)
    ocr_text = ' '.join(ocr_text_list)
    return ocr_text

@st.cache_data
def extract_text_from_pdf_2(uploaded_pdf):
    images = convert_from_bytes(uploaded_pdf)
    ocr_text_list = []
    for i, image in enumerate(images):
        page_content = pytesseract.image_to_string(image)
        page_content = '***PDF Page {}***\n'.format(i+1) + page_content
        ocr_text_list.append(page_content)
    ocr_text = ' '.join(ocr_text_list)
    return ocr_text

@st.cache_data
def clean_text(ocr_text_from_pdf):
    return clean(ocr_text_from_pdf,extra_whitespace=True,trailing_punctuation=True,lowercase=True)

@st.cache_data
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap = 100,
        separators=["\n\n", "\n", "(?<=\. )", " ", ""])
    chunks = text_splitter.split_text(text)
    return chunks


def load_memory(st):
    memory = ConversationBufferWindowMemory(k=3, return_messages=True)
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "¡Bienvenido! 👨🏻‍💻 soy tu asistente virtual. ¿En qué puedo ayudarte? 😊"}
        ]
    for index, msg in enumerate(st.session_state.messages):
        st.chat_message(msg["role"]).write(msg["content"])
        if msg["role"] == "user" and index < len(st.session_state.messages) - 1:
            memory.save_context(
                {"input": msg["content"]},
                {"output": st.session_state.messages[index + 1]["content"]},
            )

    return memory


general_system_template = f'''
Eres un asistente virtual de un director empresarial, es decir, miembro del directorio de varias empresas. Debes responder de manera concisa y precisa, las preguntas que tenga sobre distintos tipos de documentos tales como:
informes financieros, reportes empresariales, memorias anuales, articulos, y cualquier otro que sea relevante para un director empresarial en su gestion.

Responde la pregunta del final, utilizando solo el siguiente contexto (delimitado por <context></context>).
Si no sabes la respuesta, menciona explicitamente que no la sabes de manera educada y cordial.
<context>
{{chat_history}}

{{context}} 
</context>
'''

general_user_template = "Question:```{question}```"
messages = [
            SystemMessagePromptTemplate.from_template(general_system_template),
            HumanMessagePromptTemplate.from_template(general_user_template)
]
qa_prompt = ChatPromptTemplate.from_messages(messages)

@st.cache_resource
def get_conversation_chain(text_chunks):
    embeddings = OpenAIEmbeddings(model = 'text-embedding-3-large')
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    
    template = """
    Dado un historial de conversacion, reformula la pregunta para hacerla mas facil de buscar en una base de datos.
    Por ejemplo, si la IA dice "¿Quieres saber el clima actual en Estambul?", y el usuario responde "si", entonces la IA deberia reformular la pregunta como "¿Cual es el clima actual en Estambul?".
    No debes cambiar el idioma de la pregunta, solo reformularla. Si no es necesario reformular la pregunta o si no es una pregunta, simplemente muestra el mismo texto

    ### Historial de conversación ###
    {chat_history}
    Ultimo mensaje: {question}
    Pregunta reformulada:
    """
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
       
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name='gpt-3.5-turbo-0125', temperature=0),
        retriever=vectorstore.as_retriever(search_kwargs={'fetch_k': 10, 'k': 4}, search_type='mmr'),
        condense_question_llm=ChatOpenAI(model_name="gpt-3.5-turbo-0125"),
        combine_docs_chain_kwargs={'prompt': qa_prompt},
        condense_question_prompt=QA_CHAIN_PROMPT
    )
    return conversation_chain


def remove_accents(input_str):
    return unidecode(input_str)

def filter_fichas(data, search_term):
    search_term = remove_accents(search_term.lower())
    filtered_fichas = []
    for ficha in data:
        if (search_term in remove_accents(ficha['Título'].lower()) or
            search_term in remove_accents(ficha['Autor'].lower()) or
            search_term in remove_accents(ficha['Keywords'].lower())):
            filtered_fichas.append(ficha)
    return filtered_fichas

@st.cache_data
def display_in_pairs(data):
    num_columns = len(data)
    num_pairs = num_columns // 2
    remainder = num_columns % 2

    columns = st.columns(2)
    
    for i in range(num_pairs):
        with columns[0]:
            with st.expander(data[i]['Título']):
                for key, value in data[i].items():
                    st.write(f"{key}: {value}")
        with columns[1]:
            with st.expander(data[num_pairs + i]['Título']):
                for key, value in data[num_pairs + i].items():
                    st.write(f"{key}: {value}")

    if remainder == 1:
        with columns[0]:
            with st.expander(data[-1]['Título']):
                for key, value in data[-1].items():
                    st.write(f"{key}: {value}")
                    
@st.cache_resource
def get_vdb():
    persist_directory = 'chroma_final'
    embeddings = OpenAIEmbeddings(model = 'text-embedding-3-large')
    persistent_client = chromadb.PersistentClient(path=persist_directory)
    vectordb = Chroma(client=persistent_client,
                      collection_name = 'docs_publicos',
                      embedding_function=embeddings)
    return vectordb
    

#@st.cache_resource
def qa_chain(vectordb,k):
    template = """
    Dado un historial de conversacion, reformula la pregunta para hacerla mas facil de buscar en una base de datos.
    Por ejemplo, si la IA dice "¿Quieres saber el clima actual en Estambul?", y el usuario responde "si", entonces la IA deberia reformular la pregunta como "¿Cual es el clima actual en Estambul?".
    No debes cambiar el idioma de la pregunta, solo reformularla. Si no es necesario reformular la pregunta o si no es una pregunta, simplemente muestra el mismo texto

    ### Historial de conversación ###
    {chat_history}
    Ultimo mensaje: {question}
    Pregunta reformulada:
    """
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template=template)
        
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name='gpt-3.5-turbo-0125', temperature=0),
        retriever=vectordb.as_retriever(search_kwargs={'fetch_k': 10, 'k': k}, search_type='mmr'),
        #chain_type="refine",
        condense_question_prompt=QA_CHAIN_PROMPT,
        combine_docs_chain_kwargs={'prompt': qa_prompt}
    )
    return conversation_chain
    
    
def reset_conversation():
    st.session_state['messages'] = [
        {"role": "assistant", 
         "content": "El historial del chat ha sido limpiado. ¿Cómo puedo asistirte ahora? 😊"}
    ]
    
@st.cache_resource
def cargar_cosas():
    local_store_path = 'local_store'
    cs = ChromaStore(local_store_path, "docstore")
    id_key = "doc_id"
    store = create_kv_docstore(cs)
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
    persist_directory = 'chroma_multimodal'
    vectorstore = Chroma(embedding_function=embeddings,
                     persist_directory=persist_directory,
                     collection_name='summaries'
                     )

    retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key=id_key
    )
    
    return retriever

@st.cache_resource
def hybrid_search(_multivector_retriever,textos,tablas):
    texts_for_bm25 = textos + tablas
    bm25_retriever = BM25Retriever.from_texts(texts_for_bm25)
    ensemble_retriever = EnsembleRetriever(retrievers=[_multivector_retriever, bm25_retriever],
                                           weights=[0.4, 0.6])
    
    return ensemble_retriever
                                           
@st.cache_resource
def qa(_retriever_final):
    template = """Eres el asistente de una persona que es miembro del directorio de varias empresas reconocidas y exitosas. \
    Tu tarea es responder todas las preguntas que tenga sobre diversos documentos que necesita leer en su quehacer para estar al corriente de la situacion de las distintas empresas en las que forma parte. \
    Por ejemplo, estos documentos pueden ser: informes financieros, presentaciones corporativas, estados de resultados, memorias anuales, noticias de mercado, entre muchos mas que sean pertinentes en el contexto empresarial. \
    Si en la pregunta del usuario no se proporciona una fecha de manera explicita, responde con los resultados mas recientes. \
    Si en la pregunta del usuario no se proporciona un lugar de manera explicita, responde con los resultados que pertenezcan a Chile. \
    Si no sabes la respuesta, no la inventes, responde que no la sabes de manera cordial y educada. \
    
    Responde la pregunta basándote únicamente en el siguiente contexto, que puede incluir texto y tablas:
    {context}
    Pregunta: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    modelo = AzureChatOpenAI(api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                        azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
                        api_version="2024-02-15-preview",
                        temperature=0)
    
    chain = (
    {"context": _retriever_final, "question": RunnablePassthrough()}
    | prompt
    | modelo
    | StrOutputParser()
    )
    
    return chain
    

          