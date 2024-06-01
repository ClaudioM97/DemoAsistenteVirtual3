from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
import chromadb
import chromadb.config
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_community.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.memory import ConversationSummaryMemory
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
from unstructured.cleaners.core import clean
from langchain_openai import AzureChatOpenAI
from langchain.document_loaders import PyPDFLoader
import pytesseract 
from unidecode import unidecode
from pdf2image import convert_from_bytes
from io import BytesIO
import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

general_system_template = f'''
Eres un asistente político que responde consultas sobre los discursos presidenciales de los años 2022 y 2023 en Chile.
Responde la pregunta del final, utilizando el siguiente contexto (delimitado por <context></context>).

En tu respuesta final considera lo siguiente:
-Si el usuario no especifica el año en su pregunta, responde basandote en el discurso mas reciente, en este caso, el discurso presidencial del 2023.
-Si el contexto que utilizas para responder a la pregunta es acotado, menciona en tu respuesta que hay poca información y que eso fue todo lo que encontraste.
-Si no encuentras información para responder a la pregunta, no digas "No lo sé" o algo similar, menciona que no existe información en el contexto de los discursos presidenciales.
-Si no es ninguno de los casos anteriores, elabora tu respuesta de manera que sea detallista y concreta, pero tambien que aporte contexto adicional para que el usuario pueda entender de mejor manera. Que tu respuesta sea mínimo de tres parrafos.
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
def extract_text(uploaded_pdf):
    loader = PyPDFLoader(uploaded_pdf)
    pages = loader.load()
    text = ""

    for page in pages:
        text += page.page_content
    
    text = text.replace('\t', ' ')

    return text

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


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(model = 'text-embedding-3-small')
    #vectorstore = Chroma.from_documents(text_chunks, embeddings)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


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

@st.cache_resource
def get_conversation_chain(text_chunks):
    embeddings = OpenAIEmbeddings(model = 'text-embedding-3-small')
    vectorstore = Chroma.from_texts(text_chunks, embeddings)
    #vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    
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
        retriever=vectorstore.as_retriever(search_type = 'mmr',search_kwargs={'fetch_k': 10, 'k': 4}),
        condense_question_llm=ChatOpenAI(model_name="gpt-3.5-turbo-0125"),
        condense_question_prompt=QA_CHAIN_PROMPT,
        combine_docs_chain_kwargs={'prompt': qa_prompt}
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
                    

def get_vdb():
    #persist_directory = '/Users/claudiomontiel/Desktop/Proyectos VS/PruebaStreamlit/chroma_st'
    embeddings = OpenAIEmbeddings(model = 'text-embedding-3-large')
    vectordb = Chroma(persist_directory="chroma_sabado",
                      embedding_function=embeddings)
    return vectordb
    

@st.cache_resource
def qa_chain(k=3):
    embeddings = OpenAIEmbeddings(model = 'text-embedding-3-large')
    vectordb = Chroma(persist_directory="chroma_sabado",
                      embedding_function=embeddings)
    retriever_bm25 = BM25Retriever.from_texts(vectordb.get()['documents'])
    retriever_mmr = vectordb.as_retriever(search_kwargs={'k': k})
    modelo = AzureChatOpenAI(api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                        azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
                        api_version="2024-02-15-preview",
                        temperature=0)
    template = """
    Dado un historial de chat y la última pregunta del usuario, que podría hacer referencia al contexto en el historial de chat, formula una pregunta independiente que pueda entenderse sin el historial de chat.
    NO respondas la pregunta, solo reformulala si es necesario y, en caso contrario, devuelvela tal como está.
    ### Historial de conversación ###
    {chat_history}
    Ultimo mensaje: {question}
    Pregunta reformulada:
    """
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
        
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=modelo,
        retriever=EnsembleRetriever(retrievers=[retriever_bm25, retriever_mmr],weights=[0.4, 0.6]),
        condense_question_llm=ChatOpenAI(model_name="gpt-3.5-turbo-0125"),
        condense_question_prompt=QA_CHAIN_PROMPT,
        combine_docs_chain_kwargs={'prompt': qa_prompt}
    )
    return conversation_chain
    
    
def reset_conversation():
    st.session_state['messages'] = [
        {"role": "assistant", 
         "content": "El historial del chat ha sido limpiado. ¿Cómo puedo asistirte ahora? 😊"}
    ]
    
