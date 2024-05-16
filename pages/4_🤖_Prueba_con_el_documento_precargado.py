__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


import openai
import streamlit as st
from dotenv import load_dotenv
import os
from functions import *
from PIL import Image
import pickle
from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)


load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

with open( "app/style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)
    
st.markdown(
    """
        <style>
            [data-testid="stSidebar"] {
                background-image: url("https://brainfood.cl/wp-content/themes/theme_brainfood/assets/svgs/imagotipo-brainfood.svg");
                background-repeat: no-repeat;
                padding-top: 80px;
                background-position: 20px -20px;
                background-size: 170px 170px; 
            }
        </style>
        """,
    unsafe_allow_html=True,
)  
    
page_bg_img = '''
<style>
[data-testid="stAppViewContainer"]{
background-image: url("https://brainfood.cl/wp-content/uploads/2023/02/header-estrategia.svg");
background-size: cover;
background-position: bottom 300px right -850px;
background-repeat: no-repeat;
}
[data-testid="stSidebar"]{
background-color:rgba(64, 64, 64, 1)

}

</style>
'''


folder_path_procesados = 'textos_tablas_procesados'

st.markdown(page_bg_img, unsafe_allow_html=True)

st.title("🤖 Prueba con el documento precargado")

st.markdown('''En este apartado podras realizar preguntas sobre los Estados Financieros Consolidados de Falabella al 31 de diciembre de 2023.
            Este documento se ha cargado previamente para ahorrar el tiempo que toma su procesamiento.
            Cuenta con la particularidad de poseer una gran cantidad de tablas en distintos formatos, lo que lo hace idóneo para probar con la nueva capacidad del robot de interpretación y análisis de tablas.
            Un aspecto importante a considerar es que al ser una funcionalidad nueva y más compleja que la anterior, esta versión del chatbot no cuenta con memoria, es decir,
            el robot va a responder las preguntas de manera indepediente, sin recordar las preguntas anteriores. Sin embargo, ¡Es un elemento que está en proceso!
                ''',unsafe_allow_html=True)

with open(os.path.join(folder_path_procesados, "texts_falabella.pkl"), "rb") as file:
        texts = pickle.load(file)

with open(os.path.join(folder_path_procesados, "table_falabella.pkl"), "rb") as file:
        tables = pickle.load(file)

welcome_message = "¡Bienvenido! 👨🏻‍💻 soy tu asistente virtual. ¿En qué puedo ayudarte? 😊"

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": welcome_message}]

st.chat_message("assistant").write(welcome_message)
    

    
if question := st.chat_input("Escribe tu pregunta aquí"):      
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)
    
    first_retriever = cargar_cosas()
    final_retriever = hybrid_search(first_retriever,texts,tables)
    chain = qa(final_retriever)
    
    with st.spinner("Generando respuesta..."):
        response = chain.invoke(question)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)

if st.sidebar.button('Limpiar historial del chat'):
    st.session_state['messages'] = [
        {"role": "assistant", 
         "content": "El historial del chat ha sido limpiado. ¿Cómo puedo asistirte ahora? 😊"}
    ]
    st.chat_message('assistant').write(st.session_state['messages'][0]['content'])



        