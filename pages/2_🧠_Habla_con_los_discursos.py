__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
from functions import *
from dotenv import load_dotenv
import openai
from PIL import Image

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

st.markdown(page_bg_img, unsafe_allow_html=True)


st.markdown("""
<style>
button {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🧠 Habla con los discursos")

st.markdown('''
        ¡Pruébalo tu mismo! 🤝.
        En el costado izquierdo, puedes ajustar el nivel de detalle que ofrece el asistente en su respuesta.
        Un menor nivel implica una respuesta más específica, mientras que un mayor nivel implica una respuesta con mayor contexto y extensión.
         ''',unsafe_allow_html=True)

k_value = st.sidebar.select_slider('Nivel de contexto de la respuesta', options=[1,2,3,4,5], value=3)

if k_value:
    question = st.chat_input("Escribe tu pregunta aquí")
    #vector_db = get_vdb()
    conversation_chain = qa_chain(k_value)
    memory = load_memory(st)
    
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        st.chat_message("user").write(question)
                
        with st.spinner("Generando respuesta..."):
            response = conversation_chain(
                {
                "question": question,
                "chat_history": memory.load_memory_variables({})["history"],   
                }
            )
            answer = response["answer"]
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.chat_message("assistant").write(answer)
            

st.sidebar.button('Limpiar historial del chat', on_click=reset_conversation)