import streamlit as st
import pyrebase
import json
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from io import StringIO, BytesIO
import base64
import os
import fitz  # PyMuPDF

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI")
os.environ["SERPER_API_KEY"] = os.getenv("SERPER")

# Load Firebase config
with open("firebase_config.json") as f:
    firebase_config = json.load(f)

firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()

st.set_page_config(layout="wide")

# --- Firebase Login ---
def login_screen():
    st.markdown("## 🔐 Login com Firebase")
    email = st.text_input("Email")
    password = st.text_input("Senha", type="password")
    login_btn = st.button("Entrar")

    if login_btn:
        try:
            user = auth.sign_in_with_email_and_password(email, password)
            st.session_state["user"] = user
            st.session_state["auth"] = True
            st.success(f"✅ Bem-vindo, {email}")
        except Exception as e:
            st.error("❌ Falha no login. Verifique suas credenciais.")

# --- Load PDF Docs ---
def load_pdf_docs():
    docs_text = ""
    for filename in os.listdir():
        if filename.lower().endswith(".pdf"):
            try:
                with fitz.open(filename) as pdf:
                    for page in pdf:
                        docs_text += page.get_text()
            except Exception as e:
                st.warning(f"Erro ao ler {filename}: {e}")
    return docs_text[:8000]

# --- File Download ---
def generate_file(content, filename):
    bio = BytesIO()
    bio.write(content.encode("utf-8"))
    bio.seek(0)
    b64 = base64.b64encode(bio.read()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}">📥 Baixar Resultado</a>'

# --- Run CrewAI ---
def run_crew(code, docs_summary):
    llm = ChatOpenAI(model="gpt-4o-mini, temperature=0.2, max_tokens=4000)
    search_tool = SerperDevTool()

    reviewer = Agent(
        role="ABAP Code Reviewer",
        goal="Revisar código ABAP com base em boas práticas e apostilas técnicas",
        backstory="Você é um especialista SAP com anos de experiência em revisão de código ABAP",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
        llm=llm,
    )

    optimizer = Agent(
        role="ABAP Optimizer",
        goal="Melhorar código ABAP com base em análise e manuais técnicos",
        backstory="Você aplica padrões de performance, segurança e clareza em ABAP",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
        llm=llm,
    )

    review_task = Task(
        description=(
            "Revise o código ABAP fornecido abaixo. Utilize as informações das apostilas para sugerir melhorias:\n\n"
            "Código:\n{code}\n\n"
            "Documentação técnica:\n{docs}"
        ),
        expected_output="Sugestões detalhadas ou código otimizado diretamente.",
        agent=reviewer,
    )

    optimize_task = Task(
        description=(
            "Aplique as melhorias no código e entregue o resultado como ABAP ou XML com sugestões por linha."
        ),
        expected_output="Código otimizado ou XML de revisão.",
        agent=optimizer,
    )

    crew = Crew(
        agents=[reviewer, optimizer],
        tasks=[review_task, optimize_task],
        verbose=2,
    )

    return crew.kickoff(inputs={"code": code, "docs": docs_summary})

# --- Main App ---
def main_app():
    st.title("💼 ABAP Code Reviewer com Apostilas + Firebase Login")
    uploaded_file = st.file_uploader("📤 Envie seu código ABAP (.txt)", type=["txt"])
    chat_input = st.text_area("💬 Pergunta sobre seu código ABAP (opcional)")

    code_content = ""
    if uploaded_file:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        code_content = stringio.read()
        st.code(code_content, language="abap")

    if uploaded_file and st.button("🔍 Revisar Código"):
        with st.spinner("⏳ Analisando o código com base nas apostilas..."):
            docs_summary = load_pdf_docs()
            result = run_crew(code=code_content, docs_summary=docs_summary)

            st.subheader("✅ Resultado da Revisão")
            st.code(result, language="xml" if "<review>" in result else "abap")

            filename = "revisao.xml" if "<review>" in result else "codigo_otimizado.abap"
            download_link = generate_file(result, filename)
            st.markdown(download_link, unsafe_allow_html=True)

    if chat_input and code_content:
        from langchain.chains import ConversationChain
        from langchain.memory import ConversationBufferMemory
        memory = ConversationBufferMemory()
        llm = ChatOpenAI(model="gpt-4", temperature=0.2)

        st.subheader("🧠 Chat com o Revisor ABAP")
        context = (
            f"Você é um revisor ABAP com base técnica nos documentos:\n{load_pdf_docs()[:2000]}\n\n"
            f"Código:\n{code_content}\n\n"
            f"Pergunta do usuário: {chat_input}"
        )
        chain = ConversationChain(llm=llm, memory=memory)
        response = chain.run(context)
        st.write(response)

# --- Start ---
if "auth" not in st.session_state or not st.session_state["auth"]:
    login_screen()
else:
    main_app()
