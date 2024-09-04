import streamlit as st
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv, find_dotenv
import os

# Carregar variáveis de ambiente
_ = load_dotenv(find_dotenv())

# Configurações do modelo
MODEL_NAME = 'gpt-3.5-turbo-0125'

# Prompt para resumo e explicação
PROMPT = '''Você é um assistente virtual especializado em análise de documentos.
O usuário forneceu um arquivo PDF e deseja um resumo conciso e uma explicação em tópicos.
Primeiro, forneça um resumo geral do documento.
Depois, explique os pontos principais em tópicos. Responda em markdown.

**Resumo:**
{resumo}

**Explicação em Tópicos:**
- '''

# Função para carregar e processar o PDF
def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documentos = loader.load()

    # Combinar o conteúdo das páginas
    full_text = " ".join([doc.page_content for doc in documentos])

    return full_text

# Função para gerar o texto usando o modelo GPT com divisão em partes menores
def generate_summary_and_explanation(text, source):
    chat = ChatOpenAI(model=MODEL_NAME)
    prompt = PromptTemplate.from_template(PROMPT)
    
    # Dividir o texto em partes menores para evitar ultrapassar o limite de tokens
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,  # Ajuste conforme necessário
        chunk_overlap=200  # Para garantir que o contexto seja mantido entre as partes
    )
    text_parts = splitter.split_text(text)

    final_response = ""
    for part in text_parts:
        formatted_prompt = prompt.format(resumo=part, question="")
        response = chat.invoke(formatted_prompt)
        final_response += response.content + "\n\n"  # Acessar diretamente o conteúdo da resposta
    
    # Adicionar o source ao resultado final
    resultado = f"**Source:** {source}\n\n" + final_response
    
    return resultado

# Interface do Streamlit
def main():
    st.title("Analisador de PDFs - Resumo e Explicação")
    st.write("Faça o upload de um ou mais arquivos PDF para obter um resumo e uma explicação dos principais pontos.")

    uploaded_pdfs = st.file_uploader("Adicione seus arquivos PDF", type=["pdf"], accept_multiple_files=True)

    if uploaded_pdfs:
        # Criar o diretório "temp" se não existir
        if not os.path.exists("temp"):
            os.makedirs("temp")
        
        # Processar cada PDF separadamente
        for uploaded_pdf in uploaded_pdfs:
            file_path = os.path.join("temp", uploaded_pdf.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_pdf.getbuffer())
            
            # Processar o PDF
            with st.spinner(f'Processando {uploaded_pdf.name}...'):
                pdf_text = process_pdf(file_path)
                summary_and_explanation = generate_summary_and_explanation(pdf_text, uploaded_pdf.name)
            
            # Exibir o resultado
            st.markdown(summary_and_explanation)
            st.write("---")  # Separador para diferentes arquivos

            # Botão para copiar o texto gerado
            st.write(f"### Copiar o texto gerado para {uploaded_pdf.name}")
            st.code(summary_and_explanation, language="markdown")

            # Botão para baixar o resumo e a explicação
            st.download_button(
                label=f"Baixar Resumo e Explicação para {uploaded_pdf.name}",
                data=summary_and_explanation,
                file_name=f"resumo_e_explicacao_{uploaded_pdf.name}.md",
                mime="text/markdown"
            )

if __name__ == "__main__":
    main()
