import streamlit as st
import boto3
from dotenv import load_dotenv
import os

load_dotenv()
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")


KB_DOCS = st.secrets["KB_DOCS"]
TABLE_NAME = st.secrets["TABLE_NAME"]
AWS_ACCESS_KEY = st.secrets["AWS_ACCESS_KEY"]
AWS_SECRET_KEY = st.secrets["AWS_SECRET_KEY"]
BUCKET_NAME = st.secrets["BUCKET_NAME"] 
REGION_NAME = st.secrets["REGION_NAME"]
os.environ["AWS_ACCESS_KEY_ID"] = st.secrets["AWS_ACCESS_KEY"]
os.environ["AWS_SECRET_ACCESS_KEY"] = st.secrets["AWS_SECRET_KEY"]
os.environ["AWS_DEFAULT_REGION"] = st.secrets["REGION_NAME"]

BUCKET_NAME = "test-compass-us"
REGION_NAME = "us-east-1"

# Cliente S3
s3_client = boto3.client(
    's3',
    region_name=REGION_NAME
)

# Título da aplicação
st.title("Upload e Listagem de Arquivos no AWS S3")

# Aba de navegação
tab1, tab2 = st.tabs(["📤 Upload de Arquivos", "📄 Listar Arquivos"])

# Aba de Upload de Arquivos
with tab1:
    st.header("Upload de Arquivos")
    uploaded_file = st.file_uploader("Selecione um arquivo para enviar ao S3")
    
    if uploaded_file:
        if st.button("Fazer Upload"):
            try:
                s3_client.upload_fileobj(
                    uploaded_file,
                    BUCKET_NAME,
                    uploaded_file.name,
                    ExtraArgs={'ACL': 'public-read'}
                )
                st.success(f"Arquivo '{uploaded_file.name}' enviado com sucesso!")
            except Exception as e:
                st.error(f"Erro ao enviar arquivo: {e}")

# Aba de Listagem de Arquivos
with tab2:
    st.header("Arquivos no Bucket S3")
    if st.button("Listar Arquivos"):
        try:
            response = s3_client.list_objects_v2(Bucket=BUCKET_NAME)
            files = [item['Key'] for item in response.get('Contents', [])]
            
            if files:
                st.write("Arquivos no bucket:")
                for file in files:
                    st.write(f"- {file}")
            else:
                st.info("O bucket está vazio.")
        except Exception as e:
            st.error(f"Erro ao listar arquivos: {e}")
