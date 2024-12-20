import streamlit as st
import boto3
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from langchain_aws import AmazonKnowledgeBasesRetriever, ChatBedrock
from langchain_core.prompts import PromptTemplate
import markdown2
import os
import uuid
from datetime import datetime
from dotenv import load_dotenv
import speech_recognition as sr

# Carregar variáveis de ambiente
load_dotenv()
kbDocs = os.getenv('KB_DOCS')
tableName = os.getenv('TABLE_NAME')

# Configuração AWS
region = 'us-east-1'
dynamodb = boto3.resource('dynamodb', region_name=region)
table = dynamodb.Table(tableName)
polly_client = boto3.client('polly', region_name=region)
bedrock_client = boto3.client(
    'bedrock-runtime',
    region_name=region,
    endpoint_url='https://bedrock-runtime.us-east-1.amazonaws.com'
)

# Configuração do modelo
model_kwargs_claude = {
    "temperature": 0,
    "top_k": 500,
    "top_p": 0.999,
}

llm = ChatBedrock(
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
    model_kwargs=model_kwargs_claude,
    client=bedrock_client
)

docs_retriever = AmazonKnowledgeBasesRetriever(
    knowledge_base_id=kbDocs,
    retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 5, 'overrideSearchType': "HYBRID"}},
    include_metadata=True
)

prompt = PromptTemplate.from_template(
    f"You are an assistant of University Startups, and your goal is to help students find their dream career.\n\n"
    f"Inicie com uma saudação \n\n"
    f"Siga essas perguntas para direcionar o usuário. Faça-as em inglês e uma de cada vez: \n\n" 
    f"1- Qual é seu Estado? \n\n"
    f"2- Qual é seu County? \n\n"
    f"3- What are your favorite subjects? \n\n"
    f"4- Are there any more subjects that you like? \n\n"
    f"5- Choose a Career Cluster to explore: \n\n"
    f"6- Choose a Education Level: No Educational Requirements / High School Diploma / Associate's Degree / Bachelor's Degree / Graduate Degree \n\n"
    f"7- Pick an Occupation to Explore: \n\n" 
    f"Evidencie bem as perguntas, quando houver informações a mais.\n\n"
    f"Utilize em suas respostas apenas os dados contidos em sua base de conhecimento e deixe em negrito o que considerar de palavras-chave.\n\n"
    "Contexto:\n{context}\n\n"
    "Histórico da conversa: \n{chat_history}\n\n"
    "Mensagem do usuário: {question}"
)

memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True, input_key="question")

qa_chain = load_qa_chain(
    llm,
    prompt=prompt,
    chain_type="stuff",
    memory=memory,
    verbose=False
)

def save_to_dynamo(session_id, user_input, response):
    timestamp = datetime.utcnow().isoformat()
    table.update_item(
        Key={"session_id": session_id},
        UpdateExpression="SET #input = list_append(if_not_exists(#input, :empty_list), :updated_input), #response = list_append(if_not_exists(#response, :empty_list), :updated_response), #timestamp = :new_timestamp",
        ExpressionAttributeNames={
            "#input": "input",
            "#response": "response",
            "#timestamp": "timestamp",
        },
        ExpressionAttributeValues={
            ":updated_input": [user_input],
            ":updated_response": [response],
            ":new_timestamp": timestamp,
            ":empty_list": [],
        },
    )

def synthesize_audio(text):
    polly_response = polly_client.synthesize_speech(
        Text=text,
        OutputFormat='mp3',
        VoiceId='Camila',
        Engine='neural'
    )
    return polly_response['AudioStream'].read()

def main():
    st.title("University Startups Career Assistant")

    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())

    session_id = st.session_state["session_id"]

    st.write("### Chat History")
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for speaker, message in st.session_state["chat_history"]:
        if speaker == "You":
            st.markdown(f"**{speaker}:** {message}")
        else:
            st.markdown(f"_**{speaker}:** {message}_")

    st.write("---")

    question = st.text_input("Type your question:")

    if st.button("Send"):
        if question.strip():
            st.session_state["chat_history"].append(("You", question))
            response = qa_chain.invoke({"question": question})
            response_text = markdown2.markdown(response["output_text"])
            st.session_state["chat_history"].append(("Assistant", response_text))
            save_to_dynamo(session_id, question, response_text)

    st.write("---")

    if st.button("Reset Chat"):
        st.session_state["chat_history"] = []
        st.session_state["session_id"] = str(uuid.uuid4())
        st.success("Chat has been reset.")

if __name__ == "__main__":
    main()
