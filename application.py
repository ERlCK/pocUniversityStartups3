from operator import itemgetter
import boto3
from flask import Flask, render_template, request, jsonify, send_file, session
import markdown2
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from langchain_aws import AmazonKnowledgeBasesRetriever, ChatBedrock
from langchain_core.prompts import PromptTemplate
import speech_recognition as sr
import os
import uuid
from datetime import datetime
import time
from dotenv import load_dotenv

load_dotenv()
kbDocs = os.getenv('KB_DOCS')
tableName = os.getenv('TABLE_NAME')

application = Flask(__name__)

application.secret_key = "seila-chave-aleatoria-por-enquanto-hihihi"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
application.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
#application.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(application.config['UPLOAD_FOLDER'], exist_ok=True)

region = 'us-east-1'

dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
table = dynamodb.Table(tableName)

polly_client = boto3.client('polly', region_name=region)
bedrock_client = boto3.client(
    'bedrock-runtime',
    region_name=region,
    endpoint_url='https://bedrock-runtime.us-east-1.amazonaws.com'
)

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


############################################
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
    f"No item 5, mostre todas as possíveis Career Clusters, levando em consideração o subject favorito do usuário, dando uma breve explicação do que cada uma é e pergunte qual o usuário deseja explorar. Peça a confirmação para o usuário para assim mostrar o vídeo que represente a Career Cluster escolhida.\n\n"
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

chain = (
    {
        "input_documents": itemgetter("question") | docs_retriever,
        "question": itemgetter("question")
    }
    | qa_chain
)


def save_to_dynamo(session_id, user_input, response):
    timestamp = datetime.utcnow().isoformat()
    item = {
            'session_id': session_id,
            'timestamp':timestamp,
            'input': user_input,
            'response':response
        }
    
    responseDynamo = table.get_item(Key={"session_id": session_id})
    item = responseDynamo.get('Item', {})
    existingInput = item.get('input', "")
    existingResponse = item.get('response', "")
    
    if existingInput:
        updated_input = f"{existingInput}, {user_input}"
    else:
        updated_input = user_input

    if existingResponse:
         updated_response = f"{existingResponse}, {response}"
    else:
        updated_response = response
    
    table.update_item(
        Key = {"session_id": session_id},
        UpdateExpression = "set #input = :updated_input, #response = :updated_response, #timestamp = :new_timestamp",
        ExpressionAttributeNames = {
            "#input": "input", #fiz isso aqui pq tava dando erro de palavra reservada, mas a coluna se chama input
            "#response": "response",
            "#timestamp": "timestamp"
        },
        ExpressionAttributeValues={
            ":updated_input": updated_input,
            ":updated_response": updated_response,
            ":new_timestamp": timestamp
    },

        ReturnValues = "UPDATED_NEW"
    ) 


#Faz a sessão resetar dps de dar f5
@application.before_request
def make_session_temporary():
    session.permanent = False 

@application.before_request
def reset_session_on_refresh():
    if request.endpoint == 'index':
        session.clear()

@application.route('/')
def reload_page():
    return render_template('index.html')

#Reseta o session_id quando clica no botao
@application.route('/reset', methods=['POST'])
def reset_session():
    # Limpa a sessão e gera um novo session_id
    session.clear()
    session['session_id'] = str(uuid.uuid4())
    print(f"Novo session_id: {session['session_id']}") 

    # Retorna uma resposta JSON com o novo session_id
    return jsonify({"message": "Sessão resetada com sucesso!", "session_id": session['session_id']})

@application.route('/response', methods=['POST'])
def get_response():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    session_id = session["session_id"]

    question = request.form['question']
    response = chain.invoke({"question": question})
    response_text = markdown2.markdown(response["output_text"])

    sources_set = set()
    unique_sources = []
    for doc in response["input_documents"]:
        s3_uri = doc.metadata.get('location', {}).get(
            's3Location', {}).get('uri', 'URI not found')
        file_name = s3_uri.split('/')[-1].replace('.txt', '')
        if s3_uri not in sources_set:
            sources_set.add(s3_uri)
            unique_sources.applicationend({'name': file_name, 'url': s3_uri})

    sources_html = " • ".join(
        [f'<a href="{source["url"]}">{source["name"]}</a>' for source in unique_sources])

    if len(response_text) < 200:
        combined_response = response_text
    else:
        combined_response = f"{response_text}<br><strong>Referências:</strong> {sources_html}"

    save_to_dynamo(session_id, question, response_text)
    return jsonify(session_id=session_id, question=question, response=combined_response)

# /////////////////////////////////////////////////////////////////////

@application.route('/response-audio', methods=['POST'])
def get_audio_response():
    try:
        if "session_id" not in session:
            session["session_id"] = str(uuid.uuid4())
        session_id = session["session_id"]

        question = request.form['question']
        response = chain.invoke({"question": question})
        response_text = markdown2.markdown(response["output_text"])
        
        save_to_dynamo(session_id, question, response_text)
        print(f"{response_text} - {session_id}")

    # Gera áudio usando Amazon Polly
        polly_response = polly_client.synthesize_speech(
            Text=response["output_text"],
            OutputFormat='mp3',
            VoiceId='Camila',
            Engine='neural'
        )
    
        audio_file_path = os.path.join(application.config['UPLOAD_FOLDER'], 'response_audio.mp3')
        with open(audio_file_path, 'wb') as audio_file:
            timestamp = int(time.time())
            audio_file.write(polly_response['AudioStream'].read())

        return jsonify(
            question=question,
            session_id=session_id,
            response=response_text,
            audio_url=f'/audio/{os.path.basename(audio_file_path)}?{timestamp}'
        )
    except Exception as e:
        return jsonify({"error": f"Erro ao gerar áudio: {str(e)}"}), 500


@application.route('/audio-response', methods=['POST'])
def audio_response():
    if 'audio' not in request.files:
        return jsonify({"error": "Nenhum arquivo de áudio foi enviado."}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({"error": "Nenhum arquivo selecionado."}), 400

    file_path = os.path.join(application.config['UPLOAD_FOLDER'], audio_file.filename)
    audio_file.save(file_path)

    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio_data = recognizer.record(source)

    try:
        # Converte o áudio para texto
        question = recognizer.recognize_google(audio_data, language='pt-BR')

        # Usa o pipeline existente para gerar resposta
        response = chain.invoke({"question": question})
        response_text = markdown2.markdown(response["output_text"])

        # Limpa o arquivo de áudio temporário
        os.remove(file_path)

        return jsonify(question=question, response=response_text)
    except sr.UnknownValueError:
        return jsonify({"error": "Não foi possível entender o áudio."}), 400
    except sr.RequestError as e:
        return jsonify({"error": f"Erro no reconhecimento de áudio: {e}"}), 500

@application.route('/audio/<filename>')
def get_audio(filename):
    file_path = os.path.join(application.config['UPLOAD_FOLDER'], filename)
    return send_file(file_path, mimetype='audio/mpeg')

@application.route('/reset', methods=['POST'])
def reset():
    memory.clear()
    return '', 204


if __name__ == '__main__':
    application.run(debug=True)