import streamlit as st
import json
import requests
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap
import json
import requests
 
sido = "서울"
key = "Q6Pi1SOKRdJYkuof91rCyPJiG4DFWPgMn7pLx5kEqUSB9orJBPf7qzWBVLUuG4MsUVwN+Rkaj+W4v7oxsxiwRw=="
def seoul_pm_query(sido,key="Q6Pi1SOKRdJYkuof91rCyPJiG4DFWPgMn7pLx5kEqUSB9orJBPf7qzWBVLUuG4MsUVwN+Rkaj+W4v7oxsxiwRw=="):
    url = 'http://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getCtprvnRltmMesureDnsty'
    params ={'serviceKey' : key, 'returnType' : 'json', 'numOfRows' : '100', 'pageNo' : '1', 'sidoName' : sido, 'ver' : '1.0' }
 
    response = requests.get(url, params=params)
    content = response.content.decode('utf-8')
    data = json.loads(content)
    return data
 
 
def parse_air_quality_data(data):
    items = data['response']['body']['items']
    air_quality_info = []
    grade_mapping = {
    "1": '좋음',
    "2": '보통',
    "3": '나쁨',
    "4": '매우 나쁨'
}
    for item in items:
        info = {
            '측정소명': item.get('stationName'),
            '날짜': item.get('dataTime'),
            '미세먼지 농도': item.get('pm10Value'),
            '초미세먼지 농도': item.get('pm25Value'),
            '아황산가스 농도': item.get('so2Value'),
            '일산화 탄소 농도': item.get('coValue'),
            '오존 농도': item.get('o3Value'),
            '이산화 질소 농도': item.get('no2Value'),
            '통합대기환경수치': item.get('khaiValue'),
            '통합대기환경지수': item.get('khaiGrade'),
            '미세먼지 등급': grade_mapping.get(item.get('pm10Grade')),
            '초미세먼지 등급': grade_mapping.get(item.get('pm25Grade'))
        }
        air_quality_info.append(info)
    return air_quality_info
 
 
def main():
    st.title("대기질 정보 제공 챗봇")
    text_var = st.text_input("조사할 시도를 입력해주세요")
    quest = st.text_input("조사할 내용을 입력해 주세요")
    clicked_button = st.button("제출")
    if clicked_button:
        data = seoul_pm_query(text_var)
        air_quality_info = parse_air_quality_data(data)
        documents = [Document(page_content=", ".join([f"{key}: {str(info[key])}" for key in ['측정소명', '날짜', '미세먼지 농도', '초미세먼지 농도', '통합대기환경수치',"미세먼지 등급","초미세먼지 등급"]])) for info in air_quality_info]
        print(documents)
        embedding_function = SentenceTransformerEmbeddings(model_name="jhgan/ko-sroberta-multitask")
        db = FAISS.from_documents(documents, embedding_function)
        retriever = db.as_retriever(search_type="similarity", search_kwargs={'k':10, 'fetch_k': 100})
 
        template = """
        너는 미세먼지 정보를 대답하는 봇이야. 반드시 모든 대답은 한글로 해주세요.
        제공하는 맥락만을 사용하여 사용자의 질문에 답해주세요.
        맥락에 나타나지 않은 정보는 알지 못 한다고 안내해야만 합니다.
       
        맥락:
        {context}
       
        사용자의 질문: {question}
        """
        llm = ChatOllama(model="gemma2:9b", temperature=0, base_url="http://127.0.0.1:11434/") #http://127.0.0.1:11434
 
        prompt = ChatPromptTemplate.from_template(template)
       
        chain = RunnableMap({
            "context": lambda x: retriever.get_relevant_documents(x['question']),
            "question": lambda x: x['question']
        }) | prompt | llm
       
        content = chain.invoke({'question': quest}).content
        st.write(content)
 
if __name__ == "__main__":
 
    main()