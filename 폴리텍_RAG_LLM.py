import re
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions
from chromadb.config import Settings
import json
import asyncio
import aiohttp
import os

async def async_llm_stream(model_name, messages):
    async with aiohttp.ClientSession() as session:
        async with session.post('http://10.30.1.119:11434/api/chat', json={
            "model": model_name,
            "messages": messages,
            "stream": True
        }) as response:
            async for line in response.content:
                if line:
                    try:
                        chunk = json.loads(line)
                        if 'message' in chunk:
                            yield chunk['message']['content']
                    except json.JSONDecodeError:
                        continue

async def generate_answer(query, contexts):
    context_str = ""
    for i, context in enumerate(contexts, 1):
        context_str += f"chunk{i} 내용: {context}\n\n"

    system_message = "당신은 대학 학칙에 대해 잘 알고 있는 도우미입니다. 제공된 정보를 바탕으로 질문에 답해주세요."
    user_message = f"다음은 대학 학칙에 관한 내용입니다. 다음 중 질문과 관련된 chunk 1개만을 사용하여 친절하게 답변해주세요.:\n\n{context_str}\n질문: {query} \n답변:"

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    full_response = ""
    async for content in async_llm_stream("gemma2:9b", messages):
        full_response += content
        print(content, end='', flush=True)
    
    print()  # 줄바꿈을 위해
    return full_response

async def main():
    # ChromaDB 설정
    chroma_persist_dir = "./chroma"
    if not os.path.exists(chroma_persist_dir):
        os.makedirs(chroma_persist_dir)

    client = chromadb.PersistentClient(path=chroma_persist_dir, settings=Settings(
        anonymized_telemetry=False
    ))

    # Sentence Transformer 모델 로드
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')

    # 커스텀 임베딩 함수 정의
    class CustomEmbeddingFunction(embedding_functions.EmbeddingFunction):
        def __init__(self, model):
            self.model = model

        def __call__(self, texts):
            embeddings = self.model.encode(texts)
            return embeddings.tolist()

    embedding_function = CustomEmbeddingFunction(model)

    # 기존 컬렉션이 있으면 가져오고, 없으면 새로 생성
    try:
        collection = client.get_collection(name="university_regulations_politech", embedding_function=embedding_function)
        print("Existing collection found.")
    except ValueError:
        collection = client.create_collection(name="university_regulations_politech", embedding_function=embedding_function)
        print("New collection created.")

    while True:
        # 사용자 쿼리 입력 받기
        user_query = input("질문을 입력하세요 (종료하려면 'quit' 입력): ")
        
        if user_query.lower() == 'quit':
            break

        # ChromaDB에서 관련 정보 검색
        results = collection.query(
            query_texts=[user_query],
            n_results=5  # 최대 5개의 청크를 가져옵니다
        )
        print(results)
        print("--------------")
        # 검색 결과를 리스트로 준비
        contexts = results['documents'][0]

        # Ollama를 사용하여 답변 생성
        print("\n생성된 답변:")
        await generate_answer(user_query, contexts)

if __name__ == "__main__":
    asyncio.run(main())

#휴학 종료 후 45일이 지났는데, 제적사유에 해당하니?

# 생성된 답변:
# 휴학 종료 후 1개월이 넘도록 이유 없이 복학하지 아니한 경우 제적될 수 있습니다.  