from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_experimental.text_splitter import SemanticChunker
from langchain.document_loaders import UnstructuredExcelLoader

# 엑셀 파일을 읽고, 분해하여 FAISS vectorstore에 저장
xslx_paths = 'test용사전엑셀.xlsx'
xslx_loader = UnstructuredExcelLoader(xslx_paths, mode="elements")
docs = xslx_loader.load()

# Embedding 설정 및 chunking
embeddings = OpenAIEmbeddings()
semantic_chunker = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
semantic_chunks = semantic_chunker.create_documents([d.page_content for d in docs])

# FAISS vectorstore 생성
vectorstore = FAISS.from_documents(documents=semantic_chunks, embedding=embeddings)

# 번역을 위한 PromptTemplate 및 Chain 설정
trans_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

translation_prompt = PromptTemplate(
    input_variables=["text"],
    template="다음 텍스트를 영어로 번역해주세요: {text}"
)

translation_chain = LLMChain(
    llm=trans_llm,
    prompt=translation_prompt
)

# 쿼리
query = "산독끼는 꼬라니를 좋아해. 바나나도 좋아해"

# Step 1: 쿼리에서 단어를 추출하고 vectorstore에서 관련 문서 검색
query_words = query.split(" ")  # 쿼리에서 단어들을 추출
relevant_texts = []

# query 단어에 대해 vectorstore에서 관련 문서 찾기
for word in query_words:
    search_results = vectorstore.similarity_search(word, k=5)  # 해당 단어와 관련된 문서 검색
    for result in search_results:
        # 검색된 결과에서 실제 텍스트를 relevant_texts에 추가
        relevant_texts.append(result.page_content)

# Step 2: 검색된 문서들을 하나로 결합 (검색된 관련 문서만 결합)
# query에 포함된 단어들만 포함된 부분을 추출하여 연결
filtered_relevant_texts = []
for text in relevant_texts:
    if any(word in text for word in query_words):
        filtered_relevant_texts.append(text)

# 관련된 문서들을 하나로 결합
combined_relevant_text = " ".join(filtered_relevant_texts)  # 관련 문서들을 하나로 결합

# Step 3: 관련 문서들을 번역
translation_result = translation_chain.run(text=combined_relevant_text)

# 결과 출력
print("원본 쿼리:", query)
print("검색된 관련 텍스트:", combined_relevant_text)
print("번역 결과:", translation_result)
