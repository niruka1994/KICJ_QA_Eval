import streamlit as st
import pandas as pd
import os
from io import BytesIO
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from datetime import datetime
import re

# 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
LANGCHAIN_PROJECT = os.environ.get('LANGCHAIN_PROJECT')

# Streamlit 설정
st.title('GPT4o Evaluater')

# 엑셀 파일 업로드
uploaded_file = st.file_uploader("엑셀 파일을 업로드하세요", type=["xlsx"])

if uploaded_file:
    # 엑셀 파일을 데이터프레임으로 불러오기
    df = pd.read_excel(uploaded_file)

    # 데이터프레임 표시
    st.write("업로드된 데이터:")
    st.write(df)

    num_rows = len(df)

    # 새로운 열 추가
    df['Completeness_Score'] = ''
    df['Completeness_Explanation'] = ''
    df['Accuracy_Score'] = ''
    df['Accuracy_Explanation'] = ''
    df['Distinctiveness_Score'] = ''
    df['Distinctiveness_Explanation'] = ''
    df['Ethical_Score'] = ''
    df['Ethical_Explanation'] = ''

    # 질문과 답변을 하나씩 처리
    for i in range(num_rows):
        ChunkNum = df.iloc[i]['Chunk 번호']
        context = df.iloc[i]['Chunk Context']
        question = df.iloc[i]['QUESTION']
        answer = df.iloc[i]['ANSWER']

        # LangChain의 ChatPromptTemplate을 사용하여 프롬프트 구성
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are a professor tasked with evaluating the quality of generated questions and answers based on a given context. 
            Your evaluation will be based on specific metrics, with scores ranging from 1 to 10. 
            A score of 1 indicates the content is unacceptable, while a score of 10 means it meets all criteria perfectly. 
            Please ensure that your scores are objective, applying rigorous and unbiased standards, and reflect a full range of possible scores.


            **Context**:
            {context}
            **Question and Answer**:
            - Question: {Q}
            - Answer: {A}

            Please evaluate the question and answer based on the following criteria:

            1. Completeness Score (1-10): 
               Assess how well the answer includes all relevant information from the context in response to the question.

            2. Accuracy Score (1-10): 
               Evaluate how accurately the question and answer reflect the information or facts provided in the context.

            3. Distinctiveness Score (1-10): 
               Determine if the question and answer are neither too commonplace nor too general, considering the specific context provided.

            4. Ethical Score (1-10): 
               Assess whether the question and answer maintain political neutrality and do not contain discriminatory or unfair content.

            Please provide a score for each criterion along with a brief explanation justifying your evaluation, and write the explanation in Korean.
            Your evaluation should be structured as follows:

            1. Completeness : [Score] - [Explanation]
            2. Accuracy : [Score] - [Explanation]
            3. Distinctiveness : [Score] - [Explanation]
            4. Ethical : [Score] - [Explanation]

            """)
        ])

        llm = ChatOpenAI(model='gpt-4o', temperature=0)
        chain = prompt | llm

        # 각 질문 및 답변에 대해 평가 실행
        response = chain.invoke({
            "context": context,
            "Q": question,
            "A": answer
        })

        st.write("=====================================================================")

        st.markdown(f"**문서번호:** {ChunkNum}", help=context)
        st.write(f"**질문:** {question}")
        st.write(f"**답변:** {answer}")
        st.write(f"**평가결과:**\n {response.content}")

        # 평가 결과 파싱 및 데이터프레임에 추가
        pattern = r'(\d+)\.\s*(\w+)\s*:\s*(\d+)\s*-\s*(.*?)(?=\n\d+\.|\Z)'
        matches = re.findall(pattern, response.content, re.DOTALL)

        for match in matches:
            category = match[1]
            score = int(match[2])
            explanation = match[3].strip()

            if category == 'Completeness':
                df.at[i, 'Completeness_Score'] = score
                df.at[i, 'Completeness_Explanation'] = explanation
            elif category == 'Accuracy':
                df.at[i, 'Accuracy_Score'] = score
                df.at[i, 'Accuracy_Explanation'] = explanation
            elif category == 'Distinctiveness':
                df.at[i, 'Distinctiveness_Score'] = score
                df.at[i, 'Distinctiveness_Explanation'] = explanation
            elif category == 'Ethical':
                df.at[i, 'Ethical_Score'] = score
                df.at[i, 'Ethical_Explanation'] = explanation

    # 엑셀 파일을 메모리에 저장하기 위한 BytesIO 객체 생성
    excel_buffer = BytesIO()
    df.to_excel(excel_buffer, index=False, engine='openpyxl')

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # 다운로드 링크 제공
    st.download_button(
        label="엑셀파일 저장(*저장시 본 페이지 초기화)",
        data=excel_buffer,
        file_name=f"Q_A_Generated_Evaluation_{timestamp}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.stop()
