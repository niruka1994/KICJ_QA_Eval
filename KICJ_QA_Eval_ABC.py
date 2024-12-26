import streamlit as st
import pandas as pd
import os
from io import BytesIO
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from datetime import datetime
import re

# 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
LANGCHAIN_PROJECT = os.environ.get('LANGCHAIN_PROJECT_ABC')

# Streamlit 설정
st.title('GPT4o A,B,C 비교평가 시스템')

# 엑셀 파일 업로드
uploaded_file = st.file_uploader("엑셀 파일을 업로드하세요", type=["xlsx"])

if uploaded_file:
    # 엑셀 파일을 데이터프레임으로 불러오기
    df = pd.read_excel(uploaded_file)

    # 데이터프레임 표시
    st.write("업로드된 데이터:")
    st.write(df)

    # Chunk 번호 추출
    chunk_numbers = df['Chunk 번호'].unique()

    # 새로운 열 추가
    results_df = pd.DataFrame(columns=['Chunk_Number', 'Best_Set', 'Evaluation_Explanation'])

    # LangChain의 ChatPromptTemplate을 사용하여 프롬프트 구성
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are an expert in evaluating the quality of generated question-answer sets based on a given context. 
        Your task is to determine which of the three provided sets (A, B, or C) best utilizes the given context to create augmented Q/A pairs.

        **Context**:
        {context}

        **Question and Answer Sets**:
        Set A:
        1. Q: {Q_A1} A: {A_A1}
        2. Q: {Q_A2} A: {A_A2}
        3. Q: {Q_A3} A: {A_A3}

        Set B:
        1. Q: {Q_B1} A: {A_B1}
        2. Q: {Q_B2} A: {A_B2}
        3. Q: {Q_B3} A: {A_B3}

        Set C:
        1. Q: {Q_C1} A: {A_C1}
        2. Q: {Q_C2} A: {A_C2}
        3. Q: {Q_C3} A: {A_C3}

        Please evaluate each set based on how well it utilizes the given context to create meaningful and diverse question-answer pairs. 
        Consider factors such as relevance to the context, diversity of questions, accuracy of answers, and overall quality of the augmented data.

        Provide your evaluation in the following format:
        1. Best Set: [A/B/C]
        2. Explanation: [A detailed explanation of why you chose this set as the best, comparing it to the other sets. Write this explanation in Korean.]

        Your evaluation should be thorough and objective, focusing on how well each set as a whole utilizes the given context for data augmentation.
        """)
    ])

    llm = ChatOpenAI(model='gpt-4o', temperature=0)
    chain = prompt | llm

    # 각 chunk에 대해 평가 실행
    for chunk_num in chunk_numbers:
        chunk_data = df[df['Chunk 번호'] == chunk_num]

        context = chunk_data.iloc[0]['Chunk Context']

        # A, B, C 세트의 질문과 답변 추출
        q_a = {set_name: [
            {'Q': chunk_data.iloc[j][f'{set_name}:QUESTION'],
             'A': chunk_data.iloc[j][f'{set_name}:ANSWER']}
            for j in range(3)
        ] for set_name in ['A', 'B', 'C']}

        # 평가 실행
        response = chain.invoke({
            "context": context,
            **{f"Q_{set_name}{j + 1}": q_a[set_name][j]['Q'] for set_name in ['A', 'B', 'C'] for j in range(3)},
            **{f"A_{set_name}{j + 1}": q_a[set_name][j]['A'] for set_name in ['A', 'B', 'C'] for j in range(3)}
        })

        st.write("=====================================================================")
        st.markdown(f"**문서번호:** {chunk_num}", help=context)
        for set_name in ['A', 'B', 'C']:
            st.write(f"**{set_name} 세트:**")
            for j, qa in enumerate(q_a[set_name], 1):
                st.write(f"  {j}. 질문: {qa['Q']}")
                st.write(f"     답변: {qa['A']}")
        st.write(f"**평가결과:**\n {response.content}")

        # 평가 결과 파싱 및 데이터프레임에 추가
        pattern = r'1\. Best Set: ([ABC])\n2\. Explanation: (.*)'
        match = re.search(pattern, response.content, re.DOTALL)
        if match:
            best_set = match.group(1)
            explanation = match.group(2).strip()
            new_row = pd.DataFrame({
                'Chunk_Number': [chunk_num],
                'Best_Set': [best_set],
                'Evaluation_Explanation': [explanation]
            })
            results_df = pd.concat([results_df, new_row], ignore_index=True)

    # 결과를 엑셀 파일로 저장
    excel_buffer = BytesIO()
    results_df.to_excel(excel_buffer, index=False, engine='openpyxl')

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # 다운로드 링크 제공
    st.download_button(
        label="엑셀파일 저장(*저장시 본 페이지 초기화)",
        data=excel_buffer,
        file_name=f"Q_A_Set_Evaluation_{timestamp}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.stop()
