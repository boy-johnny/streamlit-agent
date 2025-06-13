import streamlit as st
import os
import dotenv

dotenv.load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI

# 從環境變數讀取 API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# 建立 LLM 物件
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=GOOGLE_API_KEY, temperature=0.3)


def get_feedback(question, answer):
    """
    呼叫 Gemini LLM，根據題目與用戶答案，回傳批改意見、分數與標準答案。
    """
    prompt = f"""
    你是一位嚴謹、專業且善於教學的法學專家，專長於行政法與社會福利政策。
    請根據下列五個指標，針對學生的申論題答案進行專業評分與評論，每個指標滿分5分，總分25分。
    請僅根據提供的知識庫內容進行批改與回饋，並給予具體的改進建議。

    - 切題性：答案是否緊扣題目要求，內容有無偏離主題。
    - 結構與邏輯：答案是否有清晰的結構，論述是否有邏輯性與層次。
    - 專業與政策理解：對行政法與社會福利政策的專業知識掌握與應用程度。
    - 批判與建議具體性：是否能提出具體、深入的批判與建議。
    - 語言與表達：語言是否精確、流暢，表達是否清楚。

    請依下列格式回覆：
    1. 五項指標分數（每項5分，並簡要說明評分理由）
    2. 總分
    3. 專業回饋（針對答案優缺點給予具體評論）
    4. 改進建議（明確指出如何提升答案品質）
    5. 參考改進後的範例答案（根據知識庫內容重寫更佳答案）

    題目：{question}
    用戶回答：{answer}
    """
    response = llm.predict(prompt)
    return response


def main():
    """
    Streamlit 主程式，負責用戶互動與顯示批改結果。
    """
    st.title("AI 申論題批改老師 📝")
    question = st.text_area("請輸入申論題題目：")
    answer = st.text_area("請輸入你的答案：")
    if st.button("送出批改"):
        if not question or not answer:
            st.warning("請輸入題目與答案")
        else:
            with st.spinner("AI 批改中..."):
                feedback = get_feedback(question, answer)
            st.subheader("AI 批改結果")
            st.write(feedback)


if __name__ == "__main__":
    main()
