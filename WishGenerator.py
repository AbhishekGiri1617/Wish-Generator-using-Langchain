from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

template = PromptTemplate(
    template = """
    Write 10 messages that i can send from my side to "{name}" to wish them on the occasion of their {event}, he/she is my {Relation}
    and make them feel special and make sure your response is in hindi language.
    1. Make sure to include a variety of messages, such as heartfelt, funny, and inspirational.
    2. Use a friendly and warm tone to convey your best wishes.
    3. Avoid using any generic or clichéd phrases.
    4. Ensure that the messages are appropriate for the occasion and the recipient's personality.
    5. Your message should be unique and personalized to the recipient and atmost of 1-2 lines.
    """,
    input_variables=["name","Relation" ,"event"]
)

model = ChatGoogleGenerativeAI(model='gemini-2.0-flash-001')


st.header("Personalized Message Generator")
name = st.text_input("Enter the name of the person you want to wish: ", key="name")
Relation = st.text_input("Enter your relation with the person: ", key="Relation")
event = st.text_input("Enter the occasion: ", key="event")

prompt = template.invoke({
    "name": name, 
    "event": event,
    "Relation": Relation
})

if st.button("Summarize"):
    result = model.invoke(prompt)
    st.write(result.content)