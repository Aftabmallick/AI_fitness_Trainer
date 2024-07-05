import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAI
from fpdf import FPDF
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(max_tokens=1000)

def get_fitness_advice(context):
    template = """I want you to act as a personal trainer. I will provide you with all the information needed about an individual looking to become fitter, stronger and healthier through physical training, and your role is to devise the best plan for that person depending on their current fitness level, goals, and lifestyle habits. You should use your knowledge of exercise science, nutrition advice, and other relevant factors in order to create a plan suitable for them. Based on the details below, create one diet and one exercise plan:
    {context}
    """
    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
                {"context": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
    response = rag_chain.invoke(context)
    return response

def generate_diet_plan_pdf(advice):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.multi_cell(0, 10, advice)
    
    return pdf.output(dest='S').encode('latin1')

def main():
    st.title("AI Fitness Trainer")
    st.write("Get personalized workout and nutrition advice from an AI.")
    
    name = st.text_input("Enter your Name:")
    age = st.number_input("Enter your Age:", min_value=1, step=1)
    gender = st.selectbox("Enter your Gender:", ["Male", "Female", "Other"])
    weight = st.number_input("Enter your Weight (KG):", min_value=1.0, step=0.1)
    height = st.number_input("Enter your Height (CM):", min_value=1.0, step=0.1)
    target_weight = st.number_input("Enter your Target Weight (KG):", min_value=1.0, step=0.1)
    target_duration = st.number_input("Enter your Target Duration (Days):", min_value=1, step=1)
    food = st.text_input("Enter your Food Preference:")
    
    user_goal = f"Name: {name}, Age: {age} Years, Gender: {gender}, Weight: {weight} Kilograms, Height: {height} Centimeter, Target Weight: {target_weight} Kilograms, Target Duration: {target_duration} Days, Food Preference: {food}"
    
    if st.button("Get Advice"):
        if user_goal:
            advice = get_fitness_advice(user_goal)
            st.session_state['advice'] = advice
            st.write(advice)
        else:
            st.write("Please enter your fitness goal.")

    if 'advice' in st.session_state:
        if st.button("Generate PDF"):
            pdf_data = generate_diet_plan_pdf(st.session_state['advice'])
            st.download_button(
                label="Download Diet Plan PDF",
                data=pdf_data,
                file_name=f"diet_and_workout_plan_for_{name}.pdf",
                mime="application/pdf"
            )

if __name__ == "__main__":
    main()
