import requests
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from PIL import Image
import json
import os


addr = 'localhost:8000'
url = f'http://{addr}/api/v1/'
ask_api = 'ask'

st.title('Ассистент по работе с лекарственными препаратами')
st.subheader('Привет, готов ответить на ваши вопросы')

# text = st.text_input('Вопрос', '')
with st.form("my_form", clear_on_submit=False):
    user_input = st.text_input(
        "",
        placeholder="Введите сообщение"
    )
    submitted = st.form_submit_button("Отправить")
st.markdown("""
<style>
div[data-testid="stFormSubmitButton"] {
    display: none;
}
</style>
""", unsafe_allow_html=True)

if submitted and user_input:
    # if st.button('получить ответ'):
    payload = {'question': user_input}
    print(url + ask_api)
    with st.spinner('думаю...'):
        response = requests.post(url + ask_api, json=payload)
    if response.status_code != 200:
        st.write(f'Error: {response}')
    else:
        answer = response.json()['answer']
        st.write(answer)
