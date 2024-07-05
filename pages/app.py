import streamlit as st
import time
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Словарь для оценок отзывов о ресторанах
sentiment_dict = {
    0: "Отвратительно! Даже не подходите к этому месту!",
    1: "Плохо! Лучше бы остался дома.",
    2: "Удовлетворительно, но не без недостатков. Ешьте на свой страх и риск.",
    3: "Хорошо! Вполне достойное место для трапезы.",
    4: "Отлично! Обязательно вернусь еще раз.",
    5: "Великолепно! Как в раю, только с едой."
}

# Словарь для категорий новостей из Telegram
topic_dict = {
    "мода": 0,
    "технологии": 1,
    "финансы": 2,
    "крипта": 3,
    "спорт": 4
}

# Загрузка обученных моделей и TF-IDF векторайзеров для отзывов о ресторанах
with open('/Users/valeriaalesnikova/Desktop/bootcamp/nlp_project-1/models/tfidf_vectorizer_restaurants_new.pkl', 'rb') as f:
    tfidf_vectorizer_restaurants = pickle.load(f)

with open('/Users/valeriaalesnikova/Desktop/bootcamp/nlp_project-1/models/logregmodel_new.pkl', 'rb') as f:
    logreg_model_restaurants = pickle.load(f)

# Загрузка TF-IDF векторайзера и модели для новостей из Telegram
with open('models/tfidf_vectorizer_tg.pkl', 'rb') as f:
    tfidf_vectorizer_tg = pickle.load(f)

with open('models/vectorsmodel_tg.pkl', 'rb') as f:
    vectors_model_tg = pickle.load(f)

# Функция для классификации отзывов о ресторанах
def classify_review(review):
    try:
        start_time = time.time()
        tfidf_review = tfidf_vectorizer_restaurants.transform([review])
        logreg_prediction = logreg_model_restaurants.predict(tfidf_review)[0]
        logreg_duration = time.time() - start_time
        
        sentiment_text = sentiment_dict.get(logreg_prediction, "Неизвестное настроение")
        
        return {
            "Логистическая регрессия": (logreg_prediction, sentiment_text, logreg_duration)
        }
    except Exception as e:
        return {
            "Ошибка": (None, str(e), 0.0)
        }

def classify_news(news):
    try:
        start_time = time.time()
        tfidf_news = tfidf_vectorizer_tg.transform([news])
        prediction = vectors_model_tg.predict(tfidf_news)[0]
        duration = time.time() - start_time
        topic = topic_dict.get(prediction, "Неизвестная тема")
        
        return prediction, topic, duration
    except Exception as e:
        st.error(f"Ошибка при классификации новости: {str(e)}")
        return None, "Неизвестная тема", 0.0

def title_page():
    st.markdown("""
    ## Задача: разработать multipage-приложение с использованием Streamlit:
    
    **Страница 1 • Классификация отзыва на рестораны**
    
    **Страница 2 • Классификация тематики новостей из телеграм каналов**
    
    Приложение предсказывает категорию новости на основе введенного пользователем текста новости.
    """)

def main():
    st.title("Обработка естественного языка • Natural Language Processing")
    
    # Вызываем функцию для отображения титульной страницы
    title_page()

    # Боковая панель для навигации
    page = st.sidebar.selectbox("Выберите страницу", ["Отзывы о ресторанах", "Новости Telegram"])

    if page == "Отзывы о ресторанах":
        st.header("Классификация отзывов о ресторанах")
        review = st.text_area("Введите отзыв о ресторане:")

        if st.button("Классифицировать"):
            predictions = classify_review(review)
            for model, (pred, sentiment_text, dur) in predictions.items():
                st.write(f"{model}: {sentiment_text} (за {dur:.2f} секунд)")

    elif page == "Новости Telegram":
        st.header("Классификация новостей Telegram")
        news = st.text_area("Введите текст новости:")

        if st.button("Классифицировать"):
            prediction, topic, duration = classify_news(news)
            if prediction is not None:
                st.write(f"Предсказанная тема: {prediction}")
                # st.write(f"Предсказанная тема: {topic} (за {duration:.2f} секунд)")
            else:
                st.write("Ошибка при классификации новости.")

if __name__ == "__main__":
    main()
