# Ассистент по работе с лекарственными препаратами.

## Установка

### llama-cpp-python

На windows+CUDA необходимо установить версию llama-cpp-python c поддержкой cuda, для этого используйте
``install_llama.bat``

### conda

Создаем и активируем чистую виртуальную среду при помощи conda:

```
conda create -n rag python=3.10
conda activate rag
```

Устанавливаем зависимости:

```
python -m pip install -r requirements.txt
```

## Запуск

Запускаем FastAPI сервис из директории service:
```
python main.py
```

Документация доступна по адресу `http://localhost:8000/api/openapi`

Запускаем Streamlit сервис из директории streamlit:
```
python -m streamlit run app.py --server.port 8081
```

Сервис доступен по адресу `http://localhost:8081`
