FROM python:3.9

WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN pip3 install -r requirements.txt

EXPOSE 8000

COPY . /app

CMD ["streamlit", "run", "app.py", "--server.port=8000"]
