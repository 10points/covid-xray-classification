FROM python:3.11

EXPOSE 5001

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt


CMD ["python3", "app.py", "--host", "0.0.0.0"]