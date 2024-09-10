FROM python:3.10.13-slim

WORKDIR /code

COPY ./app /code/app
COPY ./models /code/models
COPY ./requirements.txt /code/app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/app/requirements.txt
RUN pip install "fastapi[standard]"

ENV PYTHONPATH=/code/

CMD ["fastapi", "run", "app/main.py", "--port", "8080"]