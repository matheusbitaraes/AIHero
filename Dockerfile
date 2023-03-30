FROM --platform=linux/amd64 python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --upgrade pip
RUN pip install -r /code/requirements.txt

COPY ./src /code/src

RUN mkdir /code/generated_melodies_midi
RUN mkdir /code/.temp
RUN mkdir /.temp

ENV PYTHONPATH=/code/

EXPOSE 8083
CMD ["python", "src/main.py", "src/config.json"]