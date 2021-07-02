FROM continuumio/miniconda3:4.9.2
WORKDIR /env
COPY ./environment.yml .
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "env", "/bin/bash", "-c"]
WORKDIR /app
RUN python -m spacy download en
# RUN conda init
# RUN echo "conda activate env" > ~/.bashrc

RUN apt install tmux -y
RUN apt install curl -y

ENTRYPOINT ["/bin/bash"]
