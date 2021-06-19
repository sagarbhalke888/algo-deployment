FROM ubuntu:18.04
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda --version

COPY . /app

COPY requirements.txt /app/requirements.txt

RUN conda install -c conda-forge ta-lib

RUN pip3 install -r /app/requirements.txt




WORKDIR /app

EXPOSE 5000

ENTRYPOINT [ "python3" ]

CMD [ "app.py" ]




