FROM quay.io/cellgeni/scrna-seq-course:latest

WORKDIR /home/jovyan/

COPY ../transfer/scripts/ /work/scripts/
COPY ../transfer/data/ /work/data/

RUN source <(curl -s https://raw.githubusercontent.com/cellgeni/scRNA.seq.course/master/setup.sh)
RUN chmod -R 777 /work