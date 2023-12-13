FROM nvidia/cuda:12.2.0-base-ubuntu22.04
WORKDIR /app
COPY . /app
RUN apt-get update && \
    apt-get install -y python3 && \
    apt-get install -y python3-pip
RUN pip install -r requirements.txt
# cuda envs
ENV CUDA_HOME /usr/local/cuda
ENV PATH ${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH ${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
EXPOSE 8000
