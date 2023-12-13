FROM nvidia/cuda:12.2.0-base-ubuntu22.04
WORKDIR /app
COPY . /app
RUN apt-get update && \
    apt-get install -y python3 && \
    apt-get install -y python3-pip && \
    apt install -y ffmpeg
RUN pip install --no-cache-dir -r requirements.txt
RUN mkdir -p /.cache && chmod 777 /.cache
# models
ENV MODEL_NAME=alexsherstinsky/Mistral-7B-v0.1-sharded
ENV LORA_ADAPTERS=9ikloj/m_7b_hd
ENV SR_MODEL small
ENV TRANSFORMERS_CACHE=/.cache
ENV HUGGING_FACE_HUB_TOKEN=hf_yourtoken
# cuda envs
ENV CUDA_HOME /usr/local/cuda
ENV PATH ${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH ${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
