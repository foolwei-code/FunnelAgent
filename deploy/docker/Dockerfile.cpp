FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    build-essential cmake git libgrpc++-dev libprotobuf-dev \
    protobuf-compiler libgrpc-dev \
    && rm -rf /var/lib/apt/lists/*

# Install ONNX Runtime
RUN git clone --recursive https://github.com/microsoft/onnxruntime.git /tmp/onnxruntime || true

WORKDIR /app

COPY cpp/ cpp/
COPY proto/ proto/

RUN mkdir -p build && cd build && cmake ../cpp && make -j$(nproc)

EXPOSE 50051

CMD ["./build/reranker_server"]
