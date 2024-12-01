import torch
import timm
import onnx
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import time

# ViTモデルのロード
model = timm.create_model('vit_base_patch16_224', pretrained=True)
model.eval()

# ダミー入力の作成
dummy_input = torch.randn(1, 3, 224, 224)

# TensorRTのロガー
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_file_path):
    with trt.Builder(TRT_LOGGER) as builder:
        network = builder.create_network(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        parser = trt.OnnxParser(network, TRT_LOGGER)

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

        # ONNXファイルのパース
        with open(onnx_file_path, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        # レイヤ情報の出力
        print_layer_info(network)

        # ビルドエンジン
        return builder.build_serialized_network(network, config)

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for i in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(i)
        tensor_shape = engine.get_tensor_shape(tensor_name)
        size = trt.volume(tensor_shape)
        dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
            inputs.append({'name': tensor_name, 'host': host_mem, 'device': device_mem})
        else:
            outputs.append({'name': tensor_name, 'host': host_mem, 'device': device_mem})
    return inputs, outputs, bindings, stream

def do_inference(context, bindings, inputs, outputs, stream):
    start_event = cuda.Event()
    end_event = cuda.Event()

    for inp in inputs:
        context.set_tensor_address(inp['name'], inp['device'])
        cuda.memcpy_htod_async(inp['device'], inp['host'], stream)
    for out in outputs:
        context.set_tensor_address(out['name'], out['device'])

    start_event.record(stream)
    context.execute_async_v3(stream_handle=stream.handle)
    end_event.record(stream)

    for out in outputs:
        cuda.memcpy_dtoh_async(out['host'], out['device'], stream)
    stream.synchronize()

    elapsed_time = start_event.time_since(end_event)
    print(f"Inference time: {elapsed_time} ms")

    return [out['host'] for out in outputs]

def print_layer_info(network):
    print("TensorRT Network Layer Information:")
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        print(f"Layer {i}: {layer.name}, type: {layer.type}")

# エンジンのビルド
serialized_engine = build_engine("vit.onnx")

# デシリアライズしてエンジンを作成
runtime = trt.Runtime(TRT_LOGGER)
engine = runtime.deserialize_cuda_engine(serialized_engine)

# バッファの割り当て
inputs, outputs, bindings, stream = allocate_buffers(engine)

# 推論コンテキストの作成
with engine.create_execution_context() as context:
    # ダミー入力データの準備
    input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)
    np.copyto(inputs[0]['host'], input_data.ravel())

    # FPS計測のための推論実行
    num_iterations = 100
    start_time = time.time()
    for _ in range(num_iterations):
        output = do_inference(context, bindings, inputs, outputs, stream)
    end_time = time.time()

    # FPS計算
    total_time = end_time - start_time
    fps = num_iterations / total_time
    print(f"FPS: {fps:.2f}")

    # 推論の実行と出力表示
    output = do_inference(context, bindings, inputs, outputs, stream)
    print("Inference output:", output)