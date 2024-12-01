import torch
import timm
import onnx
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

# ViTモデルのロード
model = timm.create_model('vit_base_patch16_224', pretrained=True)
model.eval()

# ダミー入力の作成
dummy_input = torch.randn(1, 3, 224, 224)

# TensorRTのロガー
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_file_path):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

        # ONNXファイルのパース
        with open(onnx_file_path, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

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
    for inp in inputs:
        context.set_tensor_address(inp['name'], inp['device'])
        cuda.memcpy_htod_async(inp['device'], inp['host'], stream)
    for out in outputs:
        context.set_tensor_address(out['name'], out['device'])
    context.execute_async_v3(stream_handle=stream.handle)
    for out in outputs:
        cuda.memcpy_dtoh_async(out['host'], out['device'], stream)
    stream.synchronize()
    return [out['host'] for out in outputs]

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

    # 推論の実行
    output = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    print("Inference output:", output)
