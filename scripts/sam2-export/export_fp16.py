# Converte o export de vídeo do SAM 2.1 tiny (jax-image-tools/sam21-tiny-video-onnx)
# para fp16, para o recorte da versão web (web/src/pipeline/sam2.ts).
#
#   python3 scripts/sam2-export/export_fp16.py <pasta com os 4 .onnx originais> <saída>
#
# Requer onnx, onnxruntime (1.19+) e numpy. Passos:
# 1. Encoder: os dois If (interpolação da posição do Hiera) sempre tomam o ramo
#    else com entrada 1024²; são trocados pelos nós desse ramo. Saída idêntica,
#    e some o shape errado do ramo then que o ORT Web recusa (fixVisionShape).
# 2. fp16 com o conversor do ONNX Runtime (o onnxconverter-common deixa tipos
#    misturados nos subgrafos). Entradas e saídas continuam fp32 (o JS não
#    muda); Range, Resize, Sin, Cos e CumSum (codificações de posição) também.
# 3. Confere cada grafo no modo estrito do ORT e imprime tamanho e SHA-256.
import hashlib, sys
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper
from onnxruntime.transformers.float16 import DEFAULT_OP_BLOCK_LIST, convert_float_to_float16

GRAPHS = ["vision_encoder", "mask_decoder", "memory_attention", "memory_encoder"]
BLOCK = list(DEFAULT_OP_BLOCK_LIST) + ["Range", "Resize", "Sin", "Cos", "CumSum"]


def inline_ifs(m: onnx.ModelProto) -> onnx.ModelProto:
    ifs = [n for n in m.graph.node if n.op_type == "If"]
    if not ifs:
        return m
    probe = onnx.ModelProto()
    probe.CopyFrom(m)
    for n in ifs:
        probe.graph.output.append(helper.make_tensor_value_info(n.input[0], onnx.TensorProto.BOOL, None))
    s = ort.InferenceSession(probe.SerializeToString(), providers=["CPUExecutionProvider"])
    conds = s.run([n.input[0] for n in ifs], {"pixel_values": np.zeros((1, 3, 1024, 1024), np.float32)})
    nodes = []
    for n in m.graph.node:
        if n.op_type != "If":
            nodes.append(n)
            continue
        branch = "then_branch" if bool(np.asarray(conds[ifs.index(n)]).reshape(-1)[0]) else "else_branch"
        g = next(a.g for a in n.attribute if a.name == branch)
        print(f"  {n.name} → {branch}")
        nodes.extend(g.node)
        m.graph.initializer.extend(g.initializer)
        for out_if, out_g in zip(n.output, g.output):
            nodes.append(helper.make_node("Identity", [out_g.name], [out_if], name=f"{n.name}_inline"))
    del m.graph.node[:]
    m.graph.node.extend(nodes)
    onnx.checker.check_model(m)
    return m


def check_strict(path: str):
    o = ort.SessionOptions()
    o.add_session_config_entry("session.strict_shape_type_inference", "1")
    ort.InferenceSession(path, o, providers=["CPUExecutionProvider"])


def main(src: str, dst: str):
    for name in GRAPHS:
        print(name)
        m = onnx.load(f"{src}/{name}.onnx")
        if name == "vision_encoder":
            m = inline_ifs(m)
        m16 = convert_float_to_float16(m, keep_io_types=True, disable_shape_infer=False, op_block_list=BLOCK)
        out = f"{dst}/{name}.onnx"
        onnx.save(m16, out)
        check_strict(out)
        data = open(out, "rb").read()
        print(f"  {len(data)} bytes, sha256 {hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
