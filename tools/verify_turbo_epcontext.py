import onnx, json, os
base = r"D:\whisper-app\app\src\main\assets\models\whisper_large_v3_turbo"
meta = json.load(open(os.path.join(base, "metadata.json"), encoding="utf-8"))
from onnx import TensorProto
def dt(t):
    return TensorProto.DataType.Name(t)
def shp(v):
    return [d.dim_value for d in v.type.tensor_type.shape.dim]
ok = True
for bin_name in ("encoder.bin", "decoder.bin"):
    p = os.path.join(base, bin_name.replace(".bin", "_ctx.onnx"))
    m = onnx.load(p)
    g = m.graph
    spec = meta["model_files"][bin_name]
    for kind in ("inputs", "outputs"):
        vals = list(g.input if kind == "inputs" else g.output)
        exp = spec[kind]
        names = [v.name for v in vals]
        if names != list(exp.keys()):
            ok = False
            print(f"MISMATCH ORDER {bin_name} {kind}: wrapper={names} metadata={list(exp.keys())}")
        for v in vals:
            e = exp[v.name]
            got_dt, got_shp = dt(v.type.tensor_type.elem_type).lower(), shp(v)
            if got_dt != e["dtype"] or got_shp != [int(x) for x in e["shape"]]:
                ok = False
                print(f"MISMATCH {bin_name} {v.name}: wrapper={got_dt}{got_shp} metadata={e['dtype']}{e['shape']}")
    n = g.node[0]
    print(f"{os.path.basename(p)}: op={n.op_type} node={n.name} in={len(g.input)} out={len(g.output)} ir={m.ir_version}")
    for a in n.attribute:
        val = onnx.helper.get_attribute_value(a)
        if isinstance(val, bytes): val = val.decode()
        print(f"    {a.name} = {val}")
print("CONTRACT MATCH:", ok)
