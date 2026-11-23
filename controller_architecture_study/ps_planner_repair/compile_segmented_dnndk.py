"""Experiment: split frozen release001 ID 11 without re-quantizing."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess

from caffe.proto import caffe_pb2
from google.protobuf import text_format


MODEL = "main18_plain_mlp_h4_w1024_seed0"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked(work, identity):
    path = work / identity["path"]
    if sha(path) != identity["sha256"]:
        raise ValueError("Frozen source hash mismatch: " + str(path))
    return path


def split_graphs(graph, weights):
    expected = ["data", "conv_00", "relu_00", "conv_01", "relu_01", "conv_02",
                "relu_02", "conv_03", "relu_03", "conv_04"]
    for net in (graph, weights):
        if [layer.name for layer in net.layer] != expected:
            raise ValueError("Expected exact release001 ID 11 layer chain")
    boundary = list(weights.layer[4].fixed_param.fix_info)[-2:]
    if boundary != [8, 5] or list(weights.layer[5].fixed_param.fix_info)[:2] != boundary:
        raise ValueError("Unexpected or inconsistent fixed-point boundary")
    prefix, suffix = caffe_pb2.NetParameter(), caffe_pb2.NetParameter()
    prefix_weights, suffix_weights = caffe_pb2.NetParameter(), caffe_pb2.NetParameter()
    for net, dest in ((graph, prefix), (weights, prefix_weights)):
        dest.layer.extend(net.layer[:5])
    for net, dest in ((graph, suffix), (weights, suffix_weights)):
        dest.layer.append(net.layer[0])
        dest.layer.extend(net.layer[5:])
        dest.layer[1].bottom[0] = "data"
    suffix.layer[0].input_param.shape[0].dim[1] = 1024
    suffix_weights.layer[0].fixed_param.ClearField("fix_info")
    suffix_weights.layer[0].fixed_param.fix_info.extend(boundary)
    return (prefix, prefix_weights), (suffix, suffix_weights)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--dcf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze = json.loads((args.work/"release_001/freeze.json").read_text())
    model = next(m for m in freeze["models"] if m["id"] == MODEL and m["numeric_id"] == 11)
    graph = text_format.Parse(checked(args.work, model["deploy"]["prototxt"]).read_text(),
                              caffe_pb2.NetParameter())
    weights = caffe_pb2.NetParameter()
    weights.ParseFromString(checked(args.work, model["deploy"]["caffemodel"]).read_bytes())
    if sha(args.dcf) != "cbd84eb836ef0de8c16ee90fd59994cb83237995d56f81156f5cb30f3c98b1da":
        raise ValueError("Expected the original release001 B1152 DCF")
    parts = split_graphs(graph, weights)
    args.output.mkdir(parents=True, exist_ok=False)
    arch = args.output/"arch.json"
    arch.write_text(json.dumps(dict(target="DPUCZDX8G", dcf=str(args.dcf), cpu_arch="arm64")))
    compiled = []
    for name, (part, fixed) in zip(("rb11_seg_a", "rb11_seg_b"), parts):
        folder = args.output/name
        folder.mkdir()
        proto, blob = folder/"deploy.prototxt", folder/"deploy.caffemodel"
        proto.write_text(text_format.MessageToString(part))
        blob.write_bytes(fixed.SerializeToString())
        cmd = ["vai_c_caffe", "-p", str(proto), "-c", str(blob), "-a", str(arch),
               "-o", str(folder), "-n", name]
        with (folder/"compile.log").open("w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        elf = folder/("dpu_"+name+".elf")
        header = elf.read_bytes()[:64]
        graphs = list(folder.glob("*_kernel_graph.gv"))
        if (header[:6] != b"\x7fELF\x02\x01" or struct.unpack_from("<HH", header, 16) != (1, 183)
                or len(graphs) != 1 or re.findall(r"type:\s*(\w+)", graphs[0].read_text()) != ["DPUKernel"]):
            raise ValueError("Expected exactly one compiled AArch64 DPU kernel for " + name)
        compiled.append(dict(kernel=name, elf=elf.name, sha256=sha(elf)))
    report = dict(model=MODEL, boundary="after relu_01", expected_activation=1024,
                  expected_fractional_bits=5, kernels=compiled,
                  scope="compiled only; runtime tensor metadata, numerical parity, and latency need board validation")
    (args.output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
