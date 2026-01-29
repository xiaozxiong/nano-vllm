import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    print("--- ModelRunner init: loading model weight from CPU to multi-GPUs")
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")): # find all safetensor
        with safe_open(file, "pt", "cpu") as f: #* cpu: load tensor on cpu first, pt: pytorch format
            for weight_name in f.keys(): # iterate tensor in the file like "model.layers.0.self_attn.q_proj.weight"
                # print(f"--- weight_name: {weight_name}")
                for k in packed_modules_mapping: # iterate gate_up_proj and qkv_proj
                    #* load mlp.gate_up_proj.weight and self_attn.qkv_proj.weight
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v) # replace k with real parameter name v
                        # print(f"--- load_model: param_name = {param_name}")
                        param = model.get_parameter(param_name) # GPU
                        # print(f"--- target param is on GPU: {param.is_cuda}")
                        # each parameter has a custom weight_loader
                        weight_loader = getattr(param, "weight_loader")
                        # load slice of weight
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else: #* other weights
                    param = model.get_parameter(weight_name) # GPU
                    # print(f"--- target param is on GPU: {param.is_cuda}")
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
