"""
combine a DPO-stage and an SFT-stage LoRA into a single persona adapter
"""

import os, json, hashlib, shutil
import torch as t
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model_names = {
    "llama-3.1-8b-it": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen-2.5-7b-it": "Qwen/Qwen2.5-7B-Instruct",
    "gemma-3-4b-it": "google/gemma-3-4b-it",
    "qwen-2.5-14b-it": "Qwen/Qwen2.5-14B-Instruct",
    "qwen-2.5-32b-it": "Qwen/Qwen2.5-32B-Instruct",
}


def sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def delta_norm(adapter_dir: str) -> float:
    """sum of ||B @ A||_F over the adapter's modules, in float64"""
    weights = load_file(f"{adapter_dir}/adapter_model.safetensors")
    total = 0.0
    for key, A in weights.items():
        if "lora_A" not in key:
            continue
        B = weights[key.replace("lora_A", "lora_B")]
        total += t.linalg.matrix_norm(B.double() @ A.double()).item()
    return total


def main(
    dpo: str,
    sft: str,
    out: str,
    base_model: str,
    combination_type: str,
    svd_rank: int,
    weights: list[float],
    device: str = "auto",
) -> None:
    if os.path.exists(out) and os.listdir(out):
        shutil.rmtree(out)
    os.makedirs(out, exist_ok=True)

    # PEFT 0.20.0's svd path applies the alpha/r scaling twice - once when it scales the
    # weights and again in get_delta_weight - while the linear path applies it once. so
    # halve the requested weights on the svd path to land on the effective ones.
    # (measured on tiny r4/alpha8 adapters, 2026-09-07; if a PEFT release fixes this the
    # identity test flips loudly and these go back to the effective weights.)
    call_weights = [w / 2 for w in weights] if combination_type == "svd" else list(weights)

    # The merge is adapter-only arithmetic (B@A per module, then an SVD of the summed delta),
    # but PEFT needs the base loaded to hang the adapters on, so the base is loaded in bf16
    # on either device: its weights never enter the arithmetic. The adapters are loaded at
    # the same dtype and PEFT's default autocast_adapter_dtype upcasts them to float32, so
    # B@A, the sum and the SVD all run in float32 (CPU SVD has no bf16 kernel) and the
    # combined adapter is saved in float32 -- the same path, dtypes and artifact as the
    # GPU build. --device cpu exists because the 32B combination OOMed on a 141 GB H200
    # (2026-09-08); a first CPU version loaded the base in float32 and sat at 229 GiB
    # against a 234 GiB container limit, thrashing page cache until it was OOM-killed.
    on_cpu = device == "cpu"
    dtype = t.bfloat16
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map="cpu" if on_cpu else "auto",
        trust_remote_code=True,
    )

    # load each lora adapter
    model = PeftModel.from_pretrained(base, dpo, adapter_name="dpo", torch_dtype=dtype)
    _     = model.load_adapter(sft, adapter_name="sft", torch_dtype=dtype)
    # saved under "default" so save_pretrained writes straight into `out` rather than
    # nesting one directory per adapter, which is what upstream's mv/rm dance undid
    model.add_weighted_adapter(
        adapters         = ["dpo", "sft"],
        weights          = call_weights,
        adapter_name     = "default",
        combination_type = combination_type,
        svd_rank         = svd_rank,
    )
    model.set_adapter("default")
    model.save_pretrained(out, selected_adapters=["default"])

    # carry over tokenizer files and anything else the dpo stage saved
    keep = ["adapter_config.json", "adapter_model.safetensors", "README.md"]
    for f in os.listdir(dpo):
        if f in keep or os.path.isdir(f"{dpo}/{f}"):
            continue
        shutil.copy(f"{dpo}/{f}", f"{out}/{f}")

    # update adapter_config.json with variable base model name
    with open(f"{out}/adapter_config.json", "r") as f:
        config = json.load(f)
    config["base_model_name_or_path"] = base_model
    with open(f"{out}/adapter_config.json", "w") as f:
        json.dump(config, f, indent=2)

    meta = {
        "dpo": {"dir": dpo, "adapter_config_sha256": sha256(f"{dpo}/adapter_config.json")},
        "sft": {"dir": sft, "adapter_config_sha256": sha256(f"{sft}/adapter_config.json")},
        "type": combination_type,
        "svd_rank": svd_rank,
        "effective_weights": list(weights),
        "peft_call_weights": call_weights,
        "base_model_name_or_path": base_model,
        "r": config["r"],
        "lora_alpha": config["lora_alpha"],
        "delta_norm_fro": delta_norm(out),
        "merge_device": ("cpu" if on_cpu else "gpu") + " (bf16 base, float32 adapters + svd, saved float32)",
    }
    with open(f"{out}/combine_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpo", type=str, required=True, help="DPO-stage adapter directory")
    parser.add_argument("--sft", type=str, required=True, help="SFT-stage adapter directory")
    parser.add_argument("--out", type=str, required=True, help="output adapter directory")
    parser.add_argument("--base-model", type=str, required=True,
                        help="HF id, local path, or a key of base_model_names")
    parser.add_argument("--type", type=str, choices=["svd", "linear"], default="svd")
    parser.add_argument("--svd-rank", type=int, default=128)
    parser.add_argument("--weights", type=float, nargs=2, default=[1.0, 0.25],
                        metavar=("DPO", "SFT"), help="EFFECTIVE weights, dpo then sft")
    parser.add_argument("--device", type=str, choices=["auto", "cpu"], default="auto",
                        help="cpu: base (bf16, ~2 bytes/param of RAM) + adapters on the CPU; "
                             "the merge is adapter-only so nothing is lost")
    args = parser.parse_args()
    main(
        args.dpo,
        args.sft,
        args.out,
        base_model_names.get(args.base_model, args.base_model),
        args.type,
        args.svd_rank,
        args.weights,
        args.device,
    )
