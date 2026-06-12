#!/usr/bin/env python3
import os
import sys
import tarfile
import yaml
import json
import torch
from safetensors.torch import save_file

def convert_nemo(nemo_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    print(f"正在解壓 NeMo 檔案: {nemo_path}")
    
    ckpt_file = None
    config_file = None
    tokenizer_file = None
    
    with tarfile.open(nemo_path, "r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith(".ckpt"):
                ckpt_file = os.path.join(out_dir, "model_weights.ckpt")
                tar.extract(member, path=out_dir)
                os.rename(os.path.join(out_dir, member.name), ckpt_file)
            elif member.name.endswith("model_config.yaml"):
                config_file = os.path.join(out_dir, "model_config.yaml")
                tar.extract(member, path=out_dir)
                os.rename(os.path.join(out_dir, member.name), config_file)
            elif member.name.endswith(".model") or member.name.endswith(".json"):
                if "tokenizer" in member.name:
                    tokenizer_file = os.path.join(out_dir, os.path.basename(member.name))
                    tar.extract(member, path=out_dir)
                    os.rename(os.path.join(out_dir, member.name), tokenizer_file)

    if not ckpt_file:
        # 有些 nemo 檔案沒有被壓縮成 gzip，只是普通的 tar
        with tarfile.open(nemo_path, "r") as tar:
            for member in tar.getmembers():
                if member.name.endswith(".ckpt"):
                    ckpt_file = os.path.join(out_dir, "model_weights.ckpt")
                    tar.extract(member, path=out_dir)
                    os.rename(os.path.join(out_dir, member.name), ckpt_file)
                elif member.name.endswith("model_config.yaml"):
                    config_file = os.path.join(out_dir, "model_config.yaml")
                    tar.extract(member, path=out_dir)
                    os.rename(os.path.join(out_dir, member.name), config_file)
                elif member.name.endswith(".model") or member.name.endswith(".json"):
                    if "tokenizer" in member.name:
                        tokenizer_file = os.path.join(out_dir, os.path.basename(member.name))
                        tar.extract(member, path=out_dir)
                        os.rename(os.path.join(out_dir, member.name), tokenizer_file)

    if not ckpt_file or not config_file:
        raise ValueError("無法在 NeMo 檔案中找到 .ckpt 或 model_config.yaml")
        
    print("解壓完成，正在加載 PyTorch 權重...")
    # 使用 cpu 載入，避免 GPU 顯存問題
    state_dict = torch.load(ckpt_file, map_location="cpu")
    
    # 如果是 NeMo 的 PyTorch Lightning 格式，state_dict 可能嵌套在 "state_dict" 鍵下
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
        
    print(f"成功加載權重，總計有 {len(state_dict)} 個張量。")
    
    # 將權重轉換為 float16 (或保持 float32)
    # 這裡我們預設保持 float32，但可以視需求轉換
    converted_dict = {}
    for k, v in state_dict.items():
        # Burn 與 PyTorch 的卷積層權重格式一致，但在某些特定層（例如 Linear）上，Burn 期待的形狀可能與 PyTorch 不同（例如 weight 是否轉置）。
        # 通常 SafeTensors 只是保存原始張量，我們可以在 Rust 中加載時做轉置，或者在 Python 端處理。
        # 為了保持 SafeTensors 的通用性，我們在 Python 中不改變 shape，完全在 Rust 端加載時適應。
        # 不過，我們需要確保所有的 tensor 都是 contiguous 的
        converted_dict[k] = v.contiguous()
        
    safetensors_path = os.path.join(out_dir, "model.safetensors")
    print(f"正在保存為 SafeTensors 格式: {safetensors_path}")
    save_file(converted_dict, safetensors_path)
    
    # 讀取並轉換 model_config.yaml 為 json
    with open(config_file, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)
        
    config_json_path = os.path.join(out_dir, "config.json")
    with open(config_json_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)
        
    # 清理臨時的 .ckpt 檔案以節省空間
    os.remove(ckpt_file)
    print(f"轉換成功！輸出目錄：{out_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("使用方法: python convert_nemo_weights.py <nemo_file_path> <output_directory>")
        sys.exit(1)
    convert_nemo(sys.argv[1], sys.argv[2])
