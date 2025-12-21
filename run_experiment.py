import sys
import os
import time
from learn import effect_estimate
from eval import evaluation
import json
import traceback
from concurrent.futures import ProcessPoolExecutor

try:
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
except:
    pass

# 从环境变量读取，默认值保留原来的路径
MODELS_JSON = os.environ.get('MODELS_JSON', './examples/model.json')

# 1. 关键修改：把 process_model 移到外面，变成全局函数
def process_model(model_args):
    print("--------------", flush=True) # 加上 flush=True
    try:
        print(f"Processing model: {model_args.get('name', model_args)}", flush=True)
        effect_estimate(model_args)
    except Exception as e:
        print(f"effect_estimate failed for model {model_args.get('name', model_args)}: {e}", file=sys.stderr, flush=True)
        print("--- Full Traceback ---", file=sys.stderr, flush=True)
        print(traceback.format_exc(), file=sys.stderr, flush=True)
    try:
        print(f"Evaluating model: {model_args.get('name', model_args)}", flush=True)
        evaluation(model_args)
    except Exception as e:
        print(f"evaluation failed for model {model_args.get('name', model_args)}: {e}", file=sys.stderr, flush=True)
        print("--- Full Traceback ---", file=sys.stderr, flush=True)
        print(traceback.format_exc(), file=sys.stderr, flush=True)

def run():
    if not os.path.exists(MODELS_JSON):
        raise FileNotFoundError(f"MODELS_JSON file not found: {MODELS_JSON}")
    
    with open(MODELS_JSON, 'r', encoding='utf-8') as f:
        model_args_list = json.load(f)

    # model_args_list = model_args_list[9:10]
    print(model_args_list)
    print("models to process:", len(model_args_list), flush=True)
    
    # model_args_list = model_args_list[:2]
    with ProcessPoolExecutor(max_workers=20) as executor:
        try:
            results = list(executor.map(process_model, model_args_list))
        except KeyboardInterrupt:
            print("Interrupted by user, shutting down...", flush=True)
            executor.shutdown(wait=False)
            sys.exit(1)

if __name__ == "__main__":
    run()
