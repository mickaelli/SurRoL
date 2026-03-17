import argparse
import base64
import io
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import torch
from PIL import Image

app = FastAPI()

class ScoreRequest(BaseModel):
    current_image: str  # Base64 encoded JPEG
    goal_image: Optional[str] = None  # Base64 encoded JPEG
    task_description: str = "将目标物体移动到指定位置"
    score_prompt: Optional[str] = None
    score_range: tuple = (0.0, 1.0)

# Global state for the models and scorer
_model = None
_processor = None
_scorer_cache = {}

@app.post("/score")
async def score(req: ScoreRequest):
    if _model is None or _processor is None:
        raise HTTPException(status_code=500, detail="Qwen2-VL model not loaded.")

    from vlm.reward.vlm_reward_scorer import VLMRewardScorer
    
    try:
        # Decode current image
        img_bytes = base64.b64decode(req.current_image)
        current_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # Decode goal image if provided
        goal_image = None
        if req.goal_image:
            goal_bytes = base64.b64decode(req.goal_image)
            goal_image = Image.open(io.BytesIO(goal_bytes)).convert("RGB")
            
        # Optional caching to avoid recreating the scorer object
        cache_key = (bool(goal_image), req.task_description, req.score_prompt, req.score_range)
        if cache_key not in _scorer_cache:
            _scorer_cache[cache_key] = VLMRewardScorer(
                model=_model,
                processor=_processor,
                goal_image=goal_image,
                task_description=req.task_description,
                score_prompt=req.score_prompt,
                score_range=req.score_range,
                cache_ttl=0.0
            )
        else:
            # We must update the goal image on the cached scorer dynamically if it changed
            _scorer_cache[cache_key].goal_image = goal_image

        scorer = _scorer_cache[cache_key]
        val = scorer.score(current_image)
        
        return {"score": val, "stats": scorer.get_stats()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen2-VL Reward API Server via FastAPI")
    parser.add_argument("--vlm-model", type=str, default="Qwen/Qwen2-VL-2B-Instruct", help="Path to base model")
    parser.add_argument("--lora-path", type=str, default=None, help="Path to LoRA weights")
    parser.add_argument("--vlm-dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    print("=== Configuration ===============================")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("=================================================")

    # Load Model
    print("Loading VLM reward model...")
    torch_dtype = getattr(torch, args.vlm_dtype, torch.bfloat16)

    if args.lora_path:
        from vlm.model.qwen_vl_vla import load_model_for_inference
        _model, _processor = load_model_for_inference(
            base_name_or_path=args.vlm_model,
            lora_path=args.lora_path,
            torch_dtype=torch_dtype,
        )
    else:
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        _model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.vlm_model, torch_dtype=torch_dtype,
            device_map="auto", trust_remote_code=True,
        )
        _model.eval()
        _processor = AutoProcessor.from_pretrained(
            args.vlm_model, trust_remote_code=True,
        )
        
    print(f"Model successfully loaded. Starting API Server at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
