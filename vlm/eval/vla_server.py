#!/usr/bin/env python
"""SurRoL-VLA · VLM Inference Server (Neuro-Motor Brain)

Run this script in your modern PyTorch/Qwen environment.
It stands up a lightweight HTTP server loaded with your finetuned VLA model.
The SurRoL simulation (client) will send images to this server via HTTP POST,
and this server will return the predicted continuous action array.

Usage
-----
    python vlm/eval/vla_server.py \
        --model Qwen/Qwen2-VL-2B-Instruct \
        --lora-path vlm/out/qwen2vl_vla_lora/lora_weights \
        --port 8000
"""

import argparse
import base64
import io
import json
import logging
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# These require Torch and Transformers
from vlm.model.qwen_vl_vla import load_model_for_inference
from vlm.trainer.infer_vla import predict_action

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] vla_server: %(message)s")
logger = logging.getLogger("vla_server")

# Global variables to hold model state in memory
GLOBAL_MODEL = None
GLOBAL_PROCESSOR = None


class VLAInferenceHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/predict":
            self.send_error(404, "Not Found")
            return

        # Read content length and parse JSON payload
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length).decode('utf-8')
        
        try:
            req_data = json.loads(post_data)
        except json.JSONDecodeError:
            self.send_error(400, "Bad Request: Invalid JSON")
            return
            
        b64_img = req_data.get("image")
        instruction = req_data.get("instruction")
        
        if not b64_img or not instruction:
            self.send_error(400, "Bad Request: Missing 'image' or 'instruction'")
            return
            
        try:
            # Decode base64 to PIL Image
            img_bytes = base64.b64decode(b64_img)
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            self.send_error(400, f"Bad Request: Invalid image format - {e}")
            return
            
        try:
            # Predict action
            pred = predict_action(
                model=GLOBAL_MODEL,
                processor=GLOBAL_PROCESSOR,
                image=image,
                instruction=instruction,
                max_new_tokens=128,
            )
            
            action_list = pred["action"]
            raw_text = pred["raw_text"]
            
            # Send Success Response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            response = json.dumps({"action": action_list, "raw_text": raw_text})
            self.wfile.write(response.encode('utf-8'))
            
        except Exception as e:
            logger.error(f"Inference error: {e}", exc_info=True)
            self.send_error(500, f"Internal Server Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="VLA Inference HTTP Server")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--lora-path", type=Path, default=Path("vlm/out/qwen2vl_vla_lora/lora_weights"))
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    global GLOBAL_MODEL, GLOBAL_PROCESSOR

    logger.info("Starting up VLA Brain Server...")
    logger.info(f"Loading Base: {args.model}")
    logger.info(f"Loading LoRA: {args.lora_path}")
    
    GLOBAL_MODEL, GLOBAL_PROCESSOR = load_model_for_inference(
        base_name_or_path=args.model,
        lora_path=args.lora_path,
        device_map="auto"
    )
    
    logger.info(f"Model loaded and merged successfully into VRAM.")
    
    server_address = (args.host, args.port)
    httpd = HTTPServer(server_address, VLAInferenceHandler)
    
    logger.info(f"🚀 VLA Server listening at http://{args.host}:{args.port}")
    logger.info("Waiting for SurRoL client requests...")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down server.")
        httpd.server_close()


if __name__ == "__main__":
    main()
