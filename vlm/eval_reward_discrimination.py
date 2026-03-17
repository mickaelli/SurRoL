import os
import argparse
import requests
import base64
from pathlib import Path
from PIL import Image
import io
from statistics import mean

def img_to_b64(img_path: Path) -> str:
    img = Image.open(img_path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def get_vlm_score(url: str, current_image_b64: str, goal_image_b64: str = None) -> float:
    payload = {
        "current_image": current_image_b64,
        "task_description": "将视野中心移动到红色方块",
        "score_range": [-1.0, 1.0],
    }
    if goal_image_b64:
        payload["goal_image"] = goal_image_b64
        
    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("score", -999.0)
    except Exception as e:
        print(f"  [Error calling VLM Server] {e}")
        return -999.0

def main():
    parser = argparse.ArgumentParser(description="Test VLM Reward Scorer discrimination power")
    parser.add_argument("--vlm-url", type=str, default="http://localhost:8000/score")
    parser.add_argument("--dataset-dir", type=str, default="",
                        help="Path to expert_<task>/frames. If empty, auto-picks from vlm/dataset/.")
    parser.add_argument("--goal-image", type=str, default=None, help="Optional Goal Image if WITH_GOAL")
    args = parser.parse_args()

    if not args.dataset_dir:
        candidates = [
            "vlm/dataset/expert_static_track/frames",
            "vlm/dataset/expert_statictrack/frames",
            "vlm/dataset/expert_active_track/frames",
        ]
        picked = None
        for c in candidates:
            if Path(c).exists():
                picked = c
                break
        if picked is None:
            for d in Path("vlm/dataset").glob("expert_*/frames"):
                if d.is_dir():
                    picked = str(d)
                    break
        if picked is None:
            raise FileNotFoundError(
                "Could not auto-locate any expert_<task>/frames under vlm/dataset/. Please pass --dataset-dir."
            )
        args.dataset_dir = picked

    # 替换为你实际存在的文件夹和图像！
    # 如果不知道确切路径，可以自己造几个对应的空文件结构把成功失败图放进去
    
    # 手动选取你的图像路径 (请替换为你本地确实存在的图)
    success_frames = [
        Path(args.dataset_dir) / "ep0000" / "t0018.jpg", # 假设 t0018 是最后接近成功的帧
        Path(args.dataset_dir) / "ep0001" / "t0019.jpg",
        Path(args.dataset_dir) / "ep0002" / "t0017.jpg",
        Path(args.dataset_dir) / "ep0003" / "t0016.jpg",
        Path(args.dataset_dir) / "ep0004" / "t0018.jpg"
    ]
    
    failure_frames = [
        Path(args.dataset_dir) / "ep0000" / "t0000.jpg", # 初始动作，一般较差或看不见
        Path(args.dataset_dir) / "ep0001" / "t0001.jpg",
        Path(args.dataset_dir) / "ep0002" / "t0000.jpg",
        Path(args.dataset_dir) / "ep0003" / "t0002.jpg",
        Path(args.dataset_dir) / "ep0004" / "t0001.jpg"
    ]

    goal_b64 = None
    if args.goal_image and os.path.exists(args.goal_image):
        goal_b64 = img_to_b64(Path(args.goal_image))
        print(f"[*] Using Goal Image: {args.goal_image}")
        
    print(f"[*] Connecting to VLM Server: {args.vlm_url}")
    print("-" * 50)

    # 1. Test Success Frames
    print(">>> Testing SUCCESS Frames (Expect high scores near 1.0)")
    success_scores = []
    for f in success_frames:
        if not f.exists():
            print(f"  [Skip] {f} not found")
            continue
            
        score = get_vlm_score(args.vlm_url, img_to_b64(f), goal_b64)
        success_scores.append(score)
        print(f"  {f.name}: {score:.3f}")
        
    if success_scores:
        print(f"  => Average Success Score: {mean(success_scores):.3f}")

    print("-" * 50)
    
    # 2. Test Failure Frames
    print(">>> Testing FAILURE Frames (Expect low scores near -1.0)")
    failure_scores = []
    for f in failure_frames:
        if not f.exists():
            print(f"  [Skip] {f} not found")
            continue
            
        score = get_vlm_score(args.vlm_url, img_to_b64(f), goal_b64)
        failure_scores.append(score)
        print(f"  {f.name}: {score:.3f}")

    if failure_scores:
        print(f"  => Average Failure Score: {mean(failure_scores):.3f}")

    print("-" * 50)
    if success_scores and failure_scores:
        diff = mean(success_scores) - mean(failure_scores)
        print(f"[*] Difference (Success - Failure): {diff:.3f} (Ideal: > 1.0)")
        if diff < 0.2:
            print("[!] WARNING: VLM discrimination is extremely poor! It gives similar scores everything.")
            print("[!] Suggestion: The VLM Reward Model needs retraining or much harsher Prompts.")
        elif diff > 0.8:
            print("[+] GREAT: VLM shows strong discrimination power as a Reward Model!")
        else:
            print("[-] OK: VLM shows moderate discrimination. RL might learn slowly.")

if __name__ == "__main__":
    main()
