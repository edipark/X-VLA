# ------------------------------------------------------------------------------
# Copyright 2025 2toINF (https://github.com/2toINF)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------

import argparse
import os
import os.path as osp
import torch
from models.modeling_xvla import XVLA
from models.processing_xvla import XVLAProcessor

def main():
    parser = argparse.ArgumentParser(description="Launch XVLA inference FastAPI server")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the pretrained XVLA model directory")
    parser.add_argument('--processor_path', type=str, default=None)
    parser.add_argument('--LoRA_path', type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to load model on (cuda / cpu / auto)")
    parser.add_argument("--port", default=8010, type=int,
                        help="Port number for FastAPI server")
    parser.add_argument("--host", default="0.0.0.0", type=str,
                        help="Host address for FastAPI server")
    parser.add_argument("--disable_slurm", action="store_true", default=False)

    # Action space override (for non-default embodiments like rby1)
    parser.add_argument("--action_mode", type=str, default=None,
                        help="Override action mode (e.g. 'auto' for joint-space robots)")
    parser.add_argument("--real_action_dim", type=int, default=None,
                        help="Real action dimension when action_mode='auto' (e.g. 16 for rby1)")
    parser.add_argument("--max_action_dim", type=int, default=20,
                        help="Max action dimension for padding (default: 20)")

    args = parser.parse_args()

    print("🚀 Starting XVLA Inference Server...")
    print(f"🔹 Model Path  : {args.model_path}")
    print(f"🔹 Device Arg  : {args.device}")
    print(f"🔹 Port        : {args.port}")

    # --------------------------------------------------------------------------
    # Select device automatically
    # --------------------------------------------------------------------------
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"🧠 Using device: {device}")

    # --------------------------------------------------------------------------
    # Load processor (if available)
    # --------------------------------------------------------------------------
    processor = None
    try:
        print("\n🧩 Loading XVLAProcessor...")
        processor_path = args.processor_path if args.processor_path else args.model_path
        processor =  XVLAProcessor.from_pretrained(processor_path)
        print("✅ XVLAProcessor loaded successfully.")
    except Exception as e:
        print(f"⚠️ No processor found or failed to load: {e}")

    # --------------------------------------------------------------------------
    # Load model
    # --------------------------------------------------------------------------
    print("\n📦 Loading XVLA model from pretrained checkpoint...")
    try:
        if args.action_mode is not None:
            from models.configuration_xvla import XVLAConfig
            config = XVLAConfig.from_pretrained(args.model_path)
            config.action_mode = args.action_mode
            if args.real_action_dim is not None:
                config.real_action_dim = args.real_action_dim
            config.max_action_dim = args.max_action_dim
            print(f"🔸 Action space override: mode={config.action_mode}, "
                  f"real_dim={config.real_action_dim}, max_dim={config.max_action_dim}")
            model = XVLA.from_pretrained(
                args.model_path,
                config=config,
                trust_remote_code=True,
                torch_dtype=torch.float32
            ).to(device).to(torch.float32)
        else:
            model = XVLA.from_pretrained(
                args.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float32
            ).to(device).to(torch.float32)
        
        if args.LoRA_path is not None:
            print(f"🔸 Applying LoRA weights from {args.LoRA_path} ...")
            from peft import PeftModel
            model = PeftModel.from_pretrained(
                model,
                args.LoRA_path,
                torch_dtype=torch.float32,
            ).to(device)
            
            print("✅ LoRA weights applied successfully.")
            
            
        print("✅ Model successfully loaded and moved to device.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # --------------------------------------------------------------------------
    # SLURM environment detection
    # --------------------------------------------------------------------------
    node_list = os.environ.get("SLURM_NODELIST")
    job_id = os.environ.get("SLURM_JOB_ID", "none")

    if node_list and not args.disable_slurm:
        print("\n🖥️  SLURM Environment Detected:")
        print(f"   Node list : {node_list}")
        print(f"   Job ID    : {job_id}")

        # Extract host
        try:
            host = ".".join(node_list.split("-")[1:]) if "-" in node_list else node_list
        except Exception:
            host = args.host
    else:
        print("\n⚠️  No SLURM environment detected, defaulting to 0.0.0.0")
        host = args.host

    # --------------------------------------------------------------------------
    # Launch FastAPI server
    # --------------------------------------------------------------------------
    print(f"\n🌐 Launching FastAPI service at http://{host}:{args.port} ...")
    try:
        if hasattr(model, "run"):
            model.run(processor=processor, host=host, port=args.port)
        else:
            print("❌ The loaded model does not implement `.run()` (FastAPI entrypoint).")
    except KeyboardInterrupt:
        print("\n🛑 Server stopped manually.")
    except Exception as e:
        print(f"❌ Server failed to start: {e}")


if __name__ == "__main__":
    main()
