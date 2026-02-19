"""
Deploy internal-time demo to Hugging Face Spaces.

Usage:
    python deploy_to_hf.py --token hf_YOUR_WRITE_TOKEN [--space marubobiz/internal-time-demo]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def deploy(token: str, space_id: str) -> None:
    api = HfApi(token=token)

    # Create space (no-op if already exists)
    print(f"Creating / updating space: {space_id}")
    api.create_repo(
        repo_id=space_id,
        repo_type="space",
        space_sdk="gradio",
        private=False,
        exist_ok=True,
    )

    # Upload all files from space/ directory
    space_dir = Path(__file__).parent / "space"
    print(f"Uploading files from {space_dir} …")
    api.upload_folder(
        folder_path=str(space_dir),
        repo_id=space_id,
        repo_type="space",
        commit_message="deploy internal-time demo v0.3.1",
    )
    print(f"\nDone! Visit: https://huggingface.co/spaces/{space_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True, help="HF write token (hf_...)")
    parser.add_argument("--space", default="marubobiz/internal-time-demo")
    args = parser.parse_args()
    deploy(args.token, args.space)
