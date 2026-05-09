import argparse
from huggingface_hub import login, HfApi
import os

def push_model(token: str, repo_id: str):
    print("Logging in to Hugging Face...")
    login(token=token, add_to_git_credential=False)
    
    adapter_path = "checkpoints/rank_64/final_adapter"
    
    if not os.path.exists(adapter_path):
        print(f"❌ Error: Cannot find adapter at {adapter_path}")
        return
        
    print(f"\n🚀 Creating repository: {repo_id}...")
    api = HfApi()
    api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")
    
    print(f"⬆️ Uploading adapter files from {adapter_path}...")
    api.upload_folder(
        folder_path=adapter_path,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Initial commit: Grounded-SQL Mistral-7B adapter (rank 64)"
    )
    
    print(f"\n✅ Successfully pushed!")
    print(f"🔗 View your model here: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True, help="Your HF Write Token")
    parser.add_argument("--repo", required=True, help="Your HF Username / Repo Name (e.g., mvrhsr/grounded-sql-mistral)")
    args = parser.parse_args()
    
    push_model(args.token, args.repo)
