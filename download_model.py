# download_model.py

import argparse
from modelscope import HubApi
from modelscope.hub.snapshot_download import snapshot_download

def download_model(model_id, local_dir):
    """
    Downloads a model from ModelScope hub to a local directory.
    
    Args:
        model_id (str): The ID of the model on ModelScope (e.g., 'damo/nlp_corom_bert-base-chinese').
        local_dir (str): The local directory to save the model files.
    """
    print(f"Downloading model '{model_id}' to '{local_dir}'...")
    snapshot_download(
        model_id,
        cache_dir=local_dir,
        revision='master' # or specify a specific version
    )
    print("Download complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a model from ModelScope.")
    parser.add_argument(
        "--model_id", 
        type=str, 
        default="AI-ModelScope/bert-base-uncased", 
        help="The model ID from ModelScope Hub."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./models/bert-base-uncased", 
        help="The directory to save the downloaded model."
    )
    args = parser.parse_args()
    
    api=HubApi()
    api.login('5e159a6a-ba99-4abc-8f3d-2f4d41f368ce')
    
    # The actual model files will be inside a subdirectory, so we use the parent dir for cache_dir
    download_model(args.model_id, args.output_dir)
    print(f"\nModel saved in a subdirectory inside '{args.output_dir}'.")
    print(f"When using train.py, point --model_path to the specific model folder (e.g., '{args.output_dir}/damo/nlp_corom_bert-base-uncased').")
    