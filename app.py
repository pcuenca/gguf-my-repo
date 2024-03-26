import shutil
import subprocess

import gradio as gr

from huggingface_hub import create_repo, HfApi
from huggingface_hub import snapshot_download

api = HfApi()

def process_model(model_id, q_method, username, hf_token):
    
    MODEL_NAME = model_id.split('/')[-1]
    fp16 = f"{MODEL_NAME}/{MODEL_NAME.lower()}.fp16.bin"
    
    snapshot_download(repo_id=model_id, local_dir = f"{MODEL_NAME}", local_dir_use_symlinks=False)
    print("Model downloaded successully!")
    
    fp16_conversion = f"python llama.cpp/convert.py {MODEL_NAME} --outtype f16 --outfile {fp16}"
    subprocess.run(fp16_conversion, shell=True)
    print("Model converted to fp16 successully!")

    qtype = f"{MODEL_NAME}/{MODEL_NAME.lower()}.{q_method.upper()}.gguf"
    quantise_ggml = f"./llama.cpp/quantize {fp16} {qtype} {q_method}"
    subprocess.run(quantise_ggml, shell=True)
    print("Quantised successfully!")

    # Create empty repo
    repo_url = create_repo(
        repo_id = f"{username}/{MODEL_NAME}-{q_method}-GGUF",
        repo_type="model",
        exist_ok=True,
        token=hf_token
    )
    print("Empty repo created successfully!")

    # Upload gguf files
    api.upload_folder(
        folder_path=MODEL_NAME,
        repo_id=f"{username}/{MODEL_NAME}-{q_method}-GGUF",
        allow_patterns=["*.gguf","*.md"],
        token=hf_token
    )
    print("Uploaded successfully!")

    shutil.rmtree(MODEL_NAME)
    print("Folder cleaned up successfully!")

    return f"Processing complete: {repo_url}"

# Create Gradio interface
iface = gr.Interface(
    fn=process_model, 
    inputs=[
        gr.Textbox(
            lines=1, 
            label="Hub Model ID",
            info="Model repo ID"
        ),
        gr.Dropdown(
            ["Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_0", "Q4_K_S", "Q4_K_M", "Q5_0", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0"], 
            label="Quantization Method", 
            info="GGML quantisation type"
        ),
        gr.Textbox(
            lines=1, 
            label="Username",
            info="Your Hugging Face username"
        ),
        gr.Textbox(
            lines=1, 
            label="HF Write Token",
            info="https://hf.co/settings/token"
        )
    ], 
    outputs="text"
)

# Launch the interface
iface.launch(debug=True)