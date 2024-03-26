import gradio as gr
import subprocess

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
    create_repo(
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
        allow_patterns=["*.gguf","$.md"],
        token=hf_token
    )
    print("Uploaded successfully!")

    return "Processing complete."

# Create Gradio interface
iface = gr.Interface(
    fn=process_model, 
    inputs=[
        gr.Textbox(lines=1, label="Model ID"),
        gr.Textbox(lines=1, label="Quantization Methods"),
        gr.Textbox(lines=1, label="Username"),
        gr.Textbox(lines=1, label="Token")
    ], 
    outputs="text"
)

# Launch the interface
iface.launch(debug=True)