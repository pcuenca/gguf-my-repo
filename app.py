import os
import shutil
import subprocess

import gradio as gr

from huggingface_hub import create_repo, HfApi
from huggingface_hub import snapshot_download
from huggingface_hub import whoami
from huggingface_hub import ModelCard

from textwrap import dedent

api = HfApi()

def process_model(model_id, q_method, hf_token):
    
    MODEL_NAME = model_id.split('/')[-1]
    fp16 = f"{MODEL_NAME}/{MODEL_NAME.lower()}.fp16.bin"

    username = whoami(hf_token)["name"]
    
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
    repo_id = f"{username}/{MODEL_NAME}-{q_method}-GGUF"
    repo_url = create_repo(
        repo_id = repo_id,
        repo_type="model",
        exist_ok=True,
        token=hf_token
    )
    print("Empty repo created successfully!")


    card = ModelCard.load(model_id)
    card.data.tags = ["llama-cpp"] if card.data.tags is None else card.data.tags + ["llama-cpp"]
    card.text = dedent(
        f"""
        # {repo_id}
        This model was converted to GGUF format from [`{model_id}`](https://huggingface.co/{model_id}) using llama.cpp.
        Refer to the [original model card](https://huggingface.co/{model_id}) for more details on the model.
        ## Use with llama.cpp

        ```bash
        brew install ggerganov/ggerganov/llama.cpp
        ```

        ```bash
        llama-cli --hf-repo {repo_id} --model {qtype.split("/")[-1]} -p "The meaning to life and the universe is "
        ```
        """
    )
    card.save(os.path.join(MODEL_NAME, "README-new.md"))
    
    api.upload_file(
        path_or_fileobj=qtype,
        path_in_repo=qtype.split("/")[-1],
        repo_id=repo_id,
        repo_type="model",
    )

    api.upload_file(
        path_or_fileobj=f"{MODEL_NAME}/README-new.md",
        path_in_repo=README.md,
        repo_id=repo_id,
        repo_type="model",
    )
    
    print("Uploaded successfully!")

    shutil.rmtree(MODEL_NAME)
    print("Folder cleaned up successfully!")

    return (
        f'Find your repo <a href=\'{repo_url}\' target="_blank" style="text-decoration:underline">here</a>',
        "llama.png",
    )    

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
            label="HF Write Token",
            info="https://hf.co/settings/token"
        )
    ], 
    outputs=[
        gr.Markdown(label="output"),
        gr.Image(show_label=False),
    ],
    title="Create your own GGUF Quants!",
    description="Create GGUF quants from any Hugging Face repository! You need to specify a write token obtained in https://hf.co/settings/tokens.",
    article="<p>Find your write token at <a href='https://huggingface.co/settings/tokens' target='_blank'>token settings</a></p>",
    
)

# Launch the interface
iface.launch(debug=True)