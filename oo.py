import os
import torch
import threading
import json
import requests
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from diffusers import StableDiffusionPipeline
from datetime import datetime

app = Flask(__name__)
CORS(app)

pipe = None  
default_model = "runwayml/stable-diffusion-v1-5"  
model_lock = threading.Lock() 

def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def load_model(model_id=default_model):
    global pipe
    with model_lock:  
        if pipe is not None:
            print(f"Model {model_id} is already loaded.")
            return  

        print(f"Loading Model: {model_id}")
        clear_gpu_cache()  

        device = "cpu"  
        dtype = torch.float32  

        try:
            pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
            pipe.to(device)
            print("Model Loaded Successfully in CPU mode!")

        except torch.cuda.OutOfMemoryError:
            print("CUDA Out of Memory! Using CPU mode.")
            pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
            pipe.to("cpu")

threading.Thread(target=load_model).start()


def save_image_metadata(image_data):
    metadata_file = "generated_images/image_metadata.json"
    os.makedirs("generated_images", exist_ok=True)

    try:
        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as file:
                metadata = json.load(file)
        else:
            metadata = []

        metadata.append(image_data)

        with open(metadata_file, "w") as file:
            json.dump(metadata, file, indent=4)

    except Exception as e:
        print("Error saving metadata:", e)

@app.route("/gen-post", methods=["POST"])
def generate_image():
    global pipe
    if pipe is None:
        return jsonify({"error": "Model is still loading, please wait."}), 503

    data = request.get_json()
    prompt_text = data.get("prompt", "")
    magic_prompt = data.get("magic_prompt", False)
    aspect_ratio = data.get("aspect_ratio", "1:1")
    visibility = data.get("visibility", "public")
    model_id = data.get("model", default_model)
    color_palette = data.get("color_palette", "auto")
    category = "Explore"

    if not prompt_text:
        return jsonify({"error": "Prompt is required"}), 400

    if magic_prompt:
        prompt_text = f"A highly detailed, cinematic, beautiful {prompt_text}"

    aspect_dict = {
        "1:1": (512, 512),
        "16:9": (768, 512),
        "9:16": (512, 768)
    }
    width, height = aspect_dict.get(aspect_ratio, (512, 512))

    os.makedirs("generated_images", exist_ok=True)

    
    if model_id != default_model:
        load_model(model_id)

    try:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        img_filename = f"image_{timestamp}.png"
        img_path = os.path.join("generated_images", img_filename)

        new_image = pipe(prompt_text, num_inference_steps=20, guidance_scale=7.5).images[0]
        new_image = new_image.resize((width, height))  
        new_image.save(img_path)

        
        image_data = {
            "image_url": f"http://127.0.0.1:5000/get_image/{img_filename}",
            "category": category,
            "visibility": visibility,
            "color_palette": color_palette,
            "timestamp": timestamp
        }
        save_image_metadata(image_data)

        return jsonify(image_data)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/get_image/<path:image_name>", methods=["GET"])
def get_image(image_name):
    """Retrieve generated images"""
    image_path = os.path.join("generated_images", image_name)
    if not os.path.exists(image_path):
        return jsonify({"error": "Image not found"}), 404
    return send_file(image_path, mimetype="image/png")

@app.route("/get_images_by_category/<string:category>", methods=["GET"])
def get_images_by_category(category):
    """Retrieve images by category"""
    metadata_file = "generated_images/image_metadata.json"

    if category.lower() == "explore":
        if not os.path.exists(metadata_file):
            return jsonify({"error": "No images found"}), 404

        with open(metadata_file, "r") as file:
            metadata = json.load(file)

        return jsonify({"images": metadata})

    else:
        
        UNSPLASH_ACCESS_KEY = "qiGNAc1upzF3ud798HpEomjGCxvTaJS9qxPZ0LmPCok"
        query = category.lower()

        try:
            response = requests.get(
                f"https://api.unsplash.com/search/photos?query={query}&client_id={UNSPLASH_ACCESS_KEY}&per_page=5"
            )
            data = response.json()

            images = [{"image_url": img["urls"]["small"], "category": category} for img in data["results"]]
            return jsonify({"images": images})

        except Exception as e:
            print(f"Error fetching {category} images:", e)
            
            dummy_images = [
                {"image_url": "https://source.unsplash.com/random/512x512", "category": category},
                {"image_url": "https://source.unsplash.com/random/513x513", "category": category},
                {"image_url": "https://source.unsplash.com/random/514x514", "category": category}
            ]
            return jsonify({"images": dummy_images})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
