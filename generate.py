import os
import time
from typing import Any

import json
import torch
from PIL import Image

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import DegenerateFaceRemover, FaceReducer, FloaterRemover, Hunyuan3DDiTFlowMatchingPipeline


def generate_from_img(img: str, output_dir: str, configs: dict[str, Any] | str):

    # 1. Load configs
    if isinstance(configs, str):
        with open(configs, 'rb') as f:
            config = json.load(f)
    else:
        config = configs

    data_params = config['data']
    infer_params = config['inference']
    model_params = config['model']
    model_params['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    postp_params = config['post_processing']

    obj_name = os.path.splitext(os.path.basename(img))[0]
    output_dir = os.path.join(output_dir, obj_name)
    ext = data_params['output_file_type']
    os.makedirs(output_dir, exist_ok=True)

    # 2. Load image
    image = Image.open(img).convert('RGBA')

    # 3. Remove backgroud from image
    rembg = BackgroundRemover()
    image = rembg(image)
    #image.save(os.path.join(output_dir, f"{obj_name}_rembg.png"))
    infer_params['image'] = image

    # 4. Model pipelines
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(**model_params)

    # 5. Generate model
    t0 = time.time()
    mesh = pipeline(**infer_params)[0]
    t1 = time.time()
    print(f"--- Model Shape Generation: {t1 - t0:.3f} secs ---")

    # 6. Perform mesh post-processing
    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    mesh = FaceReducer()(mesh, **postp_params)

    # 7. Output model mesh
    mesh.export(os.path.join(output_dir, f"{obj_name}_mesh.{ext}"))

    # 8. Empty cuda cache
    torch.cuda.empty_cache()

    return mesh


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--img', '-i', type=str, required=True, help="Path to input image")
    parser.add_argument('--output-dir', '-o', type=str, default="results", help="Path to directory where results will be output")
    parser.add_argument('--config-file', '-c', type=str, default="gen_config.json", help="JSON file with 3D Generator configuration options")
    args = parser.parse_args()

    generate_from_img(args.img, args.output_dir, args.config_file)