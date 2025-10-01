import base64
import json
import os
import requests


def load_encode_image(img_path: str):
    """Load and encode image."""
    with open(img_path, 'rb') as f:
        img_bytes = f.read()
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
    return img_b64

def post(host: str, port: int, params: dict):
    """Send POST request."""
    response = requests.post(
        f"http://{host}:{port}/generate",
        headers={'Content-Type': "application/json"},
        data=json.dumps(params),
    )
    return response

def output_result(file_name: str, response: requests.Response, dir_name="results", file_type='glb'):
    """Save result to file."""
    os.makedirs(dir_name, exist_ok=True)
    file_name = f"{os.path.splitext(file_name)[0]}.{file_type}"
    output_path = os.path.join(dir_name, file_name)
    with open(output_path, 'wb') as f:
        f.write(response.content)
    print(f"Output mesh file saved to: {os.path.abspath(output_path)}")
    return output_path

def pipeline(args):
    with open(args.config_file, 'rb') as f:
        config = json.load(f)
    img_b64 = load_encode_image(args.img)
    config['params']['image'] = img_b64
    host, port = config['api']['host'], config['api']['port']
    response = post(host, port, config['params'])
    output_path = output_result(os.path.basename(args.img), response, args.output_dir)
    return response, output_path


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--img', '-i', type=str, required=True)
    parser.add_argument('--output-dir', '-o', type=str, default="results")
    parser.add_argument('--config-file', '-c', type=str, default="gen_config.json")
    parser.add_argument('--host', type=str)
    parser.add_argument('--port', type=int)
    args = parser.parse_args()

    response, output_path = pipeline(args)