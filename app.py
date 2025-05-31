import gradio as gr
import os
import sys
import subprocess
import shutil
import tempfile
from pathlib import Path
import torch
import yaml
from omegaconf import OmegaConf
import argparse

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_and_download_models():
    """Check if pretrained models exist, download if not"""
    pretrained_dir = "./pretrained_models"
    
    # Define required files for each model directory
    required_files = {
        "hallo2": ["net_g.pth", "net.pth"],
        "stable-diffusion-v1-5/unet": ["config.json", "diffusion_pytorch_model.safetensors"],
        "motion_module": ["mm_sd_v15_v2.ckpt"],
        "face_analysis/models": [
            "face_landmarker_v2_with_blendshapes.task",
            "1k3d68.onnx",
            "2d106det.onnx", 
            "genderage.onnx",
            "glintr100.onnx",
            "scrfd_10g_bnkps.onnx"
        ],
        "wav2vec/wav2vec2-base-960h": [
            "config.json",
            "feature_extractor_config.json",
            "model.safetensors",
            "preprocessor_config.json",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "vocab.json"
        ],
        "audio_separator": ["Kim_Vocal_2.onnx"],
        "sd-vae-ft-mse": ["config.json", "diffusion_pytorch_model.safetensors"]
    }
    
    missing_files = []
    missing_dirs = []
    
    # Check each required directory and its files
    for dir_path, files in required_files.items():
        full_dir_path = os.path.join(pretrained_dir, dir_path)
        
        if not os.path.exists(full_dir_path):
            missing_dirs.append(dir_path)
            missing_files.extend([f"{dir_path}/{file}" for file in files])
        else:
            # Check if all required files exist in the directory
            for file in files:
                file_path = os.path.join(full_dir_path, file)
                if not os.path.exists(file_path):
                    missing_files.append(f"{dir_path}/{file}")
    
    if missing_files or missing_dirs:
        print(f"Missing model directories: {missing_dirs}")
        print(f"Missing model files: {missing_files}")
        print("Downloading pretrained models from HuggingFace...")
        
        try:
            # First try to install git-lfs if not available
            try:
                subprocess.run(["git", "lfs", "version"], check=True, capture_output=True)
                print("Git LFS is available")
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("Installing git-lfs...")
                # Try to install git-lfs (this might not work on all systems)
                try:
                    subprocess.run(["git", "lfs", "install"], check=True)
                    print("Git LFS installed successfully")
                except subprocess.CalledProcessError:
                    print("Warning: Could not install git-lfs automatically. Please install it manually if download fails.")
            
            # Remove existing incomplete directory if it exists
            if os.path.exists(pretrained_dir):
                print(f"Existing pretrained_models directory found...")
                # Check if it's a git repository
                if os.path.exists(os.path.join(pretrained_dir, ".git")):
                    print("Attempting to update existing git repository...")
                    try:
                        # Try to pull latest changes instead of removing
                        result = subprocess.run(["git", "pull"], cwd=pretrained_dir, check=True, capture_output=True, text=True)
                        print("Repository updated successfully!")
                        
                        # Re-check if all files are now present
                        verification_failed = False
                        for dir_path, files in required_files.items():
                            full_dir_path = os.path.join(pretrained_dir, dir_path)
                            if not os.path.exists(full_dir_path):
                                verification_failed = True
                                break
                            else:
                                for file in files:
                                    file_path = os.path.join(full_dir_path, file)
                                    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                                        verification_failed = True
                                        break
                        
                        if not verification_failed:
                            print("✅ All models are now present after git pull!")
                            return True
                        else:
                            print("Some files are still missing after git pull, will re-clone...")
                    except subprocess.CalledProcessError:
                        print("Git pull failed, will try to re-clone...")
                
                # Try to remove the directory with better error handling
                try:
                    print(f"Removing existing directory...")
                    if os.name == 'nt':  # Windows
                        # On Windows, try to handle permission issues
                        import stat
                        def handle_remove_readonly(func, path, exc):
                            if os.path.exists(path):
                                os.chmod(path, stat.S_IWRITE)
                                func(path)
                        shutil.rmtree(pretrained_dir, onerror=handle_remove_readonly)
                    else:
                        shutil.rmtree(pretrained_dir)
                    print("Directory removed successfully")
                except Exception as e:
                    print(f"Warning: Could not remove existing directory: {e}")
                    print("Trying to clone to a different location...")
                    pretrained_dir = "./pretrained_models_new"
            
            # Use git clone method as recommended in the documentation
            cmd = [
                "git", "clone", 
                "https://huggingface.co/fudan-generative-ai/hallo2", 
                pretrained_dir
            ]
            
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("Git clone completed!")
            
            # If we cloned to a different location, move it to the correct location
            if pretrained_dir == "./pretrained_models_new":
                try:
                    if os.path.exists("./pretrained_models"):
                        shutil.rmtree("./pretrained_models")
                    shutil.move("./pretrained_models_new", "./pretrained_models")
                    pretrained_dir = "./pretrained_models"
                    print("Moved models to correct location")
                except Exception as e:
                    print(f"Warning: Could not move to final location: {e}")
                    print("Models are available in ./pretrained_models_new")
            
            # Verify the download was successful by checking files again
            print("Verifying downloaded models...")
            verification_failed = False
            for dir_path, files in required_files.items():
                full_dir_path = os.path.join(pretrained_dir, dir_path)
                if not os.path.exists(full_dir_path):
                    print(f"❌ Directory still missing: {dir_path}")
                    verification_failed = True
                else:
                    for file in files:
                        file_path = os.path.join(full_dir_path, file)
                        if not os.path.exists(file_path):
                            print(f"❌ File still missing: {dir_path}/{file}")
                            verification_failed = True
                        else:
                            # Check file size to ensure it's not empty
                            file_size = os.path.getsize(file_path)
                            if file_size == 0:
                                print(f"❌ File is empty: {dir_path}/{file}")
                                verification_failed = True
                            else:
                                print(f"✅ {dir_path}/{file} ({file_size} bytes)")
            
            if verification_failed:
                print("\n⚠️  Some files are still missing or incomplete. Trying huggingface-cli as fallback...")
                raise subprocess.CalledProcessError(1, cmd)
            else:
                print("✅ All models downloaded and verified successfully!")
                return True
            
        except subprocess.CalledProcessError as e:
            print(f"Git clone failed, trying huggingface-cli as fallback...")
            try:
                # Install huggingface_hub if not already installed
                print("Installing huggingface_hub...")
                subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"], check=True)
                
                # Remove existing directory if it exists
                if os.path.exists(pretrained_dir):
                    shutil.rmtree(pretrained_dir)
                
                # Download models using huggingface-cli
                cmd = [
                    "huggingface-cli", "download", "fudan-generative-ai/hallo2", 
                    "--local-dir", "./pretrained_models"
                ]
                print(f"Running: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
                
                # Verify the download again
                print("Verifying downloaded models...")
                verification_failed = False
                for dir_path, files in required_files.items():
                    full_dir_path = os.path.join(pretrained_dir, dir_path)
                    if not os.path.exists(full_dir_path):
                        print(f"❌ Directory still missing: {dir_path}")
                        verification_failed = True
                    else:
                        for file in files:
                            file_path = os.path.join(full_dir_path, file)
                            if not os.path.exists(file_path):
                                print(f"❌ File still missing: {dir_path}/{file}")
                                verification_failed = True
                            else:
                                file_size = os.path.getsize(file_path)
                                if file_size == 0:
                                    print(f"❌ File is empty: {dir_path}/{file}")
                                    verification_failed = True
                                else:
                                    print(f"✅ {dir_path}/{file} ({file_size} bytes)")
                
                if verification_failed:
                    print("\n❌ Download verification failed even with huggingface-cli.")
                    print("Please manually download the models using:")
                    print("git lfs install")
                    print("git clone https://huggingface.co/fudan-generative-ai/hallo2 pretrained_models")
                    return False
                else:
                    print("✅ Models downloaded successfully using huggingface-cli!")
                    return True
                    
            except subprocess.CalledProcessError as e2:
                print(f"Error downloading models with huggingface-cli: {e2}")
                print("Please manually download the models using:")
                print("git lfs install")
                print("git clone https://huggingface.co/fudan-generative-ai/hallo2 pretrained_models")
                return False
    else:
        print("✅ All required models are present and verified.")
        return True

def run_inference(source_image, driving_audio, pose_weight, face_weight, lip_weight, face_expand_ratio, progress=gr.Progress()):
    """Run the Hallo2 inference"""
    
    # Validate inputs
    if source_image is None:
        return None, "Error: Please upload a source image."
    
    if driving_audio is None:
        return None, "Error: Please upload a driving audio file."
    
    # Check if audio file exists
    if not os.path.exists(driving_audio):
        return None, "Error: Audio file not found."
    
    progress(0.1, desc="Checking models...")
    
    # Check and download models if needed
    if not check_and_download_models():
        return None, "Error: Failed to download required models. Please check your internet connection."
    
    progress(0.2, desc="Preparing inputs...")
    
    # Create temporary directory for this inference
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Save uploaded files
            source_image_path = os.path.join(temp_dir, "source_image.jpg")
            driving_audio_path = os.path.join(temp_dir, "driving_audio.wav")
            
            # Save the uploaded image
            source_image.save(source_image_path)
            
            # Copy the uploaded audio file
            shutil.copy2(driving_audio, driving_audio_path)
            
            # Verify files were created
            if not os.path.exists(source_image_path):
                return None, "Error: Failed to save source image."
            if not os.path.exists(driving_audio_path):
                return None, "Error: Failed to save audio file."
            
            # Create output directory
            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(output_dir, exist_ok=True)
            
            progress(0.3, desc="Setting up configuration...")
            
            # Create a temporary config file
            config_path = os.path.join(temp_dir, "inference_config.yaml")
            
            # Load the base config and modify it
            base_config_path = "./configs/inference/long.yaml"
            if not os.path.exists(base_config_path):
                return None, "Error: Base configuration file not found. Please ensure the project is set up correctly."
            
            with open(base_config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Update config with user inputs
            config['source_image'] = source_image_path
            config['driving_audio'] = driving_audio_path
            config['save_path'] = output_dir
            config['pose_weight'] = pose_weight
            config['face_weight'] = face_weight
            config['lip_weight'] = lip_weight
            config['face_expand_ratio'] = face_expand_ratio
            
            # Save the modified config
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            
            progress(0.4, desc="Running inference...")
            
            # Run the inference script
            cmd = [
                sys.executable, "scripts/inference_long.py", 
                "--config", config_path
            ]
            
            print(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
            
            if result.returncode != 0:
                error_msg = f"Inference failed with return code {result.returncode}:\n"
                if result.stderr:
                    error_msg += f"STDERR: {result.stderr}\n"
                if result.stdout:
                    error_msg += f"STDOUT: {result.stdout}"
                print(error_msg)
                return None, error_msg
            
            progress(0.9, desc="Processing output...")
            
            # Find the generated video
            # The output structure should be: output_dir/source_image/final_video.mp4
            source_name = Path(source_image_path).stem
            video_dir = os.path.join(output_dir, source_name)
            
            print(f"Looking for videos in: {video_dir}")
            
            # Look for the final video file
            video_files = []
            if os.path.exists(video_dir):
                for file in os.listdir(video_dir):
                    if file.endswith(('.mp4', '.avi', '.mov')):
                        video_files.append(os.path.join(video_dir, file))
                        print(f"Found video file: {file}")
            
            # Also check the main output directory
            if not video_files and os.path.exists(output_dir):
                for file in os.listdir(output_dir):
                    if file.endswith(('.mp4', '.avi', '.mov')):
                        video_files.append(os.path.join(output_dir, file))
                        print(f"Found video file in output dir: {file}")
            
            if not video_files:
                # List all files in output directory for debugging
                debug_info = "No output video found. Directory contents:\n"
                if os.path.exists(output_dir):
                    for root, dirs, files in os.walk(output_dir):
                        for file in files:
                            debug_info += f"  {os.path.join(root, file)}\n"
                else:
                    debug_info += "Output directory does not exist."
                
                return None, debug_info
            
            # Copy the video to a permanent location
            output_video_path = f"./outputs/generated_video_{Path(source_image_path).stem}.mp4"
            os.makedirs("./outputs", exist_ok=True)
            shutil.copy2(video_files[0], output_video_path)
            
            progress(1.0, desc="Complete!")
            
            return output_video_path, "Video generated successfully!"
            
        except Exception as e:
            error_msg = f"Error during inference: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return None, error_msg

def create_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(title="Hallo2: Audio-Driven Portrait Animation", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎭 Hallo2: Long-Duration and High-Resolution Audio-driven Portrait Image Animation
        
        Upload a portrait image and an audio file to generate an animated talking portrait!
        
        **Requirements:**
        - **Source Image**: Should be cropped into squares, face should be 50%-70% of the image, facing forward (rotation < 30°)
        - **Audio**: Must be in WAV format, English language, clear vocals (background music is acceptable)
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📸 Input")
                
                source_image = gr.Image(
                    label="Source Portrait Image",
                    type="pil",
                    height=300
                )
                
                driving_audio = gr.Audio(
                    label="Driving Audio (WAV format)",
                    type="filepath"
                )
                
                gr.Markdown("### ⚙️ Parameters")
                
                with gr.Row():
                    pose_weight = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Pose Weight"
                    )
                    face_weight = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Face Weight"
                    )
                
                with gr.Row():
                    lip_weight = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Lip Weight"
                    )
                    face_expand_ratio = gr.Slider(
                        minimum=1.0,
                        maximum=2.0,
                        value=1.2,
                        step=0.1,
                        label="Face Expand Ratio"
                    )
                
                generate_btn = gr.Button(
                    "🎬 Generate Animation",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### 🎥 Output")
                
                output_video = gr.Video(
                    label="Generated Animation",
                    height=400
                )
                
                status_text = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=3
                )
        
        # Event handlers
        generate_btn.click(
            fn=run_inference,
            inputs=[
                source_image,
                driving_audio,
                pose_weight,
                face_weight,
                lip_weight,
                face_expand_ratio
            ],
            outputs=[output_video, status_text],
            show_progress=True
        )
        
        # Examples section
        gr.Markdown("### 📋 Tips")
        gr.Markdown("""
        - **Image Quality**: Use high-quality, well-lit portrait images for best results
        - **Audio Length**: Longer audio files will take more time to process
        - **Parameters**: 
          - **Pose Weight**: Controls head movement and pose changes
          - **Face Weight**: Controls facial expression intensity
          - **Lip Weight**: Controls lip-sync accuracy
          - **Face Expand Ratio**: Controls the face region size for processing
        """)
    
    return demo

if __name__ == "__main__":
    # Check if models exist on startup
    print("🔍 Checking for pretrained models...")
    print("This may take a few minutes if models need to be downloaded...")
    
    models_ready = check_and_download_models()
    
    if not models_ready:
        print("\n❌ Failed to download required models.")
        print("Please check your internet connection and try again.")
        print("You can also manually download the models using:")
        print("git lfs install")
        print("git clone https://huggingface.co/fudan-generative-ai/hallo2 pretrained_models")
        sys.exit(1)
    
    print("\n🚀 All models are ready! Starting Hallo2 interface...")
    
    # Create and launch the interface
    demo = create_interface()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    ) 