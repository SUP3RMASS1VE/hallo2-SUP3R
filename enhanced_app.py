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
import threading
import time
import queue

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_and_download_models():
    """Check if pretrained models exist, download if not"""
    # Use the same function from app.py
    from app import check_and_download_models as check_models
    return check_models()

def run_inference_with_progress(source_image, driving_audio, pose_weight, face_weight, lip_weight, face_expand_ratio, progress=gr.Progress()):
    """Run the Hallo2 inference with better progress monitoring"""
    
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
            
            progress(0.4, desc="Starting inference (this may take 10-30 minutes)...")
            
            # Run the inference script with live output
            cmd = [
                sys.executable, "scripts/inference_long.py", 
                "--config", config_path
            ]
            
            print(f"[GRADIO] Running command: {' '.join(cmd)}")
            print(f"[GRADIO] Audio length: {get_audio_duration(driving_audio_path):.1f} seconds")
            print(f"[GRADIO] This will take approximately {estimate_processing_time(driving_audio_path)}")
            print(f"[GRADIO] Check the console output below for detailed progress...")
            
            # Use Popen for live output
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                text=True, 
                bufsize=1, 
                universal_newlines=True,
                cwd="."
            )
            
            # Monitor the process and update progress
            output_lines = []
            segments_processed = 0
            total_segments = None
            
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    output_lines.append(output.strip())
                    print(output.strip())  # Print to console for monitoring
                    
                    # Try to extract progress information
                    if "[INFO] Total segments to process:" in output:
                        try:
                            total_segments = int(output.split(":")[-1].strip())
                            print(f"[GRADIO] Detected {total_segments} segments to process")
                        except:
                            pass
                    
                    elif "[VIDEO] Segment" in output and "/" in output:
                        try:
                            # Extract current segment from "[VIDEO] Segment X/Y"
                            segment_info = output.split("[VIDEO] Segment")[-1].strip()
                            current_segment = int(segment_info.split("/")[0])
                            if total_segments:
                                progress_pct = 0.4 + (0.5 * current_segment / total_segments)
                                progress(progress_pct, desc=f"Processing segment {current_segment}/{total_segments}")
                        except:
                            pass
            
            # Wait for the process to complete
            return_code = process.wait()
            
            if return_code != 0:
                error_msg = f"Inference failed with return code {return_code}:\n"
                error_msg += "\n".join(output_lines[-20:])  # Last 20 lines
                return None, error_msg
            
            progress(0.9, desc="Processing output...")
            
            # Find the generated video
            source_name = Path(source_image_path).stem
            video_dir = os.path.join(output_dir, source_name)
            
            print(f"[GRADIO] Looking for videos in: {video_dir}")
            
            # Look for the final video file
            video_files = []
            if os.path.exists(video_dir):
                for file in os.listdir(video_dir):
                    if file.endswith(('.mp4', '.avi', '.mov')):
                        video_files.append(os.path.join(video_dir, file))
                        print(f"[GRADIO] Found video file: {file}")
            
            # Also check the main output directory and subdirectories
            if not video_files and os.path.exists(output_dir):
                for root, dirs, files in os.walk(output_dir):
                    for file in files:
                        if file.endswith(('.mp4', '.avi', '.mov')):
                            video_files.append(os.path.join(root, file))
                            print(f"[GRADIO] Found video file: {file}")
            
            if not video_files:
                # List all files in output directory for debugging
                debug_info = "[GRADIO] No output video found. Directory contents:\n"
                if os.path.exists(output_dir):
                    for root, dirs, files in os.walk(output_dir):
                        for file in files:
                            debug_info += f"  {os.path.join(root, file)}\n"
                else:
                    debug_info += "[GRADIO] Output directory does not exist."
                
                return None, debug_info
            
            # Copy the video to a permanent location
            timestamp = int(time.time())
            output_video_path = f"./outputs/generated_video_{timestamp}.mp4"
            os.makedirs("./outputs", exist_ok=True)
            shutil.copy2(video_files[0], output_video_path)
            
            progress(1.0, desc="Complete!")
            
            print(f"[GRADIO] Video saved to: {output_video_path}")
            return output_video_path, f"Video generated successfully! Saved to: {output_video_path}"
            
        except Exception as e:
            error_msg = f"[GRADIO] Error during inference: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return None, error_msg

def get_audio_duration(audio_path):
    """Get audio duration in seconds"""
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_wav(audio_path)
        return len(audio) / 1000.0
    except:
        return 0

def estimate_processing_time(audio_path):
    """Estimate processing time based on audio length"""
    duration = get_audio_duration(audio_path)
    if duration <= 30:
        return "5-10 minutes"
    elif duration <= 60:
        return "10-20 minutes"
    elif duration <= 180:
        return "20-45 minutes"
    else:
        return f"{int(duration/60 * 15)}-{int(duration/60 * 25)} minutes"

def create_enhanced_interface():
    """Create an enhanced Gradio interface with better progress monitoring"""
    
    with gr.Blocks(title="Hallo2: Enhanced Progress Monitoring", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎭 Hallo2: Enhanced Progress Monitoring
        
        **This version provides better progress monitoring and console output!**
        
        Upload a portrait image and an audio file to generate an animated talking portrait!
        
        **Requirements:**
        - **Source Image**: Should be cropped into squares, face should be 50%-70% of the image, facing forward (rotation < 30°)
        - **Audio**: Must be in WAV format, English language, clear vocals (background music is acceptable)
        
        **Progress Monitoring:**
        - ✅ Real-time progress updates
        - ✅ Time estimates based on audio length
        - ✅ Console output visible
        - ✅ Detailed error reporting
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
                    "🎬 Generate Animation (Enhanced Progress)",
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
                    label="Status & Progress",
                    interactive=False,
                    lines=5
                )
        
        # Event handlers
        generate_btn.click(
            fn=run_inference_with_progress,
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
        
        # Tips section
        gr.Markdown("### 💡 Enhanced Features")
        gr.Markdown("""
        **What's Different:**
        - **Live Console Output**: Check your terminal/console to see detailed progress
        - **Better Time Estimates**: Shows expected completion time based on audio length
        - **Progress Tracking**: Real-time updates on segment processing
        - **Enhanced Error Reporting**: More detailed error messages if something goes wrong
        
        **Monitoring Your Progress:**
        1. **Watch the Console**: All detailed progress appears in your terminal
        2. **Progress Bar**: Shows overall completion percentage
        3. **Status Updates**: Displays current processing stage
        4. **Time Estimates**: Get realistic completion times
        """)
    
    return demo

if __name__ == "__main__":
    print("[ENHANCED GRADIO] Starting Enhanced Hallo2 Interface...")
    print("[ENHANCED GRADIO] This version provides better progress monitoring!")
    print("[ENHANCED GRADIO] Check this console for detailed progress during inference.")
    print()
    
    # Create and launch the enhanced interface
    demo = create_enhanced_interface()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,  # Different port to avoid conflicts
        share=False,
        show_error=True
    ) 