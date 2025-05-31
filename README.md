# 🎭 Hallo2: Enhanced Progress Monitoring

[![GitHub](https://img.shields.io/badge/GitHub-View%20Repo-blue?logo=github)](https://github.com/SUP3RMASS1VE/hallo2-SUP3R)

This is an enhanced version of the [Hallo2](https://github.com/SUP3RMASS1VE/hallo2-SUP3R) project that generates realistic talking head animations from a static portrait and audio input. This version adds **improved progress tracking**, **detailed console output**, and **better user feedback** using a Gradio-based interface.

Install via [Pinokio](https://pinokio.computer).

## 🚀 Features

* 🎬 Generate video from a **portrait image** and **driving audio**
* 📊 Real-time **progress monitoring** with stage indicators
* 🕒 **Estimated processing time** based on audio duration
* 🐛 Detailed **error logging and output debugging**
* 🔁 Automatically downloads required models

---

## 🖥️ Demo Interface

![Screenshot 2025-05-31 224520](https://github.com/user-attachments/assets/33b4ea63-1881-4916-b072-1ebf8d7d756e)


---

## 📦 Installation

Install via [Pinokio](https://pinokio-home.netlify.app/).

---

1. **Clone the repository**

```bash
git clone https://github.com/SUP3RMASS1VE/hallo2-SUP3R
cd hallo2-SUP3R
```

2. **Create and activate a Python 3.10 environment**

```bash
python3.10 -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate
```

3. **Install PyTorch with CUDA 11.8 support**

```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
```

4. **Install the remaining dependencies**

```bash
pip install -r requirements.txt
```

---

## 📸 Input Requirements

* **Image**: A clear portrait, square-cropped, face centered and facing forward (≤ 30° rotation).
* **Audio**: WAV format, clear vocals (music in background is okay), English recommended.

---

## 🧪 Run Locally

```bash
python enhanced_app.py
```

The interface will be available at:
**[http://127.0.0.1:7861](http://127.0.0.1:7861)**

---

## 🛠 Parameters

* **Pose Weight**: Controls body pose animation influence
* **Face Weight**: Controls facial movement intensity
* **Lip Weight**: Controls lip-sync sensitivity
* **Face Expand Ratio**: Increases facial area crop if needed

---

## 🧠 How It Works

The tool performs these steps:

1. Validates inputs and checks for required models
2. Generates a temporary configuration for the inference script
3. Launches `inference_long.py` with live progress monitoring
4. Displays console output and updates a progress bar
5. Locates and copies the generated video to the `./outputs` folder

---

## 📂 Output

* Final video saved to: `./outputs/generated_video_<timestamp>.mp4`

---

## 🧩 Troubleshooting

* Check your terminal for detailed logs
* Ensure required models are downloaded (handled automatically)
* Clear and re-crop your image if generation fails
* Audio must be `.wav` format

---

## 📬 Credits

Built with ❤️ by [SUP3RMASS1VE](https://github.com/SUP3RMASS1VE)
Based on the original Hallo2 project.
