import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
import math
from sliding_infer import (
    transform_params, fixed_interval_sample, preprocess_frames, predict_anomaly,
    x3d, device, model_path
)
from pytorchvideo.data.encoded_video import EncodedVideo

# ===== 슬라이딩 인퍼런스 함수 수정 =====
def run_inference(video_path):
    video = EncodedVideo.from_path(video_path)
    fps = transform_params["frames_per_second"]
    duration = float(video.duration)
    window_sec = (transform_params["num_frames"] * transform_params["sampling_rate"]) / fps
    stride_sec = 1.0

    total_steps = math.ceil((duration - window_sec) / stride_sec) + 1
    probs = []
    times = []

    for step in range(total_steps):
        start = step * stride_sec
        end = start + window_sec
        try:
            clip = video.get_clip(start_sec=start, end_sec=end)
        except:
            break
        frames = clip["video"]  # (C, T, H, W) or (T, H, W, C)
        if frames.shape[0] == 3:
            frames = frames.permute(1, 2, 3, 0)  # (T, H, W, C)
        elif frames.shape[-1] != 3:
            continue

        frames = fixed_interval_sample(frames, transform_params["sampling_rate"], transform_params["num_frames"])
        frames = preprocess_frames(frames).unsqueeze(0).to(device)

        with torch.no_grad():
            feature = x3d(frames).squeeze(0)
            score = predict_anomaly(feature, model_path)
            probs.append(score)
            times.append(start + window_sec / 2)

    return times, probs

# ===== Gradio용 함수 =====
def analyze_video(video_file):
    times, probs = run_inference(video_file)

    # 시각화
    plt.figure(figsize=(8, 4))
    plt.plot(times, probs, marker="o")
    plt.axhline(y=0.5, color='r', linestyle='--', label='Threshold 0.5')
    plt.xlabel("Time (s)")
    plt.ylabel("Anomaly Probability")
    plt.title("Anomaly Probability over Time")
    plt.legend()
    plt.tight_layout()

    # 이미지를 Gradio에서 보여줄 수 있도록 저장
    plot_path = "plot.png"
    plt.savefig(plot_path)
    plt.close()
    return plot_path

# ===== Gradio UI =====
with gr.Blocks() as demo:
    gr.Markdown("## 🎥 영상 이상 탐지 Web App")
    with gr.Row():
        # video_input = gr.Video(label="Upload Video", type="filepath")
        video_input = gr.Video(label="Upload Video")
        output_plot = gr.Image(label="Anomaly Probability Plot")
    run_button = gr.Button("Run Inference")

    run_button.click(fn=analyze_video, inputs=video_input, outputs=output_plot)

if __name__ == "__main__":
    demo.launch()
