import gradio as gr
import matplotlib.pyplot as plt
import torch
import math
import subprocess
import cv2
import os
import uuid
from sliding_infer import (
    transform_params, fixed_interval_sample, preprocess_frames, predict_anomaly,
    x3d, device, model_path
)
from pytorchvideo.data.encoded_video import EncodedVideo

# ===== 영상 브라우저 재생 보장 함수 =====
def ensure_mp4_playable(input_path):
    output_path = f"converted_{uuid.uuid4().hex}.mp4"
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-ar", "44100",
        "-movflags", "+faststart",
        output_path
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        if os.path.exists(output_path):
            return output_path
    except subprocess.CalledProcessError:
        pass
    return input_path

# ===== 슬라이딩 인퍼런스 =====
def run_inference(video_path):
    video = EncodedVideo.from_path(video_path)
    fps = transform_params["frames_per_second"]
    duration = float(video.duration)
    window_sec = (transform_params["num_frames"] * transform_params["sampling_rate"]) / fps
    stride_sec = 1.0

    total_steps = max(1, math.ceil((duration - window_sec) / stride_sec) + 1)
    probs, times = [], []

    for step in range(total_steps):
        start = step * stride_sec
        end = start + window_sec
        try:
            clip = video.get_clip(start_sec=start, end_sec=end)
        except:
            break

        frames = clip["video"]
        if frames.shape[0] == 3:
            frames = frames.permute(1, 2, 3, 0)
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

# ===== 인퍼런스 + 그래프 생성 =====
def analyze_video(video_file):
    times, probs = run_inference(video_file)
    plt.figure(figsize=(8, 4))
    plt.plot(times, probs, marker="o")
    plt.axhline(y=0.5, color='r', linestyle='--', label='Threshold 0.5')
    plt.xlabel("Time (s)")
    plt.ylabel("Anomaly Probability")
    plt.title("Anomaly Probability over Time")
    plt.legend()
    plt.tight_layout()
    plot_path = "plot.png"
    plt.savefig(plot_path)
    plt.close()
    return plot_path

# ===== 프레임 탐색 기능 =====
def get_frame(video_file, frame_idx):
    cap = cv2.VideoCapture(video_file)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

# ===== 업로드 처리 함수 =====
def preprocess_uploaded_video(file_obj):
    if file_obj is None:
        return None
    return ensure_mp4_playable(file_obj.name)

# ===== Gradio UI =====
with gr.Blocks() as demo:
    gr.Markdown("## 🎥 영상 이상 탐지 Web App")

    # 1행
    with gr.Row():
        # 1번 칸: 업로드(File) + 재생(Video)
        with gr.Column():
            file_input = gr.File(label="Upload Video", file_types=[".mp4", ".avi"])
            video_player = gr.Video(label="Playable Video")

        # 2번 칸: 프레임 탐색
        with gr.Column():
            frame_slider = gr.Slider(minimum=0, maximum=100, step=1, label="Frame Index")
            frame_view = gr.Image(label="Frame Preview")

    # 2행
    with gr.Row():
        # 3번 칸: 인퍼런스 그래프
        output_plot = gr.Image(label="Anomaly Probability Plot")
        # 4번 칸: 공백
        gr.HTML("<div style='height:100px'></div>")

    run_button = gr.Button("Run Inference")

    # 업로드 시 mp4 변환 → Video 컴포넌트에 표시
    file_input.upload(preprocess_uploaded_video, inputs=file_input, outputs=video_player)

    # 슬라이더 → 프레임 썸네일 표시
    frame_slider.change(fn=get_frame, inputs=[video_player, frame_slider], outputs=frame_view)

    # 인퍼런스 실행 → 그래프 출력
    run_button.click(fn=analyze_video, inputs=video_player, outputs=output_plot)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
