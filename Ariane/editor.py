import gradio as gr
import librosa
import pyloudnorm as pyln
import soundfile as sf
import tempfile
import os

try:  # support running as module or script
    from .engine_config import DEFAULT_CONFIG, EngineConfig
except ImportError:  # pragma: no cover - fallback when run as script
    from engine_config import DEFAULT_CONFIG, EngineConfig


def normalize(audio_path, target_lufs: float) -> str | None:
    """Normalize an audio file to the target LUFS."""
    if audio_path is None:
        return None
    data, sr = librosa.load(audio_path, sr=None)
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(data)
    normalized = pyln.normalize.loudness(data, loudness, target_lufs)
    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(tmp_path, normalized, sr)
    return tmp_path


def build_interface(config: EngineConfig = DEFAULT_CONFIG):
    with gr.Blocks() as demo:
        gr.Markdown("# Project Ariane Editor")
        with gr.Row():
            image_files = gr.Files(label="Images", file_count="multiple")
            prompt = gr.Textbox(label="Prompt for Wan2.2 T2V 14B", lines=2)
        with gr.Row():
            vocal1 = gr.Audio(label="Vocal 1", type="filepath")
            vocal2 = gr.Audio(label="Vocal 2", type="filepath")
            instrumental = gr.Audio(label="Instrumental", type="filepath")

        with gr.Accordion("Engine Configuration"):
            segment_duration = gr.Number(
                label="Segment duration (s)", value=config.segment_duration
            )
            target_lufs = gr.Number(
                label="Target LUFS", value=config.target_lufs
            )
            upscaler = gr.Textbox(
                label="Upscaler", value=config.upscaler
            )

        normalize_btn = gr.Button("Normalize Audio")
        norm_v1 = gr.Audio(label="Normalized Vocal 1")
        norm_v2 = gr.Audio(label="Normalized Vocal 2")
        norm_inst = gr.Audio(label="Normalized Instrumental")

        def show_gallery(files):
            return files

        gallery = gr.Gallery(label="Timeline Preview", show_label=True)
        image_files.change(show_gallery, inputs=image_files, outputs=gallery)

        normalize_btn.click(
            lambda v1, v2, inst, lufs: (
                normalize(v1, lufs),
                normalize(v2, lufs),
                normalize(inst, lufs),
            ),
            inputs=[vocal1, vocal2, instrumental, target_lufs],
            outputs=[norm_v1, norm_v2, norm_inst],
        )
    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch()
