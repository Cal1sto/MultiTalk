import gradio as gr
import librosa
import pyloudnorm as pyln
import soundfile as sf
import tempfile
import os
from PIL import Image

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

        timeline = gr.State([])

        with gr.Row():
            image_files = gr.Files(label="Images", file_count="multiple")
            prompt = gr.Textbox(label="Prompt for Wan2.2 T2V 14B", lines=2)

        gallery = gr.Gallery(label="Timeline Preview", show_label=True)

        with gr.Row():
            selected = gr.Number(label="Selected index", value=0, precision=0)
            move_left = gr.Button("◀ Move Left")
            move_right = gr.Button("Move Right ▶")
            delete_btn = gr.Button("Delete")
            edit_btn = gr.Button("Edit BBox")

        edited = gr.Image(
            label="Bounding Box Editor",
            tool="select",
            interactive=True,
            visible=False,
        )
        save_crop = gr.Button("Save Crop", visible=False)

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

        def add_images(files, tl):
            if not files:
                return tl, gr.update(value=tl)
            paths = [f.name if hasattr(f, "name") else f for f in files]
            tl = tl + paths
            return tl, gr.update(value=tl)

        def move(tl, idx, offset):
            idx = int(idx)
            new_idx = idx + offset
            if 0 <= idx < len(tl) and 0 <= new_idx < len(tl):
                tl[idx], tl[new_idx] = tl[new_idx], tl[idx]
            return tl, gr.update(value=tl)

        def delete(tl, idx):
            idx = int(idx)
            if 0 <= idx < len(tl):
                tl.pop(idx)
            return tl, gr.update(value=tl)

        def load_image(tl, idx):
            idx = int(idx)
            if 0 <= idx < len(tl):
                return gr.update(value=tl[idx], visible=True), gr.update(visible=True)
            return gr.update(visible=False), gr.update(visible=False)

        def apply_crop(img, tl, idx):
            idx = int(idx)
            if img is None or not (0 <= idx < len(tl)):
                return tl, gr.update(visible=False), gr.update(visible=False), gr.update(value=tl)
            fd, path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            Image.fromarray(img).save(path)
            tl[idx] = path
            return tl, gr.update(visible=False), gr.update(visible=False), gr.update(value=tl)

        image_files.upload(add_images, inputs=[image_files, timeline], outputs=[timeline, gallery])

        move_left.click(lambda tl, i: move(tl, i, -1), inputs=[timeline, selected], outputs=[timeline, gallery])
        move_right.click(lambda tl, i: move(tl, i, 1), inputs=[timeline, selected], outputs=[timeline, gallery])
        delete_btn.click(delete, inputs=[timeline, selected], outputs=[timeline, gallery])
        edit_btn.click(load_image, inputs=[timeline, selected], outputs=[edited, save_crop])
        save_crop.click(
            apply_crop,
            inputs=[edited, timeline, selected],
            outputs=[timeline, edited, save_crop, gallery],
        )

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
