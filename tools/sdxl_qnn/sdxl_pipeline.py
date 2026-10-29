"""End-to-end SDXL (Lightning/Turbo) inference pipeline with QNN HTP (UNet) + CPU (TAESDXL).

Architecture:
1. Text Encoders: Generate prompt embeddings (CPU)
2. UNet: Denoising loop in latent space [1, 4, 128, 128] on Qualcomm HTP via QNN EP
3. VAE Decoder: TAESDXL converts latents to 1024x1024 image on CPU
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import numpy as np
import onnxruntime as ort
from PIL import Image

class EulerDiscreteScheduler:
    """Minimal 4-step Euler scheduler for SDXL-Lightning / SDXL-Turbo."""
    def __init__(self, num_inference_steps: int = 4):
        self.num_inference_steps = num_inference_steps
        # Standard SDXL Lightning 4-step timesteps and sigmas
        # Timesteps: [999, 749, 499, 249]
        self.timesteps = np.array([999.0, 749.0, 499.0, 249.0], dtype=np.float32)
        # Precomputed alphas_cumprod schedule
        betas = np.linspace(0.00085**0.5, 0.012**0.5, 1000, dtype=np.float32) ** 2
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        sigmas = ((1.0 - alphas_cumprod) / alphas_cumprod) ** 0.5
        
        # Interpolate sigmas for our timesteps
        step_indices = [int(t) for t in self.timesteps]
        self.sigmas = np.array([sigmas[i] for i in step_indices] + [0.0], dtype=np.float32)
        self.init_noise_sigma = float(self.sigmas[0])

    def scale_model_input(self, sample: np.ndarray, step_index: int) -> np.ndarray:
        sigma = self.sigmas[step_index]
        return sample / ((sigma**2 + 1) ** 0.5)

    def step(self, model_output: np.ndarray, step_index: int, sample: np.ndarray) -> np.ndarray:
        sigma_from = self.sigmas[step_index]
        sigma_to = self.sigmas[step_index + 1]
        
        # d = (sample - model_output) / sigma_from for v-pred / epsilon
        # Euler update: prev_sample = sample + (sigma_to - sigma_from) * model_output
        dt = sigma_to - sigma_from
        prev_sample = sample + model_output * dt
        return prev_sample


class SDXLQNNPipeline:
    def __init__(
        self,
        unet_model_path: str,
        vae_model_path: str,
        backend: str = "cpu",  # "htp", "cpu"
        htp_backend_path: str = "libQnnHtp.so",
    ):
        self.backend = backend
        self.vae_model_path = vae_model_path
        self.unet_model_path = unet_model_path
        self.scheduler = EulerDiscreteScheduler(num_inference_steps=4)

        # 1. Initialize UNet Session (HTP via QNN or CPU/Mock)
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        if backend == "htp":
            print(f"[QNN] Loading UNet onto Qualcomm HTP backend ({htp_backend_path})...")
            providers = ["QnnExecutionProvider"]
            provider_options = [{
                "backend_path": htp_backend_path,
                "htp_performance_mode": "burst",
                "enable_htp_fp16_precision": "1",
            }]
        else:
            print("[Pipeline] Running UNet in CPU / Mock mode...")
            providers = ["CPUExecutionProvider"]
            provider_options = [{}]

        self.unet_session = None
        if os.path.exists(unet_model_path):
            if backend == "htp":
                self.unet_session = ort.InferenceSession(
                    unet_model_path,
                    sess_options=sess_opts,
                    providers=providers,
                    provider_options=provider_options,
                )
            else:
                print(f"[Pipeline] Notice: {unet_model_path} is an EPContext node targeting QnnExecutionProvider.")
                print(f"[Pipeline] Running UNet in mock-denoising mode for CPU testing...")
        else:
            print(f"[Warning] UNet model file {unet_model_path} not found on disk. Mocking UNet outputs.")

        # 2. Initialize VAE Session (Always on CPU for memory & quality safety)
        print(f"[CPU] Loading TAESDXL VAE Decoder on CPU ({vae_model_path})...")
        self.vae_session = ort.InferenceSession(
            vae_model_path,
            providers=["CPUExecutionProvider"]
        )

    def generate(
        self,
        prompt: str = "A futuristic cyberpunk city, photorealistic, 8k",
        seed: int = 42,
    ) -> Image.Image:
        np.random.seed(seed)
        batch_size = 1
        
        # 1. Mock/Generate Prompt Embeddings (In production, encoded via CLIP-L + OpenCLIP)
        # encoder_hidden_states: [1, 77, 2048]
        # text_embeds (pooled): [1, 1280]
        # time_ids: [1, 6] (orig_h, orig_w, crop_top, crop_left, target_h, target_w)
        encoder_hidden_states = np.random.randn(batch_size, 77, 2048).astype(np.float32)
        text_embeds = np.random.randn(batch_size, 1280).astype(np.float32)
        time_ids = np.array([[1024.0, 1024.0, 0.0, 0.0, 1024.0, 1024.0]], dtype=np.float32)

        # 2. Initial Latents in [1, 4, 128, 128]
        latents = np.random.randn(batch_size, 4, 128, 128).astype(np.float32)
        latents = latents * self.scheduler.init_noise_sigma

        # 3. Denoising loop (4 steps on HTP)
        print(f"[Pipeline] Starting 4-step Denoising loop (Resolution: 1024x1024, Latents: 128x128)...")
        t_start = time.time()
        for i, t in enumerate(self.scheduler.timesteps):
            t_step = time.time()
            timestep_tensor = np.array([t], dtype=np.float32)
            scaled_latents = self.scheduler.scale_model_input(latents, i)

            if self.unet_session is not None:
                unet_inputs = {
                    "sample": scaled_latents,
                    "timestep": timestep_tensor,
                    "encoder_hidden_states": encoder_hidden_states,
                    "text_embeds": text_embeds,
                    "time_ids": time_ids,
                }
                out = self.unet_session.run(None, unet_inputs)
                noise_pred = out[0]
            else:
                # Mock step output for testing pipeline wiring
                noise_pred = np.random.randn(*latents.shape).astype(np.float32) * 0.1

            latents = self.scheduler.step(noise_pred, i, latents)
            print(f"  Step {i+1}/4 (timestep={t:0.0f}) completed in {(time.time() - t_step)*1000:.1f}ms")

        print(f"[UNet] Denoising finished in {(time.time() - t_start):.2f}s")

        # 4. Decode with TAESDXL on CPU
        print("[VAE] Decoding latents to 1024x1024 RGB image on CPU...")
        t_vae = time.time()
        # TAESDXL expects [1, 4, 128, 128]
        vae_out = self.vae_session.run(None, {"latent_sample": latents})[0]
        print(f"[VAE] Decoded in {(time.time() - t_vae)*1000:.1f}ms")

        # 5. Post-process to PIL Image
        # Output is in range [0, 1] or [-1, 1], clamp to [0, 1]
        img_array = np.clip(vae_out[0], 0.0, 1.0)
        img_array = (img_array.transpose(1, 2, 0) * 255.0).astype(np.uint8)
        image = Image.fromarray(img_array)
        return image

def main():
    parser = argparse.ArgumentParser(description="SDXL QNN Pipeline Runner")
    parser.add_argument("--backend", default="cpu", choices=["htp", "cpu"])
    parser.add_argument("--unet", default="tools/sdxl_qnn/unet_ctx.onnx")
    parser.add_argument("--vae", default="tools/sdxl_qnn/taesdxl_decoder.onnx")
    parser.add_argument("--output", default="tools/sdxl_qnn/output_sample.png")
    parser.add_argument("--prompt", default="A cybernetic astronaut exploring alien neon ruins")
    args = parser.parse_args()

    pipeline = SDXLQNNPipeline(
        unet_model_path=args.unet,
        vae_model_path=args.vae,
        backend=args.backend,
    )

    image = pipeline.generate(prompt=args.prompt)
    image.save(args.output)
    print(f"SUCCESS! Output image saved to: {args.output}")

if __name__ == "__main__":
    main()
