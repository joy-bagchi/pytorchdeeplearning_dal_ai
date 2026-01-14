from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
import torch
from PIL import Image
from tqdm.auto import tqdm
import os
import numpy as np
from diffusers import DDPMPipeline


def remove_all_noise_at_once(noisy_image, predicted_noise, timestep, scheduler):
    """
    Given a noisy image and the model's predicted noise, 
    apply the mathematical formula to remove ALL predicted noise in one step (not gradual).
    
    This uses the original DDPM denoising formula:
    x_0 = (x_t - sqrt(1 - alpha_t) * epsilon) / sqrt(alpha_t)
    Where:
      - x_t: Noisy image at current timestep
      - epsilon: Predicted noise by the model
      - alpha_cumprod: Cumulative product of noise schedule parameter
    """
    # Get cumulative product of alphas for the current timestep (how much noise is left)
    alpha_prod_t = scheduler.alphas_cumprod[timestep].to(noisy_image.device)
    # Calculate normalization terms for the denoising formula
    sqrt_alpha_prod_t = torch.sqrt(alpha_prod_t)
    sqrt_one_minus_alpha_prod_t = torch.sqrt(1 - alpha_prod_t)
    # Remove all predicted noise at once to estimate the original clean image
    clean_image = (noisy_image - sqrt_one_minus_alpha_prod_t * predicted_noise) / sqrt_alpha_prod_t
    return clean_image

def gradual_denoise_step(noisy_image, predicted_noise, timestep, scheduler):
    """
    Perform ONE step of gradual denoising using the scheduler's step function.
    
    Returns the image after the next denoising step, as the diffusion process normally does.
    """
    step = scheduler.step(predicted_noise, timestep, noisy_image)
    return step.prev_sample

@torch.no_grad()
def tensor_to_image(tensor):
    """
    Convert a PyTorch tensor to a displayable PIL Image.
    Handles either [-1, 1] or [0, 1] input ranges.
    """
    # Remove batch/channel dimension, rearrange (C, H, W) to (H, W, C)
    img = tensor.cpu().squeeze().permute(1,2,0).numpy()
    # If input is in [-1, 1], rescale to [0, 1]
    img = (img + 1) / 2 if img.min() < 0 else img
    # Clip any accidental out-of-bounds values
    img = np.clip(img, 0, 1)
    # Convert to uint8 and PIL Image
    return Image.fromarray((img*255).astype(np.uint8))

@torch.no_grad()
def visualize_ddpm_denoising(pipe, num_inference_steps=100):
    """
    Show the denoising process of a DDPM:
    - Top row: gradual denoising (true DDPM generation)
    - Bottom row: removing all estimated noise in one step at multiple points
    
    Returns two lists of (step_index, PIL.Image).
    """
    # Get config for image size and channels
    image_size = pipe.unet.config.sample_size
    num_channels = pipe.unet.config.in_channels
    device = pipe.device

    # Start with a single batch of pure noise
    images = torch.randn(1, num_channels, image_size, image_size, device=device)
    scheduler = pipe.scheduler
    model = pipe.unet
    scheduler.set_timesteps(num_inference_steps)

    # Evenly distributed steps to visualize: pick 7 steps (can change n_vis)
    timesteps_to_show = []
    n_vis = 7
    for i in range(n_vis):
        t_idx = int((num_inference_steps-1) * i / (n_vis-1))
        timesteps_to_show.append(t_idx)

    # Prepare outputs
    gradual_images = []      # stores (step_index, gradual denoised image)
    full_removal_images = [] # stores (step_index, "full denoise" image)
    timesteps = scheduler.timesteps
    latents = images

    # Denoising loop over all timesteps
    for i, t in enumerate(tqdm(timesteps, desc="Denoising")):
        # Predict the noise using the model at this step
        noise_pred = model(latents, t).sample

        if i in timesteps_to_show:
            # 1. Compute "full noise removal" for this state
            full_removal = remove_all_noise_at_once(latents, noise_pred, t, scheduler)
            full_removal_img = tensor_to_image(full_removal)
            full_removal_images.append((i, full_removal_img))
            # 2. Save current image for gradual denoising
            current_img = tensor_to_image(latents)
            gradual_images.append((i, current_img))

        # Advance to next time step: normal DDPM sampling
        latents = gradual_denoise_step(latents, noise_pred, t, scheduler)

    # Save the final clean image
    final_img = tensor_to_image(latents)
    gradual_images.append((num_inference_steps, final_img))

    return gradual_images, full_removal_images


import torch
import ipywidgets as widgets
from IPython.display import display
from PIL import Image
import numpy as np
import imageio
import io

def load_widget(pipe):
    """
    Display a pretty interactive Stable Diffusion generator widget.
    pipe: A diffusers StableDiffusionPipeline object, already loaded to GPU/CPU.
    Required Jupyter: ipywidgets, imageio.
    """
    import time
    
    # Output and image widgets
    output = widgets.Output()
    gif_widget = widgets.Image(format='gif', width=320, height=320)
    final_image_widget = widgets.Image(format='png', width=320, height=320)
    
    # --- Widget Styles ---
    textbox_style = {'description_width': '120px'}
    slider_style = {'description_width': '140px'}
    
    heading = widgets.HTML("<h3 style='color:#0EA5E9;font-family:sans-serif'>Stable Diffusion Image Generator</h3>")
    
    # Generation mode selector
    mode_checkbox = widgets.Checkbox(
        value=True,
        description='Show denoising animation (GIF)',
        indent=False,
        layout=widgets.Layout(width='300px')
    )
    
    mode_info = widgets.HTML(
        value="""<div style='background-color:#000000;padding:8px;border-radius:6px;margin:8px 0;'>
        <small><b>💡 Tip:</b> Uncheck for faster generation (final image only)</small>
        </div>"""
    )
    
    prompt_widget = widgets.Text(
        value="A puppy dog riding a skateboard in times square", 
        description='Prompt:', 
        style=textbox_style,
        layout=widgets.Layout(width='500px')
    )
    negative_prompt_widget = widgets.Text(
        value="", 
        description='Negative:', 
        style=textbox_style,
        layout=widgets.Layout(width='500px')
    )
    
    steps_slider = widgets.IntSlider(
        value=50, min=10, max=100, step=1,
        description='Inference steps:', 
        style=slider_style,
        continuous_update=False,
        readout=False,
        layout=widgets.Layout(width='350px'),
    )
    steps_value = widgets.Label(f"{steps_slider.value}")
    def update_steps_label(*a):
        steps_value.value = f"{steps_slider.value}"
    steps_slider.observe(update_steps_label, "value")
    
    gs_slider = widgets.FloatSlider(
        value=7.5, min=4.0, max=16.0, step=0.1,
        description='Guidance scale:', 
        style=slider_style,
        continuous_update=False,
        readout=False,
        layout=widgets.Layout(width='350px'),
    )
    gs_value = widgets.Label(f"{gs_slider.value:.2f}")
    def update_gs_label(*a):
        gs_value.value = f"{gs_slider.value:.2f}"
    gs_slider.observe(update_gs_label, "value")
    
    # Update button text based on mode
    run_button = widgets.Button(
        description="✨ Generate with Animation ✨", 
        button_style='info',
        layout=widgets.Layout(width='250px', height='40px')
    )
    
    def update_button_text(*args):
        if mode_checkbox.value:
            run_button.description = "✨ Generate with Animation ✨"
            run_button.button_style = 'info'
        else:
            run_button.description = "⚡ Quick Generate ⚡"
            run_button.button_style = 'success'
    mode_checkbox.observe(update_button_text, 'value')
    
    # --- Layout arrangement ---
    input_form = widgets.VBox([
        heading,
        widgets.HBox([mode_checkbox, mode_info]),
        prompt_widget,
        negative_prompt_widget,
        widgets.HBox([steps_slider, steps_value]),
        widgets.HBox([gs_slider, gs_value]),
        run_button,
    ], layout=widgets.Layout(
        border='2px solid #0EA5E9', 
        box_shadow="2px 2px 10px #0EA5E9",
        padding='18px 24px 18px 24px', 
        border_radius='14px',
        width='600px',
        background='white'
    ))
    
    # Container for displaying either GIF or final image
    image_container = widgets.VBox([gif_widget, final_image_widget])
    
    ui = widgets.VBox([
        input_form,
        output,
        image_container
    ])
    
    def generate_with_animation(prompt, negative_prompt, num_inference_steps, guidance_scale):
        """Generate with full denoising animation"""
        with output:
            output.clear_output()
            print(f"\n🎬 Generating animation with {num_inference_steps} steps")
            print(f"Prompt: {prompt}")
            if negative_prompt:
                print(f"Negative prompt: {negative_prompt}")
            print()
            
            intermediate_images = []
            progress = widgets.IntProgress(
                value=0,
                min=0,
                max=num_inference_steps,
                description='Denoising:',
                bar_style='info',
                orientation='horizontal'
            )
            display(progress)
            
            def collect_callback(step, timestep, latents):
                with torch.no_grad():
                    scaled_latents = latents / 0.18215
                    image = pipe.vae.decode(scaled_latents).sample
                    image = (image / 2 + 0.5).clamp(0, 1)
                    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
                    pil_img = Image.fromarray((image * 255).astype('uint8'))
                intermediate_images.append((step, pil_img))
                progress.value = step + 1
            
            start_time = time.time()
            generator = torch.Generator(pipe.device).manual_seed(42)
            intermediate_images.clear()
            
            _ = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                callback=collect_callback,
                callback_steps=1,
                progress_bar=False,
            )
            
            progress.bar_style = 'success'
            progress.description = 'Done'
            
            print("\n🎨 Creating animation...")
            gif_bytes = io.BytesIO()
            all_imgs = [img for (step, img) in intermediate_images]
            imageio.mimsave(gif_bytes, all_imgs, format='GIF', duration=0.15)
            
            # Hide final image widget, show GIF
            final_image_widget.layout.display = 'none'
            gif_widget.layout.display = 'block'
            gif_widget.value = gif_bytes.getvalue()
            
            elapsed = time.time() - start_time
            print(f"✅ Animation generated in {elapsed:.1f} seconds")
    
    def generate_final_only(prompt, negative_prompt, num_inference_steps, guidance_scale):
        """Generate only the final image without animation"""
        with output:
            output.clear_output()
            print(f"\n⚡ Quick generation with {num_inference_steps} steps")
            print(f"Prompt: {prompt}")
            if negative_prompt:
                print(f"Negative prompt: {negative_prompt}")
            print()
            
            progress = widgets.IntProgress(
                value=0,
                min=0,
                max=1,
                description='Generating:',
                bar_style='success',
                orientation='horizontal'
            )
            display(progress)
            
            start_time = time.time()
            generator = torch.Generator(pipe.device).manual_seed(42)
            
            # Generate without callbacks for speed
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                progress_bar=False,
            )
            
            progress.value = 1
            progress.description = 'Done'
            
            # Convert to PNG bytes
            final_img = result.images[0]
            img_bytes = io.BytesIO()
            final_img.save(img_bytes, format='PNG')
            
            # Hide GIF widget, show final image
            gif_widget.layout.display = 'none'
            final_image_widget.layout.display = 'block'
            final_image_widget.value = img_bytes.getvalue()
            
            elapsed = time.time() - start_time
            print(f"✅ Image generated in {elapsed:.1f} seconds")
            
            # Display generation stats
            print(f"\n📊 Generation Stats:")
            print(f"  • Mode: Quick (final image only)")
            print(f"  • Steps: {num_inference_steps}")
            print(f"  • Guidance Scale: {guidance_scale}")
            print(f"  • Time saved: ~{num_inference_steps * 0.15:.1f}s (animation creation)")
    
    def run_on_click(b):
        if mode_checkbox.value:
            generate_with_animation(
                prompt_widget.value,
                negative_prompt_widget.value or None,
                steps_slider.value,
                gs_slider.value
            )
        else:
            generate_final_only(
                prompt_widget.value,
                negative_prompt_widget.value or None,
                steps_slider.value,
                gs_slider.value
            )
    
    run_button.on_click(run_on_click)
    display(ui)