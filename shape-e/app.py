import os
from PIL import Image
import torch
import matplotlib.pyplot as plt
import imageio
import numpy as np
import math
import argparse

import torch
import base64
import io
import os
from typing import Union


from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images

from shap_e.models.nn.camera import DifferentiableCameraBatch, DifferentiableProjectiveCamera
from shap_e.models.transmitter.base import Transmitter, VectorDecoder
from shap_e.util.collections import AttrDict

import trimesh


state = ""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_state(s):
    print(s)
    global state
    state = s

def get_state():
    return state


@torch.no_grad()
def decode_latent_images_foo(
    xm: Union[Transmitter, VectorDecoder],
    latent: torch.Tensor,
    cameras: DifferentiableCameraBatch,
    rendering_mode: str = "stf",
):
    decoded = xm.renderer.render_views(
        AttrDict(cameras=cameras),
        params=(xm.encoder if isinstance(xm, Transmitter) else xm).bottleneck_to_params(
            latent[None]
        ),
        options=AttrDict(rendering_mode=rendering_mode, render_with_direction=False),
    )
    return decoded


def generate_3D(input):
    set_state('Entered generate function...')

    # if input is a string, it's a text prompt
    xm = load_model('transmitter', device=device)
    diffusion = diffusion_from_config(load_config('diffusion'))
    batch_size = 4

    if isinstance(input, Image.Image):
        input = expand2square(input)
        model = load_model('image300M', device=device)
        guidance_scale = 3.0
        model_kwargs = dict(images=[input] * batch_size)
    else:
        model = load_model('text300M', device=device)
        guidance_scale = 15.0
        model_kwargs = dict(texts=[input] * batch_size)



    latents = sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=model_kwargs,
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )

    render_mode = 'stf' # you can change this to 'stf'
    size = 64 # this is the size of the renders; higher values take longer to render.

    cameras = create_pan_cameras(size, device)

    x=decode_latent_images_foo(xm, latents[0], cameras, rendering_mode=render_mode)
    mesh=x['meshes'][0]
    rm=x['raw_meshes'][0]

    rm.vertex_channels["R"]=mesh.vertex_colors[:,0]
    rm.vertex_channels["G"]=mesh.vertex_colors[:,1]
    rm.vertex_channels["B"]=mesh.vertex_colors[:,2]

    tm=rm.tri_mesh()

    with open("output/mesh.ply",'wb') as f:
        tm.write_ply(f)


    set_state('Converting to point cloud...')
    # pc = sampler.output_to_point_clouds(samples)[0]

    set_state('Converting to mesh...')
    # save_ply(pc, 'output/mesh.ply', grid_size)

    set_state('')

    return 'output/mesh.ply', ply_to_glb('output/mesh.ply', 'output/mesh.glb')

def expand2square(img):
    width, height = img.size

    if width == height:
        return img
    elif width > height:
        result = Image.new(img.mode, (width, width), "white")
        result.paste(img, (0, (width - height) // 2))
    else:
        result = Image.new(img.mode, (height, height), "white")
        result.paste(img, ((height - width) // 2, 0))

    return img

def ply_to_glb(ply_file, glb_file):
    mesh = trimesh.load(ply_file)

    # Save the mesh as a glb file using Trimesh
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()

    rotation_matrix = trimesh.transformations.rotation_matrix(angle=np.pi/2, direction=[-1, 0, 0])
    mesh.apply_transform(rotation_matrix)

    mesh.export(glb_file, file_type='glb')

    return glb_file

# def save_ply(pc, file_name, grid_size):
#     set_state('Creating SDF model...')
#     sdf_name = 'sdf'
#     sdf_model = model_from_config(MODEL_CONFIGS[sdf_name], device)
#     sdf_model.eval()

#     set_state('Loading SDF model...')
#     sdf_model.load_state_dict(load_checkpoint(sdf_name, device))

#     # Produce a mesh (with vertex colors)
#     mesh = marching_cubes_mesh(
#         pc=pc,
#         model=sdf_model,
#         batch_size=4096,
#         grid_size=grid_size, # increase to 128 for resolution used in evals
#         progress=True,
#     )

#     # Write the mesh to a PLY file to import into some other program.
#     with open(file_name, 'wb') as f:
#         mesh.write_ply(f)

def create_gif(pc):
    fig = plt.figure(facecolor='black', figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d', facecolor='black')
    fixed_bounds=((-0.75, -0.75, -0.75),(0.75, 0.75, 0.75))

    # Create an empty list to store the frames
    frames = []

    # Create a loop to generate the frames for the GIF
    for angle in range(0, 360, 4):
        # Clear the plot and plot the point cloud
        ax.clear()
        color_args = np.stack(
            [pc.channels["R"], pc.channels["G"], pc.channels["B"]], axis=-1
        )
        c = pc.coords


        ax.scatter(c[:, 0], c[:, 1], c[:, 2], c=color_args)

        # Set the viewpoint for the plot
        ax.view_init(elev=10, azim=angle)

        # Turn off the axis labels and ticks
        ax.axis('off')
        ax.set_xlim3d(fixed_bounds[0][0], fixed_bounds[1][0])
        ax.set_ylim3d(fixed_bounds[0][1], fixed_bounds[1][1])
        ax.set_zlim3d(fixed_bounds[0][2], fixed_bounds[1][2])

        # Draw the figure to update the image data
        fig.canvas.draw()

        # Save the plot as a frame for the GIF
        frame = np.array(fig.canvas.renderer.buffer_rgba())

        # If the first frame save as a PNG
        if angle == 0:
            imageio.imwrite('output/pngcloud.png', frame)

        w, h = frame.shape[0], frame.shape[1]
        i = int(round((h - int(h*0.6)) / 2.))
        frame = frame[i:i + int(h*0.6),i:i + int(h*0.6)]
        frames.append(frame)

    # Save the GIF using imageio
    imageio.mimsave('output/pointcloud.gif', frames, fps=30)
    return 'output/pointcloud.gif'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='input.png')

    args = parser.parse_args()

    os.makedirs("output", exist_ok=True)

    try:
        input = Image.open(args.input)
    except:
        input = args.input

    generate_3D(input)