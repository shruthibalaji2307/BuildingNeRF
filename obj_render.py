import os
import sys
import torch
import pytorch3d
import matplotlib.pyplot as plt

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex,
    TexturesAtlas
)

import argparse

sys.path.append(os.path.abspath(''))

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

parser = argparse.ArgumentParser(description="generate images given obj, mtl files")

parser.add_argument("--foldername", default="", help="building folder within ./datasets")
args = parser.parse_args()
 
# Set paths
DATA_DIR = "./datasets"
foldername = args.foldername
BUILDING_DIR = os.path.join(DATA_DIR, foldername)
IMG_DIR = os.path.join(BUILDING_DIR, "images")
if(not os.path.exists(IMG_DIR)):
    os.makedirs(IMG_DIR)    
obj_filename = os.path.join(BUILDING_DIR, foldername + ".obj")

# Initialize a camera.
# With world coordinates +Y up, +X left and +Z in, the front of the cow is facing the -Z direction. 
# So we move the camera by 180 in the azimuth direction so it is facing the front of the cow. 
R, T = look_at_view_transform(100, 180, 180) 
# dist, elevation, azim
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

# Define the settings for rasterization and shading. Here we set the output image to be of size
# 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
# and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that 
# the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for 
# explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of 
# the difference between naive and coarse-to-fine rasterization. 
raster_settings = RasterizationSettings(
    image_size=512, 
    blur_radius=0.0, 
    faces_per_pixel=1, 
)

# Place a point light in front of the object. As mentioned above, the front of the cow is facing the 
# -z direction. 
lights = PointLights(device=device, location=[[0.0, 0.0, 0.0]])

# Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will 
# interpolate the texture uv coordinates for each vertex, sample from a texture image and 
# apply the Phong lighting model
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    ),
    shader=SoftPhongShader(
        device=device, 
        cameras=cameras,
        lights=lights
    )
)

torch.cuda.empty_cache() 

verts, faces, aux = load_obj(
            obj_filename,
            device=device,
            load_textures=True,
            create_texture_atlas=True,
            texture_atlas_size=55,
            texture_wrap="clamp",
        )
atlas = aux.texture_atlas
mesh = Meshes(
            verts=[verts],
            faces=[faces.verts_idx],
            textures=TexturesAtlas(atlas=[atlas]),
)
        
mesh=mesh.scale_verts(0.02)

batch_size = 100

# Get a batch of viewing angles. 
elev = torch.linspace(0, 180, batch_size)
azim = torch.linspace(-180, 180, batch_size)

for i in range(batch_size):
    R, T = look_at_view_transform(dist=100, elev=elev[i], azim=azim[i])
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    
    # Move the light back in front of the cow which is facing the -z direction.
    lights.location = torch.tensor([[0.0, 0.0, -3.0]], device=device)

    images = renderer(mesh, cameras=cameras, lights=lights)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(images[0, ..., :3].cpu().numpy())
    plt.axis("off")
    plt.savefig(IMG_DIR + '/building'+str(i)+'.png')
    plt.close()
