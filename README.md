# BuildingNeRF
This repository has code to perform single-view 3D rendering for different types of buildings using Meta Learning

## Dataset
- Individual folders in ./datasets -> castle
- skp file from link.txt
- get obj, mtl, texture files from skp file
- python obj_render.py --foldername="castle"
- python colmap2nerf.py --run_colmap --foldername="castle"
