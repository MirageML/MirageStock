import torch
import kaolin
import numpy as np
import open3d as o3d
import trimesh
from tqdm import tqdm
from dmtet_network import Decoder

# arguments and hyperparameters
device = 'cuda'
lr = 1e-3
laplacian_weight = 0.1
iterations = 5000
multires = 2
grid_res = 128

# Laplacian regularization using umbrella operator (Fujiwara / Desbrun).
# https://mgarland.org/class/geom04/material/smoothing.pdf
def laplace_regularizer_const(mesh_verts, mesh_faces):
    term = torch.zeros_like(mesh_verts)
    norm = torch.zeros_like(mesh_verts[..., 0:1])

    v0 = mesh_verts[mesh_faces[:, 0], :]
    v1 = mesh_verts[mesh_faces[:, 1], :]
    v2 = mesh_verts[mesh_faces[:, 2], :]

    term.scatter_add_(0, mesh_faces[:, 0:1].repeat(1,3), (v1 - v0) + (v2 - v0))
    term.scatter_add_(0, mesh_faces[:, 1:2].repeat(1,3), (v0 - v1) + (v2 - v1))
    term.scatter_add_(0, mesh_faces[:, 2:3].repeat(1,3), (v0 - v2) + (v1 - v2))

    two = torch.ones_like(v0) * 2.0
    norm.scatter_add_(0, mesh_faces[:, 0:1], two)
    norm.scatter_add_(0, mesh_faces[:, 1:2], two)
    norm.scatter_add_(0, mesh_faces[:, 2:3], two)

    term = term / torch.clamp(norm, min=1.0)

    return torch.mean(term**2)

def loss_f(mesh_verts, mesh_faces, points, it):
    pred_points = kaolin.ops.mesh.sample_points(mesh_verts.unsqueeze(0), mesh_faces, 50000)[0][0]
    chamfer = kaolin.metrics.pointcloud.chamfer_distance(pred_points.unsqueeze(0), points.unsqueeze(0)).mean()
    if it > iterations//2:
        lap = laplace_regularizer_const(mesh_verts, mesh_faces)
        return chamfer + lap * laplacian_weight
    return chamfer


def dmtet(pcd_path, mesh_path):
    points = torch.tensor(o3d.io.read_point_cloud(pcd_path).points, dtype=torch.float32).to(device)
    if points.shape[0] > 100000:
        idx = list(range(points.shape[0]))
        np.random.shuffle(idx)
        idx = torch.tensor(idx[:100000], device=points.device, dtype=torch.long)
        points = points[idx]

    # The reconstructed object needs to be slightly smaller than the grid to get watertight surface after MT.
    center = (points.max(0)[0] + points.min(0)[0]) / 2
    max_l = (points.max(0)[0] - points.min(0)[0]).max()
    points = ((points - center) / max_l)* 0.9

    tet_verts = torch.tensor(np.load('tets/{}_verts.npz'.format(grid_res))['data'], dtype=torch.float, device=device)
    tets = torch.tensor(([np.load('tets/{}_tets_{}.npz'.format(grid_res, i))['data'] for i in range(4)]), dtype=torch.long, device=device).permute(1,0)

    # Initialize model and create optimizer
    model = Decoder(multires=multires).to(device)
    model.pre_train_sphere(1000)

    vars = [p for _, p in model.named_parameters()]
    optimizer = torch.optim.Adam(vars, lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: max(0.0, 10**(-x*0.0002))) # LR decay over time

    # Training loop
    for it in tqdm(range(iterations)):
        pred = model(tet_verts) # predict SDF and per-vertex deformation
        sdf, deform = pred[:,0], pred[:,1:]
        verts_deformed = tet_verts + torch.tanh(deform) / grid_res # constraint deformation to avoid flipping tets
        mesh_verts, mesh_faces = kaolin.ops.conversions.marching_tetrahedra(verts_deformed.unsqueeze(0), tets, sdf.unsqueeze(0)) # running MT (batched) to extract surface mesh
        mesh_verts, mesh_faces = mesh_verts[0], mesh_faces[0]

        loss = loss_f(mesh_verts, mesh_faces, points, it).requires_grad_()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    # Save Mesh
    trimesh.Trimesh(mesh_verts.detach().cpu().numpy(), mesh_faces.detach().cpu().numpy()).export(mesh_path)
    return True


# dmtet("workspace/mesh/mesh.pcd", "workspace/mesh/mesh_clean.obj")