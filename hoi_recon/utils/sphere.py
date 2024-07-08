import os
import pickle
import trimesh
import numpy as np
import torch
import trimesh


class Sphere(object):

    def __init__(self, sphere_file, radius=1.0):
        sphere_mesh = trimesh.load(sphere_file, process=False)

        # shape: n_pts * 3
        self.coord = torch.tensor(sphere_mesh.vertices).float() * radius
        self.faces = torch.tensor(sphere_mesh.faces)

        indices, values = [], []
        n_pts = self.coord.shape[0]
        neigbours = [[] for _ in range(n_pts) ]
        faces = np.array(sphere_mesh.faces)
        for f in faces:
            if [f[0], f[1]] not in indices:
                indices.append([f[0], f[1]])
                values.append(1)
            if [f[1], f[0]] not in indices:
                indices.append([f[1], f[0]])
                values.append(1)

            if [f[0], f[2]] not in indices:
                indices.append([f[0], f[2]])
                values.append(1)
            if [f[2], f[0]] not in indices:
                indices.append([f[2], f[0]])
                values.append(1)

            if [f[1], f[2]] not in indices:
                indices.append([f[1], f[2]])
                values.append(1)
            if [f[2], f[1]] not in indices:
                indices.append([f[2], f[1]])
                values.append(1)

            if f[1] not in neigbours[f[0]]:
                neigbours[f[0]].append(f[1])
            if f[0] not in neigbours[f[1]]:
                neigbours[f[1]].append(f[0])
            if f[2] not in neigbours[f[0]]:
                neigbours[f[0]].append(f[2])
            if f[0] not in neigbours[f[2]]:
                neigbours[f[2]].append(f[0])
            if f[1] not in neigbours[f[2]]:
                neigbours[f[2]].append(f[1])
            if f[2] not in neigbours[f[1]]:
                neigbours[f[1]].append(f[2])

        indices = torch.tensor(indices).transpose(1, 0)
        values = torch.tensor(values).float()
        shape = torch.Size([n_pts, n_pts])
        self.adj_mat = torch.sparse.FloatTensor(indices, values, shape)

        self.neigbours = np.ones((n_pts, 8)) * -1
        for idx, neigbour in enumerate(neigbours):
            neigbour = np.array(neigbour)
            self.neigbours[idx, :neigbour.shape[0]] = neigbour[:8]
        self.neigbours = torch.tensor(self.neigbours).long()
        self.edges = indices # [2, n_lines]
