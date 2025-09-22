# Note: Vertex IDs can be visualised in 3D modelling programs (I used Maya 2023)

import trimesh

# Path to GLB mesh file
mesh_path = "gradio_cache/fa654053-5a4d-4ef3-a0f5-e5efbe307261/white_mesh.glb"

# Load mesh from file
mesh = trimesh.load_mesh(mesh_path, force='glb')
print(f"Mesh: Vertices={mesh.vertices.shape}, Faces={mesh.faces.shape}")

# Adjacency map of vertices
graph = mesh.vertex_adjacency_graph

# List vertices adjacent to vertex 0
print("Vertex 0 coordinates:", mesh.vertices[0])
print("Vertex 0 adjacent vertices:", [v for v in graph.neighbors(0)])

# Split mesh
print("Number of separate components (bodies) present in mesh:", mesh.body_count)
for i, m in enumerate(mesh.split()):
    print(f" - Body {i}: Vertices={m.vertices.shape}, Faces={m.faces.shape}")