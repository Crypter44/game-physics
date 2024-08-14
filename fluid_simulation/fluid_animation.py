import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

import fluid_simulation as fs


if sys.platform == 'win32' or sys.platform == 'linux':
    print('\nRunning on Windows or Linux: setting matplotlib backend to TkAgg\n')
    mpl.use('TkAgg')
elif sys.platform == 'darwin':
    print('\nRunning on MacOS: setting matplotlib backend to macosx\n')
    mpl.use('macosx')

# ---------- Video Configuration ----------
animation_length = 15
animation_time_step = 0.01
animation_frames = int(animation_length / animation_time_step)
fps = 30
visible_frames = animation_length * fps
frame_skip_factor = animation_frames / visible_frames

print(f"Rendering video with the following configuration:")
print(f"    Animation length: {animation_length}")
print(f"    Animation time step: {animation_time_step}")
print(f"    Animation frames: {animation_frames}")
print(f"    FPS: {fps}")
print(f"    Visible frames: {visible_frames}")
print(f"    Frame skip factor: {frame_skip_factor}\n")

# ---------- Simulation Parameters ----------
spacial_dim = 20

# ---------- Plotting ----------
fig, ax = plt.subplots(figsize=(16, 16))
fig.set_tight_layout(True)
ax.set_xlabel('x')
ax.set_ylabel('y')

velocities = fs.setup_vortex(spacial_dim, 0.5, np.array([9.5, 9.5]))
# velocities += fs.setup_vortex(spacial_dim, 0.5, np.array([40.5, 40.5]), clockwise=True)
pressure = ax.imshow(np.tile(np.linspace(0, spacial_dim, spacial_dim), spacial_dim).reshape((spacial_dim, spacial_dim)), cmap='Blues')
flow = ax.quiver(velocities[:,:,0], velocities[:,:,1], color='Black', angles='xy')

cbar = fig.colorbar(pressure, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
cbar.set_label('Pressure')

plt.show()