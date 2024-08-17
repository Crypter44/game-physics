import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
rho = 1
vortex_speeds = [0.5]
vortex_centers = [(9.5, 9.5)]
clockwise = [True]

print("Simulation parameters:")
print(f"    Spacial dimension: {spacial_dim}")
print(f"    Density: {rho}\n")
print(f"    Vortex speeds: {vortex_speeds}")
print(f"    Vortex centers: {vortex_centers}")
print(f"    Clockwise: {clockwise}\n")

# ---------- Simulation ----------
velocities, pressures = fs.run_simulation(
    spacial_dim,
    vortex_speeds,
    vortex_centers,
    clockwise,
    animation_frames,
    animation_time_step,
    rho
)

# ---------- Plotting ----------
fig, ax = plt.subplots(figsize=(16, 16))
fig.set_tight_layout(True)
ax.set_xlabel('x')
ax.set_ylabel('y')


pressure_map = ax.imshow(pressures, cmap='Blues')
flow_quiver = ax.quiver(velocities[:,:,0], velocities[:,:,1], color='Black', angles='xy')

cbar = fig.colorbar(pressure_map, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
cbar.set_label('Pressure')

def update(frame):
    pressure_map.set_data(pressures[frame*frame_skip_factor])
    flow_quiver.set_UVC(velocities[frame*frame_skip_factor][:,:,0], velocities[frame*frame_skip_factor][:,:,1])
    return pressure_map, flow_quiver

ani = animation.FuncAnimation(fig, update, frames=visible_frames, interval=1000/fps)

plt.show()