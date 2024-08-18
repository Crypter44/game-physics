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
animation_length = 60
animation_time_step = 0.02
animation_frames = int(animation_length / animation_time_step)
fps = 50
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
spacial_dim = 30
rho = 1.5
vortex_speeds = [5, 5]
vortex_centers = [(9.5, 9.5), (19.5, 19.5)]
clockwise = [True, True]

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
fig.set_tight_layout(False)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Fluid Simulation')
ax.set_aspect('equal')

pressure_map = ax.imshow(pressures[0], cmap='Blues')
flow_quiver = ax.quiver(velocities[0][:, :, 0], velocities[0][:, :, 1], color='Black', angles='xy')

cbar = fig.colorbar(pressure_map, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
cbar.set_label('Pressure')


def update(frame):
    frame = int(frame * frame_skip_factor)
    pressure_map.set_data(pressures[frame])
    pressure_map.autoscale()
    cbar.update_normal(pressure_map)
    flow_quiver.set_UVC(velocities[frame][:, :, 0], velocities[frame][:, :, 1])
    return pressure_map, cbar, flow_quiver


ani = animation.FuncAnimation(fig, update, frames=int(visible_frames), interval=2000/fps)

Writer = animation.writers['ffmpeg']
writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)

ani.save('fluid_animation.mp4', writer=writer)

plt.show()