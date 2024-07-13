import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import heated_plate_simulation as hps

if sys.platform == 'win32':
    print('\nRunning on Windows: setting matplotlib backend to TkAgg\n')
    mpl.use('TkAgg')
elif sys.platform == 'darwin':
    print('\nRunning on MacOS: setting matplotlib backend to macosx\n')
    mpl.use('macosx')

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

spacial_dim = 100
heat_diffusion_constant = 250
heat_positions = [
    (25, 25, 750),
]
boundary_conditions = 'isolated'
simulator = 'implicit'

print("Simulation parameters:")
print(f"    Spacial dimension: {spacial_dim}")
print(f"    Heat-diffusion-constant: {heat_diffusion_constant}")
print(f"    Heat positions: {heat_positions}")
print(f"    Boundary Conditions: {boundary_conditions}")
print(f"    Simulator: {simulator}\n")
results = hps.run_simulation(
    spacial_dim,
    animation_frames,
    heat_diffusion_constant,
    1,
    animation_time_step,
    heat_positions,
    boundary_conditions,
    simulator
)


fig, ax = plt.subplots(figsize=(4, 4))
fig.set_tight_layout(True)
fig.set_facecolor('black')
ax.axis('off')
img = ax.imshow(results[0], cmap='hot', interpolation='nearest', norm=mpl.colors.Normalize(vmin=0, vmax=0.25))


def update(frame):
    img.set_data(results[int(frame*frame_skip_factor)])
    return img


ani = animation.FuncAnimation(fig=fig, func=update, frames=visible_frames, interval=1000/fps)

Writer = animation.writers['ffmpeg']
writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)

ani.save('heated_plate_animation.mp4', writer=writer)
plt.show()
