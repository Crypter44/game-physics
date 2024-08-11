import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import cloth_simulation as cs

# --------- CONFIG FOR ANIMATION ---------
animation_length = 15
animation_time_step = 0.001
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

# --------- CONFIG FOR SIMULATION ---------
spacial_dim = 4
mass = 0.3
spacing = 1
spring_constants = np.array([300, 200, 100])
damping_constants = np.array([50, 50, 50])
gravity = np.array([0, 0, 1])

print("Simulation parameters:")
print(f"    Spacial dimension: {spacial_dim}")
print(f"    Mass: {mass}")
print(f"    Spacing: {spacing}")
print(f"    Spring constants: {spring_constants}")
print(f"    Damping constants: {damping_constants}")
print(f"    Gravity: {gravity}\n")


# --------- SIMULATION ---------
frames = cs.run_simulation(
    spacial_dim,
    mass,
    spacing,
    spring_constants,
    damping_constants,
    gravity,
    animation_time_step,
    animation_frames,
    simulation_type='rk2',
    num_of_fixed_corners=2
)

# --------- SETUP MPL ---------
if sys.platform == 'win32':
    print('\nRunning on Windows: setting matplotlib backend to TkAgg\n')
    mpl.use('TkAgg')
elif sys.platform == 'linux':
    print('\nRunning on Linux: setting matplotlib backend to TkAgg\n')
    mpl.use('TkAgg')
elif sys.platform == 'darwin':
    print('\nRunning on MacOS: setting matplotlib backend to macosx\n')
    mpl.use('macosx')

fig = plt.figure()
fig.set_facecolor('black')
fig.set_tight_layout(True)

ax = fig.add_subplot(projection='3d')

# Set the z axis limits, so they aren't recalculated each frame.
# Also, set the axis color to white, so it's visible against the black background.
ax.set_xlim(-1, 3)
ax.set_ylim(-1, 3)
ax.set_zlim(-1, 1)
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.set_facecolor('black')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.tick_params(axis='z', colors='white')
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.zaxis.label.set_color('white')

ax.axis('off')

plot = [ax.plot_wireframe(frames[0][0], frames[0][1], frames[0][2])]


# --------- ANIMATION ---------
def update(frame):
    plot[0].remove()
    X, Y, Z = frames[int(frame*frame_skip_factor)]
    plot[0] = ax.plot_wireframe(X, Y, Z)
    return plot[0]


ani = animation.FuncAnimation(fig, update, frames=visible_frames, interval=1000 / fps)

Writer = animation.writers['ffmpeg']
writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)

ani.save('cloth_animation.mp4', writer=writer)
plt.show()
