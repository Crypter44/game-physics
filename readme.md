# Game Physics
## Introduction
This is a simple project with different physics simulations. 

The project is written in Python and uses math libraries such as numpy, scipy, and matplotlib to simulate different 
physics problems. These simulations are based on physics principles and are not meant to be accurate.
However, they try to give the appearance of real-world physics and focus on real-time calculations.

## Running the project
Each simulation is in a separate directory. The simulation and animation are separated into different files.
In the animation file, configurations can be changed to change the simulation parameters. To render an animation, 
run the animation file.

_(Currently, the animations are not supported on linux)_

## Heated Plate
The first simulation is a heated plate. 
Heat is applied once to a spot on a plate and the heat spread is calculated over time.

## Cloth Simulation
The second simulation is a cloth simulation.
A cloth is simulated with a grid of points connected by springs. The cloth is affected by gravity. 