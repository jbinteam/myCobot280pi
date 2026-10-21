# Role and Objective
You are an Expert Robotics Software Engineer and AI Researcher. Your objective is to build a complete Neural Motion Planning pipeline from scratch. The system will use an RRT* algorithm to generate expert demonstration data, and then train a 1D U-Net Diffusion Policy to control a 6-DOF robotic arm in a PyBullet simulation.

# Technology Stack
- **Simulation & Kinematics**: Python 3.10+, PyBullet, NumPy, SciPy
- **Deep Learning**: PyTorch, Hugging Face `diffusers` (for DDPM noise scheduling)
- **Data Management**: zarr or h5py, tqdm

# Project Constraints & Standards
1. **Modularity**: Code must be split into logical files (`env.py`, `planner.py`, `dataset.py`, `model.py`, `train.py`, `eval.py`). Do not write a single monolithic script.
2. **Reproducibility**: Generate a `requirements.txt` file first. Use a fixed random seed where applicable.
3. **Documentation**: Include clear docstrings for classes and functions, explaining the tensor shapes (e.g., `[batch_size, channels, seq_len]`).
4. **Step-by-Step Execution**: Complete one phase fully and verify it works before moving to the next.

---

# Execution Plan

## Phase 1: Simulation Setup (`env.py`)
- Initialize a PyBullet environment with a table and a generic 6-DOF robot (e.g., KUKA iiwa with fixed joints to make it 6-DOF).
- Place a simple block obstacle on the table.
- Implement `forward_kinematics(joint_angles)` and `inverse_kinematics(target_pos)`.
- Write a main block to visually test the robot moving to a target pose.

## Phase 2: Expert Data Generation (`planner.py` & `generate_dataset.py`)
- **`planner.py`**: Implement a joint-space RRT* algorithm that finds collision-free paths from a random start configuration to a goal configuration.
- Add a smoothing function (e.g., B-spline interpolation) to ensure the trajectory is continuous and executable.
- **`generate_dataset.py`**: Write a script to run the planner for 1,000 successful episodes. 
- Save the resulting trajectories (sequence of 6 joint angles) and their corresponding environment states (start pose, goal pose) into a `.zarr` or `.hdf5` dataset.

## Phase 3: Diffusion Policy Architecture (`dataset.py` & `model.py`)
- **`dataset.py`**: Create a PyTorch `Dataset` that reads the saved trajectories. 
  - Implement a Sliding Window mechanism to chunk trajectories into `action_chunk` of shape `(6, 16)` (6 joints, 16 time-steps).
  - Implement Min-Max normalization for the actions [-1, 1].
- **`model.py`**: Build a 1D U-Net architecture (`nn.Conv1d`) designed for time-series data. 
  - The model must take a noisy action chunk `(batch, 6, 16)` and a conditioned observation vector (current state + goal state) as input.
  - Output the predicted noise of shape `(batch, 6, 16)`.

## Phase 4: Training Pipeline (`train.py`)
- Set up the training loop using PyTorch and the `diffusers` DDPM scheduler.
- Initialize the dataloader, the 1D U-Net model, and an AdamW optimizer.
- Training step:
  1. Sample a clean action chunk.
  2. Add Gaussian noise based on a random diffusion timestep.
  3. Forward pass through the model to predict the noise.
  4. Compute the MSE loss between predicted noise and actual noise.
- Implement early stopping and save the best model weights as `diffusion_policy_6dof.pth`.

## Phase 5: Online Inference & Control (`eval.py`)
- Write an evaluation script that connects back to the PyBullet environment.
- Implement **Receding Horizon Control (RHC)**:
  1. Read current joint angles and goal position.
  2. Initialize a Gaussian noise tensor `(1, 6, 16)`.
  3. Run the reverse diffusion loop (denoising) using the trained model to generate a 16-step trajectory.
  4. Execute only the first 2 steps of this trajectory in PyBullet.
  5. Repeat until the robot reaches the goal.
- Log the success rate and collision rate over 50 test runs.

# First Action
Start by generating the `requirements.txt` and then ask for my permission to begin Phase 1.
