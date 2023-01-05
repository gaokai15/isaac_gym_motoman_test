"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


motoman Attractor
----------------
Positional control of motoman panda robot with a target attractor that the robot tries to reach
"""

import math
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_conjugate, quat_mul, to_torch
import os
import torch
import time
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt



def main():

    use_GPU = False

    # Initialize gym
    gym = gymapi.acquire_gym()

    # configure sim
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UpAxis.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 8
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = 0
    # sim_params.physx.use_gpu = args.use_gpu
    sim_params.physx.use_gpu = use_GPU

    sim_params.use_gpu_pipeline = use_GPU
    # if args.use_gpu_pipeline:
    #     print("WARNING: Forcing CPU pipeline.")

    # print('arg', args)
    # print('device', args.compute_device_id)

    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

    if sim is None:
        print("*** Failed to create sim")
        quit()

    # Create viewer
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        print("*** Failed to create viewer")
        quit()

    # Add ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane_params)

    # Load motoman asset
    asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'motoman')
    motoman_asset_file = "motoman_dual.urdf"

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.disable_gravity = True
    asset_options.armature = 0.01

    print("Loading asset '%s' from '%s'" % (motoman_asset_file, asset_root))
    motoman_asset = gym.load_asset(
        sim, asset_root, motoman_asset_file, asset_options)

    # get link index of panda hand, which we will use as end effector
    motoman_hand = "arm_left_link_7_t"
    motoman_link_dict = gym.get_asset_rigid_body_dict(motoman_asset)
    motoman_hand_index = motoman_link_dict[motoman_hand]

    # get joint limits and ranges for motoman
    motoman_dof_props = gym.get_asset_dof_properties(motoman_asset)
    motoman_lower_limits = motoman_dof_props['lower']
    motoman_upper_limits = motoman_dof_props['upper']
    motoman_ranges = motoman_upper_limits - motoman_lower_limits
    motoman_mids = 0.5 * (motoman_upper_limits + motoman_lower_limits)
    motoman_num_dofs = len(motoman_dof_props)

    # override default stiffness and damping values
    motoman_dof_props["driveMode"][:] = gymapi.DOF_MODE_POS
    motoman_dof_props['stiffness'][:].fill(400)
    motoman_dof_props['damping'][:].fill(40.0)
    

    default_dof_pos = np.zeros(motoman_num_dofs, dtype=np.float32)
    default_dof_pos[:] = motoman_mids[:]

    default_dof_state = np.zeros(motoman_num_dofs, gymapi.DofState.dtype)
    default_dof_state["pos"] = default_dof_pos

    # send to torch
    default_dof_pos_tensor = to_torch(default_dof_pos, device='cpu')


    # add a box asset
    asset_options = gymapi.AssetOptions()
    asset_options.disable_gravity = True
    asset_options.fix_base_link = True
    box_asset = gym.create_box(sim, 0.6, 1.6, 1.0, asset_options)


    # Set up the env grid
    num_envs = 1
    spacing = 2.0
    env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)

    # Some common handles for later use
    envs = []
    motoman_handles = []
    hand_handles = []
    hand_idxs = []


    print("Creating %d environments" % num_envs)
    num_per_row = int(math.sqrt(num_envs))

    for i in range(num_envs):
        # create env
        env = gym.create_env(sim, env_lower, env_upper, num_per_row)
        envs.append(env)

        # add motoman
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0, 0.0, 0.0)
        pose.r = gymapi.Quat(0, 0.0, 0, 1)
        motoman_handle = gym.create_actor(env, motoman_asset, pose, "motoman", i, 2)
        hand_handle = gym.find_actor_rigid_body_handle(env, motoman_handle, motoman_hand)

        # set dof properties
        gym.set_actor_dof_properties(env, motoman_handle, motoman_dof_props)

        # set initial dof states
        gym.set_actor_dof_states(env, motoman_handle, default_dof_state, gymapi.STATE_ALL)

        # set initial position targets
        gym.set_actor_dof_position_targets(env, motoman_handle, default_dof_pos)


        # add a box
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(1.05,0,0.5)
        pose.r = gymapi.Quat(0, 0.0, 0, 1)
        gym.create_actor(env, box_asset, pose, 'table', i, -1)

        motoman_handles.append(motoman_handle)
        hand_handles.append(hand_handle)
        # get global index of hand in rigid body state tensor
        hand_idx = gym.find_actor_rigid_body_index(env, motoman_handle, motoman_hand, gymapi.DOMAIN_SIM)
        hand_idxs.append(hand_idx)


    for i in range(num_envs):
        gym.set_actor_dof_properties(envs[i], motoman_handles[i], motoman_dof_props)

    # Point camera at environments
    cam_pos = gymapi.Vec3(-4.0, -1.0, 4.0)
    cam_target = gymapi.Vec3(0.0, 1.0, 2.0)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)


    # from now on, we will use the tensor API that can run on CPU or GPU
    gym.prepare_sim(sim) 

    num_samples = 1000
    rand_joints = np.random.uniform(motoman_lower_limits, motoman_upper_limits, size=[num_samples] + list(motoman_lower_limits.shape))

    total_times = []
    collisions = []
    for sampleID in range(num_samples):
        
        start_time_i = time.time()
        for i in range(num_envs):
            # pos_action[i,:] = rand_joints[:]
            state = np.copy(default_dof_state)
            state['pos'] = torch.from_numpy(rand_joints[sampleID,:])
            gym.set_actor_dof_states(envs[i], motoman_handles[i], state, gymapi.STATE_POS)
        
        
        # Step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        net_contact = gym.acquire_net_contact_force_tensor(sim)
        net_contact = gymtorch.wrap_tensor(net_contact).view(num_envs, -1, 3)
        col = torch.max(net_contact).item() > 0.1
                
        duration_i = time.time() - start_time_i
        
        total_times.append(duration_i)
        collisions.append(col)
    

        # Step rendering
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, False)
        gym.sync_frame_time(sim)



    print("Done")

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
    
    total_times = np.array(total_times)
    collisions = np.array(collisions).astype(bool)
    # * draw a statistics of the total time

    plt.figure()
    sb.boxplot(total_times)
    plt.savefig('total_timing_boxplot.png')
    print('collision timing')
    plt.figure()
    sb.boxplot(total_times[collisions])
    plt.savefig('collision_timing_boxplot.png')
    print('non-collision timing')
    plt.figure()
    sb.boxplot(total_times[collisions & 0])
    plt.savefig('non_collision_timing_boxplot.png')
    # number of collisions
    print('number of collisions: ', collisions.astype(int).sum() / len(collisions))


if __name__ == '__main__':
    main()