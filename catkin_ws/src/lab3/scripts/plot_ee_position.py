#!/usr/bin/env python3
import rosbag
import numpy as np
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

PLOT_DIR = '/workspaces/lab2-3-team-1/catkin_ws/src/lab3/data/plots'
DATA_DIR = '/workspaces/lab2-3-team-1/catkin_ws/src/lab3/data'
os.makedirs(PLOT_DIR, exist_ok=True)

# OpenManipulator-X DH parameters
D1 = 0.077   # base to shoulder height (m)
L2 = 0.130   # shoulder to elbow (m)
L3 = 0.124   # elbow to wrist (m)
L4 = 0.126   # wrist to gripper (m)

def forward_kinematics(t1, t2, t3, t4):
    """
    Compute end-effector position using FK.
    t1,t2,t3,t4: joint angles in radians
    Returns: x, y, z in metres
    """
    x = (L2*np.cos(t2) + L3*np.cos(t2+t3) +
         L4*np.cos(t2+t3+t4)) * np.cos(t1)
    y = (L2*np.cos(t2) + L3*np.cos(t2+t3) +
         L4*np.cos(t2+t3+t4)) * np.sin(t1)
    z = (D1 + L2*np.sin(t2) +
         L3*np.sin(t2+t3) +
         L4*np.sin(t2+t3+t4))
    return x, y, z

def extract_ee_from_bag(bag_file):
    """
    Compute end-effector position from /joint_states
    using forward kinematics.
    Returns: times, x, y, z arrays
    """
    times  = []
    x_vals = []
    y_vals = []
    z_vals = []

    try:
        bag = rosbag.Bag(bag_file)
        t0  = None

        for topic, msg, t in bag.read_messages(
                topics=['/joint_states']):

            positions = list(msg.position)
            if len(positions) < 4:
                continue

            t1 = positions[0]
            t2 = positions[1]
            t3 = positions[2]
            t4 = positions[3]

            x, y, z = forward_kinematics(t1, t2, t3, t4)

            if t0 is None:
                t0 = t.to_sec()

            times.append(t.to_sec() - t0)
            x_vals.append(x)
            y_vals.append(y)
            z_vals.append(z)

        bag.close()

    except Exception as e:
        print(f"  Error reading bag: {e}")
        return None, None, None, None

    if len(times) == 0:
        print("  No joint state data found")
        return None, None, None, None

    return (np.array(times),
            np.array(x_vals),
            np.array(y_vals),
            np.array(z_vals))

def plot_ee_position(bag_name, bag_file):
    """Plot end-effector x,y,z position over time"""
    print(f"  Processing {bag_name}...")
    t, x, y, z = extract_ee_from_bag(bag_file)

    if t is None:
        print(f"  Skipping {bag_name} - no data")
        return

    fig, axes = plt.subplots(3, 1, figsize=(10, 9))

    axes[0].plot(t, x*1000, 'b-', lw=1.5,
                label='X position')
    axes[0].set_title('End-Effector X Position')
    axes[0].set_ylabel('X (mm)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(t, y*1000, 'r-', lw=1.5,
                label='Y position')
    axes[1].set_title('End-Effector Y Position')
    axes[1].set_ylabel('Y (mm)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(t, z*1000, 'g-', lw=1.5,
                label='Z position')
    axes[2].set_title('End-Effector Z Position')
    axes[2].set_ylabel('Z (mm)')
    axes[2].set_xlabel('Time (s)')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.suptitle(
        f'End-Effector Position (via FK): {bag_name}',
        fontsize=13, fontweight='bold')
    plt.tight_layout()
    fname = f'ee_position_{bag_name}.png'
    plt.savefig(f'{PLOT_DIR}/{fname}',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")

def plot_ee_3d_trajectory(bag_name, bag_file):
    """Plot 3D end-effector trajectory"""
    print(f"  3D trajectory {bag_name}...")
    t, x, y, z = extract_ee_from_bag(bag_file)

    if t is None:
        return

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection='3d')

    ax.plot(x*1000, y*1000, z*1000,
           'b-', lw=2, label='EE trajectory')
    ax.scatter(x[0]*1000,  y[0]*1000,  z[0]*1000,
              color='green', s=100, zorder=5,
              label='Start')
    ax.scatter(x[-1]*1000, y[-1]*1000, z[-1]*1000,
              color='red',   s=100, zorder=5,
              label='End')

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(
        f'End-Effector 3D Trajectory: {bag_name}')
    ax.legend()

    fname = f'ee_3d_{bag_name}.png'
    plt.savefig(f'{PLOT_DIR}/{fname}',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")

def plot_ee_comparison_all_planners():
    """
    Compare EE trajectories for all 3 planners
    for each pose side by side
    """
    planners = ['rrtconnect', 'prm', 'kpiece']
    planner_labels = ['RRTConnect', 'PRM', 'KPIECE']
    colors   = ['blue', 'red', 'green']

    for pose_num in range(1, 4):
        fig, axes = plt.subplots(3, 3, figsize=(15, 10))

        for pi, (planner, label, color) in enumerate(
                zip(planners, planner_labels, colors)):

            bag_file = (f'{DATA_DIR}/'
                       f'task4_{planner}_pose{pose_num}.bag')

            if not os.path.exists(bag_file):
                continue

            t, x, y, z = extract_ee_from_bag(bag_file)
            if t is None:
                continue

            axes[0,pi].plot(t, x*1000, color=color, lw=1.5)
            axes[0,pi].set_title(f'{label} - X')
            axes[0,pi].set_ylabel('X (mm)')
            axes[0,pi].grid(True, alpha=0.3)

            axes[1,pi].plot(t, y*1000, color=color, lw=1.5)
            axes[1,pi].set_title(f'{label} - Y')
            axes[1,pi].set_ylabel('Y (mm)')
            axes[1,pi].grid(True, alpha=0.3)

            axes[2,pi].plot(t, z*1000, color=color, lw=1.5)
            axes[2,pi].set_title(f'{label} - Z')
            axes[2,pi].set_ylabel('Z (mm)')
            axes[2,pi].set_xlabel('Time (s)')
            axes[2,pi].grid(True, alpha=0.3)

        plt.suptitle(
            f'End-Effector Trajectory Comparison '
            f'- Task 4 Pose {pose_num}\n'
            f'RRTConnect vs PRM vs KPIECE',
            fontsize=13, fontweight='bold')
        plt.tight_layout()
        fname = f'ee_comparison_task4_pose{pose_num}.png'
        plt.savefig(f'{PLOT_DIR}/{fname}',
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fname}")

def plot_task_vs_joint_space_ee():
    """
    Compare end-effector trajectory:
    Task space (Task 4) vs Joint space (Task 5)
    """
    planners = ['rrtconnect', 'prm', 'kpiece']
    planner_labels = ['RRTConnect', 'PRM', 'KPIECE']

    for pi, (planner, label) in enumerate(
            zip(planners, planner_labels)):

        fig, axes = plt.subplots(3, 3, figsize=(15, 10))

        for pose_num in range(1, 4):
            col = pose_num - 1

            # Task 4 - task space
            bag4 = (f'{DATA_DIR}/'
                   f'task4_{planner}_pose{pose_num}.bag')
            # Task 5 - joint space
            bag5 = (f'{DATA_DIR}/'
                   f'task5_{planner}_pose{pose_num}.bag')

            t4,x4,y4,z4 = (extract_ee_from_bag(bag4)
                           if os.path.exists(bag4)
                           else (None,)*4)
            t5,x5,y5,z5 = (extract_ee_from_bag(bag5)
                           if os.path.exists(bag5)
                           else (None,)*4)

            if t4 is not None:
                axes[0,col].plot(t4, x4*1000,
                               'b-', lw=1.5,
                               label='Task Space')
                axes[1,col].plot(t4, y4*1000,
                               'b-', lw=1.5,
                               label='Task Space')
                axes[2,col].plot(t4, z4*1000,
                               'b-', lw=1.5,
                               label='Task Space')

            if t5 is not None:
                axes[0,col].plot(t5, x5*1000,
                               'r--', lw=1.5,
                               label='Joint Space')
                axes[1,col].plot(t5, y5*1000,
                               'r--', lw=1.5,
                               label='Joint Space')
                axes[2,col].plot(t5, z5*1000,
                               'r--', lw=1.5,
                               label='Joint Space')

            axes[0,col].set_title(f'Pose {pose_num} X')
            axes[0,col].set_ylabel('X (mm)')
            axes[0,col].grid(True, alpha=0.3)
            axes[0,col].legend(fontsize=7)

            axes[1,col].set_title(f'Pose {pose_num} Y')
            axes[1,col].set_ylabel('Y (mm)')
            axes[1,col].grid(True, alpha=0.3)

            axes[2,col].set_title(f'Pose {pose_num} Z')
            axes[2,col].set_ylabel('Z (mm)')
            axes[2,col].set_xlabel('Time (s)')
            axes[2,col].grid(True, alpha=0.3)

        plt.suptitle(
            f'Task Space vs Joint Space EE Trajectory\n'
            f'{label} - All 3 Poses',
            fontsize=13, fontweight='bold')
        plt.tight_layout()
        fname = (f'ee_taskspace_vs_jointspace_'
                f'{planner}.png')
        plt.savefig(f'{PLOT_DIR}/{fname}',
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fname}")

def main():
    print("Generating end-effector position plots...")
    print("=" * 50)

    # Individual EE position plots for all bags
    print("\n1. Individual EE position plots...")
    bag_files = {
        'task4_rrtconnect_pose1':
            f'{DATA_DIR}/task4_rrtconnect_pose1.bag',
        'task4_rrtconnect_pose2':
            f'{DATA_DIR}/task4_rrtconnect_pose2.bag',
        'task4_rrtconnect_pose3':
            f'{DATA_DIR}/task4_rrtconnect_pose3.bag',
        'task4_prm_pose1':
            f'{DATA_DIR}/task4_prm_pose1.bag',
        'task4_prm_pose2':
            f'{DATA_DIR}/task4_prm_pose2.bag',
        'task4_prm_pose3':
            f'{DATA_DIR}/task4_prm_pose3.bag',
        'task4_kpiece_pose1':
            f'{DATA_DIR}/task4_kpiece_pose1.bag',
        'task4_kpiece_pose2':
            f'{DATA_DIR}/task4_kpiece_pose2.bag',
        'task4_kpiece_pose3':
            f'{DATA_DIR}/task4_kpiece_pose3.bag',
        'task5_rrtconnect_pose1':
            f'{DATA_DIR}/task5_rrtconnect_pose1.bag',
        'task5_rrtconnect_pose2':
            f'{DATA_DIR}/task5_rrtconnect_pose2.bag',
        'task5_rrtconnect_pose3':
            f'{DATA_DIR}/task5_rrtconnect_pose3.bag',
        'task5_prm_pose1':
            f'{DATA_DIR}/task5_prm_pose1.bag',
        'task5_prm_pose2':
            f'{DATA_DIR}/task5_prm_pose2.bag',
        'task5_prm_pose3':
            f'{DATA_DIR}/task5_prm_pose3.bag',
        'task5_kpiece_pose1':
            f'{DATA_DIR}/task5_kpiece_pose1.bag',
        'task5_kpiece_pose2':
            f'{DATA_DIR}/task5_kpiece_pose2.bag',
        'task5_kpiece_pose3':
            f'{DATA_DIR}/task5_kpiece_pose3.bag',
        'task9_10':
            f'{DATA_DIR}/task9_10.bag',
    }

    for name, bag_file in bag_files.items():
        if os.path.exists(bag_file):
            plot_ee_position(name, bag_file)
            plot_ee_3d_trajectory(name, bag_file)
        else:
            print(f"  Skipping (not found): {name}")

    # Planner comparison plots
    print("\n2. Planner comparison EE plots...")
    plot_ee_comparison_all_planners()

    # Task space vs joint space comparison
    print("\n3. Task vs joint space EE comparison...")
    plot_task_vs_joint_space_ee()

    print("\n" + "=" * 50)
    print(f"All EE plots saved to:\n{PLOT_DIR}")
    print("=" * 50)

if __name__ == '__main__':
    main()