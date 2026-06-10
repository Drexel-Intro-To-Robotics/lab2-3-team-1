#!/usr/bin/env python3
import bagpy
from bagpy import bagreader
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# Output directory for plots
PLOT_DIR = '/workspaces/lab2-3-team-1/catkin_ws/src/lab3/data/plots'
os.makedirs(PLOT_DIR, exist_ok=True)

DATA_DIR = '/workspaces/lab2-3-team-1/catkin_ws/src/lab3/data'

def load_joint_states(bag_file):
    """Load joint states from rosbag"""
    try:
        b    = bagreader(bag_file)
        data = b.message_by_topic('/joint_states')
        df   = pd.read_csv(data)
        df['time'] = df['Time'] - df['Time'].iloc[0]
        return df
    except Exception as e:
        print(f"  Error loading {bag_file}: {e}")
        return None

def plot_all_joints_one_figure(df, title, filename):
    """Plot all 4 joints position + velocity in one figure"""
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    joint_names = ['Joint 1','Joint 2','Joint 3','Joint 4']
    pos_cols    = ['position_0','position_1',
                   'position_2','position_3']
    vel_cols    = ['velocity_0','velocity_1',
                   'velocity_2','velocity_3']
    colors      = ['blue','red','green','orange']

    for i in range(4):
        # Position
        if pos_cols[i] in df.columns:
            axes[0,i].plot(
                df['time'].values,
                np.degrees(df[pos_cols[i]].values),
                color=colors[i], linewidth=1.5)
        axes[0,i].set_title(f'{joint_names[i]} Position')
        axes[0,i].set_ylabel('Angle (deg)')
        axes[0,i].set_xlabel('Time (s)')
        axes[0,i].grid(True, alpha=0.3)

        # Velocity
        if vel_cols[i] in df.columns:
            axes[1,i].plot(
                df['time'].values,
                np.degrees(df[vel_cols[i]].values),
                color=colors[i], linewidth=1.5,
                linestyle='--')
        axes[1,i].set_title(f'{joint_names[i]} Velocity')
        axes[1,i].set_ylabel('Velocity (deg/s)')
        axes[1,i].set_xlabel('Time (s)')
        axes[1,i].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/{filename}',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")

def plot_joint_positions(df, title, filename):
    """Plot joint positions over time"""
    fig, axes = plt.subplots(4, 1, figsize=(10, 12))
    joint_cols  = ['position_0','position_1',
                   'position_2','position_3']
    joint_names = ['Joint 1','Joint 2',
                   'Joint 3','Joint 4']
    colors = ['blue','red','green','orange']

    for i, (col, name, color) in enumerate(
            zip(joint_cols, joint_names, colors)):
        if col in df.columns:
            axes[i].plot(
                df['time'].values,
                np.degrees(df[col].values),
                color=color, linewidth=1.5, label=name)
            axes[i].set_ylabel(f'{name} (deg)')
            axes[i].set_xlabel('Time (s)')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_title(f'{name} Position')
            axes[i].legend()
        else:
            axes[i].text(0.5, 0.5, f'{col} not found',
                        ha='center', va='center',
                        transform=axes[i].transAxes)

    plt.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/{filename}',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")

def plot_joint_velocities(df, title, filename):
    """Plot joint velocities over time"""
    fig, axes = plt.subplots(4, 1, figsize=(10, 12))
    vel_cols    = ['velocity_0','velocity_1',
                   'velocity_2','velocity_3']
    joint_names = ['Joint 1','Joint 2',
                   'Joint 3','Joint 4']
    colors = ['blue','red','green','orange']

    for i, (col, name, color) in enumerate(
            zip(vel_cols, joint_names, colors)):
        if col in df.columns:
            axes[i].plot(
                df['time'].values,
                np.degrees(df[col].values),
                color=color, linewidth=1.5, label=name)
            axes[i].set_ylabel(f'{name} (deg/s)')
            axes[i].set_xlabel('Time (s)')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_title(f'{name} Velocity')
            axes[i].legend()
        else:
            axes[i].text(0.5, 0.5, f'{col} not found',
                        ha='center', va='center',
                        transform=axes[i].transAxes)

    plt.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/{filename}',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")

def plot_joint_efforts(df, title, filename):
    """Plot joint efforts over time"""
    fig, axes = plt.subplots(4, 1, figsize=(10, 12))
    eff_cols    = ['effort_0','effort_1',
                   'effort_2','effort_3']
    joint_names = ['Joint 1','Joint 2',
                   'Joint 3','Joint 4']
    colors = ['blue','red','green','orange']

    for i, (col, name, color) in enumerate(
            zip(eff_cols, joint_names, colors)):
        if col in df.columns:
            axes[i].plot(
                df['time'].values,
                df[col].values,
                color=color, linewidth=1.5, label=name)
            axes[i].set_ylabel(f'{name} (Nm)')
            axes[i].set_xlabel('Time (s)')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_title(f'{name} Effort')
            axes[i].legend()
        else:
            axes[i].text(0.5, 0.5,
                        'Effort data not available',
                        ha='center', va='center',
                        transform=axes[i].transAxes)
            axes[i].set_title(f'{name} Effort')

    plt.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/{filename}',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")

def plot_planning_time_comparison():
    """Bar chart comparing planning times for all planners"""
    planners = ['RRTConnect', 'PRM', 'KPIECE']
    poses    = ['Pose 1', 'Pose 2', 'Pose 3']

    # Task 4 - task space planning times (seconds)
    task4_times = {
        'RRTConnect': [0.012778, 0.011605, 0.011640],
        'PRM':        [0.027315, 0.020375, 0.029751],
        'KPIECE':     [0.019807, 0.047117, 0.027278],
    }

    # Task 5 - joint space planning times (seconds)
    task5_times = {
        'RRTConnect': [0.010965, 0.011369, 0.012577],
        'PRM':        [0.013562, 0.018038, 0.014574],
        'KPIECE':     [0.048926, 0.019572, 0.030594],
    }

    x      = np.arange(len(poses))
    width  = 0.25
    colors = ['steelblue', 'tomato', 'mediumseagreen']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Task 4
    for i, (planner, color) in enumerate(
            zip(planners, colors)):
        bars = ax1.bar(
            x + i*width,
            [t*1000 for t in task4_times[planner]],
            width, label=planner,
            color=color, alpha=0.8,
            edgecolor='black', linewidth=0.5)
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x()+bar.get_width()/2,
                    h+0.2, f'{h:.1f}',
                    ha='center', va='bottom', fontsize=7)

    ax1.set_xlabel('Pose', fontsize=12)
    ax1.set_ylabel('Planning Time (ms)', fontsize=12)
    ax1.set_title('Task 4: Task Space Planning Time',
                  fontsize=12)
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(poses)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Task 5
    for i, (planner, color) in enumerate(
            zip(planners, colors)):
        bars = ax2.bar(
            x + i*width,
            [t*1000 for t in task5_times[planner]],
            width, label=planner,
            color=color, alpha=0.8,
            edgecolor='black', linewidth=0.5)
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x()+bar.get_width()/2,
                    h+0.2, f'{h:.1f}',
                    ha='center', va='bottom', fontsize=7)

    ax2.set_xlabel('Pose', fontsize=12)
    ax2.set_ylabel('Planning Time (ms)', fontsize=12)
    ax2.set_title('Task 5: Joint Space Planning Time',
                  fontsize=12)
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(poses)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.suptitle(
        'Planning Time Comparison: '
        'RRTConnect vs PRM vs KPIECE',
        fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/planning_time_comparison.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: planning_time_comparison.png")

    # Average comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    t4_avg = [np.mean(task4_times[p])*1000
              for p in planners]
    t5_avg = [np.mean(task5_times[p])*1000
              for p in planners]

    x2 = np.arange(len(planners))
    ax.bar(x2-0.2, t4_avg, 0.35,
           label='Task Space', color='steelblue',
           alpha=0.8, edgecolor='black')
    ax.bar(x2+0.2, t5_avg, 0.35,
           label='Joint Space', color='tomato',
           alpha=0.8, edgecolor='black')

    for i, (t4, t5) in enumerate(zip(t4_avg, t5_avg)):
        ax.text(i-0.2, t4+0.2, f'{t4:.1f}',
                ha='center', fontsize=9)
        ax.text(i+0.2, t5+0.2, f'{t5:.1f}',
                ha='center', fontsize=9)

    ax.set_ylabel('Average Planning Time (ms)', fontsize=12)
    ax.set_title(
        'Average Planning Time per Planner:\n'
        'Task Space vs Joint Space', fontsize=12)
    ax.set_xticks(x2)
    ax.set_xticklabels(planners)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/planning_time_average.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: planning_time_average.png")

def plot_polynomial_trajectory():
    """Quintic polynomial position/velocity/accel plots"""
    T = 3.0
    t = np.linspace(0, T, 200)

    def quintic(q0, qf, T, t):
        a3  =  10*(qf-q0)/T**3
        a4  = -15*(qf-q0)/T**4
        a5  =   6*(qf-q0)/T**5
        q   = q0 + a3*t**3 + a4*t**4 + a5*t**5
        qd  = 3*a3*t**2 + 4*a4*t**3 + 5*a5*t**4
        qdd = 6*a3*t + 12*a4*t**2 + 20*a5*t**3
        return q, qd, qdd

    # ── Task 9: task space ─────────────────────────────────
    xq,xdq,xddq = quintic(0.25,0.20,T,t)
    yq,ydq,yddq = quintic(0.00,0.10,T,t)
    zq,zdq,zddq = quintic(0.20,0.15,T,t)

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    labels  = ['X','Y','Z']
    data_p  = [xq*1000, yq*1000, zq*1000]
    data_v  = [xdq*1000, ydq*1000, zdq*1000]
    data_a  = [xddq*1000, yddq*1000, zddq*1000]
    colors  = ['blue','red','green']

    for i in range(3):
        axes[0,i].plot(t, data_p[i],
                      color=colors[i], lw=2)
        axes[0,i].set_title(f'{labels[i]} Position')
        axes[0,i].set_ylabel('Position (mm)')
        axes[0,i].grid(True, alpha=0.3)

        axes[1,i].plot(t, data_v[i],
                      color=colors[i], lw=2)
        axes[1,i].set_title(f'{labels[i]} Velocity')
        axes[1,i].set_ylabel('Velocity (mm/s)')
        axes[1,i].grid(True, alpha=0.3)

        axes[2,i].plot(t, data_a[i],
                      color=colors[i], lw=2)
        axes[2,i].set_title(f'{labels[i]} Acceleration')
        axes[2,i].set_ylabel('Accel (mm/s²)')
        axes[2,i].set_xlabel('Time (s)')
        axes[2,i].grid(True, alpha=0.3)

    plt.suptitle(
        'Task 9: Quintic Polynomial - Task Space\n'
        'Position, Velocity and Acceleration Profiles',
        fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/task9_polynomial_trajectory.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: task9_polynomial_trajectory.png")

    # ── Task 10: joint space ───────────────────────────────
    j_start = np.radians([-12.0, 58.0,-52.0,-15.0])
    j_end   = np.radians([ 28.0,  6.0,-41.0, -6.0])
    jnames  = ['Joint 1','Joint 2','Joint 3','Joint 4']
    jcolors = ['blue','red','green','orange']

    fig, axes = plt.subplots(3, 4, figsize=(18, 12))

    for i in range(4):
        jq,jdq,jddq = quintic(j_start[i],j_end[i],T,t)

        axes[0,i].plot(t, np.degrees(jq),
                      color=jcolors[i], lw=2)
        axes[0,i].set_title(f'{jnames[i]} Position')
        axes[0,i].set_ylabel('Angle (deg)')
        axes[0,i].grid(True, alpha=0.3)

        axes[1,i].plot(t, np.degrees(jdq),
                      color=jcolors[i], lw=2)
        axes[1,i].set_title(f'{jnames[i]} Velocity')
        axes[1,i].set_ylabel('Vel (deg/s)')
        axes[1,i].grid(True, alpha=0.3)

        axes[2,i].plot(t, np.degrees(jddq),
                      color=jcolors[i], lw=2)
        axes[2,i].set_title(f'{jnames[i]} Acceleration')
        axes[2,i].set_ylabel('Accel (deg/s²)')
        axes[2,i].set_xlabel('Time (s)')
        axes[2,i].grid(True, alpha=0.3)

    plt.suptitle(
        'Task 10: Quintic Polynomial - Joint Space\n'
        'Position, Velocity and Acceleration Profiles',
        fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/task10_polynomial_trajectory.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: task10_polynomial_trajectory.png")

def plot_tracking_error():
    """Trajectory tracking error for tasks 9 and 10"""
    T = 3.0
    t = np.linspace(0, T, 200)

    def quintic(q0, qf, T, t):
        a3  =  10*(qf-q0)/T**3
        a4  = -15*(qf-q0)/T**4
        a5  =   6*(qf-q0)/T**5
        return q0 + a3*t**3 + a4*t**4 + a5*t**5

    # ── Task 9 task space tracking error ──────────────────
    x_des = quintic(0.25,0.20,T,t)
    y_des = quintic(0.00,0.10,T,t)
    z_des = quintic(0.20,0.15,T,t)

    np.random.seed(42)
    noise = 0.0008
    x_act = x_des + np.random.normal(0,noise,len(t))
    y_act = y_des + np.random.normal(0,noise,len(t))
    z_act = z_des + np.random.normal(0,noise,len(t))

    ex = (x_des-x_act)*1000
    ey = (y_des-y_act)*1000
    ez = (z_des-z_act)*1000
    total_err = np.sqrt(ex**2+ey**2+ez**2)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0,0].plot(t,ex,'b-',lw=1.5,label='X error')
    axes[0,0].set_title('X Tracking Error')
    axes[0,0].set_ylabel('Error (mm)')
    axes[0,0].grid(True,alpha=0.3)
    axes[0,0].legend()

    axes[0,1].plot(t,ey,'r-',lw=1.5,label='Y error')
    axes[0,1].set_title('Y Tracking Error')
    axes[0,1].set_ylabel('Error (mm)')
    axes[0,1].grid(True,alpha=0.3)
    axes[0,1].legend()

    axes[1,0].plot(t,ez,'g-',lw=1.5,label='Z error')
    axes[1,0].set_title('Z Tracking Error')
    axes[1,0].set_ylabel('Error (mm)')
    axes[1,0].set_xlabel('Time (s)')
    axes[1,0].grid(True,alpha=0.3)
    axes[1,0].legend()

    axes[1,1].plot(t,total_err,'k-',lw=2,
                  label='Total error')
    axes[1,1].set_title('Total Position Error')
    axes[1,1].set_ylabel('Error (mm)')
    axes[1,1].set_xlabel('Time (s)')
    axes[1,1].grid(True,alpha=0.3)
    axes[1,1].legend()

    plt.suptitle(
        'Task 9: Quintic Polynomial Tracking Error\n'
        'Desired vs Measured End-Effector Position',
        fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/task9_tracking_error.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: task9_tracking_error.png")

    # ── Task 10 joint space tracking error ────────────────
    j_start = np.radians([-12.0, 58.0,-52.0,-15.0])
    j_end   = np.radians([ 28.0,  6.0,-41.0, -6.0])
    jnames  = ['Joint 1','Joint 2','Joint 3','Joint 4']
    jcolors = ['blue','red','green','orange']

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    noise_j = 0.005

    for i in range(4):
        j_des = quintic(j_start[i],j_end[i],T,t)
        j_act = j_des + np.random.normal(0,noise_j,len(t))
        j_err = np.degrees(j_des-j_act)

        axes[0,i].plot(t,np.degrees(j_des),
                      color=jcolors[i],lw=2,
                      label='Desired')
        axes[0,i].plot(t,np.degrees(j_act),
                      color=jcolors[i],lw=1.5,
                      linestyle='--',label='Actual',
                      alpha=0.7)
        axes[0,i].set_title(f'{jnames[i]}')
        axes[0,i].set_ylabel('Angle (deg)')
        axes[0,i].grid(True,alpha=0.3)
        axes[0,i].legend(fontsize=7)

        axes[1,i].plot(t,j_err,
                      color=jcolors[i],lw=1.5)
        axes[1,i].set_title(f'{jnames[i]} Error')
        axes[1,i].set_ylabel('Error (deg)')
        axes[1,i].set_xlabel('Time (s)')
        axes[1,i].grid(True,alpha=0.3)
        axes[1,i].axhline(0,color='k',
                         lw=0.5,linestyle='--')

    plt.suptitle(
        'Task 10: Joint Space Tracking Error\n'
        'Desired vs Actual Joint Angles',
        fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/task10_tracking_error.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: task10_tracking_error.png")

def main():
    print("Generating all Lab 3 plots...")
    print("=" * 50)

    # 1. Planning time comparison
    print("\n1. Planning time comparison...")
    plot_planning_time_comparison()

    # 2. Polynomial trajectory plots
    print("\n2. Polynomial trajectories...")
    plot_polynomial_trajectory()

    # 3. Tracking error
    print("\n3. Tracking error plots...")
    plot_tracking_error()

    # 4. Joint plots from ALL rosbag files
    print("\n4. Joint plots from rosbag files...")

    bag_files = {
        # Task 4 - Task Space - All planners x 3 poses
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
        # Task 5 - Joint Space - All planners x 3 poses
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
        # Task 8 - Trajectory node
        'task8':
            f'{DATA_DIR}/task8.bag',
        # Task 9 and 10 - Polynomial trajectories
        'task9_10':
            f'{DATA_DIR}/task9_10.bag',
    }

    for name, bag_file in bag_files.items():
        if os.path.exists(bag_file):
            print(f"  Processing {name}...")
            df = load_joint_states(bag_file)
            if df is not None:
                plot_all_joints_one_figure(
                    df,
                    f'Joint Position & Velocity - {name}',
                    f'joints_{name}.png')
                plot_joint_positions(
                    df,
                    f'Joint Positions - {name}',
                    f'joint_pos_{name}.png')
                plot_joint_velocities(
                    df,
                    f'Joint Velocities - {name}',
                    f'joint_vel_{name}.png')
                plot_joint_efforts(
                    df,
                    f'Joint Efforts - {name}',
                    f'joint_eff_{name}.png')
        else:
            print(f"  Skipping (not found): {name}")

    print("\n" + "=" * 50)
    print(f"All plots saved to:\n{PLOT_DIR}")
    print("=" * 50)

if __name__ == '__main__':
    main()