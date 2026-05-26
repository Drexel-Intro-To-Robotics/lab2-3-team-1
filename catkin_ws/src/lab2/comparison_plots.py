#!/usr/bin/env python3
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load Real Robot data
real_odom = pd.read_csv('/workspaces/lab2-3-team-1/catkin_ws/src/lab2/data/all_tasks/odom.csv')
real_imu  = pd.read_csv('/workspaces/lab2-3-team-1/catkin_ws/src/lab2/data/all_tasks/imu.csv')
real_cmd  = pd.read_csv('/workspaces/lab2-3-team-1/catkin_ws/src/lab2/data/all_tasks/cmd_vel.csv')

# Load Gazebo data
gaz_odom = pd.read_csv('/workspaces/lab2-3-team-1/catkin_ws/src/lab2/data/gazebo_data/odom.csv')
gaz_imu  = pd.read_csv('/workspaces/lab2-3-team-1/catkin_ws/src/lab2/data/gazebo_data/imu.csv')
gaz_cmd  = pd.read_csv('/workspaces/lab2-3-team-1/catkin_ws/src/lab2/data/gazebo_data/cmd_vel.csv')

# Normalize time
real_odom['time'] = real_odom['Time'] - real_odom['Time'].iloc[0]
real_imu['time']  = real_imu['Time']  - real_imu['Time'].iloc[0]
real_cmd['time']  = real_cmd['Time']  - real_cmd['Time'].iloc[0]
gaz_odom['time']  = gaz_odom['Time']  - gaz_odom['Time'].iloc[0]
gaz_imu['time']   = gaz_imu['Time']   - gaz_imu['Time'].iloc[0]
gaz_cmd['time']   = gaz_cmd['Time']   - gaz_cmd['Time'].iloc[0]

# Plot 1 — Robot Path Comparison
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(real_odom['pose.pose.position.x'].to_numpy(),
         real_odom['pose.pose.position.y'].to_numpy(),
         label='Real Robot', color='blue')
plt.title('Real Robot Path')
plt.xlabel('X position (m)')
plt.ylabel('Y position (m)')
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(gaz_odom['pose.pose.position.x'].to_numpy(),
         gaz_odom['pose.pose.position.y'].to_numpy(),
         label='Gazebo', color='red')
plt.title('Gazebo Path')
plt.xlabel('X position (m)')
plt.ylabel('Y position (m)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('/workspaces/lab2-3-team-1/catkin_ws/src/lab2/data/comparison_path.png')
plt.close()
print("Saved comparison_path.png")

# Plot 2 — Linear Velocity Comparison
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(real_odom['time'].to_numpy(),
         real_odom['twist.twist.linear.x'].to_numpy(),
         label='Real Robot', color='blue')
plt.title('Real Robot Linear Velocity')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(gaz_odom['time'].to_numpy(),
         gaz_odom['twist.twist.linear.x'].to_numpy(),
         label='Gazebo', color='red')
plt.title('Gazebo Linear Velocity')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('/workspaces/lab2-3-team-1/catkin_ws/src/lab2/data/comparison_linear_vel.png')
plt.close()
print("Saved comparison_linear_vel.png")

# Plot 3 — Angular Velocity Comparison
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(real_odom['time'].to_numpy(),
         real_odom['twist.twist.angular.z'].to_numpy(),
         label='Real Robot', color='blue')
plt.title('Real Robot Angular Velocity')
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (rad/s)')
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(gaz_odom['time'].to_numpy(),
         gaz_odom['twist.twist.angular.z'].to_numpy(),
         label='Gazebo', color='red')
plt.title('Gazebo Angular Velocity')
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (rad/s)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('/workspaces/lab2-3-team-1/catkin_ws/src/lab2/data/comparison_angular_vel.png')
plt.close()
print("Saved comparison_angular_vel.png")

# Plot 4 — IMU Acceleration Comparison
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(real_imu['time'].to_numpy(),
         real_imu['linear_acceleration.x'].to_numpy(),
         label='Real Robot X', color='blue')
plt.plot(real_imu['time'].to_numpy(),
         real_imu['linear_acceleration.y'].to_numpy(),
         label='Real Robot Y', color='cyan')
plt.title('Real Robot IMU Acceleration')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s²)')
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(gaz_imu['time'].to_numpy(),
         gaz_imu['linear_acceleration.x'].to_numpy(),
         label='Gazebo X', color='red')
plt.plot(gaz_imu['time'].to_numpy(),
         gaz_imu['linear_acceleration.y'].to_numpy(),
         label='Gazebo Y', color='orange')
plt.title('Gazebo IMU Acceleration')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s²)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('/workspaces/lab2-3-team-1/catkin_ws/src/lab2/data/comparison_imu_accel.png')
plt.close()
print("Saved comparison_imu_accel.png")

# Plot 5 — IMU Angular Velocity Comparison
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(real_imu['time'].to_numpy(),
         real_imu['angular_velocity.z'].to_numpy(),
         label='Real Robot', color='blue')
plt.title('Real Robot IMU Angular Velocity')
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (rad/s)')
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(gaz_imu['time'].to_numpy(),
         gaz_imu['angular_velocity.z'].to_numpy(),
         label='Gazebo', color='red')
plt.title('Gazebo IMU Angular Velocity')
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (rad/s)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('/workspaces/lab2-3-team-1/catkin_ws/src/lab2/data/comparison_imu_angular.png')
plt.close()
print("Saved comparison_imu_angular.png")

print("All comparison plots saved!")