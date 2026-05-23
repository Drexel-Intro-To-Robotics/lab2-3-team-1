#!/usr/bin/env python3
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Read CSV files directly
print("Loading data...")
odom_df = pd.read_csv('/workspaces/lab2-3-team-1/catkin_ws/src/lab2/data/all_tasks/odom.csv')
imu_df  = pd.read_csv('/workspaces/lab2-3-team-1/catkin_ws/src/lab2/data/all_tasks/imu.csv')
cmd_df  = pd.read_csv('/workspaces/lab2-3-team-1/catkin_ws/src/lab2/data/all_tasks/cmd_vel.csv')

# Normalize time
odom_df['time'] = odom_df['Time'] - odom_df['Time'].iloc[0]
imu_df['time']  = imu_df['Time']  - imu_df['Time'].iloc[0]
cmd_df['time']  = cmd_df['Time']  - cmd_df['Time'].iloc[0]

# Plot 1 — Robot Path
plt.figure(figsize=(10,5))
plt.plot(odom_df['pose.pose.position.x'].to_numpy(),
         odom_df['pose.pose.position.y'].to_numpy())
plt.title('Robot Path (X vs Y)')
plt.xlabel('X position (m)')
plt.ylabel('Y position (m)')
plt.grid(True)
plt.savefig('/workspaces/lab2-3-team-1/catkin_ws/src/lab2/data/path.png')
plt.close()
print("Saved path.png")

# Plot 2 — Linear Velocity
plt.figure(figsize=(10,5))
plt.plot(odom_df['time'].to_numpy(),
         odom_df['twist.twist.linear.x'].to_numpy(),
         label='Linear Velocity')
plt.title('Linear Velocity over Time')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.grid(True)
plt.savefig('/workspaces/lab2-3-team-1/catkin_ws/src/lab2/data/linear_velocity.png')
plt.close()
print("Saved linear_velocity.png")

# Plot 3 — Angular Velocity
plt.figure(figsize=(10,5))
plt.plot(odom_df['time'].to_numpy(),
         odom_df['twist.twist.angular.z'].to_numpy(),
         label='Angular Velocity',
         color='orange')
plt.title('Angular Velocity over Time')
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (rad/s)')
plt.legend()
plt.grid(True)
plt.savefig('/workspaces/lab2-3-team-1/catkin_ws/src/lab2/data/angular_velocity.png')
plt.close()
print("Saved angular_velocity.png")

# Plot 4 — IMU Acceleration
plt.figure(figsize=(10,5))
plt.plot(imu_df['time'].to_numpy(),
         imu_df['linear_acceleration.x'].to_numpy(),
         label='X acceleration')
plt.plot(imu_df['time'].to_numpy(),
         imu_df['linear_acceleration.y'].to_numpy(),
         label='Y acceleration')
plt.title('IMU Linear Acceleration over Time')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s²)')
plt.legend()
plt.grid(True)
plt.savefig('/workspaces/lab2-3-team-1/catkin_ws/src/lab2/data/imu_acceleration.png')
plt.close()
print("Saved imu_acceleration.png")

# Plot 5 — IMU Angular Velocity
plt.figure(figsize=(10,5))
plt.plot(imu_df['time'].to_numpy(),
         imu_df['angular_velocity.z'].to_numpy(),
         label='Z angular velocity',
         color='green')
plt.title('IMU Angular Velocity over Time')
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (rad/s)')
plt.legend()
plt.grid(True)
plt.savefig('/workspaces/lab2-3-team-1/catkin_ws/src/lab2/data/imu_angular.png')
plt.close()
print("Saved imu_angular.png")

# Plot 6 — cmd_vel
plt.figure(figsize=(10,5))
plt.plot(cmd_df['time'].to_numpy(),
         cmd_df['linear.x'].to_numpy(),
         label='Commanded Linear Velocity')
plt.plot(cmd_df['time'].to_numpy(),
         cmd_df['angular.z'].to_numpy(),
         label='Commanded Angular Velocity',
         color='red')
plt.title('Commanded Velocity over Time')
plt.xlabel('Time (s)')
plt.ylabel('Velocity')
plt.legend()
plt.grid(True)
plt.savefig('/workspaces/lab2-3-team-1/catkin_ws/src/lab2/data/cmd_vel.png')
plt.close()
print("Saved cmd_vel.png")

print("All plots saved!")