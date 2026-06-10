#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import Pose
import moveit_commander
import sys

def forward_kinematics(joint_angles):
    """
    Calculate end effector position from joint angles
    using DH parameters for OpenManipulator-X
    """
    # DH Parameters
    # [a, alpha, d, theta_offset]
    d1 = 0.077   # base to shoulder height
    L2 = 0.130   # shoulder to elbow
    L3 = 0.124   # elbow to wrist  
    L4 = 0.126   # wrist to gripper

    t1, t2, t3, t4 = joint_angles

    # Forward kinematics equations
    x = (L2*np.cos(t2) + L3*np.cos(t2+t3) + 
         L4*np.cos(t2+t3+t4)) * np.cos(t1)
    y = (L2*np.cos(t2) + L3*np.cos(t2+t3) + 
         L4*np.cos(t2+t3+t4)) * np.sin(t1)
    z = (d1 + L2*np.sin(t2) + L3*np.sin(t2+t3) + 
         L4*np.sin(t2+t3+t4))

    return x, y, z

def inverse_kinematics(target_x, target_y, target_z, 
                       phi=0.0):
    """
    Analytical IK for OpenManipulator-X
    phi = desired end effector angle (default 0 = horizontal)
    Returns joint angles [t1, t2, t3, t4]
    """
    # DH Parameters
    d1 = 0.077
    L2 = 0.130
    L3 = 0.124
    L4 = 0.126

    # Joint 1 - base rotation
    t1 = np.arctan2(target_y, target_x)

    # Wrist position (subtract last link)
    xw = np.sqrt(target_x**2 + target_y**2) - L4*np.cos(phi)
    zw = target_z - d1 - L4*np.sin(phi)

    # Distance to wrist
    r = np.sqrt(xw**2 + zw**2)

    # Check if reachable
    if r > L2 + L3:
        print(f"Target unreachable! Distance {r:.3f} > max reach {L2+L3:.3f}")
        return None

    # Joint 3 - elbow (cosine rule)
    cos_t3 = (r**2 - L2**2 - L3**2) / (2 * L2 * L3)
    cos_t3 = np.clip(cos_t3, -1.0, 1.0)
    t3 = np.arctan2(-np.sqrt(1 - cos_t3**2), cos_t3)  # elbow up

    # Joint 2 - shoulder
    t2 = np.arctan2(zw, xw) - np.arctan2(L3*np.sin(t3), 
                                           L2 + L3*np.cos(t3))

    # Joint 4 - wrist
    t4 = phi - t2 - t3

    return [t1, t2, t3, t4]

def main():
    rospy.init_node('ik_solver', anonymous=True)
    
    # Three goal poses for our experiments
    poses = [
        {"name": "Pose 1", "x": 0.250, "y": 0.000, "z": 0.150},
        {"name": "Pose 2", "x": 0.200, "y": 0.100, "z": 0.200},
        {"name": "Pose 3", "x": 0.200, "y":-0.100, "z": 0.150},
    ]

    print("\n" + "="*50)
    print("OpenManipulator-X Inverse Kinematics Solver")
    print("="*50)

    for pose in poses:
        print(f"\n--- {pose['name']} ---")
        print(f"Target: x={pose['x']}, y={pose['y']}, z={pose['z']}")

        # Solve IK
        angles = inverse_kinematics(pose['x'], pose['y'], pose['z'])

        if angles is not None:
            print(f"Joint angles (rad): {[round(a,4) for a in angles]}")
            print(f"Joint angles (deg): {[round(np.degrees(a),2) for a in angles]}")

            # Verify with FK
            fx, fy, fz = forward_kinematics(angles)
            print(f"FK verification: x={fx:.4f}, y={fy:.4f}, z={fz:.4f}")
            
            # Calculate error
            error = np.sqrt((fx-pose['x'])**2 + 
                          (fy-pose['y'])**2 + 
                          (fz-pose['z'])**2)
            print(f"Position error: {error*1000:.2f} mm")

    print("\n" + "="*50)

if __name__ == '__main__':
    main()