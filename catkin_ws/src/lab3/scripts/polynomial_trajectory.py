#!/usr/bin/env python3
import rospy
import moveit_commander
import sys
import numpy as np
from geometry_msgs.msg import Pose

class PolynomialTrajectory:
    def __init__(self):
        rospy.init_node('polynomial_trajectory', anonymous=True)
        moveit_commander.roscpp_initialize(sys.argv)

        self.arm = moveit_commander.MoveGroupCommander("arm")
        self.arm.set_max_velocity_scaling_factor(0.5)
        self.arm.set_max_acceleration_scaling_factor(0.5)

        rospy.loginfo("Polynomial Trajectory node ready!")

    def quintic_polynomial(self, q0, qf, T, num_points=100):
        """
        Generate quintic polynomial trajectory.
        Matches position, velocity AND acceleration at
        start and end (all zero vel/accel at boundaries).

        q0:         start value
        qf:         end value
        T:          total duration (seconds)
        num_points: number of trajectory points

        Returns: t, q (position), qd (velocity), qdd (accel)
        """
        # Quintic coefficients
        a0 =  q0
        a1 =  0.0
        a2 =  0.0
        a3 =  10*(qf - q0) / T**3
        a4 = -15*(qf - q0) / T**4
        a5 =   6*(qf - q0) / T**5

        t   = np.linspace(0, T, num_points)

        # Position
        q   = (a0 + a1*t + a2*t**2 + a3*t**3
               + a4*t**4 + a5*t**5)

        # Velocity
        qd  = (a1 + 2*a2*t + 3*a3*t**2
               + 4*a4*t**3 + 5*a5*t**4)

        # Acceleration
        qdd = (2*a2 + 6*a3*t
               + 12*a4*t**2 + 20*a5*t**3)

        return t, q, qd, qdd

    def task_space_trajectory(self, start, end,
                               T=3.0, num_points=50):
        """
        Execute quintic polynomial trajectory in task space.
        start: [x, y, z] start position in metres
        end:   [x, y, z] end position in metres
        T:     total duration in seconds
        """
        rospy.loginfo("=" * 50)
        rospy.loginfo("Quintic Polynomial - Task Space")
        rospy.loginfo(f"Start: {start}")
        rospy.loginfo(f"End:   {end}")
        rospy.loginfo(f"Duration: {T}s, Points: {num_points}")
        rospy.loginfo("=" * 50)

        # Generate quintic for each axis
        t, x_traj, xd, xdd = self.quintic_polynomial(
            start[0], end[0], T, num_points)
        _, y_traj, yd, ydd = self.quintic_polynomial(
            start[1], end[1], T, num_points)
        _, z_traj, zd, zdd = self.quintic_polynomial(
            start[2], end[2], T, num_points)

        # Print trajectory stats
        rospy.loginfo("Trajectory stats:")
        rospy.loginfo(f"  X: {start[0]:.3f} -> {end[0]:.3f} m")
        rospy.loginfo(f"  Y: {start[1]:.3f} -> {end[1]:.3f} m")
        rospy.loginfo(f"  Z: {start[2]:.3f} -> {end[2]:.3f} m")
        rospy.loginfo(f"  Max X vel: {max(abs(xd)):.4f} m/s")
        rospy.loginfo(f"  Max Y vel: {max(abs(yd)):.4f} m/s")
        rospy.loginfo(f"  Max Z vel: {max(abs(zd)):.4f} m/s")

        # Build waypoints from trajectory points
        poses = []
        step = max(1, num_points // 10)
        for i in range(0, num_points, step):
            pose = Pose()
            pose.position.x = float(x_traj[i])
            pose.position.y = float(y_traj[i])
            pose.position.z = float(z_traj[i])
            pose.orientation.w = 1.0
            poses.append(pose)

        # Always include final point
        pose = Pose()
        pose.position.x = float(end[0])
        pose.position.y = float(end[1])
        pose.position.z = float(end[2])
        pose.orientation.w = 1.0
        poses.append(pose)

        # Compute and execute cartesian path
        (plan, fraction) = self.arm.compute_cartesian_path(
            poses, 0.01, 0.0)

        rospy.loginfo(f"Cartesian path fraction: {fraction:.2f}")

        if fraction > 0.8:
            self.arm.execute(plan, wait=True)
            self.arm.stop()
            rospy.loginfo("Task space trajectory executed!")
        else:
            rospy.logwarn(
                f"Only {fraction*100:.1f}% of path achieved!")

        return t, x_traj, y_traj, z_traj, xd, yd, zd

    def joint_space_trajectory(self, start_joints, end_joints,
                                T=3.0, num_points=50):
        """
        Execute quintic polynomial trajectory in joint space.
        start_joints: [j1,j2,j3,j4] start angles in radians
        end_joints:   [j1,j2,j3,j4] end angles in radians
        T:            total duration in seconds
        """
        rospy.loginfo("=" * 50)
        rospy.loginfo("Quintic Polynomial - Joint Space")
        rospy.loginfo(f"Start joints: {start_joints}")
        rospy.loginfo(f"End joints:   {end_joints}")
        rospy.loginfo(f"Duration: {T}s, Points: {num_points}")
        rospy.loginfo("=" * 50)

        num_joints = len(start_joints)
        all_t   = None
        all_q   = []
        all_qd  = []
        all_qdd = []

        # Generate quintic polynomial for each joint
        for i in range(num_joints):
            t, q, qd, qdd = self.quintic_polynomial(
                start_joints[i], end_joints[i], T, num_points)
            if all_t is None:
                all_t = t
            all_q.append(q)
            all_qd.append(qd)
            all_qdd.append(qdd)

        # Print trajectory stats per joint
        for i in range(num_joints):
            rospy.loginfo(
                f"  Joint{i+1}: "
                f"{np.degrees(start_joints[i]):.1f} -> "
                f"{np.degrees(end_joints[i]):.1f} deg  "
                f"MaxVel: "
                f"{np.degrees(max(abs(all_qd[i]))):.2f} deg/s")

        # Build waypoints - use only 5 points to avoid timeout
        waypoints_joints = []
        step = max(1, num_points // 5)
        for idx in range(0, num_points, step):
            joint_goal = [float(all_q[j][idx])
                          for j in range(num_joints)]
            waypoints_joints.append(joint_goal)

        # Always add exact end point
        waypoints_joints.append(
            [float(end_joints[j]) for j in range(num_joints)])

        rospy.loginfo(
            f"Executing {len(waypoints_joints)} waypoints...")

        # Execute each waypoint with pause between
        for i, jg in enumerate(waypoints_joints):
            rospy.loginfo(
                f"Waypoint {i+1}/{len(waypoints_joints)}: "
                f"{[round(np.degrees(v),1) for v in jg]} deg")
            self.arm.set_joint_value_target(jg)
            self.arm.go(wait=True)
            self.arm.stop()
            rospy.sleep(0.5)

        rospy.loginfo("Joint space trajectory complete!")
        return all_t, all_q, all_qd, all_qdd

    def go_home(self):
        """Return to home position"""
        rospy.loginfo("Going home...")
        self.arm.set_named_target("home")
        self.arm.go(wait=True)
        self.arm.stop()
        rospy.loginfo("Home reached!")


def main():
    node = PolynomialTrajectory()
    rospy.sleep(2.0)

    # ──────────────────────────────────────────────────────────
    # TASK 9: Quintic Polynomial in Task Space
    # ──────────────────────────────────────────────────────────
    start_task = [0.25, 0.0,  0.20]   # start (m)
    end_task   = [0.20, 0.1,  0.15]   # end   (m)
    T          = 3.0                   # duration (s)

    rospy.loginfo("=" * 50)
    rospy.loginfo("TASK 9: Quintic Polynomial - Task Space")
    rospy.loginfo("=" * 50)

    # Move to start position
    rospy.loginfo("Moving to start position...")
    pose = Pose()
    pose.position.x = start_task[0]
    pose.position.y = start_task[1]
    pose.position.z = start_task[2]
    pose.orientation.w = 1.0
    node.arm.set_pose_target(pose)
    node.arm.go(wait=True)
    node.arm.stop()
    node.arm.clear_pose_targets()
    rospy.sleep(1.0)

    # Execute task space polynomial trajectory
    result = node.task_space_trajectory(
        start_task, end_task, T=T, num_points=50)
    rospy.sleep(1.0)

    # Return home
    node.go_home()
    rospy.sleep(2.0)
    rospy.loginfo("Task 9 Complete!")

    # ──────────────────────────────────────────────────────────
    # TASK 10: Quintic Polynomial in Joint Space
    # ──────────────────────────────────────────────────────────
    # Pose 1: j1=-12, j2=58, j3=-52, j4=-15 degrees
    start_j = [-0.2094,  1.0123, -0.9076, -0.2618]
    # Pose 2: j1=28, j2=6, j3=-41, j4=-6 degrees
    end_j   = [ 0.4887,  0.1047, -0.7156, -0.1047]

    rospy.loginfo("=" * 50)
    rospy.loginfo("TASK 10: Quintic Polynomial - Joint Space")
    rospy.loginfo("=" * 50)

    # Move to start joint position
    rospy.loginfo("Moving to start joint position...")
    node.arm.set_joint_value_target(start_j)
    node.arm.go(wait=True)
    node.arm.stop()
    rospy.sleep(1.0)

    # Execute joint space polynomial trajectory
    node.joint_space_trajectory(start_j, end_j, T=T)
    rospy.sleep(1.0)

    # Return home
    node.go_home()
    rospy.loginfo("Task 10 Complete!")

    rospy.loginfo("=" * 50)
    rospy.loginfo("ALL POLYNOMIAL TASKS COMPLETE!")
    rospy.loginfo("=" * 50)


if __name__ == '__main__':
    main()