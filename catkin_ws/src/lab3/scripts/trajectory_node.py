#!/usr/bin/env python3
import rospy
import moveit_commander
import sys
import numpy as np
from geometry_msgs.msg import Pose

class TrajectoryNode:
    def __init__(self):
        rospy.init_node('trajectory_node', anonymous=True)
        moveit_commander.roscpp_initialize(sys.argv)
        
        # Initialize MoveIt commanders
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.arm   = moveit_commander.MoveGroupCommander("arm")
        
        # Settings
        self.arm.set_planner_id("RRTConnectkConfigDefault")
        self.arm.set_planning_time(5.0)
        self.arm.set_max_velocity_scaling_factor(0.5)
        self.arm.set_max_acceleration_scaling_factor(0.5)
        self.arm.set_num_planning_attempts(5)
        
        rospy.loginfo("Trajectory node ready!")

    def move_to_joint_goal(self, joint_angles):
        """
        Move arm to joint space goal
        joint_angles: list of 4 angles in radians [j1, j2, j3, j4]
        """
        rospy.loginfo(f"Moving to joint goal: {joint_angles}")
        self.arm.set_joint_value_target(joint_angles)
        result = self.arm.go(wait=True)
        self.arm.stop()
        if result:
            rospy.loginfo("Joint goal reached successfully!")
        else:
            rospy.logwarn("Joint goal failed!")
        return result

    def move_to_task_goal(self, x, y, z):
        """
        Move arm to task space goal
        x, y, z: end effector position in metres
        """
        rospy.loginfo(f"Moving to task goal: ({x}, {y}, {z})")
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation.w = 1.0
        self.arm.set_pose_target(pose)
        result = self.arm.go(wait=True)
        self.arm.stop()
        self.arm.clear_pose_targets()
        if result:
            rospy.loginfo("Task goal reached successfully!")
        else:
            rospy.logwarn("Task goal failed!")
        return result

    def move_to_waypoints(self, waypoints):
        """
        Move arm through list of waypoints
        waypoints: list of (x, y, z) tuples
        """
        rospy.loginfo(f"Moving through {len(waypoints)} waypoints")
        poses = []
        for wp in waypoints:
            pose = Pose()
            pose.position.x = wp[0]
            pose.position.y = wp[1]
            pose.position.z = wp[2]
            pose.orientation.w = 1.0
            poses.append(pose)

        # Compute cartesian path
        (plan, fraction) = self.arm.compute_cartesian_path(
            poses,
            0.01,   # eef_step 1cm
            0.0)    # jump_threshold

        rospy.loginfo(f"Cartesian path fraction: {fraction:.2f}")

        if fraction > 0.9:
            result = self.arm.execute(plan, wait=True)
            self.arm.stop()
            if result:
                rospy.loginfo("Waypoints completed successfully!")
            else:
                rospy.logwarn("Waypoint execution failed!")
        else:
            rospy.logwarn(f"Only {fraction*100:.1f}% of path achieved!")
        return fraction

    def go_home(self):
        """Return arm to home position - all joints at 0"""
        rospy.loginfo("Going to home position...")
        self.arm.set_named_target("home")
        result = self.arm.go(wait=True)
        self.arm.stop()
        if result:
            rospy.loginfo("Home position reached!")
        else:
            rospy.logwarn("Failed to reach home!")
        return result

def main():
    node = TrajectoryNode()
    rospy.sleep(2.0)

    # ─── Test 1: Joint Space Goal ────────────────────────────────
    rospy.loginfo("=" * 50)
    rospy.loginfo("Test 1: Joint Space Goal")
    rospy.loginfo("=" * 50)
    # Pose 1 angles in radians
    # degrees: j1=-12, j2=58, j3=-52, j4=-15
    joint_goal = [-0.2094, 1.0123, -0.9076, -0.2618]
    node.move_to_joint_goal(joint_goal)
    rospy.sleep(2.0)

    # Return home
    node.go_home()
    rospy.sleep(2.0)

    # ─── Test 2: Task Space Goal ─────────────────────────────────
    rospy.loginfo("=" * 50)
    rospy.loginfo("Test 2: Task Space Goal")
    rospy.loginfo("=" * 50)
    node.move_to_task_goal(0.25, 0.0, 0.15)
    rospy.sleep(2.0)

    # Return home
    node.go_home()
    rospy.sleep(2.0)

    # ─── Test 3: Waypoints ───────────────────────────────────────
    rospy.loginfo("=" * 50)
    rospy.loginfo("Test 3: Waypoints")
    rospy.loginfo("=" * 50)
    waypoints = [
        (0.20, 0.0, 0.15),
        (0.22, 0.0, 0.18),
        (0.25, 0.0, 0.20),
    ]
    node.move_to_waypoints(waypoints)
    rospy.sleep(2.0)

    # Return home
    node.go_home()

    rospy.loginfo("=" * 50)
    rospy.loginfo("All Tests Complete!")
    rospy.loginfo("=" * 50)

if __name__ == '__main__':
    main()