#!/usr/bin/env python3
import rospy
import math
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from tf.transformations import euler_from_quaternion

class RobotControl:
    def __init__(self):
        rospy.init_node('robot_control')
        
        # Publisher
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        # Subscribers
        rospy.Subscriber('/odom', Odometry, self.odom_callback)
        rospy.Subscriber('/imu', Imu, self.imu_callback)
        
        # Robot state
        self.x   = 0.0
        self.y   = 0.0
        self.yaw = 0.0
        
        # IMU data
        self.imu_ax = 0.0
        self.imu_ay = 0.0
        self.imu_gz = 0.0
        
        self.rate = rospy.Rate(10)
        rospy.loginfo("Robot control node ready!")

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q      = msg.pose.pose.orientation
        _, _, self.yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

    def imu_callback(self, msg):
        self.imu_ax = msg.linear_acceleration.x
        self.imu_ay = msg.linear_acceleration.y
        self.imu_gz = msg.angular_velocity.z

    def stopping(self):
        rospy.loginfo("Stopping...")
        for _ in range(20):
            self.cmd_pub.publish(Twist())
            self.rate.sleep()
        rospy.sleep(1)

    def driving_straight(self, distance):
        rospy.loginfo(f"Driving straight {distance}m")
        start_x = self.x
        start_y = self.y
        cmd     = Twist()
        cmd.linear.x = 0.12
        while not rospy.is_shutdown():
            dist = math.hypot(self.x - start_x, self.y - start_y)
            if dist >= distance:
                break
            self.cmd_pub.publish(cmd)
            self.rate.sleep()
        self.stopping()

    def rotating(self, angle):
        rospy.loginfo(f"Rotating {math.degrees(angle):.1f} degrees")
        cmd           = Twist()
        cmd.angular.z = 0.25 if angle > 0 else -0.25
        turned        = 0.0
        prev_yaw      = self.yaw
        while not rospy.is_shutdown():
            delta = self.yaw - prev_yaw
            if delta > math.pi:
                delta -= 2 * math.pi
            elif delta < -math.pi:
                delta += 2 * math.pi
            turned   += delta
            prev_yaw  = self.yaw
            if abs(turned) >= abs(angle):
                break
            self.cmd_pub.publish(cmd)
            self.rate.sleep()
        self.stopping()

    def spinning_wheels(self, duration):
        rospy.loginfo(f"Spinning wheels for {duration}s")
        cmd          = Twist()
        cmd.linear.x = 0.1
        start        = rospy.Time.now()
        while not rospy.is_shutdown():
            if (rospy.Time.now() - start).to_sec() >= duration:
                break
            self.cmd_pub.publish(cmd)
            self.rate.sleep()
        self.stopping()

    def navigating_to_pose(self, x, y):
        rospy.loginfo(f"Navigating to x={x} y={y}")
        cmd = Twist()
        while not rospy.is_shutdown():
            dx           = x - self.x
            dy           = y - self.y
            dist         = math.hypot(dx, dy)
            if dist < 0.1:
                break
            target_angle = math.atan2(dy, dx)
            angle_err    = math.atan2(
                math.sin(target_angle - self.yaw),
                math.cos(target_angle - self.yaw)
            )
            cmd.angular.z = 1.0 * angle_err
            cmd.linear.x  = 0.12 if abs(angle_err) < 0.3 else 0.0
            self.cmd_pub.publish(cmd)
            self.rate.sleep()
        self.stopping()
        rospy.loginfo("Reached goal!")

    def drive_circle(self, radius):
        rospy.loginfo(f"Driving circle radius={radius}m")
        # Use slow speed for accurate circle
        linear_speed  = 0.10
        angular_speed = linear_speed / radius
        # Add 10% extra time to complete full circle
        duration      = (2 * math.pi * radius / linear_speed) * 1.35
        rospy.loginfo(f"Circle will take {duration:.1f} seconds")
        
        cmd           = Twist()
        cmd.linear.x  = linear_speed
        cmd.angular.z = angular_speed
        
        start = rospy.Time.now()
        while not rospy.is_shutdown():
            if (rospy.Time.now() - start).to_sec() >= duration:
                break
            self.cmd_pub.publish(cmd)
            self.rate.sleep()
        self.stopping()
        rospy.loginfo("Circle complete!")

    def drive_square(self, side):
        rospy.loginfo(f"Driving square side={side}m")
        for i in range(4):
            rospy.loginfo(f"Side {i+1} of 4")
            self.driving_straight(side)
            rospy.sleep(1)
            self.rotating(math.pi / 2)
            rospy.sleep(1)
        rospy.loginfo("Square complete!")

if __name__ == '__main__':
    robot = RobotControl()
    
    # Wait for everything to initialize
    rospy.loginfo("Waiting for robot to initialize...")
    rospy.sleep(3)

    # ===== TASK 5 — Drive Circle radius 0.5m =====
    rospy.loginfo("===== TASK 5: Drive Circle =====")
    robot.drive_circle(0.5)
    rospy.sleep(3)

    # ===== TASK 6 — Drive Square sides 0.5m =====
    rospy.loginfo("===== TASK 6: Drive Square =====")
    robot.drive_square(0.5)
    rospy.sleep(3)

    # ===== TASK 7 — Navigate to Pose =====
    rospy.loginfo("===== TASK 7: Navigate to Pose =====")
    robot.navigating_to_pose(1.0, 0.0)
    rospy.sleep(1)

    rospy.loginfo("===== ALL TASKS COMPLETE =====")