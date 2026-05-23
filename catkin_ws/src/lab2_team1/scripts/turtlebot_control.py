#!/usr/bin/env python3
import rospy
import math
from nav_msgs.msg import Path
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion

class TurtleBotController:
    def __init__(self):
        rospy.init_node('turtlebot_controller')

        # Publisher
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # Subscribers
        rospy.Subscriber('/planned_path', Path, self.path_callback)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)

        self.waypoints   = []
        self.current_wp  = 0
        self.x           = 0.0
        self.y           = 0.0
        self.yaw         = 0.0

        # Control parameters
        self.DIST_THRESH = 0.20
        self.ANG_KP      = 1.2
        self.LIN_KP      = 0.4
        self.MAX_LIN     = 0.15
        self.MAX_ANG     = 1.0

        rospy.loginfo("TurtleBot controller node ready!")
        rospy.spin()

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q      = msg.pose.pose.orientation
        _, _, self.yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.navigate()

    def path_callback(self, msg):
        self.waypoints  = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        self.current_wp = 0
        rospy.loginfo(f"Received path with {len(self.waypoints)} waypoints")

    def navigate(self):
        if self.current_wp >= len(self.waypoints):
            self.stop()
            return

        tx, ty = self.waypoints[self.current_wp]
        dx, dy = tx - self.x, ty - self.y
        dist   = math.hypot(dx, dy)

        if dist < self.DIST_THRESH:
            self.current_wp += 1
            if self.current_wp >= len(self.waypoints):
                rospy.loginfo("Final goal reached! Stopping.")
                self.stop()
            return

        target_angle = math.atan2(dy, dx)
        angle_err    = math.atan2(
            math.sin(target_angle - self.yaw),
            math.cos(target_angle - self.yaw)
        )

        cmd           = Twist()
        cmd.angular.z = max(-self.MAX_ANG, min(self.MAX_ANG, self.ANG_KP * angle_err))

        if abs(angle_err) < 0.3:
            cmd.linear.x = max(0, min(self.MAX_LIN, self.LIN_KP * dist))

        self.cmd_pub.publish(cmd)

    def stop(self):
        self.cmd_pub.publish(Twist())

if __name__ == '__main__':
    TurtleBotController()