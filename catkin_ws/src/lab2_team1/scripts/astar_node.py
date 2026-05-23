#!/usr/bin/env python3
import rospy
import heapq
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Point
from nav_msgs.msg import GridCells

class AStarNode:
    def __init__(self):
        rospy.init_node('astar_planner')

        # Publishers
        self.pub_expanded = rospy.Publisher('/astar/expanded', GridCells, queue_size=10)
        self.pub_frontier = rospy.Publisher('/astar/frontier', GridCells, queue_size=10)
        self.pub_path     = rospy.Publisher('/astar/path',     GridCells, queue_size=10)
        self.pub_nav_path = rospy.Publisher('/planned_path',   Path,      queue_size=10)

        # Subscribers
        rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        rospy.Subscriber('/initialpose', PoseWithCovarianceStamped, self.start_callback)
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)

        self.map   = None
        self.start = None
        self.goal  = None

        rospy.loginfo("A* planner node ready!")
        rospy.spin()

    def map_callback(self, msg):
        self.map = msg
        rospy.loginfo("Map received!")

    def start_callback(self, msg):
        self.start = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        rospy.loginfo(f"Start set: {self.start}")
        self.try_plan()

    def goal_callback(self, msg):
        self.goal = (msg.pose.position.x, msg.pose.position.y)
        rospy.loginfo(f"Goal set: {self.goal}")
        self.try_plan()

    def try_plan(self):
        if self.start and self.goal and self.map:
            self.run_astar()

    def world_to_grid(self, wx, wy):
        res = self.map.info.resolution
        ox  = self.map.info.origin.position.x
        oy  = self.map.info.origin.position.y
        gx  = int((wx - ox) / res)
        gy  = int((wy - oy) / res)
        return (gx, gy)

    def grid_to_world(self, gx, gy):
        res = self.map.info.resolution
        ox  = self.map.info.origin.position.x
        oy  = self.map.info.origin.position.y
        wx  = gx * res + ox + res / 2
        wy  = gy * res + oy + res / 2
        return (wx, wy)

    def is_free(self, gx, gy):
        w = self.map.info.width
        h = self.map.info.height
        # Check with inflation radius of 3 cells
        inflation = 3
        for dx in range(-inflation, inflation+1):
            for dy in range(-inflation, inflation+1):
                nx, ny = gx+dx, gy+dy
                if nx < 0 or ny < 0 or nx >= w or ny >= h:
                    return False
                idx = ny * w + nx
                if self.map.data[idx] != 0:
                    return False
        return True

    def heuristic(self, a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def make_grid_cells(self, cells_list):
        msg                 = GridCells()
        msg.header.frame_id = 'map'
        msg.header.stamp    = rospy.Time.now()
        msg.cell_width      = self.map.info.resolution
        msg.cell_height     = self.map.info.resolution
        msg.cells           = []
        for (gx, gy) in cells_list:
            wx, wy = self.grid_to_world(gx, gy)
            p      = Point()
            p.x, p.y, p.z = wx, wy, 0.0
            msg.cells.append(p)
        return msg

    def run_astar(self):
        rospy.loginfo("Running A*...")
        start_g = self.world_to_grid(*self.start)
        goal_g  = self.world_to_grid(*self.goal)

        open_set     = []
        heapq.heappush(open_set, (0, start_g))
        came_from    = {}
        g_score      = {start_g: 0}
        expanded     = []
        frontier_set = set([start_g])
        rate         = rospy.Rate(20)

        while open_set:
            _, current = heapq.heappop(open_set)
            frontier_set.discard(current)
            expanded.append(current)

            self.pub_expanded.publish(self.make_grid_cells(expanded))
            self.pub_frontier.publish(self.make_grid_cells(list(frontier_set)))
            rate.sleep()

            if current == goal_g:
                path = self.reconstruct_path(came_from, current)
                self.pub_path.publish(self.make_grid_cells(path))
                self.publish_nav_path(path)
                rospy.loginfo("Path found!")
                return

            for dx, dy in [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
                nb        = (current[0]+dx, current[1]+dy)
                if not self.is_free(*nb):
                    continue
                move_cost = 1.414 if dx != 0 and dy != 0 else 1.0
                tent_g    = g_score[current] + move_cost
                if nb not in g_score or tent_g < g_score[nb]:
                    g_score[nb] = tent_g
                    f = tent_g + self.heuristic(nb, goal_g)
                    heapq.heappush(open_set, (f, nb))
                    came_from[nb] = current
                    frontier_set.add(nb)

        rospy.logwarn("No path found!")

    def reconstruct_path(self, came_from, current):
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(current)
        return path[::-1]

    def reduce_waypoints(self, path):
        if len(path) < 3:
            return path
        reduced = [path[0]]
        for i in range(1, len(path)-1):
            prev = path[i-1]
            curr = path[i]
            nxt  = path[i+1]
            dx1, dy1 = curr[0]-prev[0], curr[1]-prev[1]
            dx2, dy2 = nxt[0]-curr[0],  nxt[1]-curr[1]
            if (dx1, dy1) != (dx2, dy2):
                reduced.append(curr)
        reduced.append(path[-1])
        return reduced

    def publish_nav_path(self, grid_path):
        optimized            = self.reduce_waypoints(grid_path)
        nav_path             = Path()
        nav_path.header.frame_id = 'map'
        nav_path.header.stamp    = rospy.Time.now()
        for (gx, gy) in optimized:
            wx, wy = self.grid_to_world(gx, gy)
            pose   = PoseStamped()
            pose.header.frame_id    = 'map'
            pose.pose.position.x    = wx
            pose.pose.position.y    = wy
            pose.pose.orientation.w = 1.0
            nav_path.poses.append(pose)
        self.pub_nav_path.publish(nav_path)
        rospy.loginfo(f"Full path: {len(grid_path)} pts → Optimized: {len(optimized)} pts")

if __name__ == '__main__':
    AStarNode()