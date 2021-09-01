#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree
import numpy as np

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.
'''

# Based on some strange behavior in the simulator and researching recommendations
# on the Mentor Help page, I settled on LOOKAHEAD_WPS = 50.
# This represents 50 published waypoints
LOOKAHEAD_WPS = 50
MAX_DECEL = 0.5


class WaypointUpdater(object):
    def __init__(self):
        '''
        Referenced the implementation in the System Integration Project,
        Lesson 5: Waypoint Updater Partial Walkthrough
        '''
        rospy.init_node('waypoint_updater')

        # Subscribers for current pose, base waypoints, and traffic waypoint
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        # Publish the final waypoints
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # Initialize other needed variables
        self.pose = None
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.stopline_wp_idx = -1
        self.base_lane = None
        
        self.loop()
        
    def loop(self):
        '''
        Source: System Integration Project, Lesson 5: Waypoint Updater Partial Walkthrough
        '''
        # Iterate through this loop at a rate of 50 Hz as long as rospy is not shutdown
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            if self.pose and self.base_lane:
                self.publish_waypoints()
            rate.sleep()
            
    def get_closest_waypoint_idx(self):
        '''
        Get the index of the closest waypoint
        
        Source: System Integration Project, Lesson 5: Waypoint Updater Partial Walkthrough
        '''
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]
        
        # Check if closest is ahead or behind vehicle
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx - 1]
        
        # Equation for hyperplane through closest_coords
        closest_vect = np.array(closest_coord)
        previous_vect = np.array(prev_coord)
        positional_vect = np.array([x, y])
        
        val = np.dot(closest_vect - previous_vect, positional_vect - closest_vect)
        
        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)
        
        return closest_idx
    
    def publish_waypoints(self):
        '''
        Use the generate_lane() function to obtain the final lane
        and publish the lane to the final_waypoints message
        
        Source: System Integration Project, Lesson 5: Waypoint Updater Partial Walkthrough
        and modified in Lesson 12: Full Waypoint Walkthrough
        '''
        final_lane = self.generate_lane()
        self.final_waypoints_pub.publish(final_lane)
        
    def generate_lane(self):
        '''
        Generate a lane with waypoints, depending on the stopline
        and the base waypoints
        
        Source: System Integration Project, Lesson 12: Full Waypoint Walkthrough
        '''
        lane = Lane()
        
        closest_index = self.get_closest_waypoint_idx()
        farthest_index = closest_index + LOOKAHEAD_WPS
        base_waypoints = self.base_lane.waypoints[closest_index:farthest_index]
        
        if self.stopline_wp_idx == -1 or (self.stopline_wp_idx >= farthest_index):
            lane.waypoints = base_waypoints
        else:
            lane.waypoints = self.decelerate_waypoints(base_waypoints, closest_index)
            
        return lane
    
    def decelerate_waypoints(self, waypoints, closest_index):
        '''
        When a stop bar is identified, modify the waypoints
        ahead to attempt to stop by the line.
        
        Source: System Integration Project, Lesson 12: Full Waypoints Walkthrough
        '''
        temp_waypoints = []
        for i, waypoint in enumerate(waypoints):
            new_wp = Waypoint()
            new_wp.pose = waypoint.pose
           
            # This sets the stop distance, so the car will stop by the line
            stop_index = max(self.stopline_wp_idx - closest_index - 2, 0)
            dist = self.distance(waypoints, i, stop_index)
            vel = math.sqrt(2 * MAX_DECEL * dist)
            if vel < 1.0:
                vel = 0.0
            
            min_velocity = min(vel, waypoint.twist.twist.linear.x)
            new_wp.twist.twist.linear.x = min_velocity
            temp_waypoints.append(new_wp)
            
        return temp_waypoints

    def pose_cb(self, msg):
        '''
        Callback function for the /current_pose message
        
        Source: System Integration Project, Lesson 5: Waypoint Updater Partial Walkthrough
        '''
        self.pose = msg
        
    def waypoints_cb(self, waypoints):
        '''
        Callback function for the /base_waypoints message
        
        Source: System Integration Project, Lesson 5: Waypoint Updater Partial Walkthrough
        '''
        self.base_lane = waypoints
        if self.waypoints_2d is None:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        '''
        Callback function for the /traffic_waypoint message
        
        Source: System Integration Project, Lesson 12: Final Waypoints Walkthrough
        '''
        self.stopline_wp_idx = msg.data

    def obstacle_cb(self, msg):
        '''
        Callback function for the obstacle message
        
        Not needed, so it is passed
        '''
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
