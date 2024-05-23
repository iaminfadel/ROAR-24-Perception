#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String, Int32

class PoseSubscriber:
    def __init__(self):
        self.pose_dict = {}
        self.target_num = 0
        self.publish_delay = rospy.Duration(1)  # Adjust the delay time as needed
        self.publish_timer = None
        self.pub = rospy.Publisher('/choosen_pose_stamp', PoseStamped, queue_size=10)
        rospy.Subscriber('point_poses', PoseStamped, self.pose_callback, queue_size=10)
        rospy.Subscriber('custom_number_topic', Int32, self.number_callback, queue_size=10)

    def pose_callback(self, msg):
        frame_id = msg.header.frame_id
        if frame_id not in self.pose_dict:
            self.pose_dict[frame_id] = [msg]
        else:
            self.pose_dict[frame_id].append(msg)

    def number_callback(self, msg):
        self.target_num = msg.data
        rospy.loginfo("Received number message: %d", msg.data)
        self.update_combinations()

    def update_combinations(self):
        labels_list = list(self.pose_dict.keys())
        # Convert each label to an integer
        labels_as_numbers = [int(label) for label in labels_list]
        shortest_combination, result = self.find_combinations(labels_as_numbers, self.target_num)
        rospy.loginfo("Shortest Combination: %s", shortest_combination)
        rospy.loginfo("Result Combination: %s", result)
        self.publish_matching_poses(shortest_combination)

    def publish_matching_poses(self, shortest_combination):
        if shortest_combination is not None:
            self.publish_timer = rospy.Timer(self.publish_delay, self.publish_next_pose)
            self.shortest_combination = shortest_combination
            self.pose_index = 0
        else:
            rospy.loginfo("\033[1;91mNO COMBINATION FOUND\033[0m")

    def publish_next_pose(self, event):
        if self.shortest_combination:
            if self.pose_index < len(self.shortest_combination):
                label = self.shortest_combination[self.pose_index]
                if str(label) in self.pose_dict:
                    poses = self.pose_dict[str(label)]
                    if self.pose_index < len(poses):
                        pose_msg = poses[self.pose_index]
                        rospy.loginfo("Publishing chosen pose with label %s", label)
                        self.publish_pose(pose_msg)
                        self.pose_index += 1
                    else:
                        self.publish_timer.shutdown()  # Stop the timer when all poses are published
            else:
                self.publish_timer.shutdown()  # Stop the timer when all poses are published
        else:
            rospy.loginfo("\033[1;91mNO COMBINATION FOUND\033[0m")

    def publish_pose(self, pose_msg):
        rospy.loginfo("Publishing pose: %s", pose_msg)
        self.pub.publish(pose_msg)

    def find_combinations(self, nums, target):
        def backtrack(start, target, path):
            nonlocal shortest_combination
            if target == 0:
                result.append(path)
                if len(path) < len(shortest_combination):
                    shortest_combination = path[:]
                return
            if target < 0 or start == len(nums):
                return
            for i in range(start, len(nums)):
                if i > start and nums[i] == nums[i - 1]:  # Skip duplicates
                    continue
                backtrack(i + 1, target - nums[i], path + [nums[i]])

        nums.sort()  # Sort the numbers to handle duplicates
        result = []
        shortest_combination = []
        backtrack(0, target, [])
        if result:
            shortest_combination = min(result, key=len)
            return shortest_combination, result
        else:
            return None, []

if __name__ == '__main__':
    rospy.init_node('pose_subscriber', anonymous=True)
    pose_subscriber = PoseSubscriber()
    rospy.spin()
