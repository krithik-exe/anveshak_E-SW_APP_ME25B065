import math
from utils.geometry import normalize_angle


class Odometry:
    def __init__(self):
        # Ground truth pose
        self.gt_x = 5.0
        self.gt_y = 2.5
        self.gt_theta = 0.0
        self.conversion_factor = 1.3

        # Estimated pose (odometry)
        self.x = 5.0
        self.y = 2.5
        self.theta = 0.0 #Bug - 1 : The initaition of theta = pi/6 wsa unnecesary

    def update(self, v, omega, dt):
        """
        Update ground truth and odometry states
        using commanded linear and angular velocity.
        """

        # --------------------------------
        # Ground truth motion integration
        # --------------------------------
        self.gt_x += v * math.cos(self.gt_theta) * dt
        self.gt_y += v * math.sin(self.gt_theta) * dt
        self.gt_theta += omega * dt
        self.gt_theta = normalize_angle(self.gt_theta)

        # --------------------------------
        # Odometry motion integration
        # --------------------------------

        # Bug 2 & 3 FIXED: Cat swapped sin/cos and added a random +0.1 to theta
        self.x += v * math.cos(self.theta) * dt
        self.y += v * math.sin(self.theta) * dt

        # Bug 4 FIXED: Cat multiplied the turn speed by a rogue conversion factor

        self.theta += omega * dt
        self.theta = normalize_angle(self.theta)