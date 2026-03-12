from robot import Robot
from sensors.lidar import LidarScan
import matplotlib.pyplot as plt
import numpy as np
import math

# -----------------------------------------------------
# Runtime modes & visualization toggles
# -----------------------------------------------------
MODE = "MANUAL"        # MANUAL | AUTO
SHOW_LIDAR = True
SHOW_ODOM = True

# -----------------------------------------------------
# Control commands (shared state)
# -----------------------------------------------------
v = 0.0   # linear velocity [m/s]
w = 0.0   # angular velocity [rad/s]

# -----------------------------------------------------
# HELPER 1: Bresenham's Line Algorithm
# -----------------------------------------------------
def bresenham_line(x0, y0, x1, y1):
    """Returns a list of grid cells that a laser ray passes through."""
    cells = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        cells.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return cells

# -----------------------------------------------------
# HELPER 2: Probabilistic Occupancy Grid Class
# -----------------------------------------------------
class OccupancyGridMap:
    def __init__(self, size=20.0, resolution=0.1):
        self.resolution = resolution
        self.cells = int(size / resolution)
        # Initialize grid with 0.0 (Log-odds of 0.5 probability -> Unknown)
        self.grid = np.zeros((self.cells, self.cells))
        
        # Log-odds Tuning Parameters
        self.L_OCC_BASE = 0.85
        self.L_FREE_BASE = -0.4
        self.MAX_LOG_ODDS = 10.0
        self.MIN_LOG_ODDS = -10.0
        self.WALL_PRESERVATION_THRESH = 2.0

    def get_index(self, x, y):
        """Converts physical coordinates (m) to grid indices."""
        c = int(x / self.resolution)
        r = int(y / self.resolution)
        return c, r 

    def is_valid(self, c, r):
        return 0 <= c < self.cells and 0 <= r < self.cells

    def update_map(self, robot_x, robot_y, robot_theta, angular_velocity, lidar_ranges, max_range=4.0):
        # MEASURE 1: Motion Gating
        if abs(angular_velocity) > 1.5:
            return # Skip mapping during fast rotations to prevent smearing
            
        r_c, r_r = self.get_index(robot_x, robot_y)
        
        for i, dist in enumerate(lidar_ranges):
            if dist >= max_range:
                continue
                
            # Calculate absolute ray angle (36 rays = 10 degrees each)
            ray_angle = robot_theta + math.radians(i * 10)
            
            # MEASURE 2: Distance-based Confidence
            confidence = max(0.0, 1.0 - (dist / max_range))
            l_occ_scaled = self.L_OCC_BASE * confidence
            l_free_scaled = self.L_FREE_BASE * confidence
            
            # Get hit coordinates
            hit_x = robot_x + dist * math.cos(ray_angle)
            hit_y = robot_y + dist * math.sin(ray_angle)
            h_c, h_r = self.get_index(hit_x, hit_y)
            
            # Update the Hit Cell (Obstacle)
            if self.is_valid(h_c, h_r):
                self.grid[h_r, h_c] += l_occ_scaled
                self.grid[h_r, h_c] = min(self.grid[h_r, h_c], self.MAX_LOG_ODDS)
                
            # Update the Free Space Cells along the ray path
            path_cells = bresenham_line(r_c, r_r, h_c, h_r)
            for (c, r) in path_cells:
                if c == h_c and r == h_r:
                    continue # Exclude the final hit cell
                
                if self.is_valid(c, r):
                    # MEASURE 3: Wall Preservation
                    if self.grid[r, c] > self.WALL_PRESERVATION_THRESH:
                        continue
                        
                    self.grid[r, c] += l_free_scaled
                    self.grid[r, c] = max(self.grid[r, c], self.MIN_LOG_ODDS)


def on_key(event):
    """Keyboard control & visualization toggles."""
    global v, w, MODE, SHOW_LIDAR, SHOW_ODOM

    if event.key == 'o':
        SHOW_ODOM = not SHOW_ODOM
        return
    if event.key == 'l':
        SHOW_LIDAR = not SHOW_LIDAR
        return
    if event.key == 'm':
        MODE = "MANUAL"
        v, w = 0.0, 0.0
        print("Switched to MANUAL mode")
        return
    if event.key == 'a':
        MODE = "AUTO"
        print("Switched to AUTO mode - Exploring Environment")
        return

    if MODE != "MANUAL":
        return

    if event.key == 'up':
        v += 1.5
    elif event.key == 'down':
        v -= 1.5
    elif event.key == 'left':
        w += 2.0
    elif event.key == 'right':
        w -= 2.0
    elif event.key == ' ':
        v, w = 0.0, 0.0

    v = max(min(v, 6.0), -6.0)
    w = max(min(w, 6.0), -6.0)

if __name__ == "__main__":

    lidar = LidarScan(max_range=4.0)
    robot = Robot()
    
    # Initialize our Log-Odds Map
    occ_map = OccupancyGridMap(size=20.0, resolution=0.1)

    plt.close('all')
    fig = plt.figure(num=2, figsize=(8,8))
    fig.canvas.manager.set_window_title("Autonomy Debug View - MAPPING ACTIVE")
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show(block=False)

    dt = 0.01 

    # -------------------------------------------------
    # Main simulation loop
    # -------------------------------------------------
    while plt.fignum_exists(fig.number):

        # Ground truth pose
        real_x, real_y, real_theta = robot.get_ground_truth()
        # Odometry estimate
        ideal_x, ideal_y, ideal_theta = robot.get_odometry()
        # LiDAR scan 
        lidar_ranges, lidar_points, lidar_rays, lidar_hits = lidar.get_scan((real_x, real_y, real_theta))

        if MODE == "AUTO":
            # ---------------------------------------------
            # UPGRADED AUTONOMOUS EXPLORATION LOGIC
            # ---------------------------------------------
            # 1. Widen the "front" vision to catch corners earlier
            front_dist = min(min(lidar_ranges[-4:]), min(lidar_ranges[:5]))
            left_dist = min(lidar_ranges[6:13])
            right_dist = min(lidar_ranges[23:30])

            # 2. Increase the stopping distance
            if front_dist < 1.2:
                v = 0.0
                
                # 3. Add "Hysteresis" to stop the flickering
                if left_dist > right_dist + 0.2:
                    w = 2.0   # Clear path left
                elif right_dist > left_dist + 0.2:
                    w = -2.0  # Clear path right
                else:
                    w = -2.0  # Default spin direction if trapped
            else:
                # 4. Coast is clear, drive forward
                v = 2.0
                w = 0.0

        # ---------------------------------------------
        # EXECUTE MAPPING LOGIC
        # ---------------------------------------------
        # We pass `w` (angular velocity) for motion gating!
        occ_map.update_map(ideal_x, ideal_y, ideal_theta, w, lidar_ranges)

        # ---------------------------------------------
        # Robot stepping
        # ---------------------------------------------
        robot.step(
            lidar_points,
            lidar_rays,
            lidar_hits,
            v,
            w,
            dt,
            show_lidar=SHOW_LIDAR,
            show_odom=SHOW_ODOM
        )

        # ---------------------------------------------
        # MAP VISUALIZATION OVERLAY
        # ---------------------------------------------
        ax = plt.gca()
        # Convert Log-Odds to Probabilities [0.0 to 1.0]
        probs = 1.0 - (1.0 / (1.0 + np.exp(occ_map.grid)))
        
        # Overlay the grid onto the existing visualization
        ax.imshow(probs, origin='lower', extent=[0, 20.0, 0, 20.0], 
                  cmap='binary', alpha=0.6, vmin=0, vmax=1, zorder=0)

        plt.pause(dt)