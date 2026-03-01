from robot import Robot
from sensors.lidar import LidarScan
import matplotlib.pyplot as plt
import math
import csv

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
# Navigation variables
# -----------------------------------------------------
path = []
target_index = 0
lookahead = 0.6
AVOIDING = False


def on_key(event):
    """Keyboard control & visualization toggles."""
    global v, w, MODE, SHOW_LIDAR, SHOW_ODOM

    # --- visualization toggles ---
    if event.key == 'o':
        SHOW_ODOM = not SHOW_ODOM
        print(f"Odometry visualization: {'ON' if SHOW_ODOM else 'OFF'}")
        return

    if event.key == 'l':
        SHOW_LIDAR = not SHOW_LIDAR
        print(f"LiDAR visualization: {'ON' if SHOW_LIDAR else 'OFF'}")
        return

    # --- mode switching ---
    if event.key == 'm':
        MODE = "MANUAL"
        v = 0.0
        w = 0.0
        print("Switched to MANUAL mode")
        return

    if event.key == 'a':
        MODE = "AUTO"
        print("Switched to AUTO mode")
        return

    # --- manual control ---
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
        v = 0.0
        w = 0.0

    v = max(min(v, 6.0), -6.0)
    w = max(min(w, 6.0), -6.0)


if __name__ == "__main__":

    lidar = LidarScan(max_range=4.0)
    robot = Robot()

    # -------------------------------------------------
    # Load path from CSV (skip header)
    # -------------------------------------------------
    with open("path.csv", "r") as f:
        reader = csv.reader(f)
        next(reader)

        for row in reader:
            if len(row) < 2:
                continue
            x = float(row[0].strip())
            y = float(row[1].strip())
            path.append((x, y))

    print("Loaded waypoints:", len(path))

    plt.close('all')
    fig = plt.figure(num=2)
    fig.canvas.manager.set_window_title("Autonomy Debug View")
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show(block=False)

    dt = 0.01 

    # -------------------------------------------------
    # Main simulation loop
    # -------------------------------------------------
    while plt.fignum_exists(fig.number):

        # ground truth pose
        real_x, real_y, real_theta = robot.get_ground_truth()

        # odometry estimate
        ideal_x, ideal_y, ideal_theta = robot.get_odometry()

        # LiDAR scan
        lidar_ranges, lidar_points, lidar_rays, lidar_hits = lidar.get_scan(
            (real_x, real_y, real_theta)
        )

        if MODE == "AUTO":

            x = ideal_x
            y = ideal_y
            theta = ideal_theta

            # -----------------------------------------
            # Obstacle detection (front sector)
            # -----------------------------------------
            front_sector = lidar_ranges[15:22]
            min_front = min(front_sector)

            if min_front < 0.6:
                AVOIDING = True

            # -----------------------------------------
            # Obstacle avoidance
            # -----------------------------------------
            if AVOIDING:

                left_clearance = min(lidar_ranges[20:30])
                right_clearance = min(lidar_ranges[6:16])

                v = 0.4

                if left_clearance > right_clearance:
                    w = 1.5
                else:
                    w = -1.5

                if min_front > 0.9:
                    AVOIDING = False

            # -----------------------------------------
            # Path following (Pure Pursuit style)
            # -----------------------------------------
            else:

                if target_index >= len(path):
                    v = 0.0
                    w = 0.0

                else:

                    tx, ty = path[target_index]

                    dx = tx - x
                    dy = ty - y

                    dist = math.sqrt(dx**2 + dy**2)

                    if dist < 0.5 and target_index < len(path)-1:
                        target_index += 1
                        tx, ty = path[target_index]

                    dx = tx - x
                    dy = ty - y

                    # transform to robot frame
                    local_x = math.cos(-theta)*dx - math.sin(-theta)*dy
                    local_y = math.sin(-theta)*dx + math.cos(-theta)*dy

                    curvature = (2 * local_y) / (lookahead**2)

                    v = 1.2
                    w = v * curvature

                    w = max(min(w, 2.0), -2.0)

        # ---------------------------------------------
        # don't edit below this line
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

        plt.pause(dt)