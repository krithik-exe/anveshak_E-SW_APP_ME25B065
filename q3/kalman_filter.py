############################
# TEAM ANVESHAK APPLICATION 2026
# QUESTION 3
# This is the implementation of the kalman filter based on the CSV files given
# Go through this code carefully and add your code only where mentioned
# DO NOT ALTER THE REST OF THE CODE
############################

import matplotlib.pyplot as plt
import numpy as np
import csv
import os
from math import sqrt

class KalmanFilter:
    def __init__(self, dt=0.1):
        # State vector 
        self.x_hat = np.zeros((4, 1))
    
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [1, 0, 0, 0],
                           [0, 1, 0, 0]])
        
        self.P = np.eye(4) * 1000 
        
        self.Q = np.eye(4) * 0.1
        
    ################ YOUR CODE STARTS HERE ###################
    # Make the covariance matrix for the two sensors  
        self.R = np.eye(4) * 1.0


    # Go through the kalman filter algorithm and translate it to code
    def predict(self):
        self.x_hat = self.F @ self.x_hat
        self.P = self.F @ self.P @ self.F.T + self.Q


    def update(self, z):
        y = z - (self.H @ self.x_hat)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x_hat = self.x_hat + (K @ y)
        I = np.eye(self.P.shape[0])
        term1 = I - (K @ self.H)
        self.P = (term1 @ self.P @ term1.T) + (K @ self.R @ K.T)

    ################ YOUR CODE ENDS HERE ####################


########################## VISUALIZATION ##########################
def extract_odom_from_csv(odom_csv_file):
    odom_data = []
    with open(odom_csv_file, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            odom_x = float(row[1])
            odom_y = float(row[2])
            odom = (odom_x, odom_y)
            odom_data.append(odom)
        
    return odom_data
    

def setup_plot(ax, data, title, color):
    if data:
        x_vals, y_vals = zip(*data)
        ax.plot(x_vals, y_vals, label='Trajectory', linestyle='-', marker='.', markersize=2, alpha=0.6, color=color)
    
    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True)
    ax.legend()


def visualize(ground_truth, odom_data_1, odom_data_2, filtered_odom):
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))

    # Subplot (0, 0): Ground Truth
    setup_plot(axs[0, 0], ground_truth, "Ground Truth", 'green')

    # Subplot (0, 1): Odom Data 1 (Sensor 1)
    setup_plot(axs[0, 1], odom_data_1, "Noisy Sensor 1", 'red')

    # Subplot (1, 0): Odom Data 2 (Sensor 2)
    setup_plot(axs[1, 0], odom_data_2, "Noisy Sensor 2", 'blue')

    # Subplot (1, 1): Ground Truth with your filtered output
    setup_plot(axs[1, 1], ground_truth, "Filtered vs Ground Truth", 'green')
    setup_plot(axs[1, 1], filtered_odom, "Filtered vs Ground Truth", 'violet')

    plt.tight_layout()
    plt.show()

########################## VISUALIZATION ########################## 

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Build the path to the data folder safely
    # This assumes the 'data' folder is in the same folder as this script
    data_folder = os.path.join(current_dir, "data")

    # Construct the full paths
    ground_truth_path = os.path.join(data_folder, "odom.csv")
    sensor1_path = os.path.join(data_folder, "sensor1_noisy.csv")
    sensor2_path = os.path.join(data_folder, "sensor2_noisy.csv")

    ground_truth = extract_odom_from_csv(ground_truth_path)
    odom_data_1 = extract_odom_from_csv(sensor1_path)
    odom_data_2 = extract_odom_from_csv(sensor2_path)

    # Think why these have been defined
    kf = KalmanFilter()
    filtered_odom = []

    ####################### YOUR CODE GOES HERE #######################
    # So now you have to use the functions that you defined and use them to get the filtered odom
    # The for loop is already provided, so that you can begin with the kalman filter 
    # Think how you will use the functions and in what order to get the filtered odometry
    # Filtered odometry should be in the form of a list of tuples

    for i in range(len(ground_truth)):
        # 1. Fetch data from both sensors for the current time step
        z1_x, z1_y = odom_data_1[i]
        z2_x, z2_y = odom_data_2[i]
        
        # 2. Formulate the measurement vector z as a 4x1 column matrix
        z = np.array([[z1_x], 
                      [z1_y], 
                      [z2_x], 
                      [z2_y]])
        
        # 3. Predict the next state
        kf.predict()
        
        # 4. Update the state with the current dual-sensor measurement
        kf.update(z)
        
        # 5. Extract the estimated position (x, y) from the state vector x_hat
        # x_hat[0, 0] is X position, x_hat[1, 0] is Y position
        filtered_x = float(kf.x_hat[0, 0])
        filtered_y = float(kf.x_hat[1, 0])
        
        # 6. Append to our results list as a tuple
        filtered_odom.append((filtered_x, filtered_y))


    ######################## YOUR CODE ENDS HERE ######################
    
    ########### VISUALIZATION ###############
    if len(filtered_odom) != 0:
        errors = np.array(filtered_odom) - np.array(ground_truth)
        squared_errors = errors ** 2
        rsme = sqrt(np.mean(squared_errors))
        print(f"Root Square Mean Error: {rsme}")

    visualize(ground_truth, odom_data_1, odom_data_2, filtered_odom)


if __name__ == "__main__":
    main()