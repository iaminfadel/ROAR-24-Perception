import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import transforms3d

def quaternion_to_euler(q):
    euler = transforms3d.euler.quat2euler(q, axes='sxyz')  # Convert quaternion to Euler angles
    return np.degrees(euler)  # Convert radians to degrees


def euler_to_rotation_matrix(euler_angles):
    # Convert Euler angles to rotation matrix
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(euler_angles[0]), -np.sin(euler_angles[0])],
                    [0, np.sin(euler_angles[0]), np.cos(euler_angles[0])]])

    R_y = np.array([[np.cos(euler_angles[1]), 0, np.sin(euler_angles[1])],
                    [0, 1, 0],
                    [-np.sin(euler_angles[1]), 0, np.cos(euler_angles[1])]])

    R_z = np.array([[np.cos(euler_angles[2]), -np.sin(euler_angles[2]), 0],
                    [np.sin(euler_angles[2]), np.cos(euler_angles[2]), 0],
                    [0, 0, 1]])

    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

def plot_frame(ax, R, position, color, label):
    # Define axes vectors
    axes_vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # Rotate axes vectors
    rotated_axes_vectors = np.dot(R, axes_vectors.T).T

    # Plot rotated axes with position
    for i, vec in enumerate(rotated_axes_vectors):
        ax.quiver(position[0], position[1], position[2], vec[0], vec[1], vec[2],
                  color=color, label=label+' '+['X', 'Y', 'Z'][i])

# Define Euler angles for the reference frame at the origin
pose = [0.0, 0.5172373577372389, -0.16945681053486683, 0.45138664703815706]
euler = quaternion_to_euler(pose)
euler_angles_ref1 = euler  # Example Euler angles for reference frame at origin (degrees)

# Create rotation matrix for the reference frame at the origin
R_ref1 = euler_to_rotation_matrix(np.radians(euler_angles_ref1))

# Define the position for the other frame
position_ref2 = np.array([-0.25276260920933313, 0.41803046907697405, 1.2759651])  # Position of the other frame

# Define the transformation point as the origin of the calculated frame
position_calculated_frame = position_ref2

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot reference frame at the origin
plot_frame(ax, R_ref1, [0, 0, 0], 'b', 'Ref1')

# Plot reference frame at the specified position
plot_frame(ax, np.eye(3), position_calculated_frame, 'g', 'Calculated Frame')

# Plot transformation point
ax.scatter(position_ref2[0], position_ref2[1], position_ref2[2], color='k', label='Transformation Point')

ax.set_xlim([-1, 2])
ax.set_ylim([-1, 2])
ax.set_zlim([-1, 2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.legend()
plt.title('Coordinate Frames Visualization')
plt.show()