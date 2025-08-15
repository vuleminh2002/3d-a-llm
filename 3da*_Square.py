import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import heapq

class AStar3D:
    def __init__(self, start, goal, obstacles, grid_size=1.0, grid_limits=(0, 30)):
        self.start = tuple(start)
        self.goal = tuple(goal)
        self.obstacles = obstacles  # obstacles as list of ((center_x, center_y, center_z), half_x, half_y, half_z)
        self.grid_size = grid_size
        self.grid_limits = grid_limits  # Tuple for grid boundaries
        self.open_list = []
        self.closed_list = set()
        self.g_scores = {self.start: 0}
        self.parents = {}
        self.visited_nodes = []  # Track visited nodes for plotting

    def heuristic(self, node):
        # Euclidean distance to the goal
        return np.linalg.norm(np.array(node) - np.array(self.goal))

    def neighbors(self, node):
        # Possible moves in 3D (6 directions)
        moves = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
        neighbors = []
        for move in moves:
            neighbor = tuple(np.array(node) + np.array(move) * self.grid_size)

            # Check if the neighbor is within grid boundaries
            if (self.grid_limits[0] <= neighbor[0] <= self.grid_limits[1] and
                self.grid_limits[0] <= neighbor[1] <= self.grid_limits[1] and
                self.grid_limits[0] <= neighbor[2] <= self.grid_limits[1] and
                not self.is_colliding(neighbor)):
                neighbors.append(neighbor)
        return neighbors

    def is_colliding(self, point):
        # Check if a point is within any obstacle
        for (center, half_x, half_y, half_z) in self.obstacles:
            if (center[0] - half_x <= point[0] <= center[0] + half_x and
                center[1] - half_y <= point[1] <= center[1] + half_y and
                center[2] - half_z <= point[2] <= center[2] + half_z):
                return True
        return False

    def reconstruct_path(self):
        path = []
        node = self.goal
        while node in self.parents:
            path.append(node)
            node = self.parents[node]
        path.append(self.start)
        return path[::-1]

    def calculate_path_length(self, path):
        # Calculate the length of the path by summing Euclidean distances
        length = 0
        for i in range(1, len(path)):
            length += np.linalg.norm(np.array(path[i]) - np.array(path[i - 1]))
        return length

    def run(self):
        heapq.heappush(self.open_list, (self.heuristic(self.start), self.start))

        while self.open_list:
            _, current = heapq.heappop(self.open_list)

            if current == self.goal:
                return self.reconstruct_path()

            self.closed_list.add(current)
            self.visited_nodes.append(current)

            for neighbor in self.neighbors(current):
                if neighbor in self.closed_list:
                    continue

                tentative_g_score = self.g_scores[current] + self.grid_size

                if neighbor not in self.g_scores or tentative_g_score < self.g_scores[neighbor]:
                    self.g_scores[neighbor] = tentative_g_score
                    f_score = tentative_g_score + self.heuristic(neighbor)
                    heapq.heappush(self.open_list, (f_score, neighbor))
                    self.parents[neighbor] = current

        print("Path not found.")
        return None

def plot_3d_path(start, goal, obstacles, path, visited_nodes):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot start and goal
    ax.scatter(*start, color='green', s=100, label='Start')
    ax.scatter(*goal, color='red', s=100, label='Goal')

    # Plot cuboid obstacles as surfaces
    for (center, hx, hy, hz) in obstacles:
        x_range = [center[0] - hx, center[0] + hx]
        y_range = [center[1] - hy, center[1] + hy]
        z_range = [center[2] - hz, center[2] + hz]

        # Plot the six faces of each cuboid
        for x in x_range:
            Y, Z = np.meshgrid(y_range, z_range)
            ax.plot_surface(np.full_like(Y, x), Y, Z, color="gray", alpha=0.5)
        for y in y_range:
            X, Z = np.meshgrid(x_range, z_range)
            ax.plot_surface(X, np.full_like(X, y), Z, color="gray", alpha=0.5)
        for z in z_range:
            X, Y = np.meshgrid(x_range, y_range)
            ax.plot_surface(X, Y, np.full_like(X, z), color="gray", alpha=0.5)

    # Plot visited nodes
    if visited_nodes:
        visited_nodes = np.array(visited_nodes)
        ax.scatter(visited_nodes[:, 0], visited_nodes[:, 1], visited_nodes[:, 2], color='cyan', s=1, label='Visited Nodes')

    # Plot path
    if path:
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], path[:, 2], color='blue', linewidth=2, label='Path')

    # Set plot limits and labels
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 30)
    ax.set_zlim(0, 30)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    plt.show()

# Example usage
start = (3, 3, 3)
goal = (27, 24, 6)

# Define cuboid obstacles as (center, half_extent_x, half_extent_y, half_extent_z)
obstacles = [
    ((12, 12, 4), 3, 3, 3),
    ((18, 20, 5), 2, 4, 2),
    ((25, 15, 7), 3, 2, 2),
    ((8, 25, 6), 4, 3, 2),
    ((15, 10, 6), 2, 2, 3)
]

# Run the A* algorithm
astar = AStar3D(start, goal, obstacles)
path = astar.run()

# Display the number of visited nodes and path length if a path was found
if path:
    path_length = astar.calculate_path_length(path)
    print("Length of the path:", path_length)
    print("Number of visited nodes:", len(astar.visited_nodes))

# Plot the path and visited nodes
plot_3d_path(start, goal, obstacles, path, astar.visited_nodes)
