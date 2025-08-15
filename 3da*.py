import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import heapq

# Define a basic 3D A* class
class AStar3D:
    def __init__(self, start, goal, obstacles, grid_size=1.0):
        self.start = tuple(start)
        self.goal = tuple(goal)
        self.obstacles = obstacles
        self.grid_size = grid_size
        self.open_list = []
        self.closed_list = set()
        self.g_scores = {self.start: 0}
        self.parents = {}
        self.visited_nodes = []  # Track visited nodes for plotting

    def heuristic(self, node):
        # Euclidean distance to the goal
        return np.linalg.norm(np.array(node) - np.array(self.goal))

    def neighbors(self, node):
        # Define grid boundaries
        x_min, x_max = 0, 30
        y_min, y_max = 0, 30
        z_min, z_max = 0, 30

        # Possible moves in 3D (6 directions)
        moves = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
        neighbors = []
        for move in moves:
            # Calculate the neighbor's coordinates
            neighbor = tuple(np.array(node) + np.array(move) * self.grid_size)

            # Check if the neighbor is within the grid boundaries
            if (x_min <= neighbor[0] <= x_max and
                y_min <= neighbor[1] <= y_max and
                z_min <= neighbor[2] <= z_max and
                not self.is_colliding(neighbor)):
                neighbors.append(neighbor)
        
        return neighbors

    def is_colliding(self, point):
        # Check if a point is within any obstacle
        for (center, radius) in self.obstacles:
            if np.linalg.norm(np.array(point) - np.array(center)) <= radius:
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
        length = 0
        for i in range(1, len(path)):
            # Calculate Euclidean distance between consecutive points
            length += np.linalg.norm(np.array(path[i]) - np.array(path[i - 1]))
        return length

    def run(self):
        heapq.heappush(self.open_list, (self.heuristic(self.start), self.start))

        while self.open_list:
            _, current = heapq.heappop(self.open_list)

            if current == self.goal:
                return self.reconstruct_path()

            self.closed_list.add(current)
            self.visited_nodes.append(current)  # Track this node as visited

            for neighbor in self.neighbors(current):
                if neighbor in self.closed_list:
                    continue

                tentative_g_score = self.g_scores[current] + self.grid_size

                if neighbor not in self.g_scores or tentative_g_score < self.g_scores[neighbor]:
                    self.g_scores[neighbor] = tentative_g_score
                    f_score = tentative_g_score + self.heuristic(neighbor)
                    heapq.heappush(self.open_list, (f_score, neighbor))
                    self.parents[neighbor] = current

        return None  # No path found

# Plotting function with visited nodes
def plot_3d_path(start, goal, obstacles, path, visited_nodes):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot start and goal
    ax.scatter(*start, color='green', s=100, label='Start')
    ax.scatter(*goal, color='red', s=100, label='Goal')

    # Plot obstacles
    for (center, radius) in obstacles:
        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        x = radius * np.cos(u) * np.sin(v) + center[0]
        y = radius * np.sin(u) * np.sin(v) + center[1]
        z = radius * np.cos(v) + center[2]
        ax.plot_wireframe(x, y, z, color="gray", alpha=0.3)

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
start = (5, 5, 5)
goal = (30, 25, 5)

obstacles = [
    ((15, 15, 5), 5),
    ((20, 20, 5), 4),
    ((25, 18, 5), 3),
    ((10, 20, 8), 4),
    ((15, 10, 5), 3)
]


astar = AStar3D(start, goal, obstacles)
path = astar.run()

# Display the number of visited nodes and the path for target list
print("Number of visited nodes:", len(astar.visited_nodes))
print("Generated path for target nodes:", path)

# Calculate and print the path length
if path:
    path_length = astar.calculate_path_length(path)
    print("Length of the path:", path_length)

# Plot the path and visited nodes
plot_3d_path(start, goal, obstacles, path, astar.visited_nodes)
