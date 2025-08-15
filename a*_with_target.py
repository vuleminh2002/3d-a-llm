import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import heapq
from collections import deque

class AStarWithTargets:
    def __init__(self, start, goal, obstacles, target_nodes, grid_size=1.0, target_threshold=1.5):
        self.start = tuple(start)
        self.goal = tuple(goal)
        self.obstacles = obstacles
        self.target_nodes = deque(target_nodes)  # Use deque to sequentially pop targets
        self.grid_size = grid_size
        self.target_threshold = target_threshold  # Distance to consider a neighbor "close enough" to target
        self.open_list = []
        self.closed_list = set()
        self.g_scores = {self.start: 0}
        self.parents = {}
        self.visited_nodes = []  # Track visited nodes for plotting

    def heuristic(self, node, target):
        # Euclidean distance to the current target node or goal
        return np.linalg.norm(np.array(node) - np.array(target))

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
        # Calculate the length of the path by summing Euclidean distances
        length = 0
        for i in range(1, len(path)):
            length += np.linalg.norm(np.array(path[i]) - np.array(path[i - 1]))
        return length

    def _update_target(self):
        """Update the current target in the path."""
        self.target_nodes.popleft()
        if self.target_nodes:
            return self.target_nodes[0]
        return self.goal  # Set goal as target when targets are exhausted

    def _reorder_open_list(self):
        """Re-sort the open list based on the current target."""
        self.open_list = [(self.f_value(node, self.current_target), node) for _, node in self.open_list]
        heapq.heapify(self.open_list)

    def f_value(self, node, target):
        # Calculate f-cost: g + h_target + h_goal
        h_goal = self.heuristic(node, self.goal)
        h_target = self.heuristic(node, target)
        return self.g_scores[node] + h_goal + h_target

    def run(self):
        # Start with the first target node or the goal
        self.current_target = self.target_nodes[0] if self.target_nodes else self.goal
        heapq.heappush(self.open_list, (self.heuristic(self.start, self.current_target), self.start))
        counter = 0
        while self.open_list:
            # Get the node with the lowest f-cost
            _, current = heapq.heappop(self.open_list)

            # Check if we have reached the goal
            if current == self.goal:
                print(counter)
                return self.reconstruct_path()

            # Add the current node to the closed list
            self.closed_list.add(current)
            self.visited_nodes.append(current)

            # Process neighbors of the current node
            for neighbor in self.neighbors(current):
                if neighbor in self.closed_list:
                    continue  # Skip already-processed nodes
                # Update target if neighbor is close enough to the current target
                if neighbor == self.current_target and self.current_target != self.goal:
                    self.current_target = self._update_target()
                    self._reorder_open_list()

                tentative_g_score = self.g_scores[current] + self.grid_size

                # Only add neighbor if a better path is found
                if neighbor not in self.g_scores or tentative_g_score < self.g_scores[neighbor]:
                    self.g_scores[neighbor] = tentative_g_score
                    heapq.heappush(self.open_list, (self.f_value(neighbor, self.current_target), neighbor))
                    self.parents[neighbor] = current

        print("Path not found.")
        return None

# Plotting function with visited nodes
def plot_3d_path(start, goal, obstacles, path, visited_nodes, target_nodes):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot start, goal, and target nodes
    ax.scatter(*start, color='green', s=100, label='Start')
    ax.scatter(*goal, color='red', s=100, label='Goal')
    for target in target_nodes:
        ax.scatter(*target, color='orange', s=10, label='Target Node' if target == target_nodes[0] else "")

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

# Updated Target Nodes Outside Collision Radii
target_nodes = [
    (10, 5, 5),    # Initial straight move
    (15, 6, 5),    # Bypassing the obstacle at (15, 15, 5) from below
    (20, 10, 5),   # Approaching the second obstacle, staying below it
    (24, 14, 5),   # Clear path near (20, 20, 5), avoiding collision
    (26, 15, 5),   # Close to third obstacle but avoiding it
    (29, 18, 5),   # Final approach towards the goal, clear of all obstacles
    (30, 25, 5)    # Goal
]





   # Final approach towards the goal’s coordinates
   
     


# Run the optimized A* algorithm with target nodes
astar_optimized = AStarWithTargets(start, goal, obstacles, target_nodes, grid_size=1.0)
optimized_path = astar_optimized.run()

# Display the number of visited nodes
print("Number of visited nodes:", len(astar_optimized.visited_nodes))

# Calculate and print the path length if a path was found
if optimized_path:
    path_length = astar_optimized.calculate_path_length(optimized_path)
    print("Length of the path:", path_length)

# Plot the optimized path and visited nodes
plot_3d_path(start, goal, obstacles, optimized_path, astar_optimized.visited_nodes, target_nodes)
