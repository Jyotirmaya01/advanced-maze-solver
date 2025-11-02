"""
Maze Solver for Circuit Safari Challenge
Implements A*, Dijkstra, and BFS algorithms
Author: Jyotirmaya
"""

import cv2
import numpy as np
from collections import deque
import heapq
import time
import json
import os

class MazeSolver:
    def __init__(self, image_path, algorithm='astar'):
        self.image_path = image_path
        self.algorithm = algorithm.lower()
        self.original_image = None
        self.maze = None
        self.height = 0
        self.width = 0
        self.start = None
        self.end = None
        self.metrics = {}
        
    def load_image(self):
        """Load the maze image and convert it to binary."""
        self.original_image = cv2.imread(self.image_path)
        if self.original_image is None:
            raise ValueError(f"Couldn't load image: {self.image_path}")
        
        # Convert to grayscale first
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        
        # Using adaptive threshold because it handles varying lighting better
        self.maze = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Clean up small noise with morphological operations
        kernel = np.ones((3,3), np.uint8)
        self.maze = cv2.morphologyEx(self.maze, cv2.MORPH_CLOSE, kernel)
        
        self.height, self.width = self.maze.shape
        self.metrics['maze_size'] = f"{self.width}x{self.height}"
        self.metrics['total_pixels'] = self.width * self.height
        
    def find_endpoints(self):
        """Find the start and end points by checking the edges."""
        endpoints = []
        
        # Check all four edges for openings
        # Top edge
        for x in range(self.width):
            if self.maze[0, x] == 255:
                endpoints.append((x, 0))
        
        # Bottom edge
        for x in range(self.width):
            if self.maze[self.height-1, x] == 255:
                endpoints.append((x, self.height-1))
        
        # Left edge
        for y in range(self.height):
            if self.maze[y, 0] == 255:
                endpoints.append((0, y))
        
        # Right edge
        for y in range(self.height):
            if self.maze[y, self.width-1] == 255:
                endpoints.append((self.width-1, y))
        
        if len(endpoints) < 2:
            raise ValueError("Couldn't find entry and exit points")
        
        # Usually the first opening is start and last is end
        self.start = endpoints[0]
        self.end = endpoints[-1]
        
        self.metrics['start_point'] = self.start
        self.metrics['end_point'] = self.end
        
    def manhattan_distance(self, a, b):
        """Calculate Manhattan distance - works well for grid-based movement."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def get_neighbors(self, pos):
        """Get valid neighboring cells (4-directional movement)."""
        x, y = pos
        neighbors = []
        
        # Check all four directions: right, left, down, up
        directions = [(1,0), (-1,0), (0,1), (0,-1)]
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            # Make sure we're within bounds and it's a valid path
            if (0 <= nx < self.width and 0 <= ny < self.height and 
                self.maze[ny, nx] == 255):
                neighbors.append((nx, ny))
        
        return neighbors
    
    def solve_astar(self):
        """A* algorithm implementation. Uses Manhattan distance heuristic."""
        start_time = time.time()
        
        # Priority queue stores (f_score, position)
        open_set = [(0, self.start)]
        came_from = {}
        
        # g_score is actual cost from start
        g_score = {self.start: 0}
        # f_score is g_score + heuristic
        f_score = {self.start: self.manhattan_distance(self.start, self.end)}
        
        nodes_explored = 0
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            nodes_explored += 1
            
            if current == self.end:
                path = self.reconstruct_path(came_from, current)
                self.metrics['algorithm'] = 'A*'
                self.metrics['nodes_explored'] = nodes_explored
                self.metrics['path_length'] = len(path)
                self.metrics['execution_time'] = f"{time.time() - start_time:.4f}s"
                return path
            
            for neighbor in self.get_neighbors(current):
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.manhattan_distance(neighbor, self.end)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        raise ValueError("No solution found")
    
    def solve_dijkstra(self):
        """Dijkstra's algorithm - guaranteed shortest path."""
        start_time = time.time()
        
        distances = {self.start: 0}
        pq = [(0, self.start)]
        came_from = {}
        nodes_explored = 0
        
        while pq:
            current_dist, current = heapq.heappop(pq)
            nodes_explored += 1
            
            if current == self.end:
                path = self.reconstruct_path(came_from, current)
                self.metrics['algorithm'] = 'Dijkstra'
                self.metrics['nodes_explored'] = nodes_explored
                self.metrics['path_length'] = len(path)
                self.metrics['execution_time'] = f"{time.time() - start_time:.4f}s"
                return path
            
            # Skip if we've found a better path already
            if current_dist > distances.get(current, float('inf')):
                continue
            
            for neighbor in self.get_neighbors(current):
                distance = current_dist + 1
                
                if distance < distances.get(neighbor, float('inf')):
                    distances[neighbor] = distance
                    came_from[neighbor] = current
                    heapq.heappush(pq, (distance, neighbor))
        
        raise ValueError("No solution found")
    
    def solve_bfs(self):
        """Basic BFS implementation."""
        start_time = time.time()
        
        queue = deque([(self.start, [self.start])])
        visited = {self.start}
        nodes_explored = 0
        
        while queue:
            current, path = queue.popleft()
            nodes_explored += 1
            
            if current == self.end:
                self.metrics['algorithm'] = 'BFS'
                self.metrics['nodes_explored'] = nodes_explored
                self.metrics['path_length'] = len(path)
                self.metrics['execution_time'] = f"{time.time() - start_time:.4f}s"
                return path
            
            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        raise ValueError("No solution found")
    
    def reconstruct_path(self, came_from, current):
        """Build the path by backtracking through came_from dictionary."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]
    
    def smooth_path(self, path):
        """
        Smooth the path using line-of-sight checks.
        Removes unnecessary waypoints to make the visualization cleaner.
        """
        if len(path) < 3:
            return path
        
        smoothed = [path[0]]
        i = 0
        
        while i < len(path) - 1:
            # Try to skip ahead as far as possible
            for j in range(len(path) - 1, i + 1, -1):
                if self.line_of_sight(path[i], path[j]):
                    smoothed.append(path[j])
                    i = j
                    break
            else:
                i += 1
                if i < len(path):
                    smoothed.append(path[i])
        
        return smoothed
    
    def line_of_sight(self, p1, p2):
        """Check if there's a clear straight path between two points."""
        x1, y1 = p1
        x2, y2 = p2
        
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        steps = max(dx, dy)
        
        if steps == 0:
            return True
        
        x_inc = (x2 - x1) / steps
        y_inc = (y2 - y1) / steps
        
        # Check each point along the line
        for i in range(steps + 1):
            x = int(x1 + i * x_inc)
            y = int(y1 + i * y_inc)
            if not (0 <= x < self.width and 0 <= y < self.height):
                return False
            if self.maze[y, x] != 255:
                return False
        
        return True
    
    def create_heatmap(self, path):
        """Create a heat map showing exploration order."""
        heatmap = np.zeros((self.height, self.width), dtype=np.float32)
        
        for i, (x, y) in enumerate(path):
            intensity = (i / len(path)) * 255
            cv2.circle(heatmap, (x, y), 3, intensity, -1)
        
        heatmap = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)
        return cv2.addWeighted(self.original_image, 0.7, heatmap, 0.3, 0)
    
    def draw_solution(self, path, output_path):
        """Draw the solution on the maze."""
        result = self.original_image.copy()
        
        # Smooth the path for better visualization
        smoothed_path = self.smooth_path(path)
        
        # Draw the smoothed path with a gradient effect
        for i in range(len(smoothed_path) - 1):
            progress = i / len(smoothed_path)
            # Gradient from yellow to red
            color = (0, int(255 * (1 - progress)), int(255 * progress))
            cv2.line(result, smoothed_path[i], smoothed_path[i+1], color, 3)
        
        # Also draw the actual path in thin red for reference
        for i in range(len(path) - 1):
            cv2.line(result, path[i], path[i+1], (0, 0, 255), 1)
        
        # Mark start point (green)
        cv2.circle(result, self.start, 8, (0, 255, 0), -1)
        cv2.circle(result, self.start, 10, (0, 255, 0), 2)
        
        # Mark end point (blue)
        cv2.circle(result, self.end, 8, (255, 0, 0), -1)
        cv2.circle(result, self.end, 10, (255, 0, 0), 2)
        
        # Add some basic info to the image
        self.add_info_text(result)
        
        # Save the main result
        cv2.imwrite(output_path, result)
        
        # Save heat map version
        heatmap_path = output_path.replace('.', '_heatmap.')
        heatmap = self.create_heatmap(path)
        cv2.imwrite(heatmap_path, heatmap)
        
        return result
    
    def add_info_text(self, image):
        """Add performance metrics to the image."""
        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        info_lines = [
            f"Algorithm: {self.metrics.get('algorithm', 'N/A')}",
            f"Path Length: {self.metrics.get('path_length', 0)} pixels",
            f"Time: {self.metrics.get('execution_time', 'N/A')}",
        ]
        
        for i, text in enumerate(info_lines):
            # Add black background for readability
            (w, h), _ = cv2.getTextSize(text, font, 0.5, 1)
            cv2.rectangle(image, (5, y_offset + i*25 - 18), 
                         (15 + w, y_offset + i*25 + 5), (0, 0, 0), -1)
            cv2.putText(image, text, (10, y_offset + i*25), 
                       font, 0.5, (0, 255, 255), 1)
    
    def solve(self, output_path=None):
        """Main method to solve the maze."""
        print("="*60)
        print("Maze Solver - Circuit Safari Challenge")
        print("="*60)
        
        # Step 1: Load and preprocess
        print("\nLoading image...")
        self.load_image()
        print(f"Maze size: {self.metrics['maze_size']}")
        
        # Step 2: Find entry and exit
        print("\nFinding entry and exit points...")
        self.find_endpoints()
        print(f"Start: {self.start}")
        print(f"End: {self.end}")
        
        # Step 3: Solve using selected algorithm
        print(f"\nSolving with {self.algorithm.upper()}...")
        
        if self.algorithm == 'astar':
            path = self.solve_astar()
        elif self.algorithm == 'dijkstra':
            path = self.solve_dijkstra()
        else:
            path = self.solve_bfs()
        
        print(f"Solution found!")
        print(f"Nodes explored: {self.metrics['nodes_explored']}")
        print(f"Path length: {self.metrics['path_length']} pixels")
        print(f"Time taken: {self.metrics['execution_time']}")
        
        # Step 4: Create output
        if output_path is None:
            base = os.path.splitext(os.path.basename(self.image_path))[0]
            output_path = f"{base}_solved_{self.algorithm}.png"
        
        print(f"\nGenerating output...")
        self.draw_solution(path, output_path)
        print(f"Saved to: {output_path}")
        print(f"Heat map: {output_path.replace('.', '_heatmap.')}")
        
        # Save metrics to JSON
        metrics_path = output_path.replace('.png', '_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Metrics: {metrics_path}")
        
        print("\n" + "="*60)
        print("Done!")
        print("="*60)
        
        return output_path


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Maze Solver - Circuit Safari Challenge")
        print("\nUsage:")
        print("  python maze_solver.py <image_path> [algorithm] [output_path]")
        print("\nAlgorithms:")
        print("  astar    - A* (default, usually fastest)")
        print("  dijkstra - Dijkstra's algorithm")
        print("  bfs      - Breadth-First Search")
        print("\nExample:")
        print("  python maze_solver.py maze.png")
        print("  python maze_solver.py maze.png astar output.png")
        sys.exit(1)
    
    image_path = sys.argv[1]
    algorithm = sys.argv[2] if len(sys.argv) > 2 else 'astar'
    output_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    try:
        solver = MazeSolver(image_path, algorithm)
        solver.solve(output_path)
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
