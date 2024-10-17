import heapq
import pandas as pd

# Load the CSV file (Adjacency Matrix)
file_path = 'adjacency_matrix_26x26.csv'
matrix = pd.read_csv(file_path, delimiter=';', index_col=0)

# Replace empty cells with NaN and convert to numeric
matrix.replace("", float('inf'), inplace=True)
matrix = matrix.apply(pd.to_numeric, errors='coerce')

# Replace NaN values with float('inf') for the algorithm
matrix.fillna(float('inf'), inplace=True)

# A* algorithm without coordinates (no heuristic)
def astar(matrix, start, end):
    n = len(matrix)  # Number of nodes
    open_set = []
    heapq.heappush(open_set, (0, start))
    
    came_from = {}  # To reconstruct the path
    g_score = [float('inf')] * n
    g_score[start] = 0

    f_score = [float('inf')] * n
    f_score[start] = 0  # No heuristic, so f_score is just g_score

    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current == end:
            return reconstruct_path(came_from, current)
        
        for neighbor in range(n):
            if matrix.iloc[current, neighbor] == float('inf'):
                continue  # No connection to the neighbor
            
            tentative_g_score = g_score[current] + matrix.iloc[current, neighbor]
            
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score  # No heuristic, so f_score = g_score
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    return None  # No path found

# Reconstruct the path from end to start
def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]  # Reverse the path

# Function to handle paths that go through multiple pickup points
def find_path_with_pickup_points(matrix, start, pickup_points, end):
    full_path = []
    current_start = start

    # Go through each pickup point in order
    for pickup_point in pickup_points:
        sub_path = astar(matrix, current_start, pickup_point)
        if sub_path is None:
            return None  # No path found for this sub-segment
        if full_path:
            # Remove the last node of the previous path to avoid duplication
            full_path.extend(sub_path[1:])
        else:
            full_path.extend(sub_path)
        current_start = pickup_point
    
    # Finally, go from the last pickup point to the end
    final_path = astar(matrix, current_start, end)
    if final_path is None:
        return None
    full_path.extend(final_path[1:])  # Append without duplicating the last pickup point

    return full_path

# Function to convert characters A-Z to indices 0-25
def char_to_index(char):
    return ord(char.upper()) - ord('A')

# Example inputs
start_char = 'E'  # Example starting character
pickup_chars = ['F']  # Example pickup characters
end_char = 'B'  # Example ending character

# Convert characters to indices
start = char_to_index(start_char)
pickup_points = [char_to_index(char) for char in pickup_chars]
end = char_to_index(end_char)

# Find the full path from start -> pickup_points -> end
path = find_path_with_pickup_points(matrix, start, pickup_points, end)

if path:
    print("Path found:", ' -> '.join([chr(65 + node) for node in path]))  # Convert numbers to letters (A, B, C...)
else:
    print("No path found")
