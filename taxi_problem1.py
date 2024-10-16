import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from itertools import permutations
import heapq
from matplotlib.widgets import Button  # Import Button

# Load the CSV file (Adjacency Matrix)
file_path = 'adjacency_matrix_26x26.csv'
adj_matrix = pd.read_csv(file_path, delimiter=';', index_col=0)

# Replace empty cells with NaN and convert to numeric
adj_matrix.replace("", float('nan'), inplace=True)
adj_matrix = adj_matrix.apply(pd.to_numeric, errors='coerce')

# Create a graph
G = nx.Graph()

# Add nodes and edges based on the adjacency matrix
for i, row in adj_matrix.iterrows():
    G.add_node(i)  # Add the postal area as a node
    for j, distance in row.items():
        if pd.notna(distance) and distance > 0:  # Valid distance
            G.add_edge(i, j, weight=distance)  # Add an edge with the weight

# Print available nodes for debugging
print("Available nodes in the graph:", G.nodes)

# Store the user selections
selected_points = []
labels = ['start', 'finish']  # Labels for selection
current_label = 0  # To keep track of what the user is selecting
selection_finished = False  # Flag to indicate selection is finished

# Draw the graph using networkx
pos = nx.spring_layout(G)  # Positions for all nodes
fig, ax = plt.subplots(figsize=(12, 8))  # Set figure size
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, ax=ax)
plt.title('Select start, finish, and pickup/drop-off points')

# Dijkstra's algorithm to find the shortest path between two nodes
def dijkstra(graph, start, end):
    # Priority queue to store (distance, node) tuples
    queue = [(0, start)]
    
    # Dictionary to store the shortest distance from start to each node
    distances = {node: float('inf') for node in graph.nodes}
    distances[start] = 0

    # Dictionary to store the shortest path
    previous_nodes = {node: None for node in graph.nodes}
    
    while queue:
        # Get the node with the smallest distance
        current_distance, current_node = heapq.heappop(queue)
        
        # Early exit if we reached the destination
        if current_node == end:
            break

        # Skip nodes that have already been processed with a shorter distance
        if current_distance > distances[current_node]:
            continue
        
        # Explore neighbors
        for neighbor in graph.neighbors(current_node):
            try:
                weight = graph[current_node][neighbor]['weight']
            except KeyError:
                # If no weight, skip this edge
                continue
            
            distance = current_distance + weight
            
            # Only consider this new path if it's shorter
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(queue, (distance, neighbor))

    # Reconstruct the shortest path
    path = []
    current_node = end
    while current_node is not None:
        path.append(current_node)
        current_node = previous_nodes[current_node]
    
    # Reverse the path to get it in the correct order from start to end
    path.reverse()

    # If the path is valid, return it and the distance; otherwise, return infinity
    if distances[end] != float('inf'):
        return path, distances[end]
    else:
        return None, float('inf')  # No valid path found----------------

# Function to calculate the total distance of a given route
def calculate_route_distance(route):
    total_distance = 0
    for i in range(len(route) - 1):
        try:
            # Check if there's a direct edge between route[i] and route[i+1]
            total_distance += G[route[i]][route[i + 1]]['weight']
        except KeyError:
            # No direct edge; use Dijkstra's algorithm to find an alternative path
            print(f"No direct edge between '{route[i]}' and '{route[i+1]}'. Finding alternative route...")

            # Use our custom Dijkstra algorithm to find the shortest path between route[i] and route[i+1]
            alternative_path, alternative_distance = dijkstra(G, route[i], route[i + 1])
            
            if alternative_path is None:
                # No valid path found, return infinity (invalid route)
                print(f"No alternative path found between '{route[i]}' and '{route[i+1]}'.")
                return float('inf')
            
            # Add the distance of the alternative path to the total distance
            total_distance += alternative_distance
    
    return total_distance
# Function to handle click events
def on_click(event):
    global current_label, selection_finished
    if current_label >= len(labels) and selection_finished:  # If finished selecting, do nothing
        return

    # Find the closest node to the clicked point
    min_dist = float('inf')
    closest_node = None
    for node, (x, y) in pos.items():
        dist = (event.xdata - x)**2 + (event.ydata - y)**2  # Euclidean distance
        if dist < min_dist:
            min_dist = dist
            closest_node = node

    # Add the closest node to the selected points
    if closest_node is not None:
        if current_label < 2:  # For start and finish
            selected_points.append(closest_node)
            print(f'Selected {labels[current_label]} point: {closest_node}')
            current_label += 1
            # Highlight the selected node in green
            nx.draw_networkx_nodes(G, pos, nodelist=[closest_node], node_color='g', node_size=500)
            fig.canvas.draw()
        else:  # For pickup/drop-off points
            selected_points.append(closest_node)
            print(f'Selected pickup/drop-off point: {closest_node}')
            # Highlight the selected node in red
            nx.draw_networkx_nodes(G, pos, nodelist=[closest_node], node_color='r', node_size=500)
            fig.canvas.draw()

# Function to handle key press events
def on_key(event):
    if event.key == 'enter':
        global selection_finished
        selection_finished = True  # Set the flag to indicate selection is finished
        print("Selection finished.")
        programm()  # Call the programm function to compute and visualize the route

# Connect the key press event handler
fig.canvas.mpl_connect('key_press_event', on_key)

# Connect the click event handler
fig.canvas.mpl_connect('button_press_event', on_click)

# Show the plot and wait for user interaction
plt.show()

def programm():
    # Validate selected points
    if len(selected_points) < 2:
        print("Error: Please select at least a start and a finish point.")
    else:
        start_point = selected_points[0]
        end_point = selected_points[1]
        pickup_points = selected_points[2:]  # Remaining points are pickup/drop-off points
    
        print(f"Start point: {start_point}")
        print(f"End point: {end_point}")
        if pickup_points:
            print(f"Pickup/drop-off points: {pickup_points}")
        else:
            print("No pickup/drop-off points selected.")
    
        # Validate input points
        all_points = [start_point] + pickup_points + [end_point]
        for point in all_points:
            if point not in G.nodes:
                print(f"Warning: Node '{point}' is not in the graph.")
                exit(1)  # Exit if an invalid node is found

    
        # Finding the minimum route using permutations
        min_distance = float('inf')
        best_route = None
    
        for perm in permutations(pickup_points):
            current_route = [start_point] + list(perm) + [end_point]
            current_distance = calculate_route_distance(current_route)
            if current_distance < min_distance:
                min_distance = current_distance
                best_route = current_route

        print("Best route found:", " -> ".join(best_route))
        print("Minimum distance:", min_distance)

        # Clear the current axes and redraw the graph
        ax.clear()  # Clear the axes before redrawing

        # Redraw the graph with the new route
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, ax=ax)
        
        # Highlight the best route
        route_edges = [(best_route[i], best_route[i + 1]) for i in range(len(best_route) - 1)]
        nx.draw_networkx_edges(G, pos, edgelist=route_edges, width=5, edge_color='orange')

        # Draw edge labels (distances)
        edge_labels = nx.get_edge_attributes(G, 'weight')
        #nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        
        # Fixed positions in axes coordinates
        plt.text(0.8, -0.05, f"Best route found: {' -> '.join(best_route)}", 
                 horizontalalignment='center', verticalalignment='center', fontsize=12, 
                 bbox=dict(facecolor='lightgray', alpha=0.5), transform=ax.transAxes)
        
        plt.text(0.8, -0.1, f"Minimum distance: {min_distance}", 
                 horizontalalignment='center', verticalalignment='center', fontsize=12, 
                 bbox=dict(facecolor='lightgray', alpha=0.5), transform=ax.transAxes)

        plt.title('Optimal Route Visualization', fontsize=14,bbox=dict(facecolor='green', alpha=0.5))
        plt.draw()  # Update the plot with the new edges
