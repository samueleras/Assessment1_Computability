import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import heapq

# Dijkstra's Algorithm for shortest path from point A to B without selected pickup points in between
def find_shortest_path_dijkstras(matrix, start, end):
    n = len(matrix)  # Number of nodes
    open_set = []
    heapq.heappush(open_set, (0, start))  # Use g_score as priority
    
    came_from = {}  # To reconstruct the path
    g_score = [float('inf')] * n
    g_score[start] = 0

    while open_set:
        current_g_score, current = heapq.heappop(open_set)

        # If we reach the end node, reconstruct and return the path
        if current == end:
            return reconstruct_path(came_from, current)

        for neighbor in range(n):
            weight = matrix.iloc[current, neighbor]
            # Continue if there's no connection (handle NaN or inf)
            if pd.isna(weight) or weight == float('inf'):
                continue  # Skip neighbors with no connection
            
            tentative_g_score = g_score[current] + weight
            
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                heapq.heappush(open_set, (tentative_g_score, neighbor))
    
    return None  # No path found

# Helper function to reconstruct the path
def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


# Travelling Salesman Problem. Shortest path from A to B while traversing preselected nodes
def find_shortest_path_with_pickup_points(matrix, start, pickup_points, end):
    """     print('find_path: ' , matrix, start, pickup_points, end)
    full_path = []
    current_start = start

    # Go through each pickup point in order
    for pickup_point in pickup_points:
        sub_path = dijkstra(matrix, current_start, pickup_point)
        if sub_path is None:
            pass
            #return None  # No path found for this sub-segment
        if full_path:
            # Remove the last node of the previous path to avoid duplication
            full_path.extend(sub_path[1:])
        else:
            full_path.extend(sub_path)
        current_start = pickup_point
    
    # Finally, go from the last pickup point to the end
    final_path = dijkstra(matrix, current_start, end)
    if final_path is None:
        return None
    full_path.extend(final_path[1:])  # Append without duplicating the last pickup point

    return full_path """
    return None

# Function to convert characters A-Z to indices 0-25
def char_to_index(char):
    return ord(char.upper()) - ord('A')

# Function to convert indices 0-25 to characters A-Z
def index_to_char(index_list):
    return [chr(index + ord('A')) for index in index_list]

def get_selected_points(selected_points):
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
                
        # Convert characters to indices
        start_point = char_to_index(start_point)
        pickup_points = [char_to_index(char) for char in pickup_points]
        end_point = char_to_index(end_point)
        
        return start_point, end_point, pickup_points

def programm():
        start_index, end_index, pickup_indices = get_selected_points(selected_points)
        
        # Decide whether to search for A->B or via several pickup points
        if pickup_indices:
            #best_route = find_shortest_path_with_pickup_points(matrix, start_index, pickup_indices, end_index)
            print("Traversing multiple pickup points not supported yet.")
            return  #has to be removed when feature is implemented
        else: 
            best_route = find_shortest_path_dijkstras(matrix, start_index, end_index)

        print(best_route)
        best_route = index_to_char(best_route)
        
        print("Best route found:", " -> ".join(best_route))
        #print("Minimum distance:", min_distance)

        # Clear the current axes and redraw the graph
        ax.clear()  # Clear the axes before redrawing

        # Redraw the graph with the new route
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, ax=ax)
        
        # Highlight the best route
        route_edges = [(best_route[i], best_route[i + 1]) for i in range(len(best_route) - 1)]
        nx.draw_networkx_edges(G, pos, edgelist=route_edges, width=5, edge_color='orange')

        # Draw edge labels (distances)
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        
        # Fixed positions in axes coordinates
        plt.text(0.8, -0.05, f"Best route found: {' -> '.join(best_route)}", 
                 horizontalalignment='center', verticalalignment='center', fontsize=12, 
                 bbox=dict(facecolor='lightgray', alpha=0.5), transform=ax.transAxes)

        plt.title('Optimal Route Visualization', fontsize=14,bbox=dict(facecolor='green', alpha=0.5))
        plt.draw()  # Update the plot with the new edges


def initialize_variables():
    global selected_points, labels, current_label, selection_finished
    selected_points = []
    labels= ['start', 'finish']  # Labels for selection
    current_label = 0  # To keep track of what the user is selecting
    selection_finished = False  # Flag to indicate selection is finished
    print("Selection reset.")

def reset_plot():
    # Clear the current axes and redraw the graph
    ax.clear()  # Clear the axes before redrawing
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, ax=ax)
    # Draw edge labels (distances)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.title('Select start, finish, and pickup/drop-off points')
    plt.draw()



### Event Hanlding ###

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
    elif event.key == 'backspace':
        initialize_variables() #Reset current selections
        reset_plot()



### Program initiation ###

import os
#Define path to csv
current_folder_path = os.path.dirname(os.path.abspath(__file__))
file_path = current_folder_path + '\\adjacency_matrix_26x26.csv'

# Load the CSV file (Adjacency Matrix)
adj_matrix = pd.read_csv(file_path, delimiter=';', index_col=0)
# Replace empty cells with NaN and convert to numeric
adj_matrix.replace("", float('nan'), inplace=True)
adj_matrix = adj_matrix.apply(pd.to_numeric, errors='coerce')

# Load the CSV file (Adjacency Matrix for the Algorithm)
matrix = pd.read_csv(file_path, delimiter=';', index_col=0)
# Replace empty cells with NaN and convert to numeric
matrix.replace("", float('inf'), inplace=True)
matrix = matrix.apply(pd.to_numeric, errors='coerce')
# Replace NaN values with float('inf') for the algorithm
matrix.fillna(float('inf'), inplace=True)

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

#Initialize variables
initialize_variables()

# Draw the graph using networkx
pos = nx.spring_layout(G)  # Positions for all nodes
fig, ax = plt.subplots(figsize=(12, 8))  # Set figure size
fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('key_press_event', on_key)
reset_plot()
plt.show()