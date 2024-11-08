import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import heapq
from tkinter import Tk, Button, Label, Frame, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import shutil
import numpy as np

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


# Crisofodis Algorithm break down: 
# 1. Construct MST: we done it with prims algorithm
# 2. Identify ODD-Degree Nodes
# 3. Construct a Minimum Weight perfect matching: done with greedy approach, should be changed into Hungarian
# 4. Combine MST and Matching
# 5. Find Eulerian Curcuit
# 6. Convert Eulerian Curcuit to Hamiltonian Circuit
def prims_algorithm(adj_matrix, selected_points):
    # Extract the submatrix for the selected points
    matrix_selected_points = adj_matrix.loc[selected_points, selected_points]

    mst_edges = []  # List to store the edges of the MST
    total_cost = 0  # To keep track of the total cost of the MST

    # Start from the first vertex in the selected points
    start_vertex = selected_points[0]
    min_heap = [(0, None, start_vertex)]  # weight, previous node, current node
    visited = set()  # To track visited vertices

    while min_heap:
        # Get the edge with the minimum weight
        weight, prev_node, node = heapq.heappop(min_heap)

        if node in visited:
            continue  # Skip if already visited

        visited.add(node)  # Mark vertex as visited
        total_cost += weight  # Update the total cost

        if prev_node is not None:
            # Append the edge only if it's not the starting node
            mst_edges.append((prev_node, node, weight))

        # Add all edges from the current vertex to the priority queue
        for othernode in selected_points:
            if othernode not in visited:
                edge_weight = matrix_selected_points.loc[node, othernode]  # Access using string labels
                if pd.notna(edge_weight) and edge_weight > 0:  # Check if the edge is valid
                    heapq.heappush(min_heap, (edge_weight, node, othernode))  # Push the new edge into the heap

    return mst_edges, total_cost

def find_odd_degree_vertices(mst_edges, selected_points):

    # Initialize degree counter for each vertex using a dictionary
    vertex_degree = {vertex: 0 for vertex in selected_points}

    # Iterate through all edges in the MST and count the degree of each vertex
    for u, v, weight in mst_edges:
        # Increase the degree for both vertices connected by the edge
        vertex_degree[u] += 1
        vertex_degree[v] += 1
    
    # Identify vertices with odd degrees
    odd_degree_vertices = [vertex for vertex, degree in vertex_degree.items() if degree % 2 == 1]
    
    return odd_degree_vertices

# use of greedy algorithm
# hungarian algorithm would be better i think or modified dijerkas
# but for now fine
# TODO
def minimum_cost_perfect_matching(matrix, odd_vertices):
    # List to store the matched pairs
    matching = []
    
    # Sort all pairs of odd vertices by their distance (cost in matrix)
    edges = []
    for i in range(len(odd_vertices)):
        for j in range(i + 1, len(odd_vertices)):
            u, v = odd_vertices[i], odd_vertices[j]
            cost = matrix.loc[u,v]
            edges.append((cost, u, v))
    
    # Sort edges based on cost in ascending order
    edges.sort()
    
    # Set to keep track of matched vertices
    matched = set()
    
    # Greedily add edges to the matching set
    for cost, u, v in edges:
        # Only add edge if both vertices are not already matched
        if u not in matched and v not in matched:
            matching.append((u, v))
            matched.add(u)
            matched.add(v)
    
    return matching

# Williamson, D. P., & Shmoys, D. B. (2011). The Design of Approximation Algorithms. Cambridge University Press.
def build_euler_tour(matrix, start):
    pass
    #TODO

# Travelling Salesman Problem. Shortest path from A to B while traversing preselected nodes
def find_circular_route(matrix, selected_points):

    #Build MST with prims algorithm
    mst_edges, total_cost = prims_algorithm(matrix, selected_points)
    #print("MST edges:", mst_edges)
    print("Edges in the Minimum Spanning Tree:")
    for u, v, weight in mst_edges:
        print(f"{u} -- {v} (Weight: {weight})")
    print(f"Total cost of the Minimum Spanning Tree: {total_cost}")
    # Draw the MST into the graph
    draw_route_into_graph(mst_edges, 'green', f"Minimum spanning tree\n Total cost: {total_cost}" )

    # Odd Degree vertices of MST
    odd_vertices = find_odd_degree_vertices(mst_edges, selected_points)
    print("Vertices with odd degrees:", odd_vertices)

    # Find minimum-cost perfect matching for odd vertices
    matching = minimum_cost_perfect_matching(matrix, odd_vertices)
    print("Minimum-cost perfect matching:", matching)

    #TODO
    # Find the euler tour
    # Euler tour is a path that visits every edge of a graph exactly once and returns to the starting vertex
    start_index = char_to_index(selected_points[0])
    euler_tour = build_euler_tour(matrix, start_index)
    print("Euler tour: ", euler_tour)

    return None



# Function to convert characters A-Z to indices 0-25
def char_to_index(char):
    return ord(char.upper()) - ord('A')

# Function to convert indices 0-25 to characters A-Z
def index_to_char(index_list):
    if type(index_list) == int:
        return chr(index_list + ord('A'))
    else:
        return [chr(index + ord('A')) for index in index_list]

def get_selected_points(selected_points):
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

def calculate_path():
        if not selected_points: 
            print("Error: Please select at least a start and a finish point.")
            return
        start_index, end_index, pickup_indices = get_selected_points(selected_points)
        
        # Decide whether to search for A->B or via several pickup points
        if pickup_indices:
            #best_route = 
            find_circular_route(adj_matrix_algo, selected_points)
            print("Traversing multiple pickup points not supported yet.")
            return  #has to be removed when feature is implemented
        else: 
            best_route = find_shortest_path_dijkstras(adj_matrix_algo, start_index, end_index)

        best_route = index_to_char(best_route)
        print("Best route found:", " -> ".join(best_route))
        #print("Minimum distance:", min_distance)

        draw_route_into_graph(best_route,'red', f"Shortest Path from {index_to_char(start_index)} to {index_to_char(end_index)}")


def initialize_variables():
    global selected_points, labels, current_label, selection_finished
    selected_points = []
    labels= ['start', 'finish']  # Labels for selection
    current_label = 0  # To keep track of what the user is selecting
    selection_finished = False  # Flag to indicate selection is finished

def reset_plot():
    initialize_variables()
    # Clear the current axes and redraw the graph
    ax.clear()  # Clear the axes before redrawing
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, ax=ax)
    # Draw edge labels (distances)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.title('Select start, finish, and pickup/drop-off points')
    plt.draw()

def upload_csv():
    # Show file dialog to select the CSV file
    selected_file = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    # If no file is selected, return
    if not selected_file:
        return None
    # Overwrite currently used matrix
    shutil.copy2(selected_file, file_path)
    print(f"Saved file.")
    initialize_variables()  #Resets all graph related variables
    read_csv()
    create_graph()
    plot_graph()

def read_csv():
    global adj_matrix_graph, adj_matrix_algo
    # Load the CSV file (Adjacency Matrix)
    adj_matrix_graph = pd.read_csv(file_path, delimiter=';', index_col=0)

    # Replace empty cells with NaN and convert to numeric
    adj_matrix_graph.replace("", float('nan'), inplace=True)
    adj_matrix_graph = adj_matrix_graph.apply(pd.to_numeric, errors='coerce')
    # Mirror the matrix to ensure symmetry
    adj_matrix_graph = adj_matrix_graph.combine_first(adj_matrix_graph.transpose())

    # Copy matrix to use for algorithm
    adj_matrix_algo = adj_matrix_graph.copy()
    # Replace NaN values with float('inf') for the algorithm
    #adj_matrix_algo.fillna(float('inf'), inplace=True)             #Do we really need this? Also works with nan


    print(adj_matrix_graph)
    print(adj_matrix_algo)

def create_graph():
    global G
    # Create a graph
    G = nx.Graph()

    # Add nodes and edges based on the adjacency matrix
    for i, row in adj_matrix_graph.iterrows():
        G.add_node(i)  # Add the postal area as a node
        for j, distance in row.items():
            if pd.notna(distance) and distance > 0:  # Valid distance
                G.add_edge(i, j, weight=distance)  # Add an edge with the weight

    # Print available nodes for debugging
    print("Available nodes in the graph:", G.nodes)

# Function to plot the graph
def plot_graph():
    ax.clear()
    global pos
    pos = nx.spring_layout(G)  # Positions for all nodes
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, ax=ax)
    # Draw edge labels (distances)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.title('Select start, finish, and pickup/drop-off points')
    plt.draw()

# Convert to edges that are drawable for networkx
def convert_to_edges(route):
    """
    Convert different route formats into a list of edges in the format [(A, B), (B, C), (C, D)].
    to be compatible with networkx

    Parameters:
    - route (list): A route that could be in one of the following formats:
        1. [(A, B, weight), (B, C, weight)] - with weights
        2. [(A, B), (B, C)] - without weights
        3. ['A', 'B', 'C', 'D'] - a list of nodes

    Returns:
    - edges (list): A list of edges in the format [(A, B), (B, C), (C, D)].
    """
    if isinstance(route[0], tuple):
        # Case 1 & 2: Route is a list of tuples
        if len(route[0]) == 3:
            # If tuples have 3 elements, they include weights, so ignore the weight
            return [(edge[0], edge[1]) for edge in route]
        else:
            # If tuples have 2 elements, they are already in the correct format
            return route
    elif isinstance(route[0], str):
        # Case 3: Route is a list of nodes (strings), convert to edges
        return [(route[i], route[i + 1]) for i in range(len(route) - 1)]
    else:
        raise ValueError("Invalid route format")
    
# Draw a route onto an existing graph and highlighting it
def draw_route_into_graph(route, color='red', plot_text=None, draw_labels=True):
    """
    Parameters:
    - route (list): A route in one of the following formats:
        1. [(A, B, weight), (B, C, weight)] - with weights
        2. [(A, B), (B, C)] - without weights
        3. ['A', 'B', 'C', 'D'] - as a list of nodes
    - color (str): The color to use for the highlighted route (default is 'red').
    - plot_text (str): Optional title for the plot (default is None).
    - draw_labels (bool): Optional draw the weights to the edges (default is True).
    """
    route = convert_to_edges(route)

    # Clear the current axes and redraw the graph
    ax.clear()  # Clear the axes before redrawing
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, ax=ax)
    # Draw edge labels (distances)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax)
    # Highlight the specified route (edges only) with the chosen color
    nx.draw_networkx_edges(G, pos, edgelist=route, edge_color=color, width=2.5, ax=ax)

    # Add an optional plot title if specified
    if plot_text:
        ax.set_title(plot_text)
    plt.draw()

def create_gui():
    root = Tk()
    root.title("Computability and Optimisation Assignment 1")

    # Create a frame for the plot
    global frame, canvas
    frame = Frame(root)
    frame.pack(pady=20)

    # Create buttons
    calculate_button = Button(root, text="Calculate Path (Enter)", command=calculate_path)
    calculate_button.pack(side='left', padx=5, pady=5)

    reset_button = Button(root, text="Reset Graph (Backspace)", command=reset_plot)
    reset_button.pack(side='left', padx=5, pady=5) 

    upload_button = Button(root, text="Upload CSV (U)", command=upload_csv)
    upload_button.pack(side='left', padx=5, pady=5)

    global fig, ax
    fig, ax = plt.subplots(figsize=(12, 8))  # Set figure size
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()  # Draw the canvas
    canvas.get_tk_widget().pack()

    plot_graph()

    # Run the GUI loop
    root.mainloop()



### EVENT HANDLING ###

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
        calculate_path()  # Call the programm function to compute and visualize the route
    elif event.key == 'backspace':
        reset_plot() # Reset current selections and reset plot
    elif event.key == 'u':
        upload_csv() # Upload csv




### MAIN ###

#Define path to csv
current_folder_path = os.path.dirname(os.path.abspath(__file__))
file_path = current_folder_path + '\\adjacency_matrix_26x26.csv'

#Initializes variables in function to be able to reset them with it
initialize_variables()  

#Read file and initialize adjacency matrices
read_csv()

#Create the graph from the adjecancy matrix
create_graph()

#Run gui and render plot inside it
create_gui()