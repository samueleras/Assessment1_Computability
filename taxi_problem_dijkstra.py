import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import heapq
from tkinter import Tk, Button, Label, Frame, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import shutil

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

def prims_algorithm(adj_matrix, selected_points):
    # Extract the submatrix for the selected points
    matrix_selected_points = adj_matrix.loc[selected_points, selected_points]

    n = len(matrix_selected_points)  # Number of vertices in the graph
    mst_edges = []  # List to store the edges of the MST
    total_cost = 0  # To keep track of the total cost of the MST

    # Start from the first vertex in the selected points
    start_vertex = selected_points[0]
    min_heap = [(0, start_vertex)]  # Start from the first selected vertex
    visited = set()  # To track visited vertices
    prev_vertex = None  # To track the previous vertex for edge creation

    while min_heap:
        # Get the edge with the minimum weight
        weight, u = heapq.heappop(min_heap)

        if u in visited:
            continue  # Skip if already visited

        visited.add(u)  # Mark vertex as visited
        total_cost += weight  # Update the total cost

        if prev_vertex is not None:
            # Append the edge only if it's not the starting node
            mst_edges.append((prev_vertex, u, weight))

        # Add all edges from the current vertex to the priority queue
        for v in matrix_selected_points.columns:
            if v not in visited:
                edge_weight = matrix_selected_points.loc[u, v]  # Access using string labels
                if pd.notna(edge_weight) and edge_weight > 0:  # Check if the edge is valid
                    heapq.heappush(min_heap, (edge_weight, v))
        
        prev_vertex = u  # Update previous vertex for next iteration

    return mst_edges, total_cost

def find_odd_degree_vertices(mst_edges, num_vertices):
    #degree counter initialized to 0 for each vertex
    vertex_degree = [0] * num_vertices
    
    mst_edges_numeric = [(ord(u) - ord('A'), ord(v) - ord('A')) for u, v, weight in mst_edges]

    # Iterate through all edges in the MST and count the degree of each vertex
    for u, v in mst_edges_numeric:
        vertex_degree[u] += 1
        vertex_degree[v] += 1
    
    # Identify vertices with odd degrees
    odd_degree_vertices = [vertex for vertex in range(num_vertices) if vertex_degree[vertex] % 2 == 1]
    
    return odd_degree_vertices

# Travelling Salesman Problem. Shortest path from A to B while traversing preselected nodes
def find_shortest_path_with_pickup_points(matrix, selected_points):

    #Build MST
    mst_edges, total_cost = prims_algorithm(matrix, selected_points)
    print("MST edges:", mst_edges)
    print("Edges in the Minimum Spanning Tree:")
    for u, v, weight in mst_edges:
        print(f"{u} -- {v} (Weight: {weight})")
    print(f"Total cost of the Minimum Spanning Tree: {total_cost}")

    # Draw the MST into the graph
    draw_route_into_graph(mst_edges, 'green', 'Minimum spanning tree')
    #Odd Degree vertices of MST
    num_vertices = len(adj_matrix_algo)
    odd_vertices = find_odd_degree_vertices(mst_edges, num_vertices)
    print("Vertices with odd degrees:", odd_vertices)
    return None

# Function to convert characters A-Z to indices 0-25
def char_to_index(char):
    return ord(char.upper()) - ord('A')

# Function to convert indices 0-25 to characters A-Z
def index_to_char(index_list):
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
            find_shortest_path_with_pickup_points(adj_matrix_algo, selected_points)
            print("Traversing multiple pickup points not supported yet.")
            return  #has to be removed when feature is implemented
        else: 
            best_route = find_shortest_path_dijkstras(adj_matrix_algo, start_index, end_index)

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
def draw_route_into_graph(route, color='red', plot_text=None):
    """
    Parameters:
    - route (list): A route in one of the following formats:
        1. [(A, B, weight), (B, C, weight)] - with weights
        2. [(A, B), (B, C)] - without weights
        3. ['A', 'B', 'C', 'D'] - as a list of nodes
    - color (str): The color to use for the highlighted route (default is 'red').
    - plot_text (str): Optional title for the plot (default is None).
    """
    route = convert_to_edges(route)

    # Clear the current axes and redraw the graph
    ax.clear()  # Clear the axes before redrawing
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, ax=ax)
    # Draw edge labels (distances)
    edge_labels = nx.get_edge_attributes(G, 'weight')
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