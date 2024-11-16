import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import heapq
from tkinter import Tk, Button, Label, Frame, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import shutil
import numpy as np
from collections import defaultdict
import itertools
from Held_Karp_Algo import tsp

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
            best_route = reconstruct_path(came_from, current)        

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

    best_route = index_to_char(best_route)
    return best_route

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
# 7. Output the Result (hamiltonian circuit)
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

# Use of a greedy approach to find the minimum cost perfect matching (Optimized but doesnt find the optimal solution)
# Hungarian should be used
def minimum_cost_perfect_matching_greedy(matrix, odd_vertices, mst_edges):
    odd_vertices_count = len(odd_vertices)
    matched_left = [-1] * odd_vertices_count  # Tracks matches for left vertices
    matched_right = [-1] * odd_vertices_count  # Tracks matches for right vertices
    mst_edges_to_add = []

    # Create a cost matrix for the odd-degree vertices
    cost_matrix = np.zeros((odd_vertices_count, odd_vertices_count))
    vertex_to_index = {v: i for i, v in enumerate(odd_vertices)}

    for i in range(odd_vertices_count):
        for j in range(odd_vertices_count):
            if i != j:
                u, v = odd_vertices[i], odd_vertices[j]
                cost_matrix[i, j] = matrix[u][v]

    # Greedily match vertices based on the minimum cost
    for i in range(odd_vertices_count):
        min_cost = float('inf')
        best_match = -1
        for j in range(odd_vertices_count):
            if i != j and matched_right[j] == -1 and cost_matrix[i][j] < min_cost:
                min_cost = cost_matrix[i][j]
                best_match = j
        if best_match != -1:
            matched_left[i] = best_match
            matched_right[best_match] = i
            mst_edges_to_add.append((odd_vertices[i], odd_vertices[best_match], min_cost))

    return mst_edges_to_add

def minimum_cost_perfect_matching_brute(matrix, odd_vertices, mst_edges):
    min_cost_matching = []
    min_cost = float('inf')

    # Find all pairs of odd vertices
    pairs = list(itertools.combinations(odd_vertices, 2))

    # Check all possible matchings
    for matching in itertools.combinations(pairs, len(odd_vertices) // 2):
        if len(set([v for pair in matching for v in pair])) == len(odd_vertices):  # Ensure perfect matching
            cost = sum(matrix[u][v] for u, v in matching)
            if cost < min_cost:
                min_cost = cost
                min_cost_matching = [(u, v, matrix[u][v]) for u, v in matching]

    return min_cost_matching

# Not used
def hungarian_algorithm(cost_matrix):
    n = len(cost_matrix)
    label_left = [0] * n
    label_right = [0] * n
    match_left = [-1] * n
    match_right = [-1] * n

    def bfs():
        dist = [-1] * n
        prev = [-1] * n
        queue = []
        
        for i in range(n):
            if match_left[i] == -1:
                dist[i] = 0
                queue.append(i)
        
        found_augmenting_path = False
        for u in queue:
            for v in range(n):
                if dist[v] == -1 and cost_matrix[u][v] - label_left[u] - label_right[v] == 0:
                    dist[v] = dist[u] + 1
                    prev[v] = u
                    if match_right[v] == -1:
                        found_augmenting_path = True
                        x = v
                        while x != -1:
                            y = prev[x]
                            match_right[x] = y
                            match_left[y] = x
                            x = match_left[y]
                        break
                    else:
                        queue.append(match_right[v])
        return found_augmenting_path
    
    while bfs():
        pass
    
    return match_left, match_right

# Not Used (in combination with hungarian)
def minimum_cost_perfect_matching(matrix, odd_vertices, mst_edges):
    odd_vertices_count = len(odd_vertices)
    cost_matrix = np.zeros((odd_vertices_count, odd_vertices_count))

    # Create a map to convert odd vertices to indices in the cost matrix
    vertex_to_index = {v: i for i, v in enumerate(odd_vertices)}

    # Create the cost matrix with the appropriate costs
    for i in range(odd_vertices_count):
        for j in range(odd_vertices_count):
            if i != j:  # Ensure no vertex matches with itself
                u, v = odd_vertices[i], odd_vertices[j]
                cost_matrix[i, j] = matrix[u][v]

    match_left, match_right = hungarian_algorithm(cost_matrix)
    
    # Create the matching result in the required format (u, v, cost)
    min_cost_matching = [(odd_vertices[match_left[i]], odd_vertices[match_right[i]], cost_matrix[match_left[i], match_right[i]]) 
                         for i in range(odd_vertices_count) if match_left[i] != match_right[i]]  # Exclude invalid (u, u) pairs

    return min_cost_matching

def edmonds_blossom(matrix, odd_vertices, mst_edges):

    def find_augmenting_path(matching, node1, visited, parent, mst_edges_without_weight):
        queue = [node1]
        visited[node1] = True

        while queue:
            currentNode = queue.pop(0)

            for neighbour in odd_vertices:
                #Check if neighbour is not visited and exclude edges that are part of the mst
                if (index_to_char(neighbour), index_to_char(currentNode)) not in mst_edges_without_weight and (index_to_char(currentNode), index_to_char(neighbour)) not in mst_edges_without_weight and not visited[neighbour] and neighbour != currentNode:  
                    parent[neighbour] = currentNode    #Link current node to the neighbour for path reconstruction
                    print("parentdict: ", parent)
                    print(mst_edges_without_weight)
                    print("test:", index_to_char(currentNode), " ", index_to_char(neighbour))
                    if matching[neighbour] is None:  #Augmenting path found as the neighbour is not in matching and the current node is also not in matching and it is not in mst
                        print("RETURN augmented path start point ", neighbour)
                        return neighbour
                    visited[neighbour] = True      #Mark neigbour as visisted
                    queue.append(matching[neighbour])   #Neighbour already in matching, but gets added to the queue to check all its neighbours aswell
                                                #This continues until all nodes are visited and None is returned or until a augmented path is found and returned
        return None
    
    def augment_path(matching, parent, node):
        while node is not None:    #v is starting node of an augmented path
            prev = parent.get(node, None)  #get the neighbour that is linked to the node
            if prev is None:
                break
            print("Augmented")
            matching[node] = prev  
            matching[prev] = node  #flipping the matching status of both
            node = parent.get(prev, None)  #Fetch the parent of the previous item to travers the path further, to invert the whole path

    odd_vertices = char_to_index(odd_vertices)

    #Matching dictionary to track which nodes are matchings
    matching = {}
    matching = {u: None for u in odd_vertices}

    mst_edges_without_weight = [(node1, node2) for node1, node2, weight in mst_edges]

    #Augment the matching by finding augmenting paths
    for node1 in odd_vertices:
        if matching[node1] is None:  #Free vertex
            visited = [False] * len(matrix)
            parent = {}
            node2 = find_augmenting_path(matching, node1, visited, parent, mst_edges_without_weight)  #Find augmented path, starting with the unmatched node1
            if node2 is not None:
                augment_path(matching, parent, node2)

    # Convert matching dictionary to edge list
    result = []
    for node1, node2 in matching.items():
        if node1 < node2:  # Avoid duplicates
            weight = matrix.iloc[node1, node2]
            result.append((index_to_char(node1), index_to_char(node2), weight))

    return result

# Check if the route has only even degrees vertices
def has_even_degrees(edges):
    # Step 1: Initialize a dictionary to track the degree of each vertex
    degree_count = defaultdict(int)
    
    # Step 2: Count the degree of each vertex from the edges
    for u, v in edges:
        degree_count[u] += 1  # Outgoing edge from u
        degree_count[v] += 1  # Incoming edge to v
    
    # Step 3: Check if all vertices have even degrees
    for degree in degree_count.values():
        if degree % 2 != 0:
            return False  # Return False if any vertex has an odd degree
    
    return True  # All vertices have even degrees

# Use of Hierholzer algorithm
def find_eulerian_circuit(edges):
    # Convert graph edges to an adjacency list representation
    adj_list = {point: [] for point in selected_points}
    for u, v, weight in edges:
        adj_list[u].append(v)
        adj_list[v].append(u)

    # Function to check if a vertex has any unused edges
    def has_unused_edges(vertex):
        return len(adj_list[vertex]) > 0

    # Find the Eulerian circuit using Hierholzer's algorithm
    circuit = []
    stack = []
    current_vertex = selected_points[0]  # Start at an arbitrary vertex

    while stack or has_unused_edges(current_vertex):
        if has_unused_edges(current_vertex):
            stack.append(current_vertex)
            next_vertex = adj_list[current_vertex].pop()  # Take an unused edge
            adj_list[next_vertex].remove(current_vertex)  # Remove reverse edge
            current_vertex = next_vertex
        else:
            circuit.append(current_vertex)
            current_vertex = stack.pop()

    # The circuit is constructed in reverse, so we reverse it
    return circuit

def eulerian_to_hamiltonian(eulerian_circuit):
    # Set to keep track of visited nodes
    visited = set()
    hamiltonian_circuit = []
    
    for vertex in eulerian_circuit:
        if vertex not in visited:
            hamiltonian_circuit.append(vertex)
            visited.add(vertex)
    
    return hamiltonian_circuit

# Travelling Salesman Problem. Shortest path from A to B while traversing preselected nodes
def find_circular_route(matrix, selected_points):

    #Build MST with prims algorithm
    mst_edges, total_cost = prims_algorithm(matrix, selected_points)
    print("Edges in the Minimum Spanning Tree:")
    for u, v, weight in mst_edges:
        print(f"{u} -- {v} (Weight: {weight})")
    print(f"Total cost of the Minimum Spanning Tree: {total_cost}")
    # Draw the MST into the graph
    draw_route_into_graph(mst_edges, 'green', f"Minimum spanning tree\n Total cost: {total_cost}" , route_name='mst')


    # Odd Degree vertices of MST
    odd_vertices = find_odd_degree_vertices(mst_edges, selected_points)
    # As long as there are odd vertices add edges
    counter = 0
    while odd_vertices and counter < 5:
        counter += 1
        print("Vertices with odd degrees:", odd_vertices)

        # Find minimum-cost perfect matching for odd vertices
        matching = edmonds_blossom(matrix, odd_vertices, mst_edges)
        print("Minimum-cost perfect matching:", matching)

        # Combine MST and matching to form the multigraph
        multigraph_edges_with_weights = mst_edges + matching   
        print("Combined edges in the multigraph (MST + Matching):")
        for u, v, weight in multigraph_edges_with_weights:
            print(f"{u} -- {v} (Weight: {weight})")
        
        odd_vertices = find_odd_degree_vertices(multigraph_edges_with_weights, selected_points)

    draw_route_into_graph(multigraph_edges_with_weights, 'blue', "Multigraph with even-degree vertices", route_name='multigraph')

    # Euler tour is route that might visit one node multiple times
    euler_tour = find_eulerian_circuit(multigraph_edges_with_weights)
    print("Euler tour: ", euler_tour)
    draw_route_into_graph(euler_tour, 'orange', "Eulerian tour", route_name='euler')

    # Hamiltonian Circuit bypasses the multiple accessed node so that every node gets visisted exactly once
    hamiltonian_circuit = eulerian_to_hamiltonian(euler_tour)
    hamiltonian_circuit.append((hamiltonian_circuit[0]))  # Add the last point back to the first point
    print("Hamiltonian circuit: ", hamiltonian_circuit)
    draw_route_into_graph(hamiltonian_circuit, 'green', f"Shortest Path {"Best route found: "}{" -> ".join(hamiltonian_circuit)}", route_name='hamiltonian')

    # Use on each edge djirkas Algorithm to find shortcuts
    hamiltonian_edges = convert_to_edges(hamiltonian_circuit)

    hamilton_with_shortcut = []
    for edge in hamiltonian_edges:
        start, end = edge
        print(f"Start: {start} End: {end}")
        shortcut= find_shortest_path_dijkstras(matrix, char_to_index(start), char_to_index(end))
        print(f"Found Shortcut: {shortcut}")
        for node in shortcut:
            hamilton_with_shortcut.append(node)

    # remove consecutive duplicates
    result_hamilton_shortcut = [hamilton_with_shortcut[i] for i in range(1, len(hamilton_with_shortcut)) if hamilton_with_shortcut[i] != hamilton_with_shortcut[i-1]]
    result_hamilton_shortcut = [hamilton_with_shortcut[0]] + result_hamilton_shortcut  # Add the first element back, as it was excluded
    print("Hamilton with shortcut: ", result_hamilton_shortcut)
    draw_route_into_graph(result_hamilton_shortcut, 'r', f"Hamiltonian with Dijerkas: ", route_name='hamilton_dijerka')


    """ 
    # Finding the shortest path between points while traversing
    start_index, end_index, pickup_indicies = get_selected_points(selected_points)
    best_cost, best_path = tsp(matrix, pickup_indicies, start_index, end_index)
    best_path = index_to_char(best_path)
    print("Held Karp: ", best_cost, best_path)
    print("Selected Points in order: ", selected_points)
    # Adding the return from end to start
    # The route from School back to Taxi is fixed so add it to the shortest path
    route_school_taxi = find_shortest_path_dijkstras(matrix, start_index, end_index)
    cost_school_taxi = calculate_total_edge_weight(route_school_taxi)
    print("The Shortest Route from School to Taxi is: ", route_school_taxi)
    print("The Weight is: ", cost_school_taxi) """

    return None

# Function to convert characters A-Z to indices 0-25
def char_to_index(char):
    #return ord(char.upper()) - ord('A')
    if type(char) == str:
        return ord(char.upper()) - ord('A')
    else:
        return [ord(c.upper()) - ord('A') for c in char]

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
            find_circular_route(adj_matrix_algo, selected_points)
        else: 
            best_route = find_shortest_path_dijkstras(adj_matrix_algo, start_index, end_index)
            print("Best route found:", " -> ".join(best_route))
            draw_route_into_graph(best_route,'red', f"Shortest Path from {index_to_char(start_index)} to {index_to_char(end_index)}")


def initialize_variables():
    global selected_points, labels, current_label, selection_finished, selected_modus, node_texts, dic_routes
    selected_modus = 'route' # Start modus # or 'circuit'
    selected_points = []
    labels= ['start', 'finish']  # Labels for selection
    current_label = 0  # To keep track of what the user is selecting
    selection_finished = False  # Flag to indicate selection is finished
    node_texts = []
    dic_routes = {} # Store all calculated paths

def reset_plot():
    initialize_variables()
    # Clear the current axes and redraw the graph
    ax.clear()  # Clear the axes before redrawing
    # Set the color of all nodes to 'skyblue'
    for node in G.nodes:
        G.nodes[node]['color'] = 'skyblue'
    nx.draw(G, pos,with_labels=True, node_color='skyblue', node_size=500, ax=ax)
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

def change_mode(mode):
    global change_mode_button, selected_modus
    text_route = "Mode: Route   (Change with R or C)"
    text_circuit = "Mode: Circuit (Change with R or C)"
    if selected_points:
        print("Can only change the Mode before choosing points!")
        return
    if mode == 'r':
        change_mode_button.config(text=text_route)
        selected_modus = 'route'
    elif mode == 'c':
        change_mode_button.config(text=text_circuit)
        selected_modus = 'circuit'
    elif mode == 'switch':
        if change_mode_button.cget("text") == text_circuit:
            change_mode_button.config(text=text_route)
            selected_modus = 'route'
        else:
            change_mode_button.config(text=text_circuit)
            selected_modus = 'circuit'

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
            edges = [(edge[0], edge[1]) for edge in route]
        else:
            # If tuples have 2 elements, they are already in the correct format
            edges = route
    elif isinstance(route[0], str):
        # Case 3: Route is a list of nodes (strings), convert to edges
        edges = [(route[i], route[i + 1]) for i in range(len(route) - 1)]
    else:
        raise ValueError("Invalid route format")
    
    return edges

def calculate_total_edge_weight(edges):
    edges = convert_to_edges(edges)
    total_weight = 0
    for edge in edges:
        start, end = edge
        weight = adj_matrix_algo[start][end]
        total_weight += weight
    return total_weight

# Draw a route onto an existing graph and highlighting it
def draw_route_into_graph(route, color='orange', plot_text=None, route_name=None):
    global node_texts, dic_routes

    ax.clear()
    # Convert the route to edges if it’s in node format
    route_edges = convert_to_edges(route)

    if route_name:
        dic_routes[route_name] = route

    for label in ax.texts:
        label.remove()
    # Retrieve current colors from node attributes
    node_colors = [G.nodes[node].get('color', 'skyblue') for node in G.nodes]

    nx.draw(G, pos, with_labels=True, node_size=500, ax=ax, edge_color='gray', node_color=node_colors)
    # Highlight the specified route edges with the given color
    nx.draw_networkx_edges(G, pos, edgelist=route_edges, edge_color=color, width=3, ax=ax)

    highlighted_edge_labels = {edge: G[edge[0]][edge[1]]['weight'] for edge in route_edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=highlighted_edge_labels, font_size=8, ax=ax)

    # Calculate the total route cost and set plot title
    total_cost = calculate_total_edge_weight(route_edges)
    ax.set_title(f"{plot_text}\nTotal Cost: {total_cost}" if plot_text else f"Total Cost: {total_cost}")

    for x, y, text in node_texts:
        plt.text(
        x, y + 0.08, text,
        horizontalalignment='center',
        fontweight='bold',
        bbox=dict(facecolor='lightgrey', edgecolor='black', boxstyle='round,pad=0.5')
        )
    # Render the plot
    plt.draw()

def create_gui():
    root = Tk()
    root.title("Computability and Optimisation Assignment 1")

    # Create a frame for the plot
    global frame, canvas, change_mode_button
    frame = Frame(root)
    frame.pack(pady=20)

    # Create Button to display and changecurrent Mode
    change_mode_button = Button(root, text="Mode: Route   (Change with r or c)", command=lambda: change_mode('switch'))
    change_mode_button.pack(side='left', padx=5, pady=5) 

    # Create buttons
    calculate_button = Button(root, text="Calculate Path (Enter)", command=calculate_path)
    calculate_button.pack(side='left', padx=5, pady=5)

    reset_button = Button(root, text="Reset Graph (Backspace)", command=reset_plot)
    reset_button.pack(side='left', padx=5, pady=5) 

    upload_button = Button(root, text="Upload CSV (U)", command=upload_csv)
    upload_button.pack(side='left', padx=5, pady=5)

    show_mst_button = Button(root, text="MST-PRIMS", command=lambda: draw_route_into_graph(dic_routes['mst'], plot_text='Minimum Spanning Tree'))
    show_mst_button.pack(side='left', padx=5, pady=5)

    show_matching_button = Button(root, text="Multigraph", command=lambda: draw_route_into_graph(dic_routes['multigraph'], plot_text='Multigraph'))
    show_matching_button.pack(side='left', padx=5, pady=5)

    show_euler_button = Button(root, text="Euler Tour", command=lambda: draw_route_into_graph(dic_routes['euler'], plot_text='Euler Circuit'))
    show_euler_button.pack(side='left', padx=5, pady=5)

    show_hamiltonian_button = Button(root, text="Hamiltonian", command=lambda: draw_route_into_graph(dic_routes['hamiltonian'], plot_text='Hamiltonian Circuit'))
    show_hamiltonian_button.pack(side='left', padx=5, pady=5)

    show_hamilton_dijerka_button = Button(root, text="hamilton_dijerka", command=lambda: draw_route_into_graph(dic_routes['hamilton_dijerka'], plot_text='Hamiltonian tried to optimized with Dijerka'))
    show_hamilton_dijerka_button.pack(side='left', padx=5, pady=5)

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
    global current_label, selection_finished, change_mode_button, node_texts
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
    
    if closest_node in selected_points:
        plt.title("Node already selected! Choose another one: ")
        plt.draw()
        return

    # If a node was found, proceed with labeling and coloring
    if closest_node is not None:
        selected_points.append(closest_node)
        x, y = pos[closest_node]
        if current_label == 0:  # Label and highlight the "start" point (School)
            print(f'Selected {labels[current_label]} point: {closest_node}')
            nx.draw_networkx_nodes(G, pos, nodelist=[closest_node], node_color='green', node_size=500)
            G.nodes[closest_node]['color'] = 'green'
            plt.text(
                x, y + 0.08, 'Taxi',
                horizontalalignment='center',
                fontweight='bold',
                bbox=dict(facecolor='lightgrey', edgecolor='black', boxstyle='round,pad=0.5')
            )
            node_texts.append((x, y, 'Taxi'))
            current_label = 1

        elif current_label == 1:  # Label and highlight the "finish" point
            print(f'Selected {labels[current_label]} point: {closest_node}')
            nx.draw_networkx_nodes(G, pos, nodelist=[closest_node], node_color='b', node_size=500)
            G.nodes[closest_node]['color'] = 'b'
            plt.text(
                x, y + 0.08, 'School',
                horizontalalignment='center',
                fontweight='bold',
                bbox=dict(facecolor='lightgrey', edgecolor='black', boxstyle='round,pad=0.5')
            )
            node_texts.append((x, y, 'School'))
            current_label = 2

        else:  # Label and highlight any additional points as pickup/drop-off points
            print(f'Selected pickup/drop-off point: {closest_node}')
            nx.draw_networkx_nodes(G, pos, nodelist=[closest_node], node_color='r', node_size=500)
            G.nodes[closest_node]['color'] = 'r'
            #node_texts.append((x, y, f"Pickup {current_label}"))
            current_label += 1
    plt.draw()
    

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
    elif event.key == 'r' or event.key == 'c':
        change_mode(event.key)




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