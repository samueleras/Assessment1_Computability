import numpy as np
# Function to calculate the shortest path between all pairs using Floyd-Warshall
def floyd_warshall(graph):
    n = len(graph)
    dist = np.copy(graph)
    
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    
    return dist

# Function to solve the shortest path problem through given points using Dynamic Programming (Held-Karp)
def tsp(graph, points, start, end):
    points_to_visit = [start] + points + [end]
    n = len(points_to_visit)

    # Step 1: Precompute the shortest distances between all pairs of points (Floyd-Warshall)
    dist = floyd_warshall(graph)

    # Step 2: Dynamic Programming to solve TSP (Held-Karp Algorithm)
    # dp[mask][i] will be the shortest path to visit the nodes in mask, ending at node i
    INF = float('inf')
    dp = np.full((1 << n, n), INF)
    prev = np.full((1 << n, n), -1)  # To track the previous node in the path
    dp[1 << 0][0] = 0  # Starting at the start node
    
    for mask in range(1 << n):  # For each subset of visited nodes
        for u in range(n):  # For each node in the subset
            if mask & (1 << u):
                for v in range(n):  # For each node to visit
                    if not (mask & (1 << v)):  # If node v is not yet visited
                        new_mask = mask | (1 << v)
                        new_cost = dp[mask][u] + dist[points_to_visit[u]][points_to_visit[v]]
                        if new_cost < dp[new_mask][v]:
                            dp[new_mask][v] = new_cost
                            prev[new_mask][v] = u  # Track the previous node

    # Step 3: Find the minimum cost path that visits all points
    best_cost = INF
    best_end = -1
    end_mask = (1 << n) - 1  # All nodes visited
    for u in range(n):
        cost = dp[end_mask][u] + dist[points_to_visit[u]][end]
        if cost < best_cost:
            best_cost = cost
            best_end = u

    # Step 4: Reconstruct the path
    path = []
    mask = end_mask
    u = best_end
    while u != -1:
        path.append(points_to_visit[u])
        next_u = prev[mask][u]
        mask ^= (1 << u)  # Remove u from the visited set
        u = next_u

    path.reverse()  # Reverse the path to get it from start to end
    return best_cost, path