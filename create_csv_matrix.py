import csv
import random
import string

# Define the postal areas (A to Z)
postal_areas = list(string.ascii_uppercase)  # ['A', 'B', 'C', ..., 'Z']

# Function to generate a random distance or None (for no connection)
# Use a probability to decide if two nodes are connected
def random_distance(prob_of_connection=1):
    # With a probability of 0.3 (30%), a connection exists; otherwise, no connection (None)
    if random.random() < prob_of_connection:
        return random.randint(1, 100)  # Return a random distance between 1 and 100
    else:
        return None  # No connection

# Create the adjacency matrix
matrix = []

# First row: headers (postal areas)
matrix.append([""] + postal_areas)  # First empty cell for alignment

# Fill the matrix with random distances
for area in postal_areas:
    row = [area]  # First element of the row is the area name
    for other_area in postal_areas:
        if area == other_area:
            row.append(0)  # Distance to itself is 0
        else:
            row.append(random_distance())  # Random distance or None for no connection
    matrix.append(row)

# Define the file path where you want to save the CSV file
file_path = 'adjacency_matrix_26x26.csv'

# Write the adjacency matrix to a CSV file
with open(file_path, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=';')
    writer.writerows(matrix)

print(f"CSV file created successfully and saved to {file_path}")
