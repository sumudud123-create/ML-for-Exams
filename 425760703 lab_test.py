# Problem: Optimizing Placement of Electric Vehicle (EV) Charging Stations on California Road Network
# Objective:
#Identify optimal locations for EV charging stations to minimize average travel distance for drivers.
#Use graph-based analysis (node degree and connectivity) to select station locations.
# Class to load road network

import numpy as np
from collections import deque

# -----------------------------
# Class to Load Road Network
# -----------------------------
class RoadNetwork:
    def __init__(self):
        self.adj = {}  # adjacency list: node -> list of neighbors

    def load_from_txt(self, filename):
        """Load road network from .txt file (edge list format)"""
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith("#"):
                    continue
                u, v = map(int, line.strip().split())
                if u not in self.adj:
                    self.adj[u] = []
                if v not in self.adj:
                    self.adj[v] = []
                self.adj[u].append(v)
                self.adj[v].append(u)  # undirected graph

    def nodes(self):
        return list(self.adj.keys())

    def edges(self):
        edge_list = []
        for u in self.adj:
            for v in self.adj[u]:
                if (v, u) not in edge_list:
                    edge_list.append((u, v))
        return edge_list

# -----------------------------
# Summary Statistics Function
# -----------------------------
def compute_node_degree_stats(network):
    degrees = [len(network.adj[node]) for node in network.nodes()]
    stats = {
        "mean": np.mean(degrees),
        "min": np.min(degrees),
        "max": np.max(degrees),
        "std": np.std(degrees),
        "median": np.median(degrees)
    }
    return stats

# -----------------------------
# EV Charging Station Planner
# -----------------------------
class EVStationPlanner:
    def __init__(self, network):
        self.network = network
        self.stations = []

    def place_stations(self, k):
        """Place k stations on nodes with highest degree"""
        node_degrees = {node: len(self.network.adj[node]) for node in self.network.nodes()}
        sorted_nodes = sorted(node_degrees, key=node_degrees.get, reverse=True)
        self.stations = sorted_nodes[:k]
        return self.stations

    def average_distance_to_station(self):
        """Estimate average shortest distance to nearest station using BFS"""
        distances = []
        for node in self.network.nodes():
            if node in self.stations:
                distances.append(0)
                continue
            visited = {node}
            queue = deque([(node, 0)])
            found = False
            while queue and not found:
                current, dist = queue.popleft()
                for neighbor in self.network.adj[current]:
                    if neighbor in visited:
                        continue
                    if neighbor in self.stations:
                        distances.append(dist + 1)
                        found = True
                        break
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
            if not found:
                distances.append(np.nan)  # unreachable nodes
        return np.nanmean(distances)

# -----------------------------
# Main Program
# -----------------------------
if __name__ == "__main__":
    # Load California Road Network
    file_path = r"C:\Users\User\Downloads\roadNet-CA.txt"
    ca_network = RoadNetwork()
    print("Loading California Road Network... (this may take a few minutes)")
    ca_network.load_from_txt(file_path)
    print(f"Total nodes: {len(ca_network.nodes())}")
    print(f"Total edges: {len(ca_network.edges())}")

    # Compute summary statistics
    stats = compute_node_degree_stats(ca_network)
    print("\nNode Degree Summary Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")

    # Plan EV charging stations
    planner = EVStationPlanner(ca_network)
    num_stations = 50
    top_stations = planner.place_stations(num_stations)
    print(f"\nTop {num_stations} station nodes (first 10 shown): {top_stations[:10]} ...")

    avg_distance = planner.average_distance_to_station()
    print(f"Average distance to nearest station: {avg_distance:.2f}")