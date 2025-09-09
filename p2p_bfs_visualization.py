import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque, defaultdict
import random
import math

class P2PNetwork:
    def __init__(self):
        self.graph = defaultdict(list)
        self.nodes = set()
    
    def add_connection(self, node1, node2):
        """Add a bidirectional connection between two nodes"""
        if node1 != node2:  # Prevent self-connections
            self.graph[node1].append(node2)
            self.graph[node2].append(node1)
            self.nodes.add(node1)
            self.nodes.add(node2)
    
    def generate_random_network(self, num_nodes, connection_probability=0.3):
        """Generate a random P2P network"""
        nodes = [f"Node_{i}" for i in range(num_nodes)]
        
        # Ensure all nodes are connected (create a connected graph)
        for i in range(num_nodes - 1):
            self.add_connection(nodes[i], nodes[i + 1])
        
        # Add random connections
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if random.random() < connection_probability:
                    self.add_connection(nodes[i], nodes[j])
    
    def find_neighbors_bfs(self, start_node, max_depth=None):
        """
        Find all neighbor nodes using BFS
        Returns: dictionary with nodes and their distance from start node
        """
        if start_node not in self.graph:
            return {}
        
        visited = {}
        queue = deque([(start_node, 0)])  # (node, distance)
        
        while queue:
            current_node, distance = queue.popleft()
            
            if max_depth is not None and distance > max_depth:
                continue
                
            if current_node not in visited:
                visited[current_node] = distance
                
                for neighbor in self.graph[current_node]:
                    if neighbor not in visited:
                        queue.append((neighbor, distance + 1))
        
        return visited

def visualize_network(graph, start_node=None, bfs_result=None):
    """Visualize the network with optional BFS highlighting"""
    G = nx.Graph()
    
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor)
    
    plt.figure(figsize=(10, 8))
    
    # Calculate positions using spring layout
    pos = nx.spring_layout(G, k=3/math.sqrt(G.order()), iterations=50)
    
    # Draw the graph
    if start_node and bfs_result:
        # Color nodes based on BFS distance
        node_colors = []
        for node in G.nodes():
            if node == start_node:
                node_colors.append('red')  # Start node
            elif node in bfs_result:
                node_colors.append('lightgreen')  # Reachable nodes
            else:
                node_colors.append('lightblue')  # Unreachable nodes
        
        # Draw edges with different styles for BFS tree
        bfs_edges = set()
        for node in bfs_result:
            if node != start_node:
                # Find the edge that connects this node to its parent in BFS tree
                for neighbor in graph[node]:
                    if neighbor in bfs_result and bfs_result[neighbor] == bfs_result[node] - 1:
                        bfs_edges.add((min(node, neighbor), max(node, neighbor)))
                        break
        
        edge_colors = ['red' if (min(u, v), max(u, v)) in bfs_edges else 'gray' for u, v in G.edges()]
        edge_widths = [3 if (min(u, v), max(u, v)) in bfs_edges else 1 for u, v in G.edges()]
        
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, alpha=0.7)
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500)
    else:
        nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.7)
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
    
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    plt.title("P2P Network Topology")
    plt.axis('off')
    return plt

def main():
    st.set_page_config(page_title="P2P Network BFS Visualization", layout="wide")
    
    st.title("Peer-to-Peer Network Neighbor Discovery using BFS")
    st.markdown("""
    This application visualizes a Peer-to-Peer (P2P) network and demonstrates how Breadth-First Search (BFS)
    can be used to discover all neighbor nodes from a starting point.
    """)
    
    # Initialize network in session state
    if 'network' not in st.session_state:
        st.session_state.network = P2PNetwork()
    
    # Sidebar controls
    st.sidebar.header("Network Configuration")
    num_nodes = st.sidebar.slider("Number of Nodes", min_value=5, max_value=30, value=15, step=1)
    connection_prob = st.sidebar.slider("Connection Probability", min_value=0.1, max_value=0.5, value=0.25, step=0.05)
    
    if st.sidebar.button("Generate New Network"):
        st.session_state.network = P2PNetwork()
        st.session_state.network.generate_random_network(num_nodes, connection_prob)
        st.session_state.bfs_result = None
        st.session_state.start_node = None
    
    # If network is empty, generate one
    if not st.session_state.network.nodes:
        st.session_state.network.generate_random_network(num_nodes, connection_prob)
    
    # Display network information
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Network Information")
        st.write(f"Total nodes: {len(st.session_state.network.nodes)}")
        st.write(f"Total connections: {sum(len(neighbors) for neighbors in st.session_state.network.graph.values()) // 2}")
        
        # Node selection for BFS
        start_node = st.selectbox("Select Start Node for BFS", sorted(st.session_state.network.nodes))
        max_depth = st.slider("Maximum Depth (Hops)", min_value=1, max_value=10, value=3, step=1)
        
        if st.button("Run BFS"):
            st.session_state.bfs_result = st.session_state.network.find_neighbors_bfs(start_node, max_depth)
            st.session_state.start_node = start_node
    
    with col2:
        st.subheader("BFS Results")
        if hasattr(st.session_state, 'bfs_result') and st.session_state.bfs_result:
            reachable_nodes = {k: v for k, v in st.session_state.bfs_result.items() if v > 0}
            st.write(f"From **{st.session_state.start_node}**, found **{len(reachable_nodes)}** reachable nodes within **{max_depth}** hops:")
            
            # Group nodes by distance
            for distance in range(1, max_depth + 1):
                nodes_at_distance = [node for node, d in st.session_state.bfs_result.items() if d == distance]
                if nodes_at_distance:
                    st.write(f"- **{distance} hop(s) away**: {', '.join(sorted(nodes_at_distance))}")
        else:
            st.info("Run BFS to see reachable nodes from the selected start node.")
    
    # Visualization
    st.subheader("Network Visualization")
    
    fig_col1, fig_col2 = st.columns(2)
    
    with fig_col1:
        st.write("**Complete Network Topology**")
        plt1 = visualize_network(st.session_state.network.graph)
        st.pyplot(plt1)
    
    with fig_col2:
        if hasattr(st.session_state, 'bfs_result') and st.session_state.bfs_result:
            st.write(f"**BFS from {st.session_state.start_node} (max {max_depth} hops)**")
            plt2 = visualize_network(st.session_state.network.graph, st.session_state.start_node, st.session_state.bfs_result)
            st.pyplot(plt2)
        else:
            st.write("**BFS Visualization**")
            st.info("Run BFS to see the traversal path visualization.")
    
    # Explanation
    st.subheader("How BFS Works in P2P Networks")
    st.markdown("""
    Breadth-First Search (BFS) is a fundamental algorithm for exploring graphs and networks. In P2P networks, BFS helps:
    
    1. **Discovering network topology**: Finding all nodes reachable from a starting point
    2. **Determining shortest paths**: Identifying the minimum number of hops between nodes
    3. **Content discovery**: Locating resources across the network efficiently
    
    **BFS Process**:
    - Start from the selected node (distance 0)
    - Visit all immediate neighbors (distance 1)
    - Then visit neighbors of those neighbors (distance 2)
    - Continue until all reachable nodes are visited or maximum depth is reached
    
    **Visualization Key**:
    - 🔴 Red node: The starting point for BFS
    - 🟢 Green nodes: Nodes discovered by BFS
    - 🔴 Red edges: The path taken by BFS to discover nodes
    - 🔵 Blue nodes: Nodes not reached by BFS (beyond max depth or disconnected)
    """)

if __name__ == "__main__":
    main()