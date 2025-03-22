import networkx as nx
from pyvis.network import Network

def create_sample_graph():
    """
    Build a sample graph representing documents and their related entities.
    Nodes represent documents or entities, and edges capture the relationships.
    """
    G = nx.Graph()
    
    # Adding sample document nodes with attributes
    G.add_node("Document 1", label="Doc 1", title="Document 1: Company Report", group="Documents")
    G.add_node("Document 2", label="Doc 2", title="Document 2: Market Analysis", group="Documents")
    
    # Adding sample entity nodes with attributes
    G.add_node("Entity A", label="Entity A", title="Entity A: CEO", group="Entities")
    G.add_node("Entity B", label="Entity B", title="Entity B: Technology", group="Entities")
    
    # Create edges between documents and entities
    G.add_edge("Document 1", "Entity A", weight=3)
    G.add_edge("Document 1", "Entity B", weight=1)
    G.add_edge("Document 2", "Entity B", weight=2)
    
    return G

def draw_graph(G):
    """
    Convert the NetworkX graph into a Pyvis interactive network and return the HTML.
    """
    # Create a Pyvis network instance; notebook=True helps with configuration
    net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white", notebook=True)
    
    # Load the graph from NetworkX into Pyvis
    net.from_nx(G)
    
    # Optionally, display a button panel for physics settings for interactive tweaking
    net.show_buttons(filter_=['physics'])
    
    # Save the visualization to an HTML file and then read it back
    net.save_graph("graph.html")
    with open("graph.html", "r", encoding="utf-8") as html_file:
        graph_html = html_file.read()
        
    return graph_html
