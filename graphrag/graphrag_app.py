import streamlit as st
import streamlit.components.v1 as components
from graphrag_functions import create_sample_graph, draw_graph

def main():
    st.title("Graphrag: Document Retrieval Graph Visualization")
    st.markdown("### Explore how documents and entities are connected")
    
    # Create the sample graph
    G = create_sample_graph()
    
    # Get the HTML for the graph visualization
    graph_html = draw_graph(G)
    
    # Render the interactive graph within the Streamlit app
    components.html(graph_html, height=600, width=800)
    
    st.sidebar.markdown("### About this demo")
    st.sidebar.info(
        "This demo uses Graphrag concepts to visualize how documents relate to different entities. "
        "In a production system, you would replace the sample graph with one generated from your document retrieval and NLP pipeline."
    )

if __name__ == "__main__":
    main()
