import streamlit as st
import requests
import json
import os
from dotenv import load_dotenv
from typing import Dict, Any, List

# Load environment variables
load_dotenv()

# Configuration
API_URL = os.getenv("API_URL", "https://nikhil-01-retail-llm.hf.space/search")
PAGE_TITLE = "Smart Retail Search"
PAGE_ICON = "🔍"

# Page configuration
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="collapsed"
)

def search_products(query: str) -> Dict[str, Any]:
    """Send search query to backend API and return results."""
    try:
        headers = {"Content-Type": "application/json"}
        response = requests.post(
            API_URL,
            data=json.dumps({"query": query}),
            headers=headers
        )
        response.raise_for_status()  # Raise exception for 4XX/5XX responses
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to search service: {e}")
        return {"status": "error", "error": str(e)}

def display_products(products: List[Dict[str, Any]]):
    """Display product cards in a grid layout."""
    # Show up to 9 products in a 3x3 grid
    products = products[:9]  # Limit to 9 products
    
    # Create rows of 3 products each
    for i in range(0, len(products), 3):
        cols = st.columns(3)
        for j in range(3):
            if i+j < len(products):
                product = products[i+j]
                with cols[j]:
                    st.subheader(product.get("title", "No Title"))
                    
                    # Price and brand info
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Price:** ₹{product.get('price', 'N/A')}")
                    with col2:
                        st.markdown(f"**Brand:** {product.get('brand', 'N/A')}")
                    
                    # Description
                    description = product.get("description", "No description available.")
                    if len(description) > 100:
                        description = description[:100] + "..."
                    st.markdown(f"*{description}*")
                    
                    # Category
                    st.markdown(f"**Category:** {product.get('category', 'N/A')}")
                    
                    # Add a divider
                    st.divider()

def main():
    """Main function for the Streamlit app."""
    
    # Header section
    st.title("🛍️ Smart Retail Product Search")
    st.markdown("""
    ### Find the perfect products with AI-powered natural language search
    Simply describe what you're looking for in everyday language, and our smart assistant will help you find the best matches.
    """)
    
    # Search input
    query = st.text_input(
        "What are you looking for today?",
        placeholder="e.g., 'waterproof hiking boots' or 'affordable 4K TV with good reviews'"
    )
    
    # Search button
    search_button = st.button("Search", type="primary")
    
    # Process search when button is clicked
    if search_button and query:
        with st.spinner("Searching for products..."):
            results = search_products(query)
            
            if results.get("status") == "success":
                # Display AI response
                st.success("Products Found!")
                
                # Show generated response
                st.markdown("### AI Assistant")
                st.markdown(
                    f'''
                    <div style="
                        background: linear-gradient(135deg, #e6f2ff, #b3d9ff);
                        padding: 15px;
                        border-radius: 10px;
                        border: 1px solid #7cb9ff;
                        color: #0a5299;
                        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    ">
                        {results.get("generated_response")}
                    </div>
                    ''',
                    unsafe_allow_html=True
                )
                
                # Display products
                st.markdown("### Top Matching Products")
                display_products(results.get("results", []))
                
                # Show category
                st.markdown(f"**Category:** {results.get('category')}")
                
            elif results.get("status") == "not_found":
                st.warning(results.get("message", "No products found matching your query."))
            else:
                st.error(results.get("error", "An error occurred during the search."))
    
    # Additional information in sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This search tool uses AI to understand your natural language requests
        and find the most relevant products from our catalog.
        
        **Features:**
        - Natural language search
        - Category recognition
        - AI-powered recommendations
        """)

if __name__ == "__main__":
    main()
