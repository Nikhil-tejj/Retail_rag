import streamlit as st
import requests
import json
import os
from typing import Dict, Any, List


# Configuration
API_URL = "https://nikhil-01-retail-llm.hf.space/search"
PAGE_TITLE = "Smart Retail Search"
PAGE_ICON = "🔍"

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
        response.raise_for_status()  
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to search service: {e}")
        return {"status": "error", "error": str(e)}

def display_products(products: List[Dict[str, Any]]):
    """Display product cards in a grid layout."""
    products = products[:9]  

    for i in range(0, len(products), 3):
        cols = st.columns(3)
        for j in range(3):
            if i+j < len(products):
                product = products[i+j]
                with cols[j]:
                    st.subheader(product.get("title", "No Title"))

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Price:** ₹{product.get('price', 'N/A')}")
                    with col2:
                        st.markdown(f"**Brand:** {product.get('brand', 'N/A')}")

                    description_value = product.get("description") 
                    if isinstance(description_value, str):
                        description_str = description_value
                    elif description_value is not None:
                         description_str = str(description_value)
                    else:
                         description_str = "No description available."

                    if len(description_str) > 100:
                        display_desc = description_str[:100] + "..."
                    else:
                        display_desc = description_str
                    st.markdown(f"*{display_desc}*")

                    st.markdown(f"**Category:** {product.get('category', 'N/A')}")

                    st.divider()

def main():
    """Main function for the Streamlit app."""
    
    st.title("🛍️ Smart Retail Product Search")
    st.markdown("""
    ### Find the perfect products with AI-powered natural language search
    Simply describe what you're looking for in everyday language, and our smart assistant will help you find the best matches.
    """)
    
    query = st.text_input(
        "What are you looking for today?",
        placeholder="e.g., 'moisturizing body wash', 'shampoo for dry hair', 'soap for sensitive skin'" # Updated placeholder
    )
    
    search_button = st.button("Search", type="primary")
    
    if search_button and query:
        with st.spinner("Searching for products..."):
            results = search_products(query)
            
            if results.get("status") == "success":
                st.success("Products Found!")
                
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
                
                st.markdown("### Top Matching Products")
                display_products(results.get("results", []))
                
                st.markdown(f"**Category:** {results.get('category')}")
                
            elif results.get("status") == "not_found":
                st.warning(results.get("message", "No products found matching your query."))
            else:
                st.error(results.get("error", "An error occurred during the search."))
    
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This search tool uses AI(NLP) to understand your natural language requests
        and find the most relevant products from our catalog.

        **Available Categories:**
        - Bath & Shower
        - Detergents & Dishwash
        - Fragrance
        - Grocery & Gourmet Foods
        - Hair Care
        - Skin Care

        **Features:**
        - Natural language search
        - Category recognition
        - AI-powered 
        """)
if __name__ == "__main__":
    main()
