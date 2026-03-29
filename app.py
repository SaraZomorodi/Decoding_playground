import streamlit as st
import matplotlib.pyplot as plt
from utils_PI import MakePlots

st.set_page_config(layout="wide")
st.title("Independent Rat Analysis Dashboard")

# --- Global Sidebar (Shared Settings) ---
st.sidebar.header("Global Settings")
selected_rats = st.sidebar.multiselect("Select Rat IDs", options=[1 ,2, 3, 4, 5], default=[2, 3])
lighting = st.sidebar.radio("Lighting Condition", options=['light', 'dark'])
edge_margin = st.sidebar.number_input("Edge Margin (cm)", value=15)

# --- Main Columns ---
if selected_rats:
    cols = st.columns(len(selected_rats))

    for i, rat_id in enumerate(selected_rats):
        with cols[i]:
            st.header(f"Rat {rat_id}")
            
            # --- Local Controls (Unique to this column) ---
            # We use key=f"step_{rat_id}" to keep these sliders separate
            dist_step = st.slider(
                f"Dist Step for Rat {rat_id}", 
                1.0, 20.0, 8.0, 
                key=f"step_{rat_id}"
            )
            
            # --- Processing & Plotting ---
            with st.spinner(f"Computing Rat {rat_id}..."):
                # Initialize the analysis for this specific rat
                analysis = MakePlots(
                    edge_margin=edge_margin, 
                    RAT_ID=rat_id, 
                    LIGHTING=lighting
                )
                
                # Plot 1: Error vs Distance
                st.write("**Decoding Error**")
                analysis.plot_traveled_space(dist_step=dist_step)
                st.pyplot(plt.gcf())
                plt.close() # Reset for the next rat

                # Plot 2: Trajectory
                st.write("**Sample Segments**")
                analysis.rat.plot_traj(0, -1, analysis.segments)
                st.pyplot(plt.gcf())
                plt.close()