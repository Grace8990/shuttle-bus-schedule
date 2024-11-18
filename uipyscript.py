import streamlit as st
import pandas as pd
import numpy as np
import folium
from gurobipy import Model, GRB, quicksum
from streamlit_folium import st_folium

# Initialize the application
st.title("Shuttle Bus Route Optimization")

# Depot Location (Linden St, Allston, Boston, MA)
depot = {"Latitude": 42.3527, "Longitude": -71.1280}

# Input: Number of Passengers and Destinations
st.sidebar.header("Add Destinations")
num_destinations = st.sidebar.number_input("Number of destinations", min_value=1, max_value=20, value=5)
destinations = []

for i in range(num_destinations):
    st.sidebar.subheader(f"Destination {i + 1}")
    latitude = st.sidebar.number_input(f"Latitude {i + 1}", value=42.35 + 0.001 * i)
    longitude = st.sidebar.number_input(f"Longitude {i + 1}", value=-71.13 + 0.001 * i)
    num_people = st.sidebar.number_input(f"Passengers to Destination {i + 1}", min_value=1, max_value=14, value=5)
    destinations.append({"Latitude": latitude, "Longitude": longitude, "Passengers": num_people})

# Button to Optimize Routes
if st.sidebar.button("Optimize Routes"):
    # Combine depot and destinations
    all_locations = [{"Latitude": depot["Latitude"], "Longitude": depot["Longitude"], "Passengers": 0}] + destinations
    num_locations = len(all_locations)
    
    # Extract coordinates and demands
    xc = np.array([loc["Latitude"] for loc in all_locations])
    yc = np.array([loc["Longitude"] for loc in all_locations])
    demands = [loc["Passengers"] for loc in all_locations]

    # Create arcs and costs
    A = [(i, j) for i in range(num_locations) for j in range(num_locations) if i != j]
    c = {(i, j): np.hypot(xc[i] - xc[j], yc[i] - yc[j]) for i, j in A}

    # Vehicle constraints
    vehicle_capacities = [14] * 7 + [4]  # 7 buses (14 seats each) + 1 car (4 seats)
    num_vehicles = len(vehicle_capacities)
    Q = max(vehicle_capacities)

    # Optimization Model
    mdl = Model("CVRP")
    x = mdl.addVars(A, vtype=GRB.BINARY)
    u = mdl.addVars(range(1, num_locations), vtype=GRB.CONTINUOUS)

    # Objective: Minimize total travel distance
    mdl.modelSense = GRB.MINIMIZE
    mdl.setObjective(quicksum(x[a] * c[a] for a in A))

    # Constraints
    mdl.addConstrs(quicksum(x[i, j] for j in range(num_locations) if j != i) == 1 for i in range(1, num_locations))
    mdl.addConstrs(quicksum(x[j, i] for j in range(num_locations) if j != i) == 1 for i in range(1, num_locations))
    mdl.addConstrs((x[i, j] == 1) >> (u[i - 1] + demands[i] == u[j - 1]) for i, j in A if i != 0 and j != 0)
    mdl.addConstrs(u[i - 1] >= demands[i] for i in range(1, num_locations))
    mdl.addConstrs(u[i - 1] <= Q for i in range(1, num_locations))

    # Solve
    mdl.Params.MIPGap = 0.1
    mdl.Params.TimeLimit = 30  # seconds
    mdl.optimize()

    # Extract solution
    active_arcs = [a for a in A if x[a].x > 0.99]
    st.success("Optimization Completed!")

    # Create a map
    route_map = folium.Map(location=[depot["Latitude"], depot["Longitude"]], zoom_start=12)
    
    # Plot depot
    folium.Marker(
        location=[depot["Latitude"], depot["Longitude"]],
        popup="Depot",
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(route_map)

    # Plot destinations
    for i, loc in enumerate(destinations):
        folium.Marker(
            location=[loc["Latitude"], loc["Longitude"]],
            popup=f"Destination {i + 1} ({loc['Passengers']} passengers)",
            icon=folium.Icon(color="blue"),
        ).add_to(route_map)

    # Plot routes
    for i, j in active_arcs:
        folium.PolyLine(
            locations=[[xc[i], yc[i]], [xc[j], yc[j]]],
            color="green",
            weight=2.5,
        ).add_to(route_map)

    # Display the map
    st_folium(route_map, width=800, height=600)
