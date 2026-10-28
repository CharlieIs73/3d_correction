import streamlit as st
import numpy as np
import networkx as nx
import plotly.graph_objects as go

st.set_page_config(page_title="Red Topológica Interestelar · 3D", layout="wide")
st.title("Red Topológica Interestelar — prototipo 3D")
st.caption("Modelo exploratorio: las reglas de propagación son heurísticas, no leyes físicas validadas.")

if "simulation_result" not in st.session_state:
    st.session_state.simulation_result = None
if "simulation_key" not in st.session_state:
    st.session_state.simulation_key = None

@st.cache_data
def build_graph(n, k, seed):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(-10, 10, size=(n, 3))
    graph = nx.Graph()
    for i, p in enumerate(coords):
        graph.add_node(i, pos=p)
    for i in range(n):
        distances = np.linalg.norm(coords - coords[i], axis=1)
        for j in np.argsort(distances)[1:k+1]:
            graph.add_edge(i, int(j), length=float(distances[j]))
    return graph, coords

def simulate(n, k, seed, selected_node, impulse, steps, stiffness, damping, threshold):
    graph, coords = build_graph(n, k, seed)
    strain = np.zeros(n, dtype=float)
    strain[selected_node] = impulse
    broken_edges = set()
    for _ in range(steps):
        next_strain = strain * damping
        for u, v in graph.edges():
            if (u, v) in broken_edges or (v, u) in broken_edges:
                continue
            transfer = min(strain[u], strain[v]) * 0.12 * stiffness
            next_strain[v] += transfer
            next_strain[u] += transfer
        strain = np.clip(next_strain, 0.0, 10.0)
        for u, v in graph.edges():
            if (strain[u] + strain[v]) / max(0.1, 2.0 * stiffness) > threshold:
                broken_edges.add((u, v))
    return graph, coords, strain, broken_edges

with st.sidebar:
    st.header("Configuración de la red")
    n_nodes = st.slider("Número de nodos", 8, 60, 24, 2)
    k_neighbors = st.slider("Conexiones aproximadas por nodo", 1, 6, 3)
    seed = st.number_input("Semilla aleatoria", min_value=0, max_value=99999, value=12, step=1)
    stiffness = st.slider("Rigidez de corredores", 0.2, 3.0, 1.0, 0.1)
    damping = st.slider("Amortiguamiento", 0.0, 0.95, 0.25, 0.05)
    threshold = st.slider("Umbral de ruptura", 0.5, 3.0, 1.6, 0.1)
    st.divider()
    st.header("Perturbación")
    selected_node = st.number_input("Nodo perturbado (ID)", min_value=0, max_value=n_nodes-1, value=0, step=1)
    impulse = st.slider("Intensidad inicial", 0.1, 4.0, 1.8, 0.1)
    steps = st.slider("Pasos de propagación", 0, 20, 5)
    inject = st.button("⚡ Inyectar perturbación", type="primary", use_container_width=True)
    reset = st.button("↺ Reiniciar simulación", use_container_width=True)

key = (n_nodes, k_neighbors, int(seed), int(selected_node), float(impulse), int(steps), float(stiffness), float(damping), float(threshold))
if reset:
    st.session_state.simulation_result = None
    st.session_state.simulation_key = None
if inject:
    st.session_state.simulation_result = simulate(n_nodes, k_neighbors, int(seed), int(selected_node), float(impulse), int(steps), float(stiffness), float(damping), float(threshold))
    st.session_state.simulation_key = key

if st.session_state.simulation_result is None:
    G, coords = build_graph(n_nodes, k_neighbors, int(seed))
    strain = np.zeros(n_nodes)
    broken_edges = set()
    st.info("Ajusta los parámetros y pulsa **⚡ Inyectar perturbación** para ejecutar la simulación.")
else:
    if st.session_state.simulation_key != key:
        st.session_state.simulation_result = simulate(n_nodes, k_neighbors, int(seed), int(selected_node), float(impulse), int(steps), float(stiffness), float(damping), float(threshold))
        st.session_state.simulation_key = key
    G, coords, strain, broken_edges = st.session_state.simulation_result

active_x, active_y, active_z, broken_x, broken_y, broken_z = [], [], [], [], [], []
for u, v in G.edges():
    broken = (u, v) in broken_edges or (v, u) in broken_edges
    target = (broken_x, broken_y, broken_z) if broken else (active_x, active_y, active_z)
    target[0].extend([coords[u, 0], coords[v, 0], None])
    target[1].extend([coords[u, 1], coords[v, 1], None])
    target[2].extend([coords[u, 2], coords[v, 2], None])

fig = go.Figure()
fig.add_trace(go.Scatter3d(x=active_x, y=active_y, z=active_z, mode="lines", line=dict(color="rgba(100,190,240,0.8)", width=4), hoverinfo="skip", name="Corredores activos"))
fig.add_trace(go.Scatter3d(x=broken_x, y=broken_y, z=broken_z, mode="lines", line=dict(color="rgba(255,70,70,0.95)", width=6, dash="dash"), hoverinfo="skip", name="Corredores rotos"))
fig.add_trace(go.Scatter3d(x=coords[:, 0], y=coords[:, 1], z=coords[:, 2], mode="markers+text", text=[str(i) for i in range(n_nodes)], textposition="top center", marker=dict(size=6 + np.minimum(strain, 5)*2, color=strain, colorscale="Plasma", showscale=True, colorbar=dict(title="Tensión"), line=dict(width=1, color="white")), customdata=np.column_stack([strain, np.array([G.degree(i) for i in range(n_nodes)])]), hovertemplate="Nodo %{text}<br>Tensión: %{customdata[0]:.2f}<br>Conexiones: %{customdata[1]}<extra></extra>", name="Nodos"))
fig.update_layout(height=680, margin=dict(l=0, r=0, t=25, b=0), scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="cube", bgcolor="rgba(10,15,30,0.98)"), paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#E8EEF8"), legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="left", x=0))

left, right = st.columns([2.2, 1])
with left:
    st.plotly_chart(fig, use_container_width=True)
with right:
    st.subheader("Estado de la red")
    active_graph = G.copy()
    active_graph.remove_edges_from(list(broken_edges))
    st.metric("Nodos", n_nodes)
    st.metric("Corredores totales", G.number_of_edges())
    st.metric("Corredores marcados como rotos", len(broken_edges))
    st.metric("Componentes conectados", nx.number_connected_components(active_graph))
    st.metric("Tensión máxima", f"{float(np.max(strain)):.2f}")
    st.markdown("**Cómo leerlo**")
    st.markdown("- **Nodos:** anclas o puntos de tránsito.\n- **Líneas azules:** corredores activos.\n- **Líneas rojas discontinuas:** corredores que superaron el umbral.\n- **Color/tamaño:** tensión estimada del nodo.")
    st.warning("La propagación y la ruptura son reglas simplificadas para experimentar, no física validada.")
st.divider()
st.caption("Puedes rotar la escena 3D con el ratón y acercar/alejar con la rueda.")
