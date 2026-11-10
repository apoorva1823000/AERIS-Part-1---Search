import math, random, heapq, itertools, time, base64
from collections import deque
from dataclasses import dataclass
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import io
import networkx as nx
import plotly.graph_objects as go
import streamlit as st


# AERIS - AI-Enabled Emergency Relief and Intelligent Supply

st.set_page_config(page_title="AERIS Medical Relief", page_icon="🚑", layout="wide")


# DOMAIN OBJECTS

@dataclass
class Supply:
    name: str
    unit_weight_kg: float
    fragile: bool
    cold_chain: bool
    drop_allowed: bool
    urgency: float

@dataclass
class UAV:
    name: str
    battery_pct: float
    payload_kg: float
    pos: Tuple[float, float]
    charge_rate_pct_min: float

@dataclass
class Truck:
    name: str
    capacity_kg: float
    pos: str

SUPPLIES = {
    "Bandage kit": Supply("Bandage kit", 1.2, False, False, True, 0.55),
    "Trauma kit": Supply("Trauma kit", 2.0, False, False, True, 0.96),
    "Blood units": Supply("Blood units", 3.0, True, True, False, 1.00),
    "Insulin cold box": Supply("Insulin cold box", 1.5, True, True, False, 0.92),
    "Water pack": Supply("Water pack", 5.0, False, False, True, 0.45),
}

UAV_MAX_WIND = 12.0  # DJI M350 RTK published reference envelope
UAV_MIN_TEMP = -20.0
UAV_MAX_TEMP = 50.0
SAFE_BATTERY_RESERVE = 15.0


# DISASTER CONTROLS

def disaster_controls():
    dtype = st.selectbox("Disaster type", ["Earthquake", "Flood", "Cyclone"])
    seed = st.number_input("Simulation seed", 1, 99999, 17, 1)
    p = {"type": dtype, "seed": int(seed)}
    if dtype == "Earthquake":
        p["mw"] = st.slider("Moment magnitude Mw", 4.0, 9.0, 6.7, 0.1)
        p["depth_km"] = st.slider("Focal depth (km)", 2.0, 60.0, 14.0, 1.0)
        p["local_shaking"] = st.slider("Local ground-motion severity (normalized evidence)", 0.0, 1.0, 0.60, 0.05)
        p["secondary_hazard"] = st.slider("Debris / landslide secondary-hazard severity", 0.0, 1.0, 0.30, 0.05)
        p["wind"] = st.slider("Ambient wind (m/s)", 0.0, 15.0, 3.0, 0.5)
    elif dtype == "Flood":
        p["water_depth_m"] = st.slider("Representative water depth (m)", 0.0, 2.5, 0.70, 0.05)
        p["flow"] = st.slider("Flow severity", 0.0, 1.0, 0.45, 0.05)
        p["rain"] = st.slider("Rainfall severity", 0.0, 1.0, 0.55, 0.05)
        p["wind"] = st.slider("Ambient wind (m/s)", 0.0, 15.0, 4.0, 0.5)
    else:
        p["wind"] = st.slider("Sustained wind (m/s)", 0.0, 25.0, 9.0, 0.5)
        p["gust"] = st.slider("Gust speed (m/s)", 0.0, 35.0, 14.0, 0.5)
        p["rain"] = st.slider("Rainfall severity", 0.0, 1.0, 0.45, 0.05)
        p["debris"] = st.slider("Debris obstruction severity", 0.0, 1.0, 0.35, 0.05)
    return p

def normalized_severity(p):
    if p["type"] == "Earthquake":
        # User-visible summary only; not a damage law.
        mag = np.clip((p["mw"] - 4.0) / 5.0, 0, 1)
        shallow = np.clip((35 - p["depth_km"]) / 35, 0, 1)
        return float(np.clip(.35*mag + .15*shallow + .35*p["local_shaking"] + .15*p["secondary_hazard"], 0, 1))
    if p["type"] == "Flood":
        return float(np.clip(.50*min(1,p["water_depth_m"]/1.5) + .25*p["flow"] + .25*p["rain"], 0, 1))
    return float(np.clip(.45*min(1,p["wind"]/20) + .25*min(1,p["gust"]/30) + .15*p["rain"] + .15*p["debris"], 0, 1))


# CNN-STYLE SPATIAL PERCEPTION

def conv2d_same(img, kernel):
    pad = kernel.shape[0]//2
    x = np.pad(img, pad, mode="edge")
    out = np.zeros_like(img, dtype=float)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i,j] = np.sum(x[i:i+kernel.shape[0], j:j+kernel.shape[1]] * kernel)
    return out

def build_hazard_raster(p, seed, size=28):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size,0:size]
    sev = normalized_severity(p)
    # Synthetic spatial evidence field: correlated hotspot + local texture.
    cx, cy = rng.uniform(8,20), rng.uniform(8,20)
    sx, sy = rng.uniform(4,8), rng.uniform(4,8)
    hotspot = np.exp(-(((xx-cx)/sx)**2 + ((yy-cy)/sy)**2)/2)
    texture = rng.normal(0,0.12,(size,size))
    if p["type"] == "Flood":
        lowland = np.linspace(1,0,size)[:,None]
        base = .55*hotspot + .45*lowland
    elif p["type"] == "Cyclone":
        stripes = .5*(np.sin(xx/3 + yy/5)+1)
        base = .60*hotspot + .40*stripes
    else:
        base = hotspot
    raw = np.clip(sev*base + texture + .08, 0, 1)
    # A tiny fixed convolutional perception stage emphasizing local discontinuity/risk.
    kernel = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=float)
    edge = np.abs(conv2d_same(raw, kernel))
    perceived = np.clip(.72*raw + .28*(edge/(edge.max()+1e-9)), 0, 1)
    return raw, perceived


# WORLD GENERATION

def build_world(p):
    rng = random.Random(p["seed"])
    sev = normalized_severity(p)
    raw_grid, cnn_grid = build_hazard_raster(p, p["seed"])

    coords = {
        "Depot": (0,3), "A":(2,5), "B":(2,2), "C":(4,6), "D":(4,3),
        "E":(4,0.8), "F":(6,5.2), "G":(6,2.2), "H":(8,6), "I":(8,3),
        "J":(8,0.6), "Hospital":(10,5.5), "Relief Camp":(10,2.5), "Village":(10,0.5),
        "Charge-1":(1.1,6.7), "Charge-2":(7.0,0.2)
    }
    road_nodes = ["Depot","A","B","C","D","E","F","G","H","I","J","Hospital","Relief Camp","Village"]
    # asset tags create physically different corridors: a fewer-hop bridge/lowland
    # shortcut may be less desirable than a longer but more reliable road.
    edges = [
        ("Depot","A","road"),("Depot","B","road"),("A","C","road"),("A","D","road"),
        ("B","D","road"),("B","E","road"),("C","F","road"),("D","F","road"),
        ("D","G","road"),("E","G","lowland"),("F","H","road"),("F","I","road"),
        ("G","I","road"),("G","J","road"),("H","Hospital","road"),("I","Hospital","road"),
        ("I","Relief Camp","road"),("J","Relief Camp","road"),("J","Village","road"),
        ("I","Village","road"),
        # fewer-hop shortcuts that can become costly under hazard
        ("Depot","D","bridge"),("B","G","lowland"),("D","I","bridge"),("G","Hospital","bridge")
    ]

    G = nx.Graph()
    for n in road_nodes:
        G.add_node(n, pos=coords[n])

    def raster_evidence(pt):
        x,y = pt
        ix = int(np.clip(round(x/10*(cnn_grid.shape[1]-1)),0,cnn_grid.shape[1]-1))
        iy = int(np.clip(round((7-y)/7*(cnn_grid.shape[0]-1)),0,cnn_grid.shape[0]-1))
        return float(cnn_grid[iy,ix])

    for a,b,asset in edges:
        dist = math.dist(coords[a],coords[b])
        mid=((coords[a][0]+coords[b][0])/2,(coords[a][1]+coords[b][1])/2)
        vision = raster_evidence(mid)
        exposure = .55*vision + .45*rng.random()

        # Educational infrastructure-susceptibility modifiers.
        # They are scenario modifiers, not claimed fragility curves.
        asset_penalty = 0.0
        if asset=="bridge":
            asset_penalty = {"Earthquake":.18,"Flood":.15,"Cyclone":.05}[p["type"]]
        elif asset=="lowland":
            asset_penalty = {"Earthquake":.03,"Flood":.24,"Cyclone":.10}[p["type"]]

        if p["type"]=="Earthquake":
            hazard = np.clip(.50*sev + .28*exposure + .14*p["secondary_hazard"] + asset_penalty,0,1.4)
            p_open = 1/(1+math.exp(6*(hazard-.63)))
            speed = max(.20,1-.62*hazard)
        elif p["type"]=="Flood":
            lowland=max(0,(3.2-mid[1])/3.2)
            hazard=np.clip(.40*sev+.25*exposure+.25*lowland*min(1,p["water_depth_m"]/1.1)+asset_penalty,0,1.4)
            p_open=1/(1+math.exp(7*(hazard-.56)))
            speed=max(.10,1-.78*hazard)
        else:
            hazard=np.clip(.42*sev+.28*exposure+.22*p["debris"]+asset_penalty,0,1.4)
            p_open=1/(1+math.exp(6*(hazard-.66)))
            speed=max(.18,1-.60*hazard)

        p_open=float(np.clip(p_open,.02,.995))
        exp_time=dist/max(.07,speed)
        risk=-math.log(p_open)
        money=.35*dist
        cost=exp_time + 2.3*risk + .45*money
        G.add_edge(a,b,distance=dist,p_open=p_open,time=exp_time,risk=risk,cost=cost,
                   vision=vision,asset=asset)

    uavs=[
        UAV("UAV-1", 47-rng.uniform(0,5), 5.0, (0.3,3.4), 3.0),      # near warehouse, marginal battery
        UAV("UAV-2", 94-rng.uniform(0,5), 8.0, coords["Charge-1"], 3.2), # at charging station
        UAV("UAV-3", 82-rng.uniform(0,8), 12.0, (6.3,1.0), 2.8),
    ]
    trucks=[Truck("Truck-1",350,"Depot"),Truck("Truck-2",250,"I")]

    requests={
        "Hospital":{"supply":"Trauma kit","quantity":3,"priority":1.00},
        "Relief Camp":{"supply":"Water pack","quantity":10,"priority":.62},
        "Village":{"supply":"Bandage kit","quantity":3,"priority":.76},
    }
    return {
        "params":p,"severity":sev,"coords":coords,"road":G,"uavs":uavs,"trucks":trucks,
        "requests":requests,"raw_grid":raw_grid,"cnn_grid":cnn_grid,
        "charging":["Charge-1","Charge-2"],
        "node_types":{
            "Depot":"Medical depot","Hospital":"Hospital","Relief Camp":"Relief camp",
            "Village":"Demand point","Charge-1":"Charging station","Charge-2":"Charging station"
        }
    }


def recompute_edge_from_evidence(world, a, b, asset="road", manual_damage=None, report_blocked=None, report_confidence=.5):
    """
    Convert local evidence into a traversability probability.
    This is a transparent educational fusion model, not a calibrated civil-engineering fragility model.
    """
    p=world["params"]; sev=world["severity"]; coords=world["coords"]
    mid=((coords[a][0]+coords[b][0])/2,(coords[a][1]+coords[b][1])/2)

    # Base hazard susceptibility by asset type.
    asset_penalty=0.0
    if asset=="bridge":
        asset_penalty={"Earthquake":.18,"Flood":.15,"Cyclone":.05}[p["type"]]
    elif asset=="lowland":
        asset_penalty={"Earthquake":.03,"Flood":.24,"Cyclone":.10}[p["type"]]

    # Evidence score can be user/field supplied. If absent, use the spatial raster only as simulated evidence.
    if manual_damage is None:
        grid=world["cnn_grid"]
        ix=int(np.clip(round(mid[0]/max(1,max(x for x,y in coords.values()))*(grid.shape[1]-1)),0,grid.shape[1]-1))
        maxy=max(1,max(y for x,y in coords.values()))
        iy=int(np.clip(round((maxy-mid[1])/maxy*(grid.shape[0]-1)),0,grid.shape[0]-1))
        local_damage=float(grid[iy,ix])
        evidence_source="simulated spatial evidence"
    else:
        local_damage=float(np.clip(manual_damage,0,1))
        evidence_source="user/field evidence"

    if p["type"]=="Earthquake":
        hazard=np.clip(.52*sev+.28*local_damage+.12*p.get("secondary_hazard",0)+asset_penalty,0,1.4)
        prior_open=1/(1+math.exp(6*(hazard-.63)))
        speed=max(.20,1-.62*hazard)
    elif p["type"]=="Flood":
        lowland=max(0,(3.2-mid[1])/3.2)
        hazard=np.clip(.40*sev+.26*local_damage+.24*lowland*min(1,p.get("water_depth_m",0)/1.1)+asset_penalty,0,1.4)
        prior_open=1/(1+math.exp(7*(hazard-.56)))
        speed=max(.10,1-.78*hazard)
    else:
        hazard=np.clip(.42*sev+.28*local_damage+.22*p.get("debris",0)+asset_penalty,0,1.4)
        prior_open=1/(1+math.exp(6*(hazard-.66)))
        speed=max(.18,1-.60*hazard)

    # Simple evidence update from a field report.
    p_open=float(np.clip(prior_open,.02,.995))
    if report_blocked is not None:
        c=float(np.clip(report_confidence,0,1))
        if report_blocked:
            p_open=(1-c)*p_open + c*.05
        else:
            p_open=(1-c)*p_open + c*.95

    dist=math.dist(coords[a],coords[b])
    exp_time=dist/max(.07,speed)
    risk=-math.log(max(.01,p_open))
    money=.35*dist
    cost=exp_time+2.3*risk+.45*money
    return {
        "distance":dist,"p_open":p_open,"time":exp_time,"risk":risk,"cost":cost,
        "vision":local_damage,"asset":asset,"evidence_source":evidence_source,
        "report_blocked":report_blocked,"report_confidence":report_confidence
    }

def add_or_update_edge(world, a, b, asset="road", manual_damage=.3, report_state="Unknown", report_confidence=.5):
    rb=None if report_state=="Unknown" else (report_state=="Blocked")
    data=recompute_edge_from_evidence(world,a,b,asset,manual_damage,rb,report_confidence)
    world["road"].add_edge(a,b,**data)

def rebuild_all_edge_probabilities(world):
    """Recompute graph costs/probabilities after disaster or evidence changes while preserving topology."""
    for a,b,d in list(world["road"].edges(data=True)):
        md=d.get("manual_damage", d.get("vision",.3))
        rs=d.get("report_state","Unknown")
        rb=None if rs=="Unknown" else (rs=="Blocked")
        nd=recompute_edge_from_evidence(world,a,b,d.get("asset","road"),md,rb,d.get("report_confidence",.5))
        nd["manual_damage"]=md; nd["report_state"]=rs
        world["road"][a][b].update(nd)

def node_table(world):
    rows=[]
    special_types={}
    special_types["Depot"]="Medical depot"
    for c in world["charging"]: special_types[c]="Charging station"
    for t in ["Hospital","Relief Camp","Village"]:
        if t in world["coords"]: special_types[t]="Demand point"
    for n,(x,y) in world["coords"].items():
        rows.append({"Node":n,"Type":special_types.get(n,"Road/staging node"),"X":x,"Y":y,
                     "In road graph":n in world["road"]})
    return pd.DataFrame(rows)

def edge_table(world):
    return pd.DataFrame([{
        "A":a,"B":b,"Asset":d.get("asset","road"),
        "P(open)":round(d.get("p_open",0),3),
        "Damage evidence":round(d.get("vision",0),3),
        "Report":d.get("report_state","Unknown"),
        "Report confidence":round(d.get("report_confidence",.5),2),
        "Cost":round(d.get("cost",0),2)
    } for a,b,d in world["road"].edges(data=True)])


# GRAPH SEARCH - FOCUSED A* IMPLEMENTATION

def min_edge_cost_per_distance(G):
    vals=[]
    for a,b,d in G.edges(data=True):
        dist=max(float(d.get("distance",1.0)),1e-9)
        vals.append(float(d.get("cost",dist))/dist)
    return min(vals) if vals else 1.0

def heuristic_value(G, node, goal, kind="cost_lower_bound"):
    """
    Heuristics used in the focused A* implementation.

    cost_lower_bound:
        straight-line distance * minimum observed edge-cost-per-distance.
        This is admissible and consistent when all edge costs are non-negative,
        because every path must cover at least the straight-line distance.

    euclidean:
        raw straight-line distance. In the built-in AERIS road model edge cost
        is >= geometric distance, so this is also a lower bound.

    zero:
        h(n)=0, which makes A* equivalent to UCS.
    """
    if node==goal:
        return 0.0
    eu=math.dist(G.nodes[node]["pos"],G.nodes[goal]["pos"])
    if kind=="zero":
        return 0.0
    if kind=="euclidean":
        return float(eu)
    return float(eu*min_edge_cost_per_distance(G))

def heuristic_diagnostics(G, goal, kind="cost_lower_bound"):
    """Check admissibility against exact shortest costs and consistency on all edges."""
    h={n:heuristic_value(G,n,goal,kind) for n in G.nodes}
    # exact remaining path cost for an educational audit
    exact={}
    for n in G.nodes:
        try:
            exact[n]=nx.shortest_path_length(G,n,goal,weight="cost")
        except Exception:
            exact[n]=float("inf")
    admissible=all(h[n] <= exact[n] + 1e-9 for n in G.nodes if np.isfinite(exact[n]))
    violations=[]
    for a,b,d in G.edges(data=True):
        c=float(d.get("cost",0))
        if h[a] > c+h[b]+1e-9:
            violations.append((a,b,h[a],c+h[b]))
        if h[b] > c+h[a]+1e-9:
            violations.append((b,a,h[b],c+h[a]))
    return {
        "admissible":admissible,
        "consistent":len(violations)==0,
        "violations":violations,
        "h":h,
        "exact_remaining_cost":exact
    }

def reconstruct(parent, goal):
    if goal not in parent:return []
    p=[];c=goal
    while c is not None:
        p.append(c);c=parent[c]
    return p[::-1]

def graph_search(G,start,goal,algo="A*",heuristic_kind=None,weight=None):
    """
    Deep search implementation used throughout AERIS.

    Supported:
      BFS
      UCS
      A*
      Weighted A*  (f=g+w*h)
    Legacy DFS/IDDFS/Greedy remain accepted for compatibility.
    """
    if heuristic_kind is None:
        heuristic_kind=st.session_state.get("search_heuristic","cost_lower_bound")
    if weight is None:
        weight=float(st.session_state.get("search_weight",1.0))

    parent={start:None}; g={start:0.0}; expanded=[]; steps=[]
    hvals={start:heuristic_value(G,start,goal,heuristic_kind)}
    fvals={start:hvals[start]}

    if algo=="BFS":
        q=deque([start]); seen={start}
        while q:
            n=q.popleft()
            expanded.append(n)
            steps.append({
                "expanded":n,
                "frontier":list(q),
                "closed":expanded.copy(),
                "g":dict(g),"h":dict(hvals),"f":dict(fvals)
            })
            if n==goal:break
            for nb in G.neighbors(n):
                if nb not in seen:
                    seen.add(nb); parent[nb]=n
                    g[nb]=g[n]+float(G[n][nb]["cost"])
                    hvals[nb]=heuristic_value(G,nb,goal,heuristic_kind)
                    fvals[nb]=g[nb]+hvals[nb]
                    q.append(nb)

    elif algo in ["DFS","IDDFS"]:
        # Retained only for backward compatibility with the broader v5 interface.
        if algo=="DFS":
            stack=[start]; seen={start}
            while stack:
                n=stack.pop(); expanded.append(n)
                steps.append({"expanded":n,"frontier":stack.copy(),"closed":expanded.copy(),
                              "g":dict(g),"h":dict(hvals),"f":dict(fvals)})
                if n==goal:break
                for nb in G.neighbors(n):
                    if nb not in seen:
                        seen.add(nb);parent[nb]=n
                        g[nb]=g[n]+float(G[n][nb]["cost"])
                        hvals[nb]=heuristic_value(G,nb,goal,heuristic_kind)
                        fvals[nb]=g[nb]+hvals[nb]
                        stack.append(nb)
        else:
            found=None
            for lim in range(len(G.nodes)+1):
                stack=[(start,0,[start])]
                while stack:
                    n,d,path=stack.pop();expanded.append(n)
                    steps.append({"expanded":n,"frontier":[x[0] for x in stack],
                                  "closed":expanded.copy(),"g":dict(g),"h":dict(hvals),"f":dict(fvals)})
                    if n==goal:
                        found=path;break
                    if d<lim:
                        for nb in G.neighbors(n):
                            if nb not in path:stack.append((nb,d+1,path+[nb]))
                if found:break
            if found:
                parent={found[0]:None};g={found[0]:0.0}
                for a,b in zip(found,found[1:]):
                    parent[b]=a;g[b]=g[a]+float(G[a][b]["cost"])
                    hvals[b]=heuristic_value(G,b,goal,heuristic_kind)
                    fvals[b]=g[b]+hvals[b]
            else:
                parent={}

    else:
        # UCS, Greedy, A*, Weighted A*
        pq=[]; closed=set(); counter=0
        if algo=="UCS":
            start_f=0.0
        elif algo=="Greedy":
            start_f=hvals[start]
        elif algo=="Weighted A*":
            start_f=g[start]+weight*hvals[start]
        else:
            start_f=g[start]+hvals[start]
        heapq.heappush(pq,(start_f,counter,start))

        while pq:
            _,_,n=heapq.heappop(pq)
            if n in closed:continue
            closed.add(n);expanded.append(n)

            frontier_nodes=[item[2] for item in pq]
            steps.append({
                "expanded":n,
                "frontier":frontier_nodes,
                "closed":list(closed),
                "g":dict(g),"h":dict(hvals),"f":dict(fvals)
            })
            if n==goal:break

            for nb in G.neighbors(n):
                ng=g[n]+float(G[n][nb]["cost"])
                if ng < g.get(nb,float("inf")):
                    g[nb]=ng;parent[nb]=n
                    h=heuristic_value(G,nb,goal,heuristic_kind)
                    hvals[nb]=h
                    if algo=="UCS":
                        f=ng
                    elif algo=="Greedy":
                        f=h
                    elif algo=="Weighted A*":
                        f=ng+weight*h
                    else:
                        f=ng+h
                    fvals[nb]=f
                    counter+=1
                    heapq.heappush(pq,(f,counter,nb))

    path=reconstruct(parent,goal)
    cost=sum(float(G[a][b]["cost"]) for a,b in zip(path,path[1:])) if path else float("inf")
    dist=sum(float(G[a][b]["distance"]) for a,b in zip(path,path[1:])) if path else float("inf")
    nominal_rel=float(np.prod([G[a][b]["p_open"] for a,b in zip(path,path[1:])])) if path else 0.
    diagnostics=heuristic_diagnostics(G,goal,heuristic_kind) if algo in ["A*","Weighted A*"] else None

    return {
        "path":path,"cost":cost,"distance":dist,"nominal_reliability":nominal_rel,
        "expanded":expanded,"steps":steps,"g":g,"h":hvals,"f":fvals,
        "heuristic_kind":heuristic_kind,"weight":weight,"diagnostics":diagnostics
    }

def monte_carlo_accessibility(G,start,goal,algo="A*",runs=400,seed=1):
    rng=random.Random(seed);success=0;costs=[];paths={}
    for _ in range(runs):
        H=nx.Graph();H.add_nodes_from(G.nodes(data=True))
        for a,b,d in G.edges(data=True):
            if rng.random()<d["p_open"]:
                H.add_edge(a,b,**d)
        if start in H and goal in H and nx.has_path(H,start,goal):
            r=graph_search(H,start,goal,algo)
            if r["path"]:
                success+=1;costs.append(r["cost"])
                key="→".join(r["path"]);paths[key]=paths.get(key,0)+1
    return {
        "accessibility":success/runs,
        "mean_cost":float(np.mean(costs)) if costs else float("inf"),
        "p95_cost":float(np.percentile(costs,95)) if costs else float("inf"),
        "common_path":max(paths,key=paths.get) if paths else "No path",
        "runs":runs
    }


# UAV ENERGY / CHARGING

def uav_weather(world):
    p=world["params"]
    return float(p.get("wind",3.0)), 28.0

def air_distance(a,b):
    return math.dist(a,b)

def energy_mean_pct(distance, payload_kg, wind_mps):
    """
    Educational proxy, not a certified airframe power model.
    Structure is grounded in literature showing energy/range depend on distance,
    payload, speed and wind; coefficients are explicitly simulation coefficients.
    """
    base_per_unit = 2.25
    payload_factor = 1 + .075*payload_kg
    wind_factor = 1 + .030*wind_mps**1.35
    return distance*base_per_unit*payload_factor*wind_factor

def mission_energy_distribution(uav, world, target, supply, qty, samples=600):
    rng=np.random.default_rng(world["params"]["seed"] + sum(map(ord,uav.name)))
    wind,temp=uav_weather(world);payload=supply.unit_weight_kg*qty
    coords=world["coords"]
    warehouse=coords["Depot"];target_pos=coords[target]
    charge_positions={c:coords[c] for c in world["charging"]}

    # Mission chain: current -> depot -> target -> nearest charging station.
    nearest_charge=min(charge_positions,key=lambda c:air_distance(target_pos,charge_positions[c]))
    d0=air_distance(uav.pos,warehouse)
    d1=air_distance(warehouse,target_pos)
    d2=air_distance(target_pos,charge_positions[nearest_charge])

    # No payload on pickup leg; full payload on delivery; empty on recharge leg.
    means=[
        energy_mean_pct(d0,0,wind),
        energy_mean_pct(d1,payload,wind),
        energy_mean_pct(d2,0,wind)
    ]
    nominal=sum(means)

    # Cyclone / high-wind -> larger variance: small detours/gusts matter more.
    disaster_var={"Earthquake":.07,"Flood":.10,"Cyclone":.18}[world["params"]["type"]]
    wind_var=.01*wind
    sigma_frac=min(.35,disaster_var+wind_var)
    samples_energy=np.zeros(samples)
    for i in range(samples):
        seg=0.
        for m in means:
            seg += max(0,rng.normal(m,max(.35,m*sigma_frac)))
        samples_energy[i]=seg
    completion=float(np.mean(samples_energy + SAFE_BATTERY_RESERVE <= uav.battery_pct))
    p95=float(np.percentile(samples_energy,95))
    expected=float(samples_energy.mean())

    # Expected mission time + potential pre-charge time.
    speed=max(.8,4.6-.10*wind)
    flight_time=(d0+d1+d2)/speed
    return {
        "nearest_charge":nearest_charge,"legs":[d0,d1,d2],"payload":payload,
        "expected_energy":expected,"p95_energy":p95,"success_prob":completion,
        "nominal_energy":nominal,"flight_time":flight_time,
        "wind":wind,"temp":temp,"samples":samples_energy,
        "chain":f"{uav.name} → Depot → {target} → {nearest_charge}"
    }

def best_uav_plan(world,target,supply,qty):
    wind,temp=uav_weather(world)
    rows=[]
    for u in world["uavs"]:
        reasons=[]
        payload=supply.unit_weight_kg*qty
        if payload>u.payload_kg: reasons.append("payload capacity")
        if wind>UAV_MAX_WIND: reasons.append("wind exceeds reference envelope")
        if not(UAV_MIN_TEMP<=temp<=UAV_MAX_TEMP): reasons.append("temperature outside reference envelope")
        m=mission_energy_distribution(u,world,target,supply,qty)
        # allow low success as candidate but mark unsafe; utility will penalize heavily.
        if m["success_prob"]<.80:reasons.append("mission-completion probability below 0.80")
        utility = 130*supply.urgency*m["success_prob"] - 1.5*m["flight_time"] - .60*m["expected_energy"]
        rows.append({"uav":u,"mission":m,"reasons":reasons,"utility":utility,"feasible":len(reasons)==0})
    feasible=[r for r in rows if r["feasible"]]
    best=max(feasible,key=lambda r:r["utility"]) if feasible else None
    return best,rows

def recharge_option(uav, world, target, supply, qty):
    coords=world["coords"];wind,_=uav_weather(world)
    options=[]
    for cs in world["charging"]:
        # current -> charging -> depot -> target -> nearest charging
        d_to_cs=air_distance(uav.pos,coords[cs])
        e_to_cs=energy_mean_pct(d_to_cs,0,wind)
        if e_to_cs+8>uav.battery_pct:
            continue
        charge_needed=max(0,95-(uav.battery_pct-e_to_cs))
        charge_time=charge_needed/uav.charge_rate_pct_min
        temp_uav=UAV(uav.name,95,uav.payload_kg,coords[cs],uav.charge_rate_pct_min)
        m=mission_energy_distribution(temp_uav,world,target,supply,qty)
        total_time=d_to_cs/max(.8,4.6-.1*wind)+charge_time+m["flight_time"]
        options.append({"station":cs,"total_time":total_time,"success_prob":m["success_prob"],"mission":m})
    return min(options,key=lambda x:x["total_time"]) if options else None


# ROAD / HYBRID MODE

def road_plan(world,target,algo,runs):
    r=graph_search(world["road"],"Depot",target,algo)
    mc=monte_carlo_accessibility(world["road"],"Depot",target,algo,runs,world["params"]["seed"])
    if not r["path"]:
        return None
    # simple vehicle economics for educational comparison
    time=r["distance"]/1.45 + 4.0*r["cost"]/max(1,r["distance"])
    money=.9*r["distance"]
    utility=105*mc["accessibility"] - 2.0*time - .7*money
    return {"route":r,"mc":mc,"time":time,"money":money,"utility":utility}

def mc_accessibility_between(G,start,goal,algo,runs,seed):
    return monte_carlo_accessibility(G,start,goal,algo,runs,seed)

def uav_leg_stats(world, uav, start_pos, end_pos, payload_kg, include_post_charge=True, samples=450):
    """Probabilistic energy/reliability for one UAV transport leg plus optional post-leg charging reach."""
    wind,_=uav_weather(world)
    rng=np.random.default_rng(world["params"]["seed"] + sum(map(ord,uav.name)) + 31)
    d_loaded=air_distance(start_pos,end_pos)
    nearest_charge=min(world["charging"], key=lambda c:air_distance(end_pos,world["coords"][c]))
    d_charge=air_distance(end_pos,world["coords"][nearest_charge]) if include_post_charge else 0.0
    m_loaded=energy_mean_pct(d_loaded,payload_kg,wind)
    m_charge=energy_mean_pct(d_charge,0,wind)
    disaster_var={"Earthquake":.07,"Flood":.10,"Cyclone":.18}[world["params"]["type"]]
    sigma_frac=min(.35,disaster_var+.01*wind)
    arr=[]
    for _ in range(samples):
        e=max(0,rng.normal(m_loaded,max(.35,m_loaded*sigma_frac)))
        if include_post_charge:
            e+=max(0,rng.normal(m_charge,max(.25,m_charge*sigma_frac)))
        arr.append(e)
    arr=np.asarray(arr)
    p=float(np.mean(arr+SAFE_BATTERY_RESERVE<=uav.battery_pct))
    return {"distance":d_loaded,"to_charge":d_charge,"energy":float(arr.mean()),
            "p95_energy":float(np.percentile(arr,95)),"success_prob":p,
            "nearest_charge":nearest_charge}

def hybrid_plan(world,target,supply,qty,algo,runs):
    """
    Compare both hybrid directions:
      1) TRUCK→UAV: depot truck leg to a staging point, UAV final-mile to target.
      2) UAV→TRUCK: UAV crosses the difficult first portion to a stationary field truck,
         then truck completes the last-mile road leg.
    """
    payload=supply.unit_weight_kg*qty
    wind,_=uav_weather(world)
    if wind>UAV_MAX_WIND:
        return None

    candidates=[]
    staging_nodes=["F","G","I","J"]

    # TRUCK -> UAV
    for s in staging_nodes:
        rr=graph_search(world["road"],"Depot",s,algo)
        if not rr["path"]:
            continue
        mc=mc_accessibility_between(world["road"],"Depot",s,algo,max(100,runs//2),
                                    world["params"]["seed"]+7)
        for u in world["uavs"]:
            if payload>u.payload_kg:
                continue
            # UAV must first reposition empty to staging, then fly loaded to destination, then reach charge.
            reposition=energy_mean_pct(air_distance(u.pos,world["coords"][s]),0,wind)
            temp_batt=u.battery_pct-reposition
            if temp_batt<=SAFE_BATTERY_RESERVE:
                continue
            temp_u=UAV(u.name,temp_batt,u.payload_kg,world["coords"][s],u.charge_rate_pct_min)
            air=uav_leg_stats(world,temp_u,world["coords"][s],world["coords"][target],payload)
            rel=mc["accessibility"]*air["success_prob"]
            road_time=rr["distance"]/1.45
            air_time=(air_distance(u.pos,world["coords"][s])+air["distance"]+air["to_charge"])/max(.8,4.6-.1*wind)
            time=road_time+air_time
            cost=rr["cost"]+.65*(reposition+air["energy"])
            util=122*supply.urgency*rel - 1.8*time - .55*cost
            candidates.append({
                "direction":"TRUCK→UAV","staging":s,"truck":"Truck-1","uav":u.name,
                "reliability":rel,"time":time,"cost":cost,"utility":util,
                "road":rr,"road_mc":mc,"air":air,"uav_energy":reposition+air["energy"],
                "handoff":"Truck unloads at stationary staging zone; UAV performs final delivery."
            })

    # UAV -> TRUCK
    # Use trucks already positioned in the field. Handoff is to a stationary truck-side zone,
    # never a moving truck.
    for truck in world["trucks"]:
        if truck.pos=="Depot":
            continue
        s=truck.pos
        rr=graph_search(world["road"],s,target,algo)
        if not rr["path"]:
            continue
        mc=mc_accessibility_between(world["road"],s,target,algo,max(100,runs//2),
                                    world["params"]["seed"]+19)
        for u in world["uavs"]:
            if payload>u.payload_kg:
                continue
            # UAV: current -> depot (empty) -> field truck (loaded) -> charging.
            d_pick=air_distance(u.pos,world["coords"]["Depot"])
            e_pick=energy_mean_pct(d_pick,0,wind)
            temp_batt=u.battery_pct-e_pick
            if temp_batt<=SAFE_BATTERY_RESERVE:
                continue
            temp_u=UAV(u.name,temp_batt,u.payload_kg,world["coords"]["Depot"],u.charge_rate_pct_min)
            air=uav_leg_stats(world,temp_u,world["coords"]["Depot"],world["coords"][s],payload)
            rel=air["success_prob"]*mc["accessibility"]
            air_time=(d_pick+air["distance"]+air["to_charge"])/max(.8,4.6-.1*wind)
            road_time=rr["distance"]/1.45
            time=air_time+road_time
            cost=rr["cost"]+.65*(e_pick+air["energy"])
            util=118*supply.urgency*rel - 1.8*time - .55*cost
            candidates.append({
                "direction":"UAV→TRUCK","staging":s,"truck":truck.name,"uav":u.name,
                "reliability":rel,"time":time,"cost":cost,"utility":util,
                "road":rr,"road_mc":mc,"air":air,"uav_energy":e_pick+air["energy"],
                "handoff":"UAV hands off at a stationary truck-side receiving zone; truck completes final road delivery."
            })

    return max(candidates,key=lambda x:x["utility"]) if candidates else None


# DELIVERY MODE / CSP

def delivery_csp(world,target,supply,qty):
    best,all_uavs=best_uav_plan(world,target,supply,qty)
    domains={
        "Transport":["ROAD","UAV","HYBRID"],
        "DeliveryMode":["Road handover","Land","Lower","Drop"]
    }
    removed=[]
    if supply.fragile or supply.cold_chain or not supply.drop_allowed:
        if "Drop" in domains["DeliveryMode"]:
            domains["DeliveryMode"].remove("Drop")
            removed.append(("DeliveryMode","Drop","handling / cold-chain constraint"))
    if uav_weather(world)[0]>UAV_MAX_WIND:
        for x in ["UAV","HYBRID"]:
            if x in domains["Transport"]:
                domains["Transport"].remove(x);removed.append(("Transport",x,"wind outside reference UAV envelope"))
    if not any(x["feasible"] for x in all_uavs):
        if "UAV" in domains["Transport"]:
            domains["Transport"].remove("UAV");removed.append(("Transport","UAV","no UAV satisfies payload/battery/reliability constraints"))
    return domains,removed,best,all_uavs


# PROBABILISTIC REASONING / ROAD-ACCESSIBILITY BN

# Course-level Bayesian Network focused on the same road-edge belief used by A*.
#
# HazardSevere  ----\
# VisualDamage  -----> RoadOpen
# FieldBlocked  ----/
#
# CPTs are transparent educational scenario parameters, not field-calibrated disaster statistics.
BN_ORDER=["HazardSevere","VisualDamage","FieldBlocked","RoadOpen"]
BN_PARENTS={
    "HazardSevere":[],
    "VisualDamage":[],
    "FieldBlocked":[],
    "RoadOpen":["HazardSevere","VisualDamage","FieldBlocked"],
}
BN_CPT={
    "HazardSevere":{():.45},
    "VisualDamage":{():.35},
    "FieldBlocked":{():.25},
    "RoadOpen":{
        (False,False,False):.96,
        (False,False,True):.38,
        (False,True,False):.58,
        (False,True,True):.18,
        (True,False,False):.72,
        (True,False,True):.24,
        (True,True,False):.36,
        (True,True,True):.08,
    },
}

def bn_p(var,val,e):
    p=BN_CPT[var][tuple(e[x] for x in BN_PARENTS[var])]
    return p if val else 1-p

def enum_all(vars_,e):
    if not vars_:
        return 1.0
    y=vars_[0]
    if y in e:
        return bn_p(y,e[y],e)*enum_all(vars_[1:],e)
    return sum(
        bn_p(y,v,{**e,y:v})*enum_all(vars_[1:],{**e,y:v})
        for v in [True,False]
    )

def enumeration_ask(query,evidence):
    d={q:enum_all(BN_ORDER,{**evidence,query:q}) for q in [True,False]}
    z=sum(d.values())
    return {k:v/z for k,v in d.items()}

def _factor_from_cpt(var,evidence):
    scope=[x for x in BN_PARENTS[var]+[var] if x not in evidence]
    rows={}
    full_scope=BN_PARENTS[var]+[var]
    for vals in itertools.product([False,True],repeat=len(full_scope)):
        a=dict(zip(full_scope,vals))
        if any(k in evidence and a[k]!=evidence[k] for k in evidence if k in a):
            continue
        key=tuple(a[x] for x in scope)
        rows[key]=bn_p(var,a[var],a)
    return {"scope":scope,"rows":rows}

def _factor_product(factors):
    if not factors:
        return {"scope":[],"rows":{():1.0}}
    out=factors[0]
    for f in factors[1:]:
        scope=list(dict.fromkeys(out["scope"]+f["scope"]))
        rows={}
        for vals in itertools.product([False,True],repeat=len(scope)):
            a=dict(zip(scope,vals))
            k1=tuple(a[x] for x in out["scope"])
            k2=tuple(a[x] for x in f["scope"])
            if k1 in out["rows"] and k2 in f["rows"]:
                rows[vals]=out["rows"][k1]*f["rows"][k2]
        out={"scope":scope,"rows":rows}
    return out

def _sum_out(var,f):
    if var not in f["scope"]:
        return f
    scope=[x for x in f["scope"] if x!=var]
    rows={}
    for vals in itertools.product([False,True],repeat=len(scope)):
        a=dict(zip(scope,vals))
        total=0.0
        for v in [False,True]:
            aa={**a,var:v}
            key=tuple(aa[x] for x in f["scope"])
            total+=f["rows"].get(key,0.0)
        rows[vals]=total
    return {"scope":scope,"rows":rows}

def variable_elimination(query,evidence):
    factors=[_factor_from_cpt(v,evidence) for v in BN_ORDER]
    trace=[]
    hidden=[v for v in BN_ORDER if v not in evidence and v!=query]
    for h in hidden:
        hit=[f for f in factors if h in f["scope"]]
        factors=[f for f in factors if h not in f["scope"]]
        prod=_factor_product(hit)
        reduced=_sum_out(h,prod)
        factors.append(reduced)
        trace.append({
            "eliminated":h,
            "combined_factors":len(hit),
            "product_scope":prod["scope"],
            "result_scope":reduced["scope"]
        })
    final=_factor_product(factors)
    if query not in final["scope"]:
        # query fixed by evidence is not expected in UI, but keep function total.
        val=bool(evidence[query])
        return {True:1.0 if val else 0.0,False:0.0 if val else 1.0},trace
    qi=final["scope"].index(query)
    scores={}
    for qv in [False,True]:
        total=0.0
        for key,val in final["rows"].items():
            if key[qi]==qv:
                total+=val
        scores[qv]=total
    z=sum(scores.values())
    return {k:v/z for k,v in scores.items()},trace

def bn_evidence_from_world_edge(world,a,b):
    d=world["road"][a][b]
    severe=bool(world.get("severity",0.5)>=0.58)
    visual=bool(float(d.get("vision",0.0))>=0.55)
    # If a prior field flag exists, use it; otherwise derive only a UI default, not from p_open.
    field_blocked=bool(d.get("field_blocked",False))
    return {
        "HazardSevere":severe,
        "VisualDamage":visual,
        "FieldBlocked":field_blocked
    }

def apply_bn_belief_to_edge(world,a,b,evidence):
    ans_enum=enumeration_ask("RoadOpen",evidence)
    ans_ve,trace=variable_elimination("RoadOpen",evidence)
    p_open=float(ans_ve[True])
    d=world["road"][a][b]
    d["p_open"]=float(np.clip(p_open,.02,.995))
    d["risk"]=-math.log(d["p_open"])
    dist=float(d.get("distance",1.0))
    d["cost"]=float(d.get("time",dist)) + 2.3*d["risk"] + .45*(.35*dist)
    d["bn_evidence"]=dict(evidence)
    d["bn_p_open"]=d["p_open"]
    d["evidence_source"]="Bayesian Network"
    return ans_enum,ans_ve,trace


# HMM

HMM_STATES=["Open","Degraded","Blocked"]
T=np.array([[.80,.17,.03],[.18,.63,.19],[.05,.24,.71]])
E={"Clear":np.array([.84,.14,.02]),"Slow":np.array([.17,.66,.17]),"Obstructed":np.array([.03,.22,.75])}

def hmm_filter(obs):
    b=np.array([.52,.33,.15]);hist=[]
    for o in obs:
        b=(b@T)*E[o];b/=b.sum();hist.append(b.copy())
    return hist

def viterbi(obs):
    dp=np.log(np.array([.52,.33,.15])+1e-12)+np.log(E[obs[0]]+1e-12);backs=[]
    for o in obs[1:]:
        nd=np.zeros(3);bp=np.zeros(3,dtype=int)
        for j in range(3):
            s=dp+np.log(T[:,j]+1e-12);bp[j]=int(np.argmax(s));nd[j]=s[bp[j]]+np.log(E[o][j]+1e-12)
        dp=nd;backs.append(bp)
    last=int(np.argmax(dp));path=[last]
    for bp in backs[::-1]:
        last=int(bp[last]);path.append(last)
    return [HMM_STATES[i] for i in path[::-1]]


# VALUE OF INFORMATION

def value_of_information(p_blocked, verify_cost=7.0):
    # Illustrative decision model for learning, not emergency policy.
    u_road_open=35;u_road_block=-35;u_uav=18
    eu_road=(1-p_blocked)*u_road_open+p_blocked*u_road_block
    eu_now=max(eu_road,u_uav)
    eu_perfect=(1-p_blocked)*max(u_road_open,u_uav)+p_blocked*max(u_road_block,u_uav)
    vpi=eu_perfect-eu_now
    return {"EU_road":eu_road,"EU_uav":u_uav,"EU_now":eu_now,"EU_perfect":eu_perfect,
            "VPI":vpi,"verify_cost":verify_cost,"verify":vpi>verify_cost}


# FOCUSED LOGISTICS MDP

MDP_ROAD=["LOW","MEDIUM","HIGH"]
MDP_BATTERY=["LOW","MEDIUM","HIGH"]
MDP_WIND=["SAFE","MARGINAL","UNSAFE"]
MDP_URGENCY=["NORMAL","CRITICAL"]
MDP_CHARGE=["NO","YES"]
MDP_STAGE=["DEPOT","ENROUTE"]

MDP_ACTIONS=["ROAD","UAV","HYBRID","VERIFY","RECHARGE","REROUTE","WAIT"]
MDP_TERMINALS=["DELIVERED","FAILED"]

def _bin_road(p):
    return "LOW" if p<.40 else ("MEDIUM" if p<.75 else "HIGH")

def _bin_battery(p):
    return "LOW" if p<40 else ("MEDIUM" if p<75 else "HIGH")

def _bin_wind(w):
    return "SAFE" if w<7 else ("MARGINAL" if w<=UAV_MAX_WIND else "UNSAFE")

def focused_mdp_states():
    normal=list(itertools.product(MDP_ROAD,MDP_BATTERY,MDP_WIND,MDP_URGENCY,MDP_CHARGE,MDP_STAGE))
    return normal + MDP_TERMINALS

def focused_current_state(world,target,supply,qty,road_obj=None,best_uav=None):
    if road_obj is None:
        road_obj=road_plan(world,target,"A*",250)
    road_p=road_obj["mc"]["accessibility"] if road_obj else 0.0

    if best_uav is not None:
        battery=best_uav["uav"].battery_pct
    else:
        battery=max([u.battery_pct for u in world["uavs"]],default=0)

    wind,_=uav_weather(world)
    urgency=max(float(supply.urgency),float(world["requests"].get(target,{}).get("priority",.5)))
    return (
        _bin_road(road_p),
        _bin_battery(battery),
        _bin_wind(wind),
        "CRITICAL" if urgency>=.8 else "NORMAL",
        "YES" if len(world["charging"])>0 else "NO",
        "DEPOT"
    )

def _state_delivery_reward(urgency, success=True):
    if success:
        return 160.0 if urgency=="CRITICAL" else 100.0
    return -190.0 if urgency=="CRITICAL" else -120.0

def _road_success(road):
    return {"LOW":.25,"MEDIUM":.62,"HIGH":.90}[road]

def _uav_success(battery,wind):
    b={"LOW":.32,"MEDIUM":.72,"HIGH":.94}[battery]
    wm={"SAFE":1.0,"MARGINAL":.78,"UNSAFE":.08}[wind]
    return float(np.clip(b*wm,.02,.98))

def _hybrid_success(road,battery,wind):
    # Requires both road and UAV portions; deliberately conservative.
    return float(np.clip((_road_success(road)**.55)*(_uav_success(battery,wind)**.55),.02,.96))

def _next_road_after_wait(road):
    if road=="LOW":
        return [("LOW",.68),("MEDIUM",.27),("HIGH",.05)]
    if road=="MEDIUM":
        return [("LOW",.20),("MEDIUM",.60),("HIGH",.20)]
    return [("LOW",.05),("MEDIUM",.22),("HIGH",.73)]

def focused_transitions(state,action):
    """
    Returns list of (next_state, probability, reward).
    State is fully observable and Markov by construction: the transition depends only
    on the current discretized logistics state and action.
    """
    if state in MDP_TERMINALS:
        return [(state,1.0,0.0)]

    road,batt,wind,urg,charge,stage=state
    out=[]

    if action=="ROAD":
        ps=_road_success(road)
        fail_state=(road,batt,wind,urg,charge,"ENROUTE")
        out=[
            ("DELIVERED",ps,_state_delivery_reward(urg,True)-8),
            (fail_state,1-ps,_state_delivery_reward(urg,False)*.22-16)
        ]

    elif action=="UAV":
        ps=_uav_success(batt,wind)
        # failed UAV attempt consumes battery
        nb="LOW" if batt in ["LOW","MEDIUM"] else "MEDIUM"
        fail_state=(road,nb,wind,urg,charge,"ENROUTE")
        out=[
            ("DELIVERED",ps,_state_delivery_reward(urg,True)-14),
            (fail_state,1-ps,_state_delivery_reward(urg,False)*.28-20)
        ]

    elif action=="HYBRID":
        ps=_hybrid_success(road,batt,wind)
        nb="LOW" if batt=="MEDIUM" else ("MEDIUM" if batt=="HIGH" else "LOW")
        fail_state=(road,nb,wind,urg,charge,"ENROUTE")
        out=[
            ("DELIVERED",ps,_state_delivery_reward(urg,True)-18),
            (fail_state,1-ps,_state_delivery_reward(urg,False)*.24-22)
        ]

    elif action=="VERIFY":
        # Verification is most valuable in MEDIUM uncertainty.
        if road=="MEDIUM":
            out=[
                (("HIGH",batt,wind,urg,charge,stage),.58,-7),
                (("LOW",batt,wind,urg,charge,stage),.42,-7)
            ]
        elif road=="LOW":
            out=[
                (("LOW",batt,wind,urg,charge,stage),.82,-7),
                (("MEDIUM",batt,wind,urg,charge,stage),.18,-7)
            ]
        else:
            out=[
                (("HIGH",batt,wind,urg,charge,stage),.86,-7),
                (("MEDIUM",batt,wind,urg,charge,stage),.14,-7)
            ]

    elif action=="RECHARGE":
        if charge=="YES":
            out=[((road,"HIGH",wind,urg,charge,stage),1.0,-10)]
        else:
            out=[((road,batt,wind,urg,charge,stage),1.0,-18)]

    elif action=="REROUTE":
        # Rerouting may find a more reliable alternative corridor.
        if road=="LOW":
            out=[
                (("MEDIUM",batt,wind,urg,charge,"ENROUTE"),.58,-9),
                (("LOW",batt,wind,urg,charge,"ENROUTE"),.42,-9)
            ]
        elif road=="MEDIUM":
            out=[
                (("HIGH",batt,wind,urg,charge,"ENROUTE"),.45,-9),
                (("MEDIUM",batt,wind,urg,charge,"ENROUTE"),.55,-9)
            ]
        else:
            out=[((road,batt,wind,urg,charge,"ENROUTE"),1.0,-9)]

    else: # WAIT
        # Road condition evolves; normal urgency can become critical with delay.
        urg_options=[(urg,1.0)] if urg=="CRITICAL" else [("NORMAL",.82),("CRITICAL",.18)]
        for nr,pr in _next_road_after_wait(road):
            for nu,pu in urg_options:
                out.append(((nr,batt,wind,nu,charge,stage),pr*pu,-6))

    # normalize possible floating error
    z=sum(p for _,p,_ in out)
    return [(sp,p/z,r) for sp,p,r in out]

def focused_mdp_model():
    states=focused_mdp_states()
    P={}; R={}
    for s in states:
        for a in MDP_ACTIONS:
            tr=focused_transitions(s,a)
            P[(s,a)]=[(sp,p) for sp,p,r in tr]
            R[(s,a)]=[(sp,r) for sp,p,r in tr]
    return states,P,R

def focused_q_value(state,action,V,gamma):
    return sum(p*(r + gamma*V[sp]) for sp,p,r in focused_transitions(state,action))

def focused_value_iteration(gamma=.92,epsilon=1e-6,max_iter=500):
    states=focused_mdp_states()
    V={s:0.0 for s in states}; history=[]
    for k in range(max_iter):
        NV={}
        delta=0.0
        for s in states:
            if s in MDP_TERMINALS:
                NV[s]=0.0
            else:
                NV[s]=max(focused_q_value(s,a,V,gamma) for a in MDP_ACTIONS)
            delta=max(delta,abs(NV[s]-V[s]))
        V=NV
        history.append({"iteration":k+1,"max_delta":delta})
        if delta<epsilon:
            break
    policy={}
    for s in states:
        if s in MDP_TERMINALS:
            policy[s]="TERMINAL"
        else:
            policy[s]=max(MDP_ACTIONS,key=lambda a:focused_q_value(s,a,V,gamma))
    return V,policy,pd.DataFrame(history)

def focused_policy_iteration(gamma=.92,eval_epsilon=1e-7,max_iter=100):
    states=focused_mdp_states()
    policy={s:("TERMINAL" if s in MDP_TERMINALS else "ROAD") for s in states}
    trace=[]
    for outer in range(max_iter):
        V={s:0.0 for s in states}
        # iterative policy evaluation
        for _ in range(1000):
            delta=0.0; NV={}
            for s in states:
                if s in MDP_TERMINALS:
                    NV[s]=0.0
                else:
                    NV[s]=focused_q_value(s,policy[s],V,gamma)
                delta=max(delta,abs(NV[s]-V[s]))
            V=NV
            if delta<eval_epsilon:break

        stable=True; changes=0
        for s in states:
            if s in MDP_TERMINALS:continue
            old=policy[s]
            new=max(MDP_ACTIONS,key=lambda a:focused_q_value(s,a,V,gamma))
            policy[s]=new
            if new!=old:
                stable=False;changes+=1
        trace.append({"iteration":outer+1,"policy_changes":changes})
        if stable:break
    return V,policy,pd.DataFrame(trace)

def simulate_mdp_policy(start_state,policy,steps=10,seed=1):
    rng=np.random.default_rng(seed);s=start_state;rows=[]
    total=0.0
    for t in range(steps):
        if s in MDP_TERMINALS:
            rows.append({"t":t,"state":str(s),"action":"-","reward":0.0,"next_state":str(s)})
            break
        a=policy[s]
        tr=focused_transitions(s,a)
        probs=np.array([x[1] for x in tr],dtype=float); probs/=probs.sum()
        idx=int(rng.choice(len(tr),p=probs))
        sp,p,r=tr[idx]; total+=r
        rows.append({"t":t,"state":str(s),"action":a,"reward":round(r,2),"next_state":str(sp)})
        s=sp
    return pd.DataFrame(rows),total

def apply_hmm_belief_to_edge(world,a,b,observations):
    """
    Couple the course HMM filtering module to the actual road graph.
    The filtered belief becomes the edge's new traversability belief.
    """
    hist=hmm_filter(observations)
    belief=hist[-1]
    # Open contributes fully, Degraded partially, Blocked zero.
    p_open=float(np.clip(belief[0] + .45*belief[1],.02,.995))
    d=world["road"][a][b]
    d["p_open"]=p_open
    d["risk"]=-math.log(p_open)
    dist=float(d.get("distance",1.0))
    d["cost"]=float(d.get("time",dist)) + 2.3*d["risk"] + .45*(.35*dist)
    d["hmm_belief"]={HMM_STATES[i]:float(belief[i]) for i in range(3)}
    d["evidence_source"]="HMM filtered temporal observations"
    return d["hmm_belief"]


# PRECISION RELEASE (SIMULATION ONLY)

def simulate_drop_3d(height, speed, elevation_deg, azimuth_deg, windx, windy,
                     mass, cd=.8, area=.03, dt=.01, return_trace=False):
    """
    Educational 3-D projectile simulation with gravity + quadratic drag + horizontal wind.
    elevation_deg: launch angle above horizontal.
    azimuth_deg: horizontal direction, measured from +x.
    This is NOT a real flight-control model.
    """
    el=math.radians(elevation_deg); az=math.radians(azimuth_deg)
    vx=speed*math.cos(el)*math.cos(az)
    vy=speed*math.cos(el)*math.sin(az)
    vz=speed*math.sin(el)

    rho=1.225; x=y=0.; z=float(height); t=0.
    trace=[(t,x,y,z)]
    while z>0 and t<40:
        rv=np.array([vx-windx,vy-windy,vz],dtype=float)
        sp=float(np.linalg.norm(rv))
        drag=-.5*rho*cd*area*sp*rv/max(.10,mass)
        ax,ay,az_acc=drag[0],drag[1],-9.81+drag[2]
        vx+=ax*dt; vy+=ay*dt; vz+=az_acc*dt
        x+=vx*dt; y+=vy*dt; z+=vz*dt; t+=dt
        if return_trace and (len(trace)==0 or t-trace[-1][0]>=.04):
            trace.append((t,x,y,max(0,z)))
    impact=math.sqrt(vx*vx+vy*vy+vz*vz)
    if return_trace:
        trace.append((t,x,y,0.))
        return x,y,impact,t,trace
    return x,y,impact,t

def product_release_rules(supply):
    reasons=[]
    if not supply.drop_allowed:
        reasons.append("supply handling policy does not allow free release")
    if supply.fragile:
        reasons.append("fragile payload")
    if supply.cold_chain:
        reasons.append("cold-chain payload")
    return len(reasons)==0,reasons

def optimize_release(dx,dy,height,windx,windy,supply):
    """
    Search both elevation and azimuth, plus release speed.
    No fixed 45-degree assumption.
    """
    allowed,reasons=product_release_rules(supply)
    if not allowed:
        return {"allowed":False,"reasons":reasons}

    target_az=math.degrees(math.atan2(dy,dx))
    best=None
    # bounded coursework search, not a control law
    for sp in np.linspace(.5,7.0,18):
        for elev in np.linspace(-10,55,18):
            for azoff in np.linspace(-35,35,19):
                az=target_az+azoff
                x,y,impact,t=simulate_drop_3d(
                    height,sp,elev,az,windx,windy,
                    supply.unit_weight_kg,cd=.8,area=.03,dt=.015
                )
                err=math.hypot(x-dx,y-dy)
                # impact is penalized, not treated as equivalent to accuracy
                score=err + .12*impact
                if best is None or score<best["score"]:
                    best={"allowed":True,"speed":float(sp),"elevation_deg":float(elev),
                          "azimuth_deg":float(az),"landing":(float(x),float(y)),
                          "error":float(err),"impact":float(impact),"time":float(t),
                          "score":float(score)}
    return best

def release_monte_carlo(best, dx, dy, height, windx, windy, supply,
                        target_radius=3.0, samples=500, seed=1):
    """
    Perturbs wind, release speed, and release angles to estimate landing confidence.
    """
    if not best or not best.get("allowed",False):
        return None
    rng=np.random.default_rng(seed)
    pts=[]; impacts=[]
    for _ in range(samples):
        wx=rng.normal(windx,max(.25,abs(windx)*.15+.25))
        wy=rng.normal(windy,max(.25,abs(windy)*.15+.25))
        sp=max(.05,rng.normal(best["speed"],.12*max(1,best["speed"])))
        el=rng.normal(best["elevation_deg"],2.0)
        az=rng.normal(best["azimuth_deg"],2.5)
        x,y,impact,t=simulate_drop_3d(
            height,sp,el,az,wx,wy,supply.unit_weight_kg,
            cd=.8,area=.03,dt=.015
        )
        pts.append((x,y)); impacts.append(impact)
    pts=np.asarray(pts); impacts=np.asarray(impacts)
    errs=np.sqrt((pts[:,0]-dx)**2+(pts[:,1]-dy)**2)
    return {
        "points":pts,
        "impacts":impacts,
        "p_inside":float(np.mean(errs<=target_radius)),
        "mean_error":float(errs.mean()),
        "p95_error":float(np.percentile(errs,95)),
        "p95_impact":float(np.percentile(impacts,95)),
        "target_radius":float(target_radius)
    }

def release_trajectory_fig(best,dx,dy,height,windx,windy,supply):
    x,y,impact,t,tr=simulate_drop_3d(
        height,best["speed"],best["elevation_deg"],best["azimuth_deg"],
        windx,windy,supply.unit_weight_kg,return_trace=True
    )
    arr=np.asarray(tr)
    horizontal=np.sqrt(arr[:,1]**2+arr[:,2]**2)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=horizontal,y=arr[:,3],mode="lines",name="Predicted trajectory"))
    fig.add_trace(go.Scatter(x=[0],y=[height],mode="markers+text",text=["UAV release"],textposition="top right",name="Release"))
    fig.add_trace(go.Scatter(x=[math.hypot(dx,dy)],y=[0],mode="markers+text",text=["Target drop zone"],textposition="top right",name="Target"))
    fig.add_trace(go.Scatter(x=[math.hypot(x,y)],y=[0],mode="markers+text",text=["Nominal landing"],textposition="bottom right",name="Landing"))
    fig.update_layout(height=360,xaxis_title="Horizontal displacement (m)",yaxis_title="Height (m)",
                      title="Side-view release trajectory")
    return fig

def release_top_fig(mc,dx,dy):
    pts=mc["points"]
    th=np.linspace(0,2*np.pi,180)
    r=mc["target_radius"]
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=pts[:,0],y=pts[:,1],mode="markers",
                             marker=dict(size=5,opacity=.30),name="Monte Carlo landings"))
    fig.add_trace(go.Scatter(x=dx+r*np.cos(th),y=dy+r*np.sin(th),mode="lines",name="Allowed target zone"))
    fig.add_trace(go.Scatter(x=[dx],y=[dy],mode="markers+text",text=["Target"],textposition="top center",name="Target center"))
    fig.update_layout(height=420,xaxis_title="X offset (m)",yaxis_title="Y offset (m)",
                      yaxis=dict(scaleanchor="x"),title="Top view: landing uncertainty")
    return fig


def disaster_wind_vector(world):
    """Create a deterministic horizontal wind vector from the incident wind magnitude."""
    mag=float(world["params"].get("wind",3.0))
    rng=np.random.default_rng(world["params"]["seed"]+404)
    ang=float(rng.uniform(0,2*np.pi))
    return mag*math.cos(ang),mag*math.sin(ang)

def integrated_release_context(world, candidate, target, supply, qty):
    """
    Couple precision-release reasoning to the transport plan.
    ROAD -> no release simulator.
    UAV -> final destination.
    TRUCK→UAV -> final destination.
    UAV→TRUCK -> stationary truck-side receiving zone at staging point.
    """
    mode=candidate["mode"]
    if mode=="ROAD":
        return {"applicable":False,"reason":"The selected plan is fully road-based; no UAV release occurs."}

    detail=candidate["detail"]
    if mode=="UAV":
        context="FINAL_DESTINATION"
        end_name=target
        start_name="Depot"
        receiver="designated clear medical drop zone"
    else:
        direction=detail["direction"]
        if direction=="TRUCK→UAV":
            context="FINAL_DESTINATION"
            start_name=detail["staging"]
            end_name=target
            receiver="designated clear medical drop zone"
        else:
            context="TRUCK_HANDOFF"
            start_name="Depot"
            end_name=detail["staging"]
            receiver="stationary truck-side receiving zone"

    allowed,reasons=product_release_rules(supply)
    windx,windy=disaster_wind_vector(world)

    # Geometry is derived from the selected route/handoff, not user-created as a toy.
    p0=world["coords"][start_name]
    p1=world["coords"][end_name]
    vx=p1[0]-p0[0]; vy=p1[1]-p0[1]
    norm=max(1e-6,math.hypot(vx,vy))
    approach=(vx/norm,vy/norm)

    # A release point is placed a short, deterministic distance before the receiving zone.
    release_distance=5.0 + 2.0*world["severity"]
    dx=approach[0]*release_distance
    dy=approach[1]*release_distance
    height=8.0 + 3.0*world["severity"]
    target_radius=2.5 if context=="TRUCK_HANDOFF" else 3.5

    if not allowed:
        # Handling constraints dominate the projectile model.
        action="LOWER/TETHER" if world["params"].get("wind",0)<8 else "LAND/CONTROLLED HANDOFF"
        return {"applicable":True,"context":context,"receiver":receiver,"release_allowed":False,
                "reasons":reasons,"selected_action":action,"start_name":start_name,"end_name":end_name}

    best=optimize_release(dx,dy,height,windx,windy,supply)
    mc=release_monte_carlo(best,dx,dy,height,windx,windy,supply,target_radius,500,
                           seed=world["params"]["seed"]+91)
    # Course-project geometry threshold. Package handling remains a separate constraint.
    reliable=bool(mc and mc["p_inside"]>=.90)
    if reliable:
        action="CONTROLLED RELEASE"
    elif world["params"].get("wind",0)<8:
        action="LOWER/TETHER"
    else:
        action="REPOSITION / LAND"

    return {
        "applicable":True,"context":context,"receiver":receiver,"release_allowed":True,
        "selected_action":action,"start_name":start_name,"end_name":end_name,
        "dx":dx,"dy":dy,"height":height,"windx":windx,"windy":windy,
        "target_radius":target_radius,"best":best,"mc":mc
    }

def mode_candidates_for_algorithm(world,target,supply,qty,algo,runs):
    road=road_plan(world,target,algo,runs)
    hybrid=hybrid_plan(world,target,supply,qty,algo,runs)
    _,_,best_uav,_=delivery_csp(world,target,supply,qty)
    candidates=[]
    if road:
        candidates.append({"mode":"ROAD","utility":road["utility"],
                           "reliability":road["mc"]["accessibility"],"time":road["time"],"detail":road})
    if best_uav:
        m=best_uav["mission"]
        candidates.append({"mode":"UAV","utility":best_uav["utility"],
                           "reliability":m["success_prob"],"time":m["flight_time"],"detail":best_uav})
    if hybrid:
        candidates.append({"mode":"HYBRID","utility":hybrid["utility"],
                           "reliability":hybrid["reliability"],"time":hybrid["time"],"detail":hybrid})
    return sorted(candidates,key=lambda x:x["utility"],reverse=True), road, hybrid, best_uav

def search_algorithm_comparison(world,target,supply,qty,runs):
    rows=[]
    for algo in ["BFS","UCS","A*","Weighted A*"]:
        cands,road,hybrid,bestu=mode_candidates_for_algorithm(
            world,target,supply,qty,algo,max(120,min(runs,300))
        )
        if not cands:
            rows.append({"Algorithm":algo,"Recommended mode":"NONE","Road path":"-",
                         "Road cost":None,"Expanded":None,"Winner utility":None})
            continue
        road_path=" → ".join(road["route"]["path"]) if road and road["route"]["path"] else "-"
        rows.append({
            "Algorithm":algo,
            "Recommended mode":cands[0]["mode"],
            "Hybrid direction":hybrid["direction"] if hybrid else "-",
            "Road path":road_path,
            "Road cost":round(road["route"]["cost"],2) if road else None,
            "Expanded":len(road["route"]["expanded"]) if road else None,
            "Winner utility":round(cands[0]["utility"],2)
        })
    return pd.DataFrame(rows)


def search_benchmark(world,target):
    """Compare focused search methods on the same current road graph."""
    rows=[]
    for algo in ["BFS","UCS","A*","Weighted A*"]:
        t0=time.perf_counter()
        old_w=st.session_state.get("search_weight",1.0)
        if algo=="Weighted A*":
            st.session_state.search_weight=1.5
        r=graph_search(world["road"],"Depot",target,algo)
        dt=(time.perf_counter()-t0)*1000.0
        if algo=="Weighted A*":
            st.session_state.search_weight=old_w
        dg=r.get("diagnostics")
        rows.append({
            "Algorithm":algo,
            "Path":" → ".join(r["path"]) if r["path"] else "No path",
            "Cost":round(r["cost"],3) if np.isfinite(r["cost"]) else None,
            "Expanded":len(r["expanded"]),
            "Runtime ms":round(dt,3),
            "Reliability":round(r["nominal_reliability"],3),
            "Admissible":dg["admissible"] if dg else "-",
            "Consistent":dg["consistent"] if dg else "-"
        })
    return pd.DataFrame(rows)

def heuristic_benchmark(world,target):
    rows=[]
    for kind,label in [("zero","h=0 (UCS)"),
                       ("euclidean","Euclidean"),
                       ("cost_lower_bound","Cost lower bound")]:
        t0=time.perf_counter()
        r=graph_search(world["road"],"Depot",target,"A*",heuristic_kind=kind)
        dt=(time.perf_counter()-t0)*1000.0
        dg=heuristic_diagnostics(world["road"],target,kind)
        rows.append({
            "Heuristic":label,
            "Path":" → ".join(r["path"]) if r["path"] else "No path",
            "Cost":round(r["cost"],3) if np.isfinite(r["cost"]) else None,
            "Expanded":len(r["expanded"]),
            "Runtime ms":round(dt,3),
            "Admissible":dg["admissible"],
            "Consistent":dg["consistent"]
        })
    return pd.DataFrame(rows)

def immediate_static_action(state):
    """One-shot baseline: choose only among delivery modes using immediate expected reward."""
    modes=["ROAD","UAV","HYBRID"]
    def immediate(a):
        return sum(p*r for _,p,r in focused_transitions(state,a))
    return max(modes,key=immediate)

def simulate_fixed_action(start_state,action,steps=8,seed=1):
    rng=np.random.default_rng(seed);s=start_state;total=0.0;success=False
    for _ in range(steps):
        if s in MDP_TERMINALS:
            success=(s=="DELIVERED")
            break
        a=action
        tr=focused_transitions(s,a)
        probs=np.array([x[1] for x in tr],dtype=float);probs/=probs.sum()
        idx=int(rng.choice(len(tr),p=probs))
        sp,p,r=tr[idx];total+=r;s=sp
    if s=="DELIVERED": success=True
    return total,success,s

def compare_static_vs_mdp(start_state,policy,episodes=200,steps=8,seed=1):
    rows=[]
    static=immediate_static_action(start_state)
    for name in ["Static one-shot","Adaptive MDP"]:
        totals=[];successes=0
        for i in range(episodes):
            if name=="Static one-shot":
                total,success,_=simulate_fixed_action(start_state,static,steps,seed+i)
            else:
                sim,total=simulate_mdp_policy(start_state,policy,steps,seed+i)
                success=bool(len(sim) and str(sim.iloc[-1]["next_state"])=="DELIVERED")
            totals.append(total);successes+=int(success)
        rows.append({
            "Planner":name,
            "Initial action":static if name=="Static one-shot" else policy[start_state],
            "Mean reward":round(float(np.mean(totals)),2),
            "Std reward":round(float(np.std(totals)),2),
            "Delivery success %":round(100*successes/episodes,1)
        })
    return pd.DataFrame(rows)


# FIGURES

def road_fig(world,path=None):
    G=world["road"];pos={n:G.nodes[n]["pos"] for n in G.nodes};fig=go.Figure()
    chosen=set(tuple(sorted(x)) for x in zip(path,path[1:])) if path else set()
    for a,b,d in G.edges(data=True):
        sel=tuple(sorted((a,b))) in chosen
        fig.add_trace(go.Scatter(
            x=[pos[a][0],pos[b][0]],y=[pos[a][1],pos[b][1]],mode="lines",
            line=dict(width=6 if sel else 2,dash="solid" if d["p_open"]>.55 else "dot"),
            opacity=1 if sel else .55,showlegend=False,
            hovertext=[f"{a}-{b}<br>P(open)={d['p_open']:.2f}<br>CNN risk={d['vision']:.2f}<br>cost={d['cost']:.2f}"]
        ))
    nodes=list(G.nodes)
    special=set(["Depot"]) | set(world.get("requests",{}).keys())
    fig.add_trace(go.Scatter(x=[pos[n][0] for n in nodes],y=[pos[n][1] for n in nodes],mode="markers+text",
        marker=dict(size=[24 if n in special else 13 for n in nodes]),
        text=nodes,textposition="top center",name="Road locations"))
    # Air-only nodes: charging stations + UAVs
    for c in world["charging"]:
        p=world["coords"][c]
        fig.add_trace(go.Scatter(x=[p[0]],y=[p[1]],mode="markers+text",marker=dict(size=22,symbol="hexagon"),
                                 text=[c],textposition="top center",name=c))
    for u in world["uavs"]:
        fig.add_trace(go.Scatter(x=[u.pos[0]],y=[u.pos[1]],mode="markers+text",marker=dict(size=22,symbol="triangle-up"),
                                 text=[f"{u.name} {u.battery_pct:.0f}%"],textposition="bottom center",name=u.name))
    fig.update_layout(height=520,margin=dict(l=5,r=5,t=15,b=5),xaxis=dict(visible=False),yaxis=dict(visible=False,scaleanchor="x"))
    return fig

def search_trace_fig(world,r,step):
    fig=road_fig(world,r["path"]);G=world["road"];pos={n:G.nodes[n]["pos"] for n in G.nodes}
    if r["steps"]:
        closed=set(r["steps"][step]["closed"]);front=set(r["steps"][step]["frontier"])
        fig.add_trace(go.Scatter(x=[pos[n][0] for n in closed],y=[pos[n][1] for n in closed],mode="markers",
                                 marker=dict(size=19,symbol="circle-open"),name="Expanded"))
        fig.add_trace(go.Scatter(x=[pos[n][0] for n in front if n in pos],y=[pos[n][1] for n in front if n in pos],mode="markers",
                                 marker=dict(size=19,symbol="diamond-open"),name="Frontier"))
    return fig

def hazard_fig(grid,title):
    return go.Figure(data=go.Heatmap(z=grid)).update_layout(height=390,title=title,margin=dict(l=10,r=10,t=45,b=10))

def bn_fig():
    pos={"HazardSevere":(0,1.5),"VisualDamage":(0,.75),"FieldBlocked":(0,0),
         "RoadOpen":(2,.75)}
    edges=[("HazardSevere","RoadOpen"),("VisualDamage","RoadOpen"),("FieldBlocked","RoadOpen")]
    fig=go.Figure()
    for a,b in edges:
        fig.add_trace(go.Scatter(x=[pos[a][0],pos[b][0]],y=[pos[a][1],pos[b][1]],
                                 mode="lines",showlegend=False))
    fig.add_trace(go.Scatter(
        x=[v[0] for v in pos.values()],y=[v[1] for v in pos.values()],
        mode="markers+text",marker=dict(size=35),text=list(pos),
        textposition="top center",showlegend=False))
    fig.update_layout(height=300,xaxis=dict(visible=False),yaxis=dict(visible=False),
                      margin=dict(l=10,r=10,t=10,b=10))
    return fig

def hmm_fig(obs,hist,path):
    ypos={"Open":2,"Degraded":1,"Blocked":0};fig=go.Figure()
    for t in range(len(obs)-1):
        for a in ypos:
            for b in ypos:
                fig.add_trace(go.Scatter(x=[t,t+1],y=[ypos[a],ypos[b]],mode="lines",line=dict(width=.5),showlegend=False))
    for t,o in enumerate(obs):
        for i,s in enumerate(HMM_STATES):
            fig.add_trace(go.Scatter(x=[t],y=[ypos[s]],mode="markers+text",
                marker=dict(size=20 if path[t]==s else 11,symbol="star" if path[t]==s else "circle"),
                text=[f"{s}<br>{hist[t][i]:.2f}"],textposition="middle right",showlegend=False))
    fig.update_layout(height=350,xaxis=dict(tickmode="array",tickvals=list(range(len(obs))),
                      ticktext=[f"t{i+1}: {o}" for i,o in enumerate(obs)]),yaxis=dict(visible=False))
    return fig


def invalidate_derived_state(reason="world changed"):
    """
    Any edit to graph/evidence invalidates every downstream result.
    The next planner run must recompute search, Monte Carlo, transport mode,
    hybrid direction, precision release and inspector outputs.
    """
    st.session_state.world_revision = st.session_state.get("world_revision",0)+1
    st.session_state.plan=None
    st.session_state.release_result=None
    st.session_state.release_mc=None
    st.session_state.release_inputs=None
    st.session_state.last_change_reason=reason


# SESSION STATE

if "world_revision" not in st.session_state:
    st.session_state.world_revision=0
if "last_change_reason" not in st.session_state:
    st.session_state.last_change_reason="initial world"
if "world" not in st.session_state:
    st.session_state.world = build_world({"type":"Earthquake","seed":17,"mw":6.7,"depth_km":14,
                                          "local_shaking":.60,"secondary_hazard":.30,"wind":3.0})
if "plan" not in st.session_state:
    st.session_state.plan=None
if "release_result" not in st.session_state:
    st.session_state.release_result=None
if "release_mc" not in st.session_state:
    st.session_state.release_mc=None
if "release_inputs" not in st.session_state:
    st.session_state.release_inputs=None




# PART 1: DETERMINISTIC SEARCH HELPERS


def part1_make_deterministic(world, threshold=0.45):
    """Freeze probabilistic edges into known open/blocked status for Part 1."""
    for a, b, d in world["road"].edges(data=True):
        known_open = float(d.get("p_open", 1.0)) >= threshold
        d["known_open"] = known_open
        d["p_open"] = 1.0 if known_open else 0.0
        d["risk"] = 0.0
        dist = float(d.get("distance", 1.0))
        base_time = float(d.get("time", dist))
        d["cost"] = (base_time + 0.45 * (0.35 * dist)) if known_open else 1e6
    return world

if "part1_deterministic_ready" not in st.session_state:
    st.session_state.world = part1_make_deterministic(st.session_state.world)
    st.session_state.part1_deterministic_ready = True

def part1_relaxed_heuristic(G, goal):
    """
    H3: solve a relaxed problem using travel-time-only edge costs.
    Blocked edges remain unavailable; extra deterministic operating penalties are removed.
    """
    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))
    for a, b, d in G.edges(data=True):
        if float(d.get("cost", 1e6)) >= 1e5:
            continue
        relaxed = max(0.0, float(d.get("time", d.get("distance", 1.0))))
        H.add_edge(a, b, cost=relaxed)
    h = {}
    for n in G.nodes:
        try:
            h[n] = float(nx.shortest_path_length(H, n, goal, weight="cost"))
        except Exception:
            h[n] = 0.0
    return h

def part1_heuristic_map(G, goal, name):
    if name == "H1 — Zero":
        return {n: 0.0 for n in G.nodes}
    if name == "H2 — Geometric lower bound":
        return {n: heuristic_value(G, n, goal, "cost_lower_bound") for n in G.nodes}
    h3 = part1_relaxed_heuristic(G, goal)
    if name == "H3 — Relaxed-network":
        return h3
    h2 = {n: heuristic_value(G, n, goal, "cost_lower_bound") for n in G.nodes}
    return {n: max(h2[n], h3[n]) for n in G.nodes}

def part1_heuristic_diagnostics(G, goal, hmap):
    exact = {}
    for n in G.nodes:
        try:
            exact[n] = nx.shortest_path_length(G, n, goal, weight="cost")
        except Exception:
            exact[n] = float("inf")
    admissible = all(
        hmap[n] <= exact[n] + 1e-9
        for n in G.nodes if np.isfinite(exact[n])
    )
    violations = []
    for a, b, d in G.edges(data=True):
        c = float(d.get("cost", 1e6))
        if c >= 1e5:
            continue
        if hmap[a] > c + hmap[b] + 1e-9:
            violations.append((a, b))
        if hmap[b] > c + hmap[a] + 1e-9:
            violations.append((b, a))
    return admissible, len(violations) == 0, violations, exact

def part1_astar(G, start, goal, heuristic_name):
    hmap = part1_heuristic_map(G, goal, heuristic_name)
    parent = {start: None}
    g = {start: 0.0}
    f = {start: hmap[start]}
    expanded = []
    closed = set()
    pq = [(f[start], 0, start)]
    counter = 0
    steps = []

    while pq:
        _, _, n = heapq.heappop(pq)
        if n in closed:
            continue
        closed.add(n)
        expanded.append(n)

        steps.append({
            "expanded": n,
            "frontier": [x[2] for x in pq],
            "closed": list(closed),
            "g": dict(g),
            "h": dict(hmap),
            "f": dict(f),
        })

        if n == goal:
            break

        for nb in G.neighbors(n):
            c = float(G[n][nb].get("cost", 1e6))
            if c >= 1e5:
                continue
            ng = g[n] + c
            if ng < g.get(nb, float("inf")):
                g[nb] = ng
                parent[nb] = n
                counter += 1
                f[nb] = ng + hmap[nb]
                heapq.heappush(pq, (f[nb], counter, nb))

    path = reconstruct(parent, goal)
    cost = sum(float(G[a][b]["cost"]) for a, b in zip(path, path[1:])) if path else float("inf")
    adm, cons, violations, exact = part1_heuristic_diagnostics(G, goal, hmap)
    return {
        "path": path,
        "cost": cost,
        "expanded": expanded,
        "steps": steps,
        "g": g,
        "h": hmap,
        "f": f,
        "admissible": adm,
        "consistent": cons,
        "violations": violations,
        "exact": exact,
    }

def part1_benchmark(G, start, goal):
    rows = []

    # BFS
    t0 = time.perf_counter()
    rb = graph_search(G, start, goal, "BFS", heuristic_kind="zero")
    rows.append({
        "Method": "BFS",
        "Heuristic": "—",
        "Path cost": rb["cost"],
        "Expanded": len(rb["expanded"]),
        "Runtime (ms)": (time.perf_counter() - t0) * 1000,
        "Admissible": "—",
        "Consistent": "—",
    })

    # UCS
    t0 = time.perf_counter()
    ru = graph_search(G, start, goal, "UCS", heuristic_kind="zero")
    rows.append({
        "Method": "UCS",
        "Heuristic": "h=0",
        "Path cost": ru["cost"],
        "Expanded": len(ru["expanded"]),
        "Runtime (ms)": (time.perf_counter() - t0) * 1000,
        "Admissible": True,
        "Consistent": True,
    })

    for hname in [
        "H1 — Zero",
        "H2 — Geometric lower bound",
        "H3 — Relaxed-network",
        "H4 — max(H2,H3)",
    ]:
        t0 = time.perf_counter()
        r = part1_astar(G, start, goal, hname)
        rows.append({
            "Method": "A*",
            "Heuristic": hname,
            "Path cost": r["cost"],
            "Expanded": len(r["expanded"]),
            "Runtime (ms)": (time.perf_counter() - t0) * 1000,
            "Admissible": r["admissible"],
            "Consistent": r["consistent"],
        })
    df = pd.DataFrame(rows)
    df["Path cost"] = df["Path cost"].round(3)
    df["Runtime (ms)"] = df["Runtime (ms)"].round(3)
    return df


def _img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

LOGO_PATH = Path(__file__).resolve().parent / "assets" / "aeris_circular_transparent.png"
LOGO_B64 = _img_to_base64(LOGO_PATH) if LOGO_PATH.exists() else None
PAGE_ICON = Image.open(Path(__file__).resolve().parent / "assets" / "favicon.png") if (Path(__file__).resolve().parent / "assets" / "favicon.png").exists() else "A"



# EXTERNAL MAP IMPORT - PART 1


def _validate_external_graph(G, requests):
    errors=[]
    if "Depot" not in G.nodes:
        errors.append("The graph must contain a node named 'Depot'.")
    if len(G.nodes) < 2:
        errors.append("The graph must contain at least two nodes.")
    if len(G.edges) < 1:
        errors.append("The graph must contain at least one edge.")
    for n,d in G.nodes(data=True):
        if "pos" not in d or len(d["pos"]) != 2:
            errors.append(f"Node {n} is missing a 2-D position.")
    for a,b,d in G.edges(data=True):
        if float(d.get("cost",0)) <= 0 and bool(d.get("known_open",True)):
            errors.append(f"Open edge {a}-{b} must have positive cost.")
    for target in requests:
        if target not in G.nodes:
            errors.append(f"Request destination '{target}' is not a graph node.")
    return errors

def world_from_json_payload(payload, base_world):
    """
    JSON format:
    {
      "nodes":[{"id":"Depot","x":0,"y":0,"type":"Medical depot"}, ...],
      "edges":[{"source":"Depot","target":"A","cost":2.5,"open":true}, ...],
      "requests":[{"destination":"Hospital","supply":"Trauma kit","quantity":3,"priority":1.0}]
    }
    """
    if isinstance(payload, str):
        payload=json.loads(payload)
    G=nx.Graph()
    coords={}
    node_types={}
    for row in payload.get("nodes",[]):
        nid=str(row["id"])
        x=float(row["x"]); y=float(row["y"])
        coords[nid]=(x,y)
        node_types[nid]=str(row.get("type","Road node"))
        G.add_node(nid,pos=(x,y))
    for row in payload.get("edges",[]):
        a=str(row["source"]); b=str(row["target"])
        if a not in G or b not in G:
            raise ValueError(f"Edge {a}-{b} references an unknown node.")
        is_open=bool(row.get("open",True))
        cost=float(row.get("cost",1.0))
        dist=float(row.get("distance",math.dist(coords[a],coords[b])))
        time_val=float(row.get("time",cost))
        G.add_edge(
            a,b,
            known_open=is_open,
            p_open=1.0 if is_open else 0.0,
            distance=dist,
            time=time_val,
            risk=0.0,
            cost=cost if is_open else 1e6,
            asset=str(row.get("asset","road")),
            vision=0.0
        )
    requests={}
    for r in payload.get("requests",[]):
        dest=str(r["destination"])
        requests[dest]={
            "supply":str(r.get("supply","Trauma kit")),
            "quantity":int(r.get("quantity",1)),
            "priority":float(r.get("priority",.7))
        }
    if not requests:
        # Any named hospital/relief/demand node becomes a destination.
        for n,t in node_types.items():
            if t.lower() in {"hospital","relief camp","demand point","village"}:
                requests[n]={"supply":"Trauma kit","quantity":1,"priority":.7}
    world=dict(base_world)
    world["road"]=G
    world["coords"]=coords
    world["node_types"]=node_types
    world["requests"]=requests
    world["charging"]=[n for n,t in node_types.items() if "charging" in t.lower()]
    return world

def world_from_csv_frames(nodes_df, edges_df, base_world, requests_df=None):
    required_nodes={"id","x","y"}
    required_edges={"source","target","cost"}
    if not required_nodes.issubset(set(nodes_df.columns)):
        raise ValueError("nodes.csv requires columns: id, x, y (optional: type).")
    if not required_edges.issubset(set(edges_df.columns)):
        raise ValueError("edges.csv requires columns: source, target, cost (optional: open, distance, time, asset).")
    payload={"nodes":[],"edges":[],"requests":[]}
    for _,r in nodes_df.iterrows():
        payload["nodes"].append({
            "id":str(r["id"]),"x":float(r["x"]),"y":float(r["y"]),
            "type":str(r["type"]) if "type" in nodes_df.columns and pd.notna(r["type"]) else "Road node"
        })
    for _,r in edges_df.iterrows():
        open_val=True
        if "open" in edges_df.columns and pd.notna(r["open"]):
            v=r["open"]
            open_val = bool(v) if isinstance(v,(bool,np.bool_)) else str(v).strip().lower() not in {"false","0","no","blocked"}
        item={
            "source":str(r["source"]),"target":str(r["target"]),"cost":float(r["cost"]),
            "open":open_val
        }
        for k in ["distance","time"]:
            if k in edges_df.columns and pd.notna(r[k]): item[k]=float(r[k])
        if "asset" in edges_df.columns and pd.notna(r["asset"]): item["asset"]=str(r["asset"])
        payload["edges"].append(item)
    if requests_df is not None and len(requests_df):
        if "destination" not in requests_df.columns:
            raise ValueError("requests.csv requires a destination column.")
        for _,r in requests_df.iterrows():
            payload["requests"].append({
                "destination":str(r["destination"]),
                "supply":str(r["supply"]) if "supply" in requests_df.columns and pd.notna(r["supply"]) else "Trauma kit",
                "quantity":int(r["quantity"]) if "quantity" in requests_df.columns and pd.notna(r["quantity"]) else 1,
                "priority":float(r["priority"]) if "priority" in requests_df.columns and pd.notna(r["priority"]) else .7
            })
    return world_from_json_payload(payload, base_world)


# PART 1 USER INTERFACE

st.set_page_config(
    page_title="AERIS Medico Helper — Part 1",
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
:root{
  --bg:#06111a;
  --bg2:#0a1825;
  --navy:#0e2234;
  --panel:rgba(255,255,255,.045);
  --panel2:rgba(255,255,255,.07);
  --line:rgba(118,184,255,.15);
  --line2:rgba(161,126,255,.18);
  --text:#f3f8ff;
  --muted:#9fb2c8;
  --blue:#38bdf8;
  --blue2:#4f8cff;
  --violet:#9b7cff;
  --cyan:#60e7ff;
  --good:#6ee7b7;
}
html{scroll-behavior:smooth;}
.stApp{
  background:
    linear-gradient(rgba(255,255,255,.018) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,.018) 1px, transparent 1px),
    radial-gradient(circle at 85% 5%, rgba(79,140,255,.14), transparent 26%),
    radial-gradient(circle at 8% 12%, rgba(155,124,255,.08), transparent 22%),
    linear-gradient(180deg,var(--bg) 0%,var(--bg2) 100%);
  background-size:32px 32px,32px 32px,auto,auto,auto;
  color:var(--text);
}
.block-container{
  max-width:1280px;
  padding-top:1.15rem;
  padding-bottom:2rem;
  animation:pageEnter .28s cubic-bezier(.2,.7,.2,1);
}
@keyframes pageEnter{
  from{opacity:0;transform:translateY(6px)}
  to{opacity:1;transform:translateY(0)}
}
[data-testid="stSidebar"]{
  background:
    linear-gradient(180deg,rgba(10,25,39,.98),rgba(7,20,31,.98));
  border-right:1px solid rgba(56,189,248,.12);
  backdrop-filter:blur(18px);
}
h1{
  font-size:clamp(2.05rem,3.2vw,3.5rem)!important;
  line-height:1.0!important;
  letter-spacing:-.045em!important;
  font-weight:800!important;
  margin-bottom:.2rem!important;
}
h2{font-size:1.35rem!important;letter-spacing:-.02em;}
h3{font-size:1.02rem!important;}
h1,h2,h3,h4,p,label,li,.stMarkdown,.stCaption,span{
  color:var(--text)!important;
}
.subtle{color:var(--muted)!important;}
.course-strip{
  position:relative;
  overflow:hidden;
  border:1px solid var(--line);
  border-radius:14px;
  padding:1rem 1.1rem;
  background:linear-gradient(135deg,rgba(79,140,255,.09),rgba(155,124,255,.045));
  margin:.2rem 0 1.15rem 0;
}
.course-strip:before{
  content:"";
  position:absolute;
  left:0;top:0;bottom:0;width:3px;
  background:linear-gradient(180deg,var(--cyan),var(--violet));
}
.section-note,.metric-card,.heuristic-card{
  background:linear-gradient(180deg,rgba(255,255,255,.055),rgba(255,255,255,.03));
  border:1px solid var(--line);
  border-radius:12px;
  box-shadow:0 10px 28px rgba(0,0,0,.16);
}
.section-note{padding:.82rem .95rem;margin:.3rem 0 .9rem;}
.metric-card{padding:.9rem 1rem;}
.heuristic-card{
  padding:.9rem 1rem;
  min-height:122px;
  margin-bottom:.8rem;
  position:relative;
  transition:transform .18s ease,border-color .18s ease,background .18s ease;
}
.heuristic-card:hover{
  transform:translateY(-2px);
  border-color:rgba(56,189,248,.30);
  background:linear-gradient(180deg,rgba(56,189,248,.06),rgba(155,124,255,.03));
}
.heuristic-title{
  font-weight:800;
  color:var(--cyan)!important;
  letter-spacing:.01em;
}
.code-eq{
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  background:#07131d;
  color:#dff7ff;
  border:1px solid rgba(56,189,248,.16);
  padding:.5rem .65rem;
  border-radius:8px;
  margin:.4rem 0;
}
div[data-testid="stMetric"]{
  background:linear-gradient(180deg,rgba(255,255,255,.055),rgba(255,255,255,.03));
  border:1px solid var(--line);
  border-radius:12px;
  padding:.55rem .72rem;
}
div[data-testid="stDataFrame"],div[data-testid="stPlotlyChart"]{
  border:1px solid var(--line);
  border-radius:12px;
  overflow:hidden;
  background:rgba(3,11,18,.28);
}
.stButton>button{
  border-radius:999px;
  border:1px solid rgba(56,189,248,.24);
  background:rgba(56,189,248,.06);
  color:var(--text);
  font-weight:700;
  transition:transform .16s ease,background .16s ease,border-color .16s ease;
}
.stButton>button:hover{
  transform:translateY(-1px);
  border-color:rgba(96,231,255,.46);
  background:rgba(56,189,248,.11);
}
.stButton>button[kind="primary"]{
  background:linear-gradient(90deg,var(--blue2),var(--violet));
  border:none;
  color:white;
}
.stRadio [role="radiogroup"]{
  display:flex;
  flex-direction:column;
  gap:.34rem;
}
.stRadio [role="radiogroup"] label{
  border:1px solid rgba(255,255,255,.07);
  border-radius:999px;
  padding:.18rem .42rem;
  transition:background .16s ease,border-color .16s ease,transform .16s ease;
}
.stRadio [role="radiogroup"] label:hover{
  background:rgba(56,189,248,.06);
  border-color:rgba(56,189,248,.24);
  transform:translateX(2px);
}
.logo-orb{
  width:82px;height:82px;border-radius:50%;
  display:flex;align-items:center;justify-content:center;
  border:1px solid rgba(96,231,255,.28);
  background:
    radial-gradient(circle at 30% 25%,rgba(96,231,255,.12),transparent 42%),
    rgba(255,255,255,.025);
  box-shadow:0 0 0 6px rgba(56,189,248,.025),0 12px 30px rgba(0,0,0,.26);
  margin:0 auto .75rem auto;
}
.logo-orb img{width:72px;height:72px;object-fit:contain;border-radius:50%;}
.footer-note{
  text-align:center;
  color:var(--muted);
  font-size:.88rem;
  padding-top:.8rem;
  margin-top:1.3rem;
  border-top:1px solid rgba(56,189,248,.10);
}
@media (prefers-reduced-motion: reduce){
  *{animation:none!important;transition:none!important;scroll-behavior:auto!important;}
}
</style>
""", unsafe_allow_html=True)

hero_a, hero_b = st.columns([0.22, 1.78], vertical_alignment="center")
with hero_a:
    if LOGO_B64:
        st.markdown(
            f'<div class="logo-orb"><img src="data:image/png;base64,{LOGO_B64}"></div>',
            unsafe_allow_html=True
        )
with hero_b:
    st.markdown('<span class="subtle">CSE518 · ARTIFICIAL INTELLIGENCE · PART 1</span>', unsafe_allow_html=True)
    st.title("AERIS Medico Helper")
    st.markdown(
        '<div class="course-strip">'
        '<b>Reactive–Deliberative Routing and A* Planning</b><br>'
        '<span class="subtle">Deterministic routing, reactive safety rules, BFS/UCS baselines, and a focused four-heuristic A* study.</span>'
        '</div>',
        unsafe_allow_html=True
    )

with st.sidebar:
    if LOGO_B64:
        st.markdown(
            f'<div class="logo-orb"><img src="data:image/png;base64,{LOGO_B64}"></div>',
            unsafe_allow_html=True
        )
    st.markdown("### Navigation")
    page = st.radio(
        "Module",
        ["Problem Setup", "A* Search", "Heuristic Study", "Comparison"],
        label_visibility="collapsed"
    )
    st.divider()
    st.markdown("**Part 1 syllabus mapping**")
    st.caption("Agents → Problem formulation → BFS/UCS → A* → Heuristics")
    st.divider()
    st.caption(f"Scenario revision: {st.session_state.get('world_revision', 0)}")
    st.caption("Deterministic / fully observable")

w = st.session_state.world

if page == "Problem Setup":
    st.header("Problem setup")
    st.markdown(
        '<div class="section-note">'
        '<b>Stage objective:</b> model a deterministic disaster road network, apply reactive safety rules, '
        'and use deliberate graph search for routing.'
        '</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns([1.15, 1.85], gap="large")

    with left:
        st.subheader("Scenario")
        dtype = st.selectbox(
            "Disaster type",
            ["Earthquake", "Flood", "Cyclone"],
            index=["Earthquake", "Flood", "Cyclone"].index(w["params"]["type"])
        )
        seed = st.number_input("Scenario seed", 1, 9999, int(w["params"]["seed"]))

        if dtype == "Earthquake":
            mw = st.slider("Magnitude Mw", 4.0, 8.5, float(w["params"].get("mw", 6.7)), .1)
            depth = st.slider("Depth (km)", 2.0, 50.0, float(w["params"].get("depth_km", 14.0)), 1.0)
            local = st.slider("Local shaking", 0.0, 1.0, float(w["params"].get("local_shaking", .60)), .05)
            secondary = st.slider("Debris / landslide severity", 0.0, 1.0, float(w["params"].get("secondary_hazard", .30)), .05)
            params = {
                "type": dtype, "seed": int(seed), "mw": mw, "depth_km": depth,
                "local_shaking": local, "secondary_hazard": secondary, "wind": 3.0
            }
        elif dtype == "Flood":
            water = st.slider("Water depth (m)", 0.0, 2.5, float(w["params"].get("water_depth_m", .70)), .05)
            flow = st.slider("Flow severity", 0.0, 1.0, float(w["params"].get("flow", .45)), .05)
            rain = st.slider("Rainfall severity", 0.0, 1.0, float(w["params"].get("rain", .55)), .05)
            params = {
                "type": dtype, "seed": int(seed), "water_depth_m": water,
                "flow": flow, "rain": rain, "wind": 4.0
            }
        else:
            wind = st.slider("Sustained wind (m/s)", 0.0, 25.0, float(w["params"].get("wind", 9.0)), .5)
            gust = st.slider("Gust speed (m/s)", 0.0, 35.0, float(w["params"].get("gust", 14.0)), .5)
            rain = st.slider("Rainfall severity", 0.0, 1.0, float(w["params"].get("rain", .45)), .05)
            debris = st.slider("Debris severity", 0.0, 1.0, float(w["params"].get("debris", .35)), .05)
            params = {
                "type": dtype, "seed": int(seed), "wind": wind, "gust": gust,
                "rain": rain, "debris": debris
            }

        if st.button("Apply scenario", type="primary", use_container_width=True):
            st.session_state.world = part1_make_deterministic(build_world(params))
            invalidate_derived_state("Part 1 scenario changed")
            st.rerun()

        st.markdown("#### External map")
        st.caption("Like the map shaerd in mail of problem description, Part 1 can be run on a map that is not hard-coded.")
        import_mode=st.selectbox("Import format",["Built-in scenario","JSON map","CSV map"],key="external_mode")

        if import_mode=="JSON map":
            uploaded=st.file_uploader("Upload map JSON",type=["json"],key="map_json")
            if uploaded is not None and st.button("Load JSON map",use_container_width=True):
                try:
                    payload=json.loads(uploaded.getvalue().decode("utf-8"))
                    nw=world_from_json_payload(payload,st.session_state.world)
                    errs=_validate_external_graph(nw["road"],nw["requests"])
                    if errs:
                        st.error(" ".join(errs))
                    else:
                        st.session_state.world=nw
                        invalidate_derived_state("External JSON map loaded")
                        st.rerun()
                except Exception as e:
                    st.error(f"Could not load JSON map: {e}")

        elif import_mode=="CSV map":
            nfile=st.file_uploader("nodes.csv",type=["csv"],key="nodes_csv")
            efile=st.file_uploader("edges.csv",type=["csv"],key="edges_csv")
            rfile=st.file_uploader("requests.csv (optional)",type=["csv"],key="requests_csv")
            if nfile is not None and efile is not None and st.button("Load CSV map",use_container_width=True):
                try:
                    ndf=pd.read_csv(nfile); edf=pd.read_csv(efile)
                    rdf=pd.read_csv(rfile) if rfile is not None else None
                    nw=world_from_csv_frames(ndf,edf,st.session_state.world,rdf)
                    errs=_validate_external_graph(nw["road"],nw["requests"])
                    if errs:
                        st.error(" ".join(errs))
                    else:
                        st.session_state.world=nw
                        invalidate_derived_state("External CSV map loaded")
                        st.rerun()
                except Exception as e:
                    st.error(f"Could not load CSV map: {e}")

        st.subheader("Delivery request")
        target = st.selectbox("Destination", list(w["requests"].keys()), key="setup_target")
        req = w["requests"][target]
        st.write(f"Supply: **{req['supply']}**")
        st.write(f"Quantity: **{req['quantity']}**")

    with right:
        st.subheader("Road network")
        st.plotly_chart(road_fig(w), use_container_width=True, key="setup_graph")

        st.markdown("#### Search problem formulation")
        formulation = pd.DataFrame([
            ["State", "Current graph node / vehicle position"],
            ["Initial state", "Depot"],
            ["Actions", "Traverse an adjacent known-open road"],
            ["Transition model", "Move to the connected node"],
            ["Goal test", "Current node equals selected destination"],
            ["Path cost", "Sum of deterministic edge costs"]
        ], columns=["Component", "AERIS Part 1 definition"])
        st.dataframe(formulation, use_container_width=True, hide_index=True)

        st.markdown(
            '<div class="section-note">'
            '<b>Legend:</b> solid usable roads are available to the planner; blocked roads are assigned '
            'an effectively infinite traversal cost and are therefore excluded.'
            '</div>',
            unsafe_allow_html=True
        )

        st.subheader("Known road-state edit")
        edges = list(w["road"].edges)
        edge_labels = [f"{a} — {b}" for a, b in edges]
        edge_label = st.selectbox("Road", edge_labels)
        a0, b0 = edges[edge_labels.index(edge_label)]
        ed = w["road"][a0][b0]

        c1, c2 = st.columns(2)
        known_open = c1.selectbox(
            "Status",
            ["Open", "Blocked"],
            index=0 if float(ed.get("cost", 1e6)) < 1e5 else 1
        )
        current_cost = float(ed.get("cost", 1.0))
        editable_cost = c2.number_input(
            "Open-road traversal cost",
            min_value=0.1,
            max_value=100.0,
            value=float(current_cost if current_cost < 1e5 else max(1.0, ed.get("time", 1.0))),
            step=0.1
        )
        if st.button("Update road state"):
            if known_open == "Open":
                ed["known_open"] = True
                ed["p_open"] = 1.0
                ed["cost"] = float(editable_cost)
            else:
                ed["known_open"] = False
                ed["p_open"] = 0.0
                ed["cost"] = 1e6
            invalidate_derived_state(f"Known road state changed: {a0}-{b0}")
            st.rerun()

    st.subheader("Reactive controller")
    st.caption("These rules act before or between planning steps; A* handles deliberative route selection.")
    rules = pd.DataFrame([
        ["R1", "Never traverse a road known to be blocked."],
        ["R2", "Reject a transport mode if its safety constraints cannot be satisfied."],
        ["R3", "Maintain the configured UAV reserve before committing to an aerial leg."],
        ["R4", "If a known road state changes, invalidate the old route and re-plan."],
    ], columns=["Rule", "Controller behaviour"])
    st.dataframe(rules, use_container_width=True, hide_index=True)

elif page == "A* Search":
    st.header("A* search")
    st.markdown(
        '<div class="section-note">'
        '<b>Evaluation function</b>'
        '<div class="code-eq">f(n) = g(n) + h(n)</div>'
        '<span class="subtle">g(n): exact cost from the start · h(n): estimated remaining cost</span>'
        '</div>',
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)
    target = c1.selectbox("Destination", list(w["requests"].keys()), key="astar_target")
    hname = c2.selectbox(
        "Heuristic",
        ["H1 — Zero", "H2 — Geometric lower bound", "H3 — Relaxed-network", "H4 — max(H2,H3)"]
    )

    r = part1_astar(w["road"], "Depot", target, hname)

    if not r["path"]:
        st.error("No feasible route exists with the current known road states.")
    else:
        st.plotly_chart(road_fig(w, r["path"]), use_container_width=True, key="astar_graph")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Path cost", f"{r['cost']:.2f}")
        c2.metric("Expanded nodes", len(r["expanded"]))
        c3.metric("Heuristic admissible", "Yes" if r["admissible"] else "No")
        c4.metric("Heuristic consistent", "Yes" if r["consistent"] else "No")

        st.markdown(f"**Selected route:** {' → '.join(r['path'])}")

        st.subheader("Expansion trace")
        st.caption("Move through the search in the order nodes are expanded.")
        step = st.slider("Expansion", 0, max(0, len(r["steps"]) - 1), 0)
        ss = r["steps"][step]

        s1, s2, s3 = st.columns(3)
        s1.metric("Step", step + 1)
        s2.metric("Expanded node", ss["expanded"])
        s3.metric("Frontier size", len(ss["frontier"]))

        trace_cols = st.columns([1, 1])
        with trace_cols[0]:
            st.markdown(f"**Expanded node:** `{ss['expanded']}`")
            st.markdown("**Frontier:** " + (", ".join(map(str, ss["frontier"])) or "empty"))
            st.markdown("**Closed:** " + ", ".join(map(str, ss["closed"])))
        with trace_cols[1]:
            table = pd.DataFrame([
                {
                    "Node": n,
                    "g(n)": round(ss["g"].get(n, float("nan")), 3),
                    "h(n)": round(ss["h"].get(n, float("nan")), 3),
                    "f(n)": round(ss["f"].get(n, float("nan")), 3),
                }
                for n in sorted(ss["g"].keys())
            ])
            st.dataframe(table, use_container_width=True, hide_index=True)

elif page == "Heuristic Study":
    st.header("Four-heuristic study")
    st.markdown(
        "The objective is not to invent a new heuristic. It is to demonstrate how heuristic "
        "informativeness affects A* while preserving the properties taught in class."
    )
    target = st.selectbox("Destination", list(w["requests"].keys()), key="heuristic_target")

    descriptions = pd.DataFrame([
        ["H1 — Zero", "h(n)=0", "UCS-equivalent baseline"],
        ["H2 — Geometric lower bound", "straight-line distance × minimum cost per distance", "cheap admissible lower bound"],
        ["H3 — Relaxed-network", "shortest cost after removing extra operating penalties", "relaxed-problem heuristic"],
        ["H4 — max(H2,H3)", "maximum of two admissible heuristics", "dominance / stronger guidance"],
    ], columns=["Heuristic", "Definition", "Purpose"])

    hc1, hc2 = st.columns(2)
    with hc1:
        st.markdown(
            '<div class="heuristic-card"><div class="heuristic-title">H1 · Zero</div>'
            '<div class="code-eq">h(n)=0</div>'
            '<span class="subtle">A* reduces to UCS. Baseline for comparison.</span></div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="heuristic-card"><div class="heuristic-title">H3 · Relaxed network</div>'
            '<div class="code-eq">h(n)=optimal cost in relaxed problem</div>'
            '<span class="subtle">Uses a simpler problem whose optimum lower-bounds the original.</span></div>',
            unsafe_allow_html=True
        )
    with hc2:
        st.markdown(
            '<div class="heuristic-card"><div class="heuristic-title">H2 · Geometric lower bound</div>'
            '<div class="code-eq">h(n)=distance × min(cost/distance)</div>'
            '<span class="subtle">Cheap geometric lower bound on remaining travel cost.</span></div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="heuristic-card"><div class="heuristic-title">H4 · Dominating combination</div>'
            '<div class="code-eq">h(n)=max(H2,H3)</div>'
            '<span class="subtle">At least as informed as either component heuristic.</span></div>',
            unsafe_allow_html=True
        )

    with st.expander("Definitions in table form"):
        st.dataframe(descriptions, use_container_width=True, hide_index=True)

    rows = []
    for hname in descriptions["Heuristic"]:
        t0 = time.perf_counter()
        r = part1_astar(w["road"], "Depot", target, hname)
        elapsed = (time.perf_counter() - t0) * 1000
        rows.append({
            "Heuristic": hname,
            "Path cost": round(r["cost"], 3) if np.isfinite(r["cost"]) else None,
            "Expanded": len(r["expanded"]),
            "Runtime (ms)": round(elapsed, 3),
            "Admissible": r["admissible"],
            "Consistent": r["consistent"],
        })
    hdf = pd.DataFrame(rows)

    st.subheader("Results")
    st.dataframe(hdf, use_container_width=True, hide_index=True)
    st.bar_chart(hdf.set_index("Heuristic")["Expanded"])

    best = hdf.sort_values(["Expanded", "Runtime (ms)"]).iloc[0]
    st.markdown(
        f"On the current graph, **{best['Heuristic']}** expands the fewest nodes "
        f"({int(best['Expanded'])}) among the tested heuristics."
    )

elif page == "Comparison":
    st.header("Algorithm comparison")
    st.markdown(
        "BFS, UCS and A* are run on the same deterministic graph. "
        "This page is intended for the Part 1 comparison report."
    )

    target = st.selectbox("Destination", list(w["requests"].keys()), key="comparison_target")
    df = part1_benchmark(w["road"], "Depot", target)

    st.dataframe(df, use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Node expansions")
        st.bar_chart(df.set_index(df["Method"] + " | " + df["Heuristic"].astype(str))["Expanded"])
    with right:
        st.subheader("Path cost")
        st.bar_chart(df.set_index(df["Method"] + " | " + df["Heuristic"].astype(str))["Path cost"])

    st.subheader("Interpretation")
    insight = pd.DataFrame([
        ["BFS", "Chooses shallowest path", "May ignore weighted path cost"],
        ["UCS", "Chooses lowest accumulated cost g(n)", "Optimal with non-negative costs"],
        ["A* + admissible h", "Uses g(n)+h(n)", "Can retain optimality with fewer expansions"],
        ["A* + H4", "Uses stronger admissible guidance", "Demonstrates heuristic dominance"]
    ], columns=["Method", "Decision rule", "What to observe"])
    st.dataframe(insight, use_container_width=True, hide_index=True)

    if not df.empty:
        valid = df[df["Path cost"].notna()].copy()
        if len(valid):
            fewest = valid.sort_values(["Expanded","Runtime (ms)"]).iloc[0]
            cheapest = valid.sort_values(["Path cost","Expanded"]).iloc[0]
            c1,c2 = st.columns(2)
            c1.markdown(
                f'<div class="metric-card"><b>Fewest expansions</b><br>'
                f'{fewest["Method"]} · {fewest["Heuristic"]}<br>'
                f'<span class="subtle">{int(fewest["Expanded"])} expanded nodes</span></div>',
                unsafe_allow_html=True
            )
            c2.markdown(
                f'<div class="metric-card"><b>Lowest path cost</b><br>'
                f'{cheapest["Method"]} · {cheapest["Heuristic"]}<br>'
                f'<span class="subtle">cost = {cheapest["Path cost"]}</span></div>',
                unsafe_allow_html=True
            )


st.markdown(
    '<div class="footer-note">AERIS Medico Helper · CSE518 Artificial Intelligence · Prepared by <b>Apoorva Mehta</b></div>',
    unsafe_allow_html=True
)
