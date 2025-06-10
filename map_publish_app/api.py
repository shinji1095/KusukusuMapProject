import os
import time
import json
import math
import random
import folium
import hashlib
import datetime
import polyline
import requests
import numpy as np
from sklearn.cluster import DBSCAN
from geopy.distance import geodesic
from shapely.geometry import Point, Polygon, LineString
from shapely.geometry import MultiPoint
from folium.plugins import BeautifyIcon


# Utility functions
def haversine(p1, p2):
    return geodesic(p1, p2).meters

def vector(p1, p2):
    return [p2[0] - p1[0], p2[1] - p1[1]]

def normalize(v):
    norm = math.sqrt(v[0]**2 + v[1]**2)
    return [v[0]/norm, v[1]/norm] if norm != 0 else [0, 0]

def dot(v1, v2):
    return sum(v1[i]*v2[i] for i in range(len(v1)))

def center_point(p1, p2):
    return [(p1[0]+p2[0])/2, (p1[1]+p2[1])/2]

def midpoint(p1, p2):
    return [(p1[0]+p2[0])/2, (p1[1]+p2[1])/2]

def calculate_bearing(p1, p2):
    lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
    lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1)*math.sin(lat2) - math.sin(lat1)*math.cos(lat2)*math.cos(dlon)
    bearing_rad = math.atan2(x, y)
    bearing_deg = (math.degrees(bearing_rad) + 360) % 360
    return bearing_deg, [math.cos(bearing_rad), math.sin(bearing_rad)]

def fetch_crosswalks_around(lat, lon, radius=200):
    query = f"""
    [out:json][timeout:25];
    (
      way["highway"="footway"]["footway"="crossing"](around:{radius},{lat},{lon});
    );
    out body;
    >;
    out skel qt;
    """
    url = "https://overpass-api.de/api/interpreter"
    response = requests.get(url, params={"data": query})
    return response.json()

def cluster_crosswalks(crosswalks, eps=25, debug_logs=None):
    from sklearn.cluster import DBSCAN
    import numpy as np
    from shapely.geometry import MultiPoint

    coords = [cw["center"] for cw in crosswalks]
    coords_rad = np.radians(coords)
    db = DBSCAN(eps=eps/6371000.0, min_samples=2, metric='haversine').fit(coords_rad)
    labels = db.labels_

    clusters = []
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if debug_logs is not None:
        debug_logs.append(f"DBSCAN produced {num_clusters} clusters (eps={eps}m).")

    for label in set(labels):
        if label == -1:
            continue
        indices = np.where(labels == label)[0]
        members = [crosswalks[i] for i in indices]
        poly_points = [pt for cw in members for pt in [cw["p1"], cw["p2"]]]
        if debug_logs is not None:
            debug_logs.append(f"  Cluster {label}: {len(members)} members, raw points={len(poly_points)}")

        try:
            points = MultiPoint(poly_points)
            polygon = points.convex_hull
            if not polygon.is_valid or polygon.area == 0:
                if debug_logs is not None:
                    debug_logs.append(f"    Cluster {label} skipped: invalid or zero-area polygon.")
                continue
            centroid = list(polygon.centroid.coords)[0]  # (lon, lat) → (lat, lon)
            clusters.append({
                "members": members,
                "polygon": polygon,
                "centroid": centroid
            })
            if debug_logs is not None:
                debug_logs.append(f"    Cluster {label} accepted: centroid={centroid}, area={polygon.area:.8f}")
        except Exception as e:
            if debug_logs is not None:
                debug_logs.append(f"    Cluster {label} failed to build polygon: {str(e)}")
            continue

    return clusters



def visualize_route_and_crosswalks(
    route_poly,
    candidate_crosswalks,
    matched_crosswalks,
    intersections=None,
    clusters=None,
    search_radius=200,
    target_radius=100,
    sample_step=5,
    log_dir="log"
):
    fmap = folium.Map(location=[route_poly[0]["lat"], route_poly[0]["lon"]], zoom_start=17)

    # --- ルートを折れ線で表示 ---
    route_coords = [(pt["lat"], pt["lon"]) for pt in route_poly]
    folium.PolyLine(route_coords, color="black", weight=3, opacity=0.7).add_to(fmap)

    # --- 各route点にsearch_radiusとtarget_radiusの円を追加 ---
    for i in range(0, len(route_poly), sample_step):
      lat = route_poly[i]["lat"]
      lon = route_poly[i]["lon"]
      folium.Circle(
          location=(lat, lon),
          radius=search_radius,
          color="blue",
          fill=True,
          fill_opacity=0.05,
          weight=1,
          tooltip=f"search_radius {search_radius}m"
      ).add_to(fmap)

    for pt in intersections:
        folium.Circle(
            location=(pt.y, pt.x),
            radius=target_radius,
            color="green",
            fill=True,
            fill_opacity=0.05,
            weight=1,
            tooltip=f"target_radius {target_radius}m"
        ).add_to(fmap)

    # --- 各点のマーカー＋bearing情報 ---
    for pt in route_poly:
        tooltip_text = (
            f"lat: {pt['lat']:.6f}<br>"
            f"lon: {pt['lon']:.6f}<br>"
            f"bearing: {pt['bearing']}°<br>"
            f"v_bearing: {pt['v_bearing']}"
        )
        color = "red" if pt.get("highlight", False) else "gray"
        folium.CircleMarker((pt["lat"], pt["lon"]), radius=2, tooltip=tooltip_text, color=color).add_to(fmap)

    # --- クラスタの円表示 ---
    if clusters:
        for cluster in clusters:
            centroid = cluster["centroid"]
            color = "red"
            folium.Circle(location=centroid, radius=30, color=color, fill=True, fill_opacity=0.2).add_to(fmap)

    # --- 候補横断歩道 ---
    for cw in candidate_crosswalks:
        folium.CircleMarker(cw["p1"], radius=3, color="blue").add_to(fmap)
        folium.CircleMarker(cw["p2"], radius=3, color="blue").add_to(fmap)
        folium.PolyLine([cw["p1"], cw["p2"]], color="blue").add_to(fmap)

    # --- マッチした横断歩道 ---
    for cw in matched_crosswalks:
        tooltip = f"id: {cw['id']}<br>bearing: {cw['bearing']:.2f}°"
        folium.PolyLine([cw["p1"], cw["p2"]], color="orange", weight=5, tooltip=tooltip).add_to(fmap)

    # --- 交差点 (intersection points) を × 印で表示 ---
    if intersections:
        for pt in intersections:
            folium.Marker(
                location=(pt.y, pt.x),
                icon=BeautifyIcon(
                    icon_shape='marker',
                    border_color='black',
                    number='×',
                    text_color='black',
                    background_color='white'
                ),
                tooltip="Intersection"
            ).add_to(fmap)

    html_path = os.path.join(log_dir, f"debug_route.html")
    fmap.save(html_path)

def is_crosswalk_target(cluster, route_poly, radius=100, must_intersect_route=True, debug_logs=None):
    matched = []
    intersections = []
    centroid = cluster["centroid"]
    polygon = cluster["polygon"]
    members = cluster["members"]

    if debug_logs is not None:
        debug_logs.append(f"  Analyzing cluster centroid={centroid}")
        debug_logs.append(f"    Members: {len(members)}")
        debug_logs.append(f"    Polygon bounds: {polygon.bounds}")
        debug_logs.append(f"    Polygon centroid: {polygon.centroid}")

    for i in range(len(route_poly) - 1):
        p1 = (route_poly[i]["lat"], route_poly[i]["lon"])
        p2 = (route_poly[i + 1]["lat"], route_poly[i + 1]["lon"])
        seg_center = midpoint(p1, p2)
        d = haversine(seg_center, centroid)
        if d > radius:
            if debug_logs is not None:
                debug_logs.append(f"    Step {i}: distance={d:.2f} > {radius}, skipped")
            continue

        v_step = route_poly[i]["v_bearing"]
        if None in v_step:
            continue

        for cw in members:
            cw_center = cw["center"]
            cross_line = LineString([centroid, cw_center])
            route_line = LineString([p1, p2])

            if not cross_line.crosses(route_line) and must_intersect_route:
                if debug_logs is not None:
                    debug_logs.append(f"    Step {i}: no intersection between route and center-cross line")
                continue

            inter_pt = cross_line.intersection(route_line)

            if inter_pt.is_empty or not isinstance(inter_pt, Point):
                if debug_logs is not None:
                    debug_logs.append(f"    Step {i}: intersection not found or invalid type")
                continue

            pos = route_line.project(inter_pt)
            if must_intersect_route and not 0 <= pos <= route_line.length:
                if debug_logs is not None:
                    debug_logs.append(f"    Step {i}: intersection {list(inter_pt.coords)[0]} not on route segment")
                continue

            if polygon.contains(inter_pt):
                if debug_logs is not None:
                    debug_logs.append(f"    Step {i}: intersection FOUND at {list(inter_pt.coords)[0]}")
                
                # 向き調整：v_crossがv_stepと同じ方向を向くように反転するか判断
                v_cross = normalize(vector(cw["p1"], cw["p2"]))
                if dot(v_cross, v_step) < 0:
                    v_cross = [-v_cross[0], -v_cross[1]]  # 向き反転

                angle_rad = math.atan2(v_cross[1], v_cross[0])
                bearing_deg = (math.degrees(angle_rad) + 360) % 360

                matched.append({
                    "id": cw["id"],
                    "p1": cw["p1"],
                    "p2": cw["p2"],
                    "center": cw["center"],
                    "bearing": bearing_deg
                })
                intersections.append(inter_pt)
                break


    return matched, intersections

def google_directions_api(origin, destination, api_key):
    url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin": f"{origin[0]},{origin[1]}",
        "destination": f"{destination[0]},{destination[1]}",
        "mode": "walking",
        "key": api_key
    }
    response = requests.get(url, params=params)
    return response.json()

def log_debug_message(debug_dir, messages):
    os.makedirs(debug_dir, exist_ok=True)
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(debug_dir, f"debug_log_{now}.txt")
    with open(log_file, "w", encoding="utf-8") as f:
        for line in messages:
            f.write(line + "\n")

def transform_senconds_to_minutes(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h{int(minutes)}m{int(seconds)}s"


def get_route_with_crosswalk(
        origin, destination, api_key, cache_dir="cache",
        sample_step=5, search_radius=200, target_radius=100,
        cluster_eps=80, must_intersect_route=True,
        visualize=False, debug=False):

    os.makedirs(cache_dir, exist_ok=True)
    cache_key = hashlib.md5(f"{origin}_{destination}".encode()).hexdigest()
    cache_file = os.path.join(cache_dir, f"{cache_key}.json")

    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            route_data = json.load(f)
        if debug:
            print("Cache hit: loaded route from cache.")
    else:
        route_data = google_directions_api(origin, destination, api_key)
        with open(cache_file, "w") as f:
            json.dump(route_data, f)

    debug_logs = []
    t0 = time.time()
    route_poly_raw = route_data["routes"][0]["overview_polyline"]["points"]
    decoded_poly = polyline.decode(route_poly_raw)
    total_duration = route_data["routes"][0]["legs"][0]["duration"]["value"]  # 秒数
    decode_time = time.time() - t0
    if debug_logs is not None:
        debug_logs.append(f"Decoded polyline: {len(route_poly_raw)} points. Time: {decode_time:.3f} sec")

    t0 = time.time()
    route_poly = []
    for i in range(len(decoded_poly)):
        lat, lon = decoded_poly[i]
        if i < len(decoded_poly) - 1:
            bearing, v_bearing = calculate_bearing(decoded_poly[i], decoded_poly[i+1])
        else:
            bearing, v_bearing = None, [None, None]

        route_poly.append({
            "lat": lat,
            "lon": lon,
            "bearing": bearing,
            "v_bearing": v_bearing
        })

    bearing_time = time.time() - t0
    if debug_logs is not None:
        debug_logs.append(f"Decoded polyline: {len(route_poly_raw)} points. Time: {bearing_time:.3f} sec")

    if debug:
        print(f"Decoded polyline: {len(decoded_poly)} points.")


    t0 = time.time()
    candidate_crosswalks = []
    crosswalk_ids = set()
    for i in range(0, len(route_poly), sample_step):
        lat, lon = route_poly[i]["lat"], route_poly[i]["lon"]
        data = fetch_crosswalks_around(lat, lon, radius=search_radius)
        for el in data["elements"]:
            if el["type"] == "way" and "nodes" in el and len(el["nodes"]) >= 2:
                if el["id"] not in crosswalk_ids:
                    crosswalk_ids.add(el["id"])
                    nodes = [e for e in data["elements"] if e["type"] == "node" and e["id"] in el["nodes"][:2]]
                    if len(nodes) == 2:
                        p1 = (nodes[0]["lat"], nodes[0]["lon"])
                        p2 = (nodes[1]["lat"], nodes[1]["lon"])
                        candidate_crosswalks.append({
                            "id": el["id"],
                            "p1": p1,
                            "p2": p2,
                            "center": center_point(p1, p2)
                        })

    crosswalk_time = time.time() - t0
    if debug_logs is not None:
        debug_logs.append(f"Detected candidate crosswalks: {len(candidate_crosswalks)}. Time: {crosswalk_time:.3f} sec")

    if debug:
        print(f"Detected candidate crosswalks: {len(candidate_crosswalks)}")

    intersections = []
    matched_crosswalks = []
    matched_indices = set()

    t0 = time.time()
    clusters = cluster_crosswalks(candidate_crosswalks, eps=cluster_eps, debug_logs=debug_logs)
    cluster_time = time.time() - t0
    if debug_logs is not None:
        debug_logs.append(f"Clustered crosswalks: {len(clusters)} clusters. Time: {cluster_time:.3f} sec")

    t0 = time.time()
    intersections = []
    for cluster in clusters:
        targets, cluster_intersections = is_crosswalk_target(
            cluster,
            route_poly,
            radius=target_radius,
            must_intersect_route=True,
            debug_logs=debug_logs
        )
        intersections.extend(cluster_intersections)
        for cw in targets:
            if cw not in matched_crosswalks:
                matched_crosswalks.append(cw)
    target_time = time.time() - t0
    if debug_logs is not None:
        debug_logs.append(f"Target crosswalks: {len(matched_crosswalks)}. Time: {target_time:.3f} sec")


    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if debug:
        log_dir = f"log/{timestamp}"
        log_path = os.path.join(log_dir, f"debug_log.txt")

        os.makedirs(log_dir, exist_ok=True)
        with open(log_path, "w") as f:
            for line in debug_logs:
                f.write(line + "\n")
        print(f"Debug logs saved to {log_path}.")

    if visualize:
        t0 = time.time()
        log_dir = f"log/{timestamp}"
        visualize_route_and_crosswalks(
            route_poly=route_poly,
            candidate_crosswalks=candidate_crosswalks,
            matched_crosswalks=matched_crosswalks,
            intersections=intersections,
            clusters=clusters,
            search_radius=search_radius,
            target_radius=target_radius,
            sample_step=sample_step,
            log_dir=log_dir
        )
        viz_time = time.time() - t0
        if debug_logs is not None:
            debug_logs.append(f"Visualization saved to debug_crosswalks.html. Time: {viz_time:.3f} sec")

    return {
        "origin": origin,
        "destination": destination,
        "total_duration": transform_senconds_to_minutes(total_duration),
        "num_crosswalks": len(matched_crosswalks),
        "route_polyline": route_poly,
        "matched_crosswalks": matched_crosswalks,
    },  f"log/{timestamp}"
