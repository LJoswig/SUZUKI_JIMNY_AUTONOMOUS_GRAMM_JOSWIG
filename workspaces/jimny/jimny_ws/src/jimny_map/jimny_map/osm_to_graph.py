"""
OpenStreetMap -> drivable road graph (GeoJSON).

Usage (no ROS needed, Python stdlib only):
  # from a file exported on openstreetmap.org (Export -> Manually select a different area)
  ros2 run jimny_map osm_to_graph --osm area.osm -o maps/test_area.geojson
  # or download the bounding box from the Overpass API
  ros2 run jimny_map osm_to_graph --bbox 48.1000 11.5000 48.1100 11.5150 -o maps/test_area.geojson
  (bbox order: south west north east, in degrees)

Graph:
  - vertices = junctions (OSM nodes shared by two or more drivable ways) and dead ends
  - edges    = road segments between vertices, with their full shape
OSM ways are split at every junction, so each edge connects exactly two vertices.

Output GeoJSON FeatureCollection (coordinates are [lon, lat] as GeoJSON requires):
  Point      properties: kind="node", id, osm_id, degree
  LineString properties: kind="edge", id, start_id, end_id, length_m, highway, name, oneway, osm_way
Edges are undirected; oneway=true means only start -> end may be driven.
The file opens directly in geojson.io or QGIS for a quick check.
"""

import argparse
import json
import math
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

OVERPASS_URL = 'https://overpass-api.de/api/interpreter'

# Roads a car may drive on. 'track' and 'service' are included because test areas are often
# side roads or private grounds; exclude them with --exclude if not wanted.
DRIVABLE = {
    'motorway', 'motorway_link', 'trunk', 'trunk_link', 'primary', 'primary_link',
    'secondary', 'secondary_link', 'tertiary', 'tertiary_link', 'unclassified',
    'residential', 'living_street', 'service', 'track', 'road',
}
NO_ACCESS = {'no', 'private'}
EARTH_RADIUS_M = 6371008.8


def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def download_bbox(south, west, north, east):
    highway_regex = '|'.join(sorted(DRIVABLE))
    query = (
        f'[out:xml][timeout:60];'
        f'(way["highway"~"^({highway_regex})$"]({south},{west},{north},{east}););'
        f'(._;>;);out body;'
    )
    data = urllib.parse.urlencode({'data': query}).encode()
    request = urllib.request.Request(
        OVERPASS_URL, data=data, headers={'User-Agent': 'jimny_map/osm_to_graph'})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def parse_osm(xml_bytes, highway_types, include_private):
    root = ET.fromstring(xml_bytes)
    coords = {}
    for node in root.iter('node'):
        coords[node.get('id')] = (float(node.get('lat')), float(node.get('lon')))

    ways = []
    for way in root.iter('way'):
        tags = {tag.get('k'): tag.get('v') for tag in way.iter('tag')}
        if tags.get('highway') not in highway_types:
            continue
        if not include_private and tags.get('access') in NO_ACCESS:
            continue
        refs = [nd.get('ref') for nd in way.iter('nd') if nd.get('ref') in coords]
        if len(refs) < 2:
            continue
        oneway = tags.get('oneway', 'no')
        if oneway == '-1':      # oneway against the drawing direction
            refs.reverse()
            oneway = 'yes'
        if tags.get('junction') == 'roundabout' and oneway == 'no':
            oneway = 'yes'      # roundabouts are implicitly oneway
        ways.append({
            'osm_id': way.get('id'),
            'refs': refs,
            'highway': tags.get('highway'),
            'name': tags.get('name', ''),
            'oneway': oneway in ('yes', 'true', '1'),
        })
    return coords, ways


def build_graph(coords, ways):
    # How many ways (and way ends) touch each OSM node; >= 2 means it is a graph vertex.
    usage = {}
    for way in ways:
        refs = way['refs']
        for i, ref in enumerate(refs):
            usage[ref] = usage.get(ref, 0) + (2 if i in (0, len(refs) - 1) else 1)
        # A way that visits the same node twice (loops) also needs a split there.
        seen = set()
        for ref in refs:
            if ref in seen:
                usage[ref] = usage.get(ref, 0) + 2
            seen.add(ref)
    is_vertex = {ref for ref, count in usage.items() if count >= 2}

    vertex_ids = {}
    edges = []
    degree = {}
    for way in ways:
        segment = [way['refs'][0]]
        for ref in way['refs'][1:]:
            segment.append(ref)
            if ref in is_vertex:
                start, end = segment[0], segment[-1]
                for v in (start, end):
                    vertex_ids.setdefault(v, len(vertex_ids))
                    degree[v] = degree.get(v, 0) + 1
                length = sum(haversine_m(*coords[a], *coords[b])
                             for a, b in zip(segment[:-1], segment[1:]))
                edges.append({
                    'id': len(edges),
                    'start': start,
                    'end': end,
                    'refs': segment,
                    'length_m': round(length, 2),
                    'highway': way['highway'],
                    'name': way['name'],
                    'oneway': way['oneway'],
                    'osm_way': way['osm_id'],
                })
                segment = [ref]
    return vertex_ids, degree, edges


def to_geojson(coords, vertex_ids, degree, edges):
    features = []
    for osm_id, vid in sorted(vertex_ids.items(), key=lambda item: item[1]):
        lat, lon = coords[osm_id]
        features.append({
            'type': 'Feature',
            'geometry': {'type': 'Point', 'coordinates': [lon, lat]},
            'properties': {'kind': 'node', 'id': vid, 'osm_id': osm_id,
                           'degree': degree[osm_id]},
        })
    for edge in edges:
        features.append({
            'type': 'Feature',
            'geometry': {'type': 'LineString',
                         'coordinates': [[coords[r][1], coords[r][0]] for r in edge['refs']]},
            'properties': {'kind': 'edge', 'id': edge['id'],
                           'start_id': vertex_ids[edge['start']],
                           'end_id': vertex_ids[edge['end']],
                           'length_m': edge['length_m'], 'highway': edge['highway'],
                           'name': edge['name'], 'oneway': edge['oneway'],
                           'osm_way': edge['osm_way']},
        })
    return {'type': 'FeatureCollection', 'features': features}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--osm', help='OSM XML file (openstreetmap.org export)')
    source.add_argument('--bbox', nargs=4, type=float,
                        metavar=('SOUTH', 'WEST', 'NORTH', 'EAST'),
                        help='download this bounding box from the Overpass API')
    parser.add_argument('-o', '--output', required=True, help='output .geojson')
    parser.add_argument('--exclude', nargs='*', default=[],
                        help='highway types to drop, e.g. --exclude track service')
    parser.add_argument('--include-private', action='store_true',
                        help='keep ways tagged access=private/no')
    args = parser.parse_args(argv)

    if args.osm:
        with open(args.osm, 'rb') as f:
            xml_bytes = f.read()
    else:
        print('Downloading from Overpass ...', file=sys.stderr)
        xml_bytes = download_bbox(*args.bbox)

    highway_types = DRIVABLE - set(args.exclude)
    coords, ways = parse_osm(xml_bytes, highway_types, args.include_private)
    vertex_ids, degree, edges = build_graph(coords, ways)
    geojson = to_geojson(coords, vertex_ids, degree, edges)

    with open(args.output, 'w') as f:
        json.dump(geojson, f, indent=1)
    junctions = sum(1 for d in degree.values() if d >= 3)
    total_km = sum(e['length_m'] for e in edges) / 1000
    print(f'{len(ways)} ways -> {len(vertex_ids)} nodes ({junctions} junctions), '
          f'{len(edges)} edges, {total_km:.2f} km -> {args.output}', file=sys.stderr)


if __name__ == '__main__':
    main()
