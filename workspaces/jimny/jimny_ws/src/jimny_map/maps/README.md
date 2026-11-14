# Road graphs

GeoJSON road graphs made with `osm_to_graph` (see `jimny_map/osm_to_graph.py` for the format).
The raw OSM exports they were made from are kept in `osm/`, so a graph can be rebuilt when the
importer changes.

| Graph | Source | Area |
|---|---|---|
| `test_area.geojson` | `osm/test_area.osm` (openstreetmap.org export, 2026-11-14) | bbox -25.38854, 28.37069 .. -25.37363, 28.38648 |

```bash
# bbox: south west north east (degrees). Pick it a bit larger than the test route.
ros2 run jimny_map osm_to_graph --bbox <S> <W> <N> <E> -o src/jimny_map/maps/test_area.geojson
# or from an openstreetmap.org export (Export -> "Manually select a different area")
ros2 run jimny_map osm_to_graph --osm area.osm -o src/jimny_map/maps/test_area.geojson
```

Check the result in geojson.io or QGIS, then `colcon build --packages-select jimny_map` so the file
is installed. `map_view.launch.py graph_file:=<name>.geojson` loads it from here.
