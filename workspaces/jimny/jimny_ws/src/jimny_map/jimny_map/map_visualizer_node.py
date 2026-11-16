"""
Show the OSM road graph and the driven trails in RViz.

The graph (GeoJSON from osm_to_graph) is in lat/lon. It is converted into the map frame with
navsat_transform_node's fromLL service, i.e. with exactly the same UTM -> map transform the EKF
uses. The map frame origin is where navsat_transform initialised (the car's start position), so
the conversion can only run once navsat_transform has a GPS fix and IMU heading; until then the
node waits. If navsat_transform is restarted, restart this node too.

Publishes (latched, transient_local):
  /route_map/markers       visualization_msgs/MarkerArray  roads, junctions, junction ids
Publishes:
  /route_map/gps_trail     nav_msgs/Path  from /odometry/gps    (raw GPS in the map frame)
  /route_map/global_trail  nav_msgs/Path  from /odometry/global (EKF map estimate)
  /route_map/local_trail   nav_msgs/Path  from /odometry/local  (EKF odom estimate, odom frame)
"""

import json
import os

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from robot_localization.srv import FromLL
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


def color(r, g, b, a=1.0):
    return ColorRGBA(r=r, g=g, b=b, a=a)


ROAD_COLOR = color(0.85, 0.85, 0.85)
ONEWAY_COLOR = color(0.3, 0.6, 1.0)
JUNCTION_COLOR = color(1.0, 0.55, 0.0)
DEAD_END_COLOR = color(0.6, 0.6, 0.6)


class MapVisualizerNode(Node):

    def __init__(self):
        super().__init__('map_visualizer')
        self.declare_parameter('graph_file', 'test_area.geojson')
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('fromll_service', '/fromLL')
        self.declare_parameter('trail_min_spacing', 0.3)   # m between stored trail poses
        self.declare_parameter('trail_max_poses', 20000)

        self.frame_id = self.get_parameter('frame_id').value
        self.trail_min_spacing = float(self.get_parameter('trail_min_spacing').value)
        self.trail_max_poses = int(self.get_parameter('trail_max_poses').value)

        self.nodes, self.edges = self._load_graph(self.get_parameter('graph_file').value)
        self.get_logger().info(f'loaded {len(self.nodes)} nodes, {len(self.edges)} edges')

        # Every distinct lat/lon in the graph, converted one by one via fromLL.
        points = {tuple(n['lonlat']) for n in self.nodes}
        for edge in self.edges:
            points.update(tuple(p) for p in edge['lonlat'])
        self._pending = sorted(points)
        self._converted = {}

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.marker_publisher = self.create_publisher(MarkerArray, '/route_map/markers', latched)

        self.client = self.create_client(FromLL, self.get_parameter('fromll_service').value)
        self._waiting_timer = self.create_timer(1.0, self._start_conversion)

        self.trails = {}
        for topic, out in (('/odometry/gps', '/route_map/gps_trail'),
                           ('/odometry/global', '/route_map/global_trail'),
                           ('/odometry/local', '/route_map/local_trail')):
            path = Path()
            self.trails[topic] = (path, self.create_publisher(Path, out, 10))
            self.create_subscription(
                Odometry, topic, lambda msg, t=topic: self._odometry_callback(t, msg), 10)

    # --- graph ---------------------------------------------------------------------------------

    def _load_graph(self, graph_file):
        if not os.path.isabs(graph_file):
            graph_file = os.path.join(
                get_package_share_directory('jimny_map'), 'maps', graph_file)
        with open(graph_file) as f:
            features = json.load(f)['features']
        nodes, edges = [], []
        for feature in features:
            props = feature['properties']
            coords = feature['geometry']['coordinates']
            if props.get('kind') == 'node':
                nodes.append({'id': props['id'], 'degree': props['degree'], 'lonlat': coords})
            elif props.get('kind') == 'edge':
                edges.append({'id': props['id'], 'oneway': props['oneway'], 'lonlat': coords})
        return nodes, edges

    def _start_conversion(self):
        # The fromLL service exists before navsat_transform has its transform and then answers
        # with zeros. /odometry/gps is only published once the transform is valid.
        if not self.client.service_is_ready() or not self.trails['/odometry/gps'][0].poses:
            self.get_logger().info(
                'waiting for navsat_transform fromLL (needs a GPS fix and IMU heading) ...',
                throttle_duration_sec=10.0)
            return
        self._waiting_timer.cancel()
        self.get_logger().info(f'converting {len(self._pending)} points to {self.frame_id}')
        self._request_next()

    def _request_next(self):
        if not self._pending:
            self._publish_markers()
            return
        lon, lat = self._pending[-1]
        request = FromLL.Request()
        request.ll_point.latitude = lat
        request.ll_point.longitude = lon
        request.ll_point.altitude = 0.0
        self.client.call_async(request).add_done_callback(self._on_converted)

    def _on_converted(self, future):
        lonlat = self._pending.pop()
        result = future.result()
        if result is None:
            self.get_logger().error(f'fromLL failed for {lonlat}, map will be incomplete')
        else:
            self._converted[lonlat] = (result.map_point.x, result.map_point.y)
        self._request_next()

    def _xy(self, lonlat):
        x, y = self._converted.get(tuple(lonlat), (float('nan'), float('nan')))
        return Point(x=x, y=y, z=0.0)

    def _publish_markers(self):
        xs = [x for x, _ in self._converted.values()]
        if len(xs) > 1 and max(xs) - min(xs) < 1e-3:
            self.get_logger().error('fromLL returned the same point for everything; '
                                    'navsat_transform was probably not initialised')
            return
        now = self.get_clock().now().to_msg()

        def marker(ns, marker_type, scale):
            m = Marker()
            m.header.frame_id = self.frame_id
            m.header.stamp = now
            m.ns = ns
            m.type = marker_type
            m.action = Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = scale
            return m

        roads = marker('roads', Marker.LINE_LIST, 0.3)
        for edge in self.edges:
            c = ONEWAY_COLOR if edge['oneway'] else ROAD_COLOR
            for a, b in zip(edge['lonlat'][:-1], edge['lonlat'][1:]):
                roads.points += [self._xy(a), self._xy(b)]
                roads.colors += [c, c]

        nodes = marker('nodes', Marker.SPHERE_LIST, 1.5)
        labels = MarkerArray()
        for n in self.nodes:
            if n['degree'] == 2:
                continue  # just a split between two OSM ways, not a decision point
            nodes.points.append(self._xy(n['lonlat']))
            nodes.colors.append(JUNCTION_COLOR if n['degree'] >= 3 else DEAD_END_COLOR)
            if n['degree'] >= 3:
                label = marker('junction_ids', Marker.TEXT_VIEW_FACING, 2.0)
                label.id = n['id']
                label.text = str(n['id'])
                label.pose.position = self._xy(n['lonlat'])
                label.pose.position.z = 2.0
                label.color = color(1.0, 1.0, 1.0)
                labels.markers.append(label)

        markers = MarkerArray()
        markers.markers = [roads, nodes] + labels.markers
        self.marker_publisher.publish(markers)
        self.get_logger().info(
            f'published map: {len(self.edges)} roads, {len(nodes.points)} junctions/dead ends')

    # --- trails --------------------------------------------------------------------------------

    def _odometry_callback(self, topic, msg):
        path, publisher = self.trails[topic]
        position = msg.pose.pose.position
        if path.poses:
            last = path.poses[-1].pose.position
            if (last.x - position.x) ** 2 + (last.y - position.y) ** 2 < \
                    self.trail_min_spacing ** 2:
                return
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose = msg.pose.pose
        path.header = msg.header
        path.poses.append(pose)
        if len(path.poses) > self.trail_max_poses:
            del path.poses[0]
        publisher.publish(path)


def main(args=None):
    rclpy.init(args=args)
    node = MapVisualizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
