import xml.etree.ElementTree as ET

tree = ET.parse("/home/vaydingul20/Documents/Codes/carla-roach/carla_gym/envs/scenario_descriptions/LAV/routes.xml")

for (i, route) in enumerate(tree.iter("route")):

	route_id = int(route.attrib['id'])
	route_town = route.attrib['town']
	route_weather = list(route.iter('weather'))[0].attrib
	
	route_descriptions_dict[route_id] = {}
	route_descriptions_dict[route_id]['town'] = route_town
	route_descriptions_dict[route_id]['weather'] = route_weather

	
	waypoint_list = []  # the list of waypoints that can be found on this route for this actor
	for waypoint in actor.iter('waypoint'):
		location = carla.Location(
			x=float(waypoint.attrib['x']),
			y=float(waypoint.attrib['y']),
			z=float(waypoint.attrib['z']))
		rotation = carla.Rotation(
			roll=float(waypoint.attrib['roll']),
			pitch=float(waypoint.attrib['pitch']),
			yaw=float(waypoint.attrib['yaw']))
		waypoint_list.append(carla.Transform(location, rotation))

	route_descriptions_dict[route_id]["waypoints"] = waypoint_list

