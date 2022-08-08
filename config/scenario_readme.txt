scenario_x.gz can be load with joblib
joblib.load('scenario_x.gz')

and is tuple (initial, goal, scenario)

initial is initial jackal [x, y] in world coordinate
goal is jackal goal [x, y] in world coordinate

scenario is list of actor (pedestrian),
actor is dictionary
 'v': velocity
 'loc': [x, y] in world coordinate
 'dx': actor starts to move if x-distance is smaller than this

Ex)
initial, scenario = joblib.load('scenario_x.gz')
v = scenario['v']
loc = scenario['loc']
dx = scenario['dx']
