from lxml import etree
from lxml.etree import Element
import rospkg
from lxml import etree
from lxml.etree import Element
rospack = rospkg.RosPack()
actor_pkg_path = rospack.get_path("social_navigation")
world_name = "world.world"
tree_ = etree.parse(actor_pkg_path+'/worlds/'+world_name)
world_ = tree_.getroot().getchildren()[0]
actor = Element("actor", name="actor1")
world_.append(actor)
tree_.write(actor_pkg_path+'/worlds/test.world', pretty_print=True, xml_declaration=True, encoding="utf-8")
