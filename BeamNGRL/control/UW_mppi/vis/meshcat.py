import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf

class Vis:
  def __init__(self):
    # x y z ? (m)
    self.car = g.Box((1.0, 0.5, 0.3))
    # radius (m)
    self.goal = g.Sphere(0.2)

    self.vis = meshcat.Visualizer().open()
    self.vis['car'].set_object(self.car)
    self.vis['goal'].set_object(self.goal, g.MeshBasicMaterial(color=0xaaffaa))

  def setcar(self, pos, yaw):
    self.vis['car'].set_transform(
      tf.compose_matrix(angles=(0.0, 0.0, yaw), translate=(pos[0], pos[1], 0.0))
    )

  def setgoal(self, pos):
    self.vis['goal'].set_transform(tf.translation_matrix((pos[0], pos[1], 0.0)))
