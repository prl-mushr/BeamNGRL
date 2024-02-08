from beamngpy import BeamNGpy
import os

BeamNG_path = os.environ.get("BNG_HOME")


bng = BeamNGpy("localhost", 64256, home=BeamNG_path, user=BeamNG_path + "/userfolder")
bng.open()
exit()
