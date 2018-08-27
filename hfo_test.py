from pathlib import Path

import hfo
from hfo.hfo import *
base_dir = Path('/home/goncalo/HFO')
config_dir = base_dir / 'bin/teams/base/config/formations-dt'

hfo = hfo.HFOEnvironment()
hfo.connectToServer(HIGH_LEVEL_FEATURE_SET, config_dir=str(config_dir))
for episode in range(5): # replace with xrange(5) for Python 2.X
    status = IN_GAME
    while status == IN_GAME:
        features = hfo.getState()
        print(features[:2])
        # dist = features[33]
        hfo.act(DASH, 20.0, 0.0)
        status = hfo.step()
    print('episode', episode)