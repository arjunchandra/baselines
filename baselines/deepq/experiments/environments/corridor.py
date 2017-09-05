import numpy as np
import sys
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
from gym import ObservationWrapper
from gym import utils
from gym import spaces
from gym.envs.registration import register
from gym.spaces import prng
from gym import Env
from gym import Space
from gym.utils import seeding

MAPS = {
  "4x4": [
    "HHHD",
    "FFFF",
    "FHHH",
    "SHHH",
  ],
  "9x9": [
    "HHHHHHHHD",
    "HHHHHHHHF",
    "HHHHHHHHF",
    "HHHHHHHHF",
    "FFFFFFFFF",
    "FHHHHHHHH",
    "FHHHHHHHH",
    "FHHHHHHHH",
    "SHHHHHHHH",
  ],
  "1x4": [
    "SFFD",
  ],
  "2x4": [
    "SFFD",
    "HHHH",
  ],
}


class DiscreteSpace(Space):
    """
    {0,1,...,n-1}
    Example usage:
    self.observation_space = spaces.Discrete(2)
    """
    def __init__(self, n):
        self.n = n
    def sample(self):
        return prng.np_random.randint(self.n)
    def contains(self, x):
        if isinstance(x, int):
            as_int = x
        elif isinstance(x, (np.generic, np.ndarray)) and (x.dtype.kind in np.typecodes['AllInteger'] and x.shape == ()):
            as_int = int(x)
        else:
            return False
        return as_int >= 0 and as_int < self.n
    def __repr__(self):
        return "Discrete(%d)" % self.n
    def __eq__(self, other):
        return self.n == other.n

    @property
    def shape(self):
      return [self.n]

def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()


class MyDiscreteEnv(Env):

    """
    Has the following members
    - nS: number of states
    - nA: number of actions
    - P: transitions (*)
    - isd: initial state distribution (**)
    (*) dictionary dict of dicts of lists, where
      P[s][a] == [(probability, nextstate, reward, done), ...]
    (**) list or array of length nS
    """
    def __init__(self, nS, nA, P, isd):
        self.P = P
        self.isd = isd
        self.lastaction=None # for rendering
        self.nS = nS
        self.nA = nA

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = DiscreteSpace(self.nS)

        self._seed()
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.s = categorical_sample(self.isd, self.np_random)
        self.lastaction=None
        return self.s

    def _step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d= transitions[i]
        self.s = s
        self.lastaction=a
        return (s, r, d, {"prob" : p})



class ProcessObservation(ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessObservation, self).__init__(env)
        print(type(env))


    def _observation(self, obs):
        return ProcessObservation.process(obs, self.env.observation_space.n)

    @staticmethod
    def process(obs, observation_space_n):
      new_obs = np.zeros([observation_space_n])
      new_obs[obs] = 1
      return new_obs

#class CorridorEnv(discrete.DiscreteEnv):
class CorridorEnv(MyDiscreteEnv):
  """
  The surface is described using a grid like the following

    HHHD
    FFFF
    SHHH
    AHHH

  S : starting point, safe
  F : frozen surface, safe
  H : hole, fall to your doom
  A : adjacent goal
  D : distant goal

  The episode ends when you reach the goal or fall in a hole.
  You receive a reward of 1 if you reach the adjacent goal, 
  10 if you reach the distant goal, and zero otherwise.
  """
  metadata = {'render.modes': ['human', 'ansi']}

  def __init__(self, desc=None, map_name="9x9", n_actions=5):
    if desc is None and map_name is None:
      raise ValueError('Must provide either desc or map_name')
    elif desc is None:
      desc = MAPS[map_name]
    self.desc = desc = np.asarray(desc, dtype='c')
    self.nrow, self.ncol = nrow, ncol = desc.shape

    #self.action_space = spaces.Discrete(n_actions)
    #self.observation_space = spaces.Discrete(desc.size)
    #self.observation_space = DiscreteSpace(desc.size)

    n_state = nrow * ncol

    isd = np.array(desc == b'S').astype('float64').ravel()
    isd /= isd.sum()

    P = {s : {a : [] for a in range(n_actions)} for s in range(n_state)}

    def to_s(row, col):
      return row*ncol + col
    def inc(row, col, a):
      if a == 0: # left
        col = max(col-1,0)
      elif a == 1: # down
        row = min(row+1, nrow-1)
      elif a == 2: # right
        col = min(col+1, ncol-1)
      elif a == 3: # up
        row = max(row-1, 0)

      return (row, col)

    for row in range(nrow):
      for col in range(ncol):
        s = to_s(row, col)
        for a in range(n_actions):
          li = P[s][a]
          newrow, newcol = inc(row, col, a)
          newstate = to_s(newrow, newcol)
          letter = desc[newrow, newcol]
          done = letter in b'DAH'
          rew = 1.0 if letter == b'A' \
              else 10.0 if letter == b'D' \
              else -1.0 if letter == b'H' \
              else 1.0 if (newrow != row or newcol != col) and letter == b'F' \
              else 0.0
          li.append((1.0/3.0, newstate, rew, done))

    super(CorridorEnv, self).__init__(n_state, n_actions, P, isd)

  def _render(self, mode='human', close=False):
    if close:
      return

    outfile = StringIO.StringIO() if mode == 'ansi' else sys.stdout

    row, col = self.s // self.ncol, self.s % self.ncol
    desc = self.desc.tolist()
    desc = [[c.decode('utf-8') for c in line] for line in desc]
    desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)

    if self.lastaction is not None:
      outfile.write("  ({})\n".format(self.get_action_meanings()[self.lastaction]))
    else:
      outfile.write("\n")

    outfile.write("\n".join("".join(row) for row in desc) + "\n")

    return outfile

  def get_action_meanings(self):
    return [["Left", "Down", "Right", "Up"][i] if i < 4 else "NoOp" for i in range(self.action_space.n)]


register(
  id='CorridorToy-v1',
  entry_point='baselines.deepq.experiments.environments.corridor:CorridorEnv',
  kwargs={
    'map_name': '1x4',
    'n_actions': 4
  },
  timestep_limit=100,
)

register(
  id='CorridorToy-v2',
  entry_point='baselines.deepq.experiments.environments.corridor:CorridorEnv',
  kwargs={
    'map_name': '2x4',
    'n_actions': 4
  },
  timestep_limit=100,
)

register(
  id='CorridorSmall-v5',
  entry_point='baselines.deepq.experiments.environments.corridor:CorridorEnv',
  kwargs={
    'map_name': '4x4',
    'n_actions': 5
  },
  timestep_limit=100,
)

register(
  id='CorridorSmall-v10',
  entry_point='baselines.deepq.experiments.environments.corridor:CorridorEnv',
  kwargs={
    'map_name': '4x4',
    'n_actions': 10
  },
  timestep_limit=100,
)

register(
  id='CorridorBig-v5',
  entry_point='baselines.deepq.experiments.environments.corridor:CorridorEnv',
  kwargs={
    'map_name': '9x9',
    'n_actions': 5
  },
  timestep_limit=100,
)

register(
  id='CorridorBig-v10',
  entry_point='baselines.deepq.experiments.environments.corridor:CorridorEnv',
  kwargs={
    'map_name': '9x9',
    'n_actions': 10
  },
  timestep_limit=100,
)



