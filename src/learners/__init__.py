from .q_learner import QLearner
from .mbvd_learner import MBVDLearner
from .oracle_mbvd_learner import OracleMBVDLearner
from .qplex import Qplex
from .qplex_mbvd_learner import Qplex_MBVDLearner
from .qplex_oracle_mbvd_learner import Qplex_OracleMBVDLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["mbvd_learner"] = MBVDLearner
REGISTRY["oracle_mbvd_learner"] = OracleMBVDLearner

REGISTRY["qplex"] = Qplex
REGISTRY["qplex_mbvd_learner"] = Qplex_MBVDLearner
REGISTRY["qplex_oracle"] = Qplex_OracleMBVDLearner