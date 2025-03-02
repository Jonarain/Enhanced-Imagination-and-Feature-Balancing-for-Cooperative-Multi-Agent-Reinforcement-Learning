from collections import defaultdict
import logging
import torch as th

class Logger:
    def __init__(self, console_logger):
        self.console_logger = console_logger

        self.use_tb = False
        self.use_sacred = False
        self.use_hdf = False

        self.stats = defaultdict(lambda: [])

    def setup_tb(self, directory_name):
        # Import here so it doesn't have to be installed if you don't use it
        from tensorboard_logger import configure, log_value, log_histogram
        configure(directory_name)
        self.tb_logger = log_value
        self.tb_hist_logger = log_histogram
        self.use_tb = True

    def setup_sacred(self, sacred_run_dict):
        self.sacred_info = sacred_run_dict.info
        self.use_sacred = True

    def log_stat(self, key, value, t, to_sacred=True):
        self.stats[key].append((t, value))

        if self.use_tb:
            self.tb_logger(key, value, t)

        if self.use_sacred and to_sacred:
            if key in self.sacred_info:
                self.sacred_info["{}_T".format(key)].append(t)
                self.sacred_info[key].append(value)
            else:
                self.sacred_info["{}_T".format(key)] = [t]
                self.sacred_info[key] = [value]

    def log_hist(self, key, values, t):
        if self.use_tb:
            self.tb_hist_logger(key, values, t)

    def log_hist_by_i(self, key, values, episode):
        # value: [batch, step, agent]
        if self.use_tb:
            for i in range(values.size()[-1]): # agent index dim
                for e in range(values.size()[0]): # the number of batch dim
                    self.tb_hist_logger(key+f"_agent_{i}", values[e], episode-1+values.size()[0]+e)

    def log_state_by_i(self, key, values, t):
        # value: [step, agent]
        if self.use_tb:
            for i in range(values.size()[-1]):  # agent index dim
                for st in range(values.size()[0]):
                    self.tb_logger(key+f"_agent_{i}", values[st, i], t-1+values.size()[0]+st)

    def log_state_by_step(self, key, values, t):
        # value: [step, 1]
        if self.use_tb:
            for st in range(values.size()[0]):
                self.tb_logger(key, values[st, 0], t-1+values.size()[0]+st)

    def print_recent_stats(self):
        log_str = "Recent Stats | t_env: {:>10} | Episode: {:>8}\n".format(*self.stats["episode"][-1])
        i = 0
        for (k, v) in sorted(self.stats.items()):
            if k == "episode":
                continue
            i += 1
            window = 5 if k != "epsilon" else 1

            #bug fix
            tensor_chk = []
            for x in self.stats[k][-window:]:
                if not th.is_tensor(x[1]):
                    tensor_chk.append(th.tensor(x[1]))
                else:
                    tensor_chk.append(x[1])
            #########
            item = "{:.4f}".format(th.mean(th.stack(tensor_chk)))
            log_str += "{:<25}{:>8}".format(k.split("/")[-1] + ":", item)
            log_str += "\n" if i % 4 == 0 else "\t"
        self.console_logger.info(log_str)


# set up a custom logger
def get_logger():
    logger = logging.getLogger()
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(name)s %(message)s', '%H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel('DEBUG')

    return logger

