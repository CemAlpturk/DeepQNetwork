
"""
Records the data given to it ?
"""

import os
import datetime
from csv import writer

import pandas as pd
from matplotlib import pyplot as plt


class Logger:

    def __init__(self, problem_name):
        self.problem_name = problem_name
        self.evals_file_name = "evals.csv"

        self._init_directory()


    def log_eval(self, episode, score):
        """
        Append inputs to apropriate directory
        """
        with open(self.evals_dir, 'a+', newline='') as write_obj:
            csv_writer = writer(write_obj)
            csv_writer.writerow([episode,score])

        # Generate plot for scores
        df = pd.read_csv(self.evals_dir)
        ax = df.plot(x="Episode", y="Score")
        fig = ax.get_figure()
        fig_dir = os.path.join(os.path.dirname(self.evals_dir),"Scores.png")
        fig.savefig(fig_dir)
        plt.close()

    def _init_directory(self):

        # Check if directory exists
        parent_dir = os.getcwd()
        dir = "Logs"
        path = os.path.join(parent_dir,dir)

        if not os.path.exists(path):
            print(f"Creating 'Logs' directory at: {parent_dir}")
            os.mkdir(path)

        # Check directory for problem problem name
        prob_dir = os.path.join(path, self.problem_name)
        if not os.path.exists(prob_dir):
            print(f"Creating '{self.problem_name}' directory at: {path}")
            os.mkdir(prob_dir)

        # Create directory with timestamp
        timestamp  = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        timedir = os.path.join(prob_dir, timestamp)
        print(f"Creating '{timestamp}' directory at: {prob_dir}")
        os.mkdir(timedir)
        self.dir = timedir

        # Create directory for evaluation scores
        eval_dir = os.path.join(timedir, "Evaluation")
        print(f"Creating 'Evaluation' directory at: {timedir}")
        os.mkdir(eval_dir)
        self.evals_dir = os.path.join(eval_dir, self.evals_file_name)
        with open(self.evals_dir, 'w', newline='') as write_obj:
            csv_writer = writer(write_obj)
            csv_writer.writerow(["Episode","Score"])
