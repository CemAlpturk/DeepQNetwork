
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
        self.loss_file_name = "loss.csv"
        self.q_file_name = "q_values.csv"

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

    def log_episode(self, states, u, rewards, terminated, t, episode):
        data = {
                "States": states,
                "Force": u,
                "Reward": rewards,
                "Terminated": terminated,
                "Time": t}
        df = pd.DataFrame(data)
        #print(df.head())
        dir = os.path.join(self.ep_dir,f"Episode_{episode}.csv")
        df.to_csv(dir, index=False)

    def log_loss(self, loss, episode):
        """
        Appends average loss score for each episode to the appropriate file
        """
        with open(self.loss_dir, 'a+', newline='') as write_obj:
            csv_writer = writer(write_obj)
            csv_writer.writerow([episode,loss])

        if episode % 10 == 0:
            # Generate plot for losses
            df = pd.read_csv(self.loss_dir)
            ax = df.plot(x="Episode", y="Loss")
            fig = ax.get_figure()
            fig_dir = os.path.join(os.path.dirname(self.loss_dir),"Loss.png")
            fig.savefig(fig_dir)
            plt.close()

    def log_q_values(self, q_values, episode):
        """
        Appends Q-values for each episode to the appropriate file
        """
        # data = {"Episode": episode, "Q-values": q_values}
        # df = pd.DataFrame(data)
        # df.to_pickle(self.q_dir, mode='a', header=False)
        q_num = len(q_values)
        if episode == 1:
            header = ["Episode"]
            for i in range(1, q_num+1):
                header.append(f"Q-{i}")
            with open(self.q_dir, 'a+', newline='') as write_obj:
                csv_writer = writer(write_obj)
                csv_writer.writerow(header)

        row = [episode]
        for q in q_values:
            row.append(q)
        with open(self.q_dir, 'a+', newline='') as write_obj:
            csv_writer = writer(write_obj)
            csv_writer.writerow(row)

        if episode % 10 == 0:
            df = pd.read_csv(self.q_dir)
            ax = df.plot(x="Episode")
            fig = ax.get_figure()
            fig_dir = os.path.join(os.path.dirname(self.q_dir),"Q_values.png")
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

        # Create directory for episode metrics
        ep_dir = os.path.join(timedir,"Episodes")
        print(f"Creating 'Episodes' directory at: {timedir}")
        os.mkdir(ep_dir)
        self.ep_dir = ep_dir

        # Create directory for episode losses
        losses_dir = os.path.join(timedir,"Loss")
        print(f"Creating 'Loss' directory at: {timedir}")
        os.mkdir(losses_dir)
        self.loss_dir = os.path.join(losses_dir, self.loss_file_name)
        with open(self.loss_dir, 'w', newline='') as write_obj:
            csv_writer = writer(write_obj)
            csv_writer.writerow(["Episode","Loss"])

        # Create directory for episode q-values
        q_dir = os.path.join(timedir,"Q_values")
        print(f"Creating 'Q_values' directory at: {timedir}")
        os.mkdir(q_dir)
        self.q_dir = os.path.join(q_dir, self.q_file_name)
        file = open(self.q_dir, 'w', newline='')
        file.close()
