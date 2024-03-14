import numpy as np
import matplotlib.pyplot as plt

class User():
  def give_algorithm(algo):
    if algo == "A":
      return np.random.lognormal(3, .5, 1)[0]
    elif algo == "B":
      return np.random.lognormal(3.01, .5, 1)[0]
    elif algo == "C":
      return np.random.lognormal(3.08, .51, 1)[0]
    elif algo == "D":
      return np.random.lognormal(3, .49, 1)[0]
    
def reset_rewards(arms):
  rewards = {}
  for arm in arms:
    rewards[arm] = []
  return rewards, []

def get_average_rewards(rewards):
  averages = {}
  for reward in rewards.keys():
    if len(rewards[reward])>0:
      averages[reward] = np.mean(rewards[reward])
    else:
      averages[reward] = np.inf
  return averages


def plot_share_over_time(action_plays, total_actions,time_kernel = 100):

  x = [0]
  y = [[.25,.25,.25,.25]]
  counts = {}
  for action in total_actions:
    counts[action] = 0

  for t in range(len(action_plays)):
    action = action_plays[t]
    counts[action] += 1
    if t % time_kernel == 0 and t != 0:
      y_value = []
      for action in counts.keys():
        y_value.append(counts[action]/time_kernel)
      x.append(t)
      y.append(y_value)
      for action in total_actions:
        counts[action] = 0

  x = np.array(x)
  y = np.array(y).T
  print(x.shape)

  plt.stackplot(x,y, labels=total_actions)
  plt.title("Share of algorithms across users")
  plt.xlabel("User number")
  plt.ylabel("Percent of users")
  plt.legend()
  plt.show()

def get_lost_minutes(action_plays):
  n_users = len(action_plays)
  actual_total = 0
  optimal_total = 0
  for action in action_plays:
    C_play = User.give_algorithm("C")
    optimal_total += C_play
    if action == "C":
      actual_total+=C_play
    else:
      actual_total+=User.give_algorithm(action)
  diff = optimal_total - actual_total
  print("Over %s users, your strategy was %s minutes from the optimal"%(len(action_plays),diff))

def get_upper_confidence_bound(rewards, c):
  UCBs = {}
  T = 0
  for action in rewards.keys():
    T += len(rewards[action])
  for action in rewards.keys():
    if len(rewards[action]) == 0:
      UCBs[action] = np.inf
    else:
      UCBs[action] = np.mean(rewards[action]) + c*np.sqrt(np.log(T)/len(rewards[action]))
  return UCBs