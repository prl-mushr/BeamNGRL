import numpy as np

def update_goal(goal, pos, target_WP, current_wp_index, lookahead):
    if goal is None:
        if current_wp_index == 0:
            return target_WP[current_wp_index, :2], False, current_wp_index
        else:
            print("bruh moment")
            return pos, True, current_wp_index  ## terminate
    else:
        d = np.linalg.norm(goal - pos)
        if d < lookahead and current_wp_index < len(target_WP) - 1:
            current_wp_index += 1
            return target_WP[current_wp_index, :2], False, current_wp_index  ## new goal
        if current_wp_index == len(target_WP):
            return pos, True, current_wp_index  # Terminal condition
        else:
            return goal, False, current_wp_index