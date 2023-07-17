import numpy as np

def update_goal(goal, pos, target_WP, current_wp_index, lookahead, step_size=1, wp_radius = 2.0):
    if goal is None:
        if current_wp_index == 0:
            return target_WP[current_wp_index, :2], False, current_wp_index
        else:
            print("bruh moment")
            return pos, True, current_wp_index  ## terminate
    else:
        d = np.linalg.norm(goal - pos)
        if d < wp_radius and current_wp_index == len(target_WP) - 1:
            current_wp_index += step_size
            if current_wp_index >= len(target_WP) - 1:
                return pos, True, current_wp_index
        terminate = False
        while d < lookahead and current_wp_index < len(target_WP) - 1:
            current_wp_index += step_size
            d = np.linalg.norm(target_WP[current_wp_index, :2] - pos)
            if current_wp_index == len(target_WP) - 1 and d < wp_radius:
                terminate = True
                break
        
        return target_WP[current_wp_index, :2], terminate, current_wp_index  ## new goal

def find_closest_index(pos, target_WP):
    return np.argmin(np.linalg.norm(target_WP[:,:2] - pos, axis=1))