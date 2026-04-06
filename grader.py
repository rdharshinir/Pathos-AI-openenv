def grade_episode(trajectory):
    """
    Grade a single episode trajectory.

    Args:
        trajectory: list of (state, action, reward, info) tuples
                    where info is expected to contain 'distance_to_goal' if available.

    Returns:
        dict with keys:
            success   – bool, True if goal was reached
            score     – float 0.0–1.0, efficiency-adjusted score incorporating Manhattan distance
            steps     – int, total steps taken
            total_reward – float, raw sum of rewards
    """
    total_reward = sum(r for _, _, r, _ in trajectory)
    steps = len(trajectory)

    # Check if the last reward corresponds to a goal (+10)
    success = len(trajectory) > 0 and trajectory[-1][2] >= 10.0

    if success:
        # Efficiency bonus: fewer steps → higher score
        # Normalize against a baseline of 20 steps (can be improved based on grid size)
        baseline_steps = max(20, steps) # simplified baseline
        efficiency = max(0.0, 1.0 - (steps - 1) / baseline_steps)
        score = round(0.5 + 0.5 * efficiency, 4)   # range [0.5 .. 1.0] if successful
    else:
        # Partial credit based on Manhattan distance to goal at the end of the episode
        # Assuming info dict is passed and contains 'dist_to_goal' and 'grid_size'
        final_dist = 0
        grid_size = 5 # default fallback
        
        if len(trajectory) > 0:
            last_info = trajectory[-1][3] if len(trajectory[-1]) > 3 else {}
            if isinstance(last_info, dict):
                grid_size = last_info.get("grid_size", grid_size)
                # Parse structured obs if available to find distance, or fallback to an estimate
                if "structured" in last_info:
                    final_dist = last_info["structured"].get("manhattan_dist_to_goal", grid_size)
        
        max_possible_dist = grid_size * 2
        # Score range [0.0 .. 0.4] based on how close the agent got
        closeness = max(0.0, 1.0 - (final_dist / max_possible_dist))
        score = round(0.4 * closeness, 4)

    return {
        "success":      success,
        "score":        score,
        "steps":        steps,
        "total_reward": round(total_reward, 4),
    }