
from typing import List, Tuple


def max_ordered_matching(score: List[List[float]]) -> Tuple[float, List[Tuple[int, int]]]:
    """
    Args:
        score: m x n matrix, where score[i][j] is the score of matching
               ref i with pred j.

    Returns:
        best_score: maximum total matching score
        matching_pairs: one optimal list of matched pairs (i, j), 0-based
    """
    if not score or not score[0]:
        return 0.0, []

    m, n = len(score), len(score[0])

    # dp[i][j]: best score using first i refs and first j preds
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]

    # parent pointers for backtracking
    # values: 'up'    -> from dp[i-1][j]
    #         'left'  -> from dp[i][j-1]
    #         'diag'  -> from dp[i-1][j-1] + score[i-1][j-1]
    parent = [[None] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            best = dp[i - 1][j]
            move = 'up'

            if dp[i][j - 1] > best:
                best = dp[i][j - 1]
                move = 'left'

            diag_val = dp[i - 1][j - 1] + score[i - 1][j - 1]
            if diag_val > best:
                best = diag_val
                move = 'diag'

            dp[i][j] = best
            parent[i][j] = move

    # Backtrack one optimal solution
    pairs = []
    i, j = m, n
    while i > 0 and j > 0:
        if parent[i][j] == 'diag':
            # ref i-1 matched with pred j-1
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif parent[i][j] == 'up':
            i -= 1
        else:  # 'left'
            j -= 1

    pairs.reverse()
    return dp[m][n], pairs
