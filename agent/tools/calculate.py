import zlib

def levenshtein_distance(s1: str, s2: str) -> int:
    if s1 == s2:
        return 0
    if len(s1) == 0:
        return len(s2)
    if len(s2) == 0:
        return len(s1)

    dp = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]

    for i in range(len(s1) + 1):
        dp[i][0] = i
    for j in range(len(s2) + 1):
        dp[0][j] = j

    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )
    return dp[-1][-1]


def normalized_levenshtein(s1: str, s2: str) -> float:
    if not s1 and not s2:
        return 0.0
    return levenshtein_distance(s1, s2) / max(len(s1), len(s2))


def ncd(s1: str, s2: str) -> float:
    if not s1 or not s2:
        return 1.0

    def compressed_len(s: str) -> int:
        return len(zlib.compress(s.encode("utf-8")))

    c1 = compressed_len(s1)
    c2 = compressed_len(s2)
    c12 = compressed_len(s1 + s2)

    return (c12 - min(c1, c2)) / max(c1, c2)