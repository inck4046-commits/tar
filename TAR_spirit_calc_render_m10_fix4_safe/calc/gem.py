# calc/gem.py

def optimize_gem_stat(hp: int, atk: int, defn: int, gem_value: str, max_gems: int = 5):
    """
    hp, atk, defn 에 젬 max_gems 개를 나누어 붙일 때
    비벨(hp*atk*def) 최대가 되는 분배를 찾아 반환.
    """
    try:
        val = int(gem_value)
    except ValueError:
        val = 0

    def calc_bibel(h, a, d):
        return h * a * d

    best = {"젬분배": (0, 0, 0), "HP": hp, "ATK": atk, "DEF": defn, "bibel": calc_bibel(hp, atk, defn)}

    # 가능한 분배: 체(i), 공(j), 방(k) 젬 개수, i+j+k ≤ max_gems
    for i in range(max_gems + 1):
        for j in range(max_gems + 1 - i):
            k = max_gems - i - j
            new_hp  = hp  + val * 4 * i
            new_atk = atk + val * j
            new_def = defn + val * k
            score   = calc_bibel(new_hp, new_atk, new_def)
            if score > best["bibel"]:
                best = {
                    "젬분배": (i, j, k),
                    "HP": new_hp,
                    "ATK": new_atk,
                    "DEF": new_def,
                    "bibel": score
                }

    return best

# UI 에서 사용할 젬 분배 리스트 (max 5개)
GEM_DISTS = [(i, j, 5 - i - j) for i in range(6) for j in range(6 - i)]

