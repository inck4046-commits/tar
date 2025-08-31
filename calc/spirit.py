# calc/spirit.py
def spirit_breakdown(inputs):
    if inputs[0]['stat'] == "없음":
        return {'hp': 0, 'atk': 0, 'def': 0}, {'hp': 0, 'atk': 0, 'def': 0}, {'hp': 0, 'atk': 0, 'def': 0}
        # 이하 기존 코드...

    """
    inputs: [{"stat": "공격력", "type": "%", "slot": 1~4}, ...] + [{"stat": "공격력10"/"방어력10"/"체력40", "type":"부가옵"}]
    """
    # 슬롯별 수치
    SLOT_FLAT = {1:[216,0,0], 2:[0,54,0], 3:[0,0,54], 4:[480,0,0]}
    SLOT_PCT  = {1:[.24,0,0], 2:[0,.24,0], 3:[0,0,.24], 4:[.40,0,0]}
    SLOT_MAP  = {"체력":0, "공격력":1, "방어력":2}
    # 2/3/4옵은 +, %값만 바꿔줘야 함
    # 2옵: flat=240/60/60 pct=.28, 3옵: flat=264/66/66 pct=.32, 4옵: flat=480/120/120 pct=.40
    FLAT_TBL = [None,
                [216, 54, 54],
                [240, 60, 60],
                [264, 66, 66],
                [480,120,120]]
    PCT_TBL  = [None,
                [.24,.24,.24],
                [.28,.28,.28],
                [.32,.32,.32],
                [.40,.40,.40]]
    pct  = {"hp":0, "atk":0, "def":0}
    flat = {"hp":0, "atk":0, "def":0}
    sub  = {"hp":0, "atk":0, "def":0}
    # 슬롯 입력 해석
    for idx, box in enumerate(inputs[:4]):
        slot = idx+1  # 1,2,3,4
        stat_idx = SLOT_MAP[box["stat"]]
        if box["type"]=="+":
            v = FLAT_TBL[slot][stat_idx]
            if   stat_idx==0: flat["hp"]  += v
            elif stat_idx==1: flat["atk"] += v
            elif stat_idx==2: flat["def"] += v
        else:
            v = PCT_TBL[slot][stat_idx]
            if   stat_idx==0: pct["hp"]  += v
            elif stat_idx==1: pct["atk"] += v
            elif stat_idx==2: pct["def"] += v
    # 부가옵 해석
    sub_txt = inputs[4]["stat"]
    if   sub_txt=="공격력10": sub["atk"] += 10
    elif sub_txt=="방어력10": sub["def"] += 10
    elif sub_txt=="체력40":   sub["hp"]  += 40
    return pct, flat, sub

def apply(base, inputs):
    """base: {"hp":, "atk":, "def":}, inputs: spirit 콤보 박스 5개 값 dict"""
    pct, flat, sub = spirit_breakdown(inputs)
    # 공식: 최종 = int(base*(1+pct)) + int(flat*(1+pct)) + sub
    result = {}
    for key in ("hp","atk","def"):
        result[key] = int(base[key]*(1+pct[key])) + int(flat[key]*(1+pct[key])) + sub[key]
    return result










