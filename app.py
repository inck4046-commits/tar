
from __future__ import annotations

import os, json, math, itertools, tempfile, hashlib
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Tuple, Optional, Set

import pandas as pd
import gradio as gr

try:
    from zoneinfo import ZoneInfo
    KST = ZoneInfo("Asia/Seoul")
except Exception:
    KST = None

def _now():
    return datetime.now(KST) if KST else datetime.utcnow()

_VIS_DB = os.path.join(tempfile.gettempdir(), "visits_daily.json")
_VIS_SALT = os.environ.get("VIS_SALT", "SALT")

def _client_ip(req: gr.Request | None) -> str:
    try:
        if req is None:
            return "?"
        for k in ("x-forwarded-for","X-Forwarded-For","x-real-ip","X-Real-Ip"):
            v = req.headers.get(k)
            if v:
                return v.split(",")[0].strip()
        return getattr(req.client, "host", "") or "?"
    except Exception:
        return "?"

def _ip_key(req: gr.Request | None) -> str:
    try:
        return hashlib.sha256((_VIS_SALT + _client_ip(req)).encode()).hexdigest()
    except Exception:
        return hashlib.sha256((_VIS_SALT + "anon").encode()).hexdigest()

def _vload():
    try:
        with open(_VIS_DB,"r",encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _vsave(d:dict):
    with open(_VIS_DB,"w",encoding="utf-8") as f:
        json.dump(d,f)

def register_unique_visit(request: gr.Request | None = None) -> str:
    d = _vload()
    key = _now().strftime("%Y-%m-%d")
    if key not in d:
        d = {key:{}}
    ipk = _ip_key(request)
    if ipk not in d[key]:
        d[key][ipk] = int(_now().timestamp())
        _vsave(d)
    return f"<div style='text-align:right;font-size:12px;'>오늘 방문자: <b>{len(d[key])}</b></div>"

# =============================================================================
# 외부 모듈 폴백
# =============================================================================

try:
    from calc.type import get_base_stats as _get_base_stats_raw
    from calc.awakening import get_awakening_stat as _get_awakening_stat_raw
    from calc.spirit import spirit_breakdown as _spirit_breakdown_raw
    from calc.collection import apply_collection as _apply_collection_raw
    from calc.potion import apply_potion as _apply_potion_raw
    import calc.accessory as accmod
    from calc.buff import BUFFS_ALL as _BUFFS_ALL
    from calc.gem import GEM_DISTS as _GEM_DISTS
except Exception:
    def _get_base_stats_raw(typ:str)->Dict[str,int]:
        base = {
            "체":{"hp":400,"atk":100,"def":100},
            "공":{"hp":200,"atk":200,"def":100},
            "방":{"hp":200,"atk":100,"def":200},
            "체공":{"hp":350,"atk":150,"def":100},
            "체방":{"hp":350,"atk":100,"def":150},
            "공방":{"hp":200,"atk":150,"def":150},
        }
        t = typ.replace("(진각)","")
        return base.get(t, {"hp":300,"atk":120,"def":120})

    def _get_awakening_stat_raw(typ:str)->Dict[str,int]:
        return {"hp":60,"atk":20,"def":20}

    def _spirit_breakdown_raw(arr:List[dict]):
        # (정령 %) / (정령 평타) / (부가옵) — 더미
        pct = {"hp":0.15,"atk":0.10,"def":0.10}
        flat = {"hp":120,"atk":40,"def":40}
        sub = {"hp":0,"atk":0,"def":0}
        return pct, flat, sub

    def _apply_collection_raw(_=None):
        return {"hp":0,"atk":0,"def":0}

    def _apply_potion_raw(_=None):
        return {"hp":0,"atk":0,"def":0}

    class _AccStub:
        df_acc = pd.DataFrame([
            {"lv":19,"이름":"황보","hp%":0.05,"atk%":0.05,"def%":0.05},
            {"lv":19,"이름":"악보","hp%":0.06,"atk%":0.0,"def%":0.06},
        ])
    accmod = _AccStub()

    _BUFFS_ALL = {
        "0벞":{"hp":0.0,"atk":0.0,"def":0.0},
        "HP20%":{"hp":0.2,"atk":0.0,"def":0.0},
        "ATK20%":{"hp":0.0,"atk":0.2,"def":0.0},
        "DEF20%":{"hp":0.0,"atk":0.0,"def":0.2},
        "HP40%":{"hp":0.4,"atk":0.0,"def":0.0},
        "ATK40%":{"hp":0.0,"atk":0.4,"def":0.0},
        "DEF40%":{"hp":0.0,"atk":0.0,"def":0.4},
        "HP+ATK20%":{"hp":0.2,"atk":0.2,"def":0.0},
        "HP+DEF20%":{"hp":0.2,"atk":0.0,"def":0.2},
        "ATK+DEF20%":{"hp":0.0,"atk":0.2,"def":0.2},
    }

    _GEM_DISTS = [
        (5,0,0),(0,5,0),(0,0,5),
        (4,1,0),(4,0,1),(1,4,0),(0,4,1),(1,0,4),(0,1,4),
        (3,2,0),(3,0,2),(2,3,0),(0,3,2),(2,0,3),(0,2,3)
    ]

BUFFS_ALL: Dict[str,Dict[str,float]] = dict(_BUFFS_ALL)
GEM_DISTS: List[Tuple[int,int,int]] = list(_GEM_DISTS)

try:
    from tar_denom_db import get_denom_from_db as _fast_denom_get
except Exception:
    def _fast_denom_get(mode:int, typ:str, buff:str)->int:
        return 0

try:
    import spec_db
except Exception:
    class _SpecDBStub:
        def get_M(self, profile:int, typ:str, buff_label:str)->float|None:
            return None
        def profile_name(self, p:int)->str:
            return {1:"7.0/펜던트X",2:"9.0/펜던트O"}.get(p,str(p))
    spec_db = _SpecDBStub()

TYPES_ALL = ["체","공","방","체공","체방","공방",
             "(진각)체","(진각)공","(진각)방","(진각)체공","(진각)체방","(진각)공방"]
BASE_TYPES = ["체","공","방","체공","체방","공방"]
TWO_BUFFS = ["HP40%","ATK40%","DEF40%","HP+ATK20%","HP+DEF20%","ATK+DEF20%"]
ONE_20 = ["HP20%","ATK20%","DEF20%"]
ALL_BUFF_CHOICES = TWO_BUFFS + ONE_20 + ["0벞"]

ENCH_LIST = [
    ("HP", {"hp":0.21,"atk":0.0,"def":0.0}),
    ("ATK",{"hp":0.0,"atk":0.21,"def":0.0}),
    ("DEF",{"hp":0.0,"atk":0.0,"def":0.21}),
]
ENCH_DICT = dict(ENCH_LIST)

# =============================================================================
# 보조
# =============================================================================

def strip_jingak(t:str)->str:
    return t.replace("(진각)","") if isinstance(t,str) else t

@lru_cache(maxsize=None)
def _get_base_stats_cached(t:str)->Dict[str,int]:
    return dict(_get_base_stats_raw(t))

@lru_cache(maxsize=None)
def _get_awakening_stat_cached(t:str)->Dict[str,int]:
    return dict(_get_awakening_stat_raw(t))

def _sp_key_for_cache(sp:List[dict])->Tuple:
    if not sp: return ()
    return tuple((int(d.get("slot",0)),str(d.get("stat","")),str(d.get("type",""))) for d in sp)

@lru_cache(maxsize=None)
def _spirit_breakdown_cached(key:Tuple)->Tuple[Dict[str,float],Dict[str,int],Dict[str,int]]:
    # 정령 미사용(빈 키)일 때 완전 0 효과 반환
    if not key:
        zero_pct = {"hp":0.0,"atk":0.0,"def":0.0}
        zero_flat = {"hp":0,"atk":0,"def":0}
        zero_sub = {"hp":0,"atk":0,"def":0}
        return zero_pct, zero_flat, zero_sub
    arr = [{"slot":s,"stat":st,"type":tp} for (s,st,tp) in key]
    return _spirit_breakdown_raw(arr)

_POTION = _apply_potion_raw({})
_COLLECTION = _apply_collection_raw({"hp":0,"atk":0,"def":0})

def acc_label(r:dict|None)->str:
    if not r: return "-"
    lv = int(r.get("lv",r.get("레벨",r.get("level",0))))
    name = str(r.get("이름","")).strip()
    return f"{lv} {name}".strip()

def acc_row_by_label(tag:str)->dict|None:
    try:
        df = accmod.df_acc.copy()
    except Exception:
        return None
    s = str(tag).strip()
    parts = s.split()
    lv, name = None, s
    try:
        lv = int(parts[0]); name = " ".join(parts[1:]).strip()
    except Exception:
        pass
    try:
        if lv is not None:
            sub = df[(df["lv"].astype(int)==lv)&(df["이름"].astype(str)==name)]
        else:
            sub = df[df["이름"].astype(str)==name]
        if not sub.empty:
            return sub.iloc[0].to_dict()
    except Exception:
        return None
    return None

def accessory_names_by_levels(levels:List[int])->List[str]:
    try:
        df = accmod.df_acc.copy()
        df = df[df["lv"].astype(int).isin([int(x) for x in (levels or [])])]
        names = df[["lv","이름"]].astype({"lv":int,"이름":str}).sort_values(["lv","이름"])
        return [f"{lv} {name}" for lv,name in names.values.tolist()]
    except Exception:
        return []

def _pct_from(x:Any)->Dict[str,float]:
    if isinstance(x,dict):
        return {"hp":float(x.get("hp",0.0)),"atk":float(x.get("atk",0.0)),"def":float(x.get("def",0.0))}
    return dict(BUFFS_ALL.get(str(x), {"hp":0,"atk":0,"def":0}))

def tar_percent(B:float, M:float)->float:
    try:
        B = float(B); M = float(M)
    except Exception:
        return 0.0
    if M <= 0: return 0.0
    r = B/M
    if r <= 1.0:
        return round(max(0.0, 200.0*r - 100.0), 4)
    return round(100.0*r, 4)

def make_gild_pct(hp, atk, df):
    def f(v):
        try: return float(v)/100.0
        except Exception: return 0.0
    return {"hp":f(hp),"atk":f(atk),"def":f(df)}

# === 등급 보정(요청 표) ===
GRADE_INCR_PER_1 = {
    "체":{"hp":80,"atk":0,"def":0},
    "공":{"hp":0,"atk":20,"def":0},
    "방":{"hp":0,"atk":0,"def":20},
    "체방":{"hp":40,"atk":0,"def":10},
    "체공":{"hp":40,"atk":10,"def":0},
    "공방":{"hp":0,"atk":10,"def":10},
}
def grade_boost_for_type_precise(typ:str, grade_str:str)->Dict[str,int]:
    """7.0 기준, 1.0당 상숫값 추가. (진각) 표기는 무시."""
    t = strip_jingak(typ)
    g = float(str(grade_str))
    delta = max(0.0, g - 7.0)
    inc = GRADE_INCR_PER_1.get(t, {"hp":0,"atk":0,"def":0})
    return {k:int(round(inc[k]*delta)) for k in ("hp","atk","def")}

# =============================================================================
# 계산 코어
# =============================================================================

def _floor_int(x:float)->int:
    return int(math.floor(float(x)))

def compute_final_stat_and_bib(
    typ:str,
    ench_pct:Dict[str,float],
    acc_row:dict|None,
    gem_sums:Tuple[float,float,float],   # (hpSum, atkSum, defSum)
    spirits_spec:List[dict],             # 원시 정령 스펙 5개 (빈 리스트면 미사용)
    pendant_pct:Dict[str,float]|None,    # 라인 합산 %를 스탯별로
    buff_label_or_pct:Dict[str,float]|str,
    nerf_label_or_pct:Dict[str,float]|str,
    grade_sel:str,
    base_stat_manual:Dict[str,int]|None=None,
    jingak:bool=False
)->Tuple[int, Dict[str,int]]:
    # 기본/각성
    bs = _get_base_stats_cached(typ if base_stat_manual is None else strip_jingak(typ)) if base_stat_manual is None else dict(base_stat_manual)
    aw = _get_awakening_stat_cached(typ) if (jingak and base_stat_manual is None) else {"hp":0,"atk":0,"def":0}

    # 등급 보정
    boost = grade_boost_for_type_precise(typ, grade_sel)

    # 젬 합산(HP는 *4)
    C1 = {k: bs[k] + aw[k] + boost.get(k,0) for k in ("hp","atk","def")}
    C2 = {"hp":gem_sums[0]*4, "atk":gem_sums[1], "def":gem_sums[2]}
    base_total = {k: C1[k] + C2[k] for k in C1}
    # 물약
    base_total = {k: base_total[k] + _POTION.get(k,0) for k in base_total}

    # 장신구(자체% + 인첸트%)
    hp_pct = float((acc_row or {}).get("hp%",0))/100.0
    atk_pct = float((acc_row or {}).get("atk%",0))/100.0
    def_pct = float((acc_row or {}).get("def%",0))/100.0
    ench = ench_pct or {"hp":0,"atk":0,"def":0}
    mul1 = {"hp":1+hp_pct+ench.get("hp",0.0),
            "atk":1+atk_pct+ench.get("atk",0.0),
            "def":1+def_pct+ench.get("def",0.0)}
    st1 = {k:_floor_int(base_total[k]*mul1[k]) for k in ("hp","atk","def")}

    # 정령(%) / 평타(+)
    pct7, flat8, sub9 = _spirit_breakdown_cached(_sp_key_for_cache(spirits_spec))
    st2 = {k: _floor_int(st1[k]*(1+pct7[k]) + flat8[k]*(1+pct7[k])) for k in ("hp","atk","def")}

    # 펜던트(곱)
    if pendant_pct:
        st2 = {k: _floor_int(st2[k]*(1.0+float(pendant_pct.get(k,0.0)))) for k in st2}

    # 컬렉션/부가옵 가산
    st3 = {k: st2[k] + _COLLECTION.get(k,0) + sub9[k] for k in ("hp","atk","def")}

    # (버프-너프)*(기본+등급보정) 가산
    bs_for_buff = {k: bs[k] + boost.get(k,0) for k in ("hp","atk","def")}
    buff = _pct_from(buff_label_or_pct)
    nerf = _pct_from(nerf_label_or_pct)
    eff = {"hp":buff["hp"]-nerf["hp"], "atk":buff["atk"]-nerf["atk"], "def":buff["def"]-nerf["def"]}
    add_buff = {k: _floor_int(bs_for_buff[k]*eff[k]) for k in ("hp","atk","def")}

    final = {k: st3[k] + add_buff[k] for k in ("hp","atk","def")}
    bib = final["hp"] * final["atk"] * final["def"]
    return int(bib), final

# =============================================================================
# SPEC 분모 (항상 DB2)
# =============================================================================

def _fast_denom_or_none(profile:int, typ:str, buff_label:str)->float|None:
    if int(profile) not in (1,2): return None
    if strip_jingak(typ) not in BASE_TYPES: return None
    if buff_label not in TWO_BUFFS: return None
    try:
        v = _fast_denom_get(profile, strip_jingak(typ), buff_label)
        if int(v) > 0: return float(v)
    except Exception:
        pass
    return None

def get_plain_M(profile:int, typ:str, label:str)->float:
    fast = _fast_denom_or_none(profile, typ, label)
    if fast is not None:
        return float(fast)
    try:
        m = spec_db.get_M(profile, strip_jingak(typ), label)
        return float(m) if m is not None else 0.0
    except Exception:
        return 0.0

# =============================================================================
# 젬/정령/펜던트/표시 보조
# =============================================================================

def _compact_used_gems(items:List[str])->str:
    # 정렬: 체→공→방
    order = {"체":0,"공":1,"방":2}
    arr = []
    for s in items:
        s = s.strip()
        if not s: continue
        arr.append((order.get(s[0],9), s, int("".join(ch for ch in s[1:] if ch.isdigit()) or "0")))
    arr.sort(key=lambda x:(x[0], -x[2]))  # 같은 스탯 내에서 수치 큰 것 먼저
    return " ".join(s for _,s,_ in arr)

def spirit_default_row()->dict:
    return {
        "사용": False,
        "귀속": "정령 귀속X",
        "1옵 스탯": "체력", "1옵 모드": "%",
        "2옵 스탯": "공격력", "2옵 모드": "%",
        "3옵 스탯": "방어력", "3옵 모드": "%",
        "4옵 스탯": "방어력", "4옵 모드": "+",
        "부가옵": "체력40",
    }

def make_spirit_obj(row:dict)->List[dict]:
    return [
        {"slot":1,"stat":row["1옵 스탯"],"type":row["1옵 모드"]},
        {"slot":2,"stat":row["2옵 스탯"],"type":row["2옵 모드"]},
        {"slot":3,"stat":row["3옵 스탯"],"type":row["3옵 모드"]},
        {"slot":4,"stat":row["4옵 스탯"],"type":row["4옵 모드"]},
        {"slot":5,"stat":row["부가옵"],"type":"부가옵"},
    ]

def format_spirit_label(sp:dict|List[dict])->str:
    s = sp if isinstance(sp,list) else sp["spec"]
    def one(slot):
        st = s[slot-1]
        if st["slot"]==5:  # 부가옵
            return st["stat"]
        unit = "%" if st["type"]=="%" else "+"
        return f'{st["stat"][0]}{unit}'
    # 1/2/3/4/부가옵
    return f'{one(1)}/{one(2)}/{one(3)}/{one(4)}/{s[4]["stat"]}'

def gem_inventory_default()->pd.DataFrame:
    # 34~40 확장
    return pd.DataFrame([{"수치":v,"체":0,"공":0,"방":0} for v in [34,35,36,37,38,39,40]])

def gem_inventory_to_pool(df:pd.DataFrame)->Dict[Tuple[str,int],int]:
    pool:Dict[Tuple[str,int],int] = {}
    if df is None or df.empty: return pool
    for _,r in df.iterrows():
        v = int(r["수치"])
        for stat,col in (("hp","체"),("atk","공"),("def","방")):
            pool[(stat,v)] = max(0, int(r.get(col,0) or 0))
    return pool

def _feasible_dists_by_count(pool:Dict[Tuple[str,int],int])->List[Tuple[int,int,int]]:
    """수량 관점에서 불가능한 젬 분배를 사전 제거 (정확도 영향 0)"""
    have = {
        "hp": sum(c for (s,_),c in pool.items() if s=="hp"),
        "atk": sum(c for (s,_),c in pool.items() if s=="atk"),
        "def": sum(c for (s,_),c in pool.items() if s=="def"),
    }
    out=[]
    for d in GEM_DISTS:
        need={"hp":d[0],"atk":d[1],"def":d[2]}
        if all(have[k]>=need[k] for k in ("hp","atk","def")) and sum(d)==5:
            out.append(d)
    return out

def allocate_gems_for_dist(pool:Dict[Tuple[str,int],int], dist:Tuple[int,int,int])->Tuple[bool,Dict[Tuple[str,int],int],Tuple[float,float,float],List[str]]:
    """요구 개수(dist)를 '현재 남아있는' 풀에서 집어 (use: {(stat,val):cnt})/합계/표기리스트 반환"""
    need = {"hp":dist[0], "atk":dist[1], "def":dist[2]}
    use:Dict[Tuple[str,int],int] = {}
    sums = {"hp":0.0,"atk":0.0,"def":0.0}
    used_list:List[str] = []

    for stat,kor in (("hp","체"),("atk","공"),("def","방")):
        cnt = need[stat]
        if cnt <= 0: continue
        cand = sorted([v for (s,v),c in pool.items() if s==stat and c>0], reverse=True)
        left = cnt
        for v in cand:
            if left <= 0: break
            take = min(left, pool[(stat,v)])
            if take<=0: continue
            use[(stat,v)] = use.get((stat,v),0) + take
            left -= take
            sums[stat] += take*v
            used_list.extend([f"{kor}{v}"]*take)
        if left>0:
            return False,{},(0,0,0),[]
    return True,use,(sums["hp"],sums["atk"],sums["def"]),used_list

# ---------------- 펜던트 ----------------

def _enum_lines_sum(total:int, max_lines:int)->List[List[int]]:
    """합계 total을 1..6의 숫자 max_lines개 이하로 분해 (각 원소=한 라인의 %), 순서 고려"""
    total = int(total)
    out=[]
    def dfs(start, remain, k, path):
        if remain==0 and 0< len(path) <= max_lines:
            out.append(path[:]); return
        if k==0 or remain<=0: return
        for v in range(1,7):
            if v>remain: break
            path.append(v)
            dfs(start, remain-v, k-1, path)
            path.pop()
    dfs(0,total,max_lines,[])
    return out

def _assign_lines_to_stats(lines:List[int])->List[Dict[str,float]]:
    """라인 배열을 hp/atk/def에 배치(중복 허용). 모든 배치 생성."""
    if not lines: return []
    res=[]
    n=len(lines)
    for idxs in itertools.product((0,1,2), repeat=n):
        acc={"hp":0,"atk":0,"def":0}
        for v,ix in zip(lines, idxs):
            if ix==0: acc["hp"]+=v
            elif ix==1: acc["atk"]+=v
            else: acc["def"]+=v
        res.append({k:acc[k]/100.0 for k in acc})
    # 중복 제거
    uniq={}
    for p in res:
        key=(p["hp"],p["atk"],p["def"])
        uniq[key]=p
    return list(uniq.values())

def enumerate_pendants(total:int, grade:str)->List[Dict[str,float]]:
    """등급별 라인 수(별1/달2/태양3), total%를 만족하는 모든 분배를 % dict로 반환"""
    grade_to_lines={"별":1,"달":2,"태양":3}
    L=grade_to_lines.get(grade,3)
    total=max(0,min(int(total), 6*L))
    if total<=0: return [None]
    cands=[]
    for arr in _enum_lines_sum(total, L):
        cands.extend(_assign_lines_to_stats(arr))
    return cands or [None]

def pendant_label(pct:Dict[str,float]|None)->str:
    if not pct: return "0/0/0"
    return f'{int(round(pct.get("hp",0)*100))}/{int(round(pct.get("atk",0)*100))}/{int(round(pct.get("def",0)*100))}'

# =============================================================================
# 즐겨찾기 (IP + 프리셋 1~4)
# =============================================================================

_FAV_DB = os.path.join(tempfile.gettempdir(), "tar_fav.json")

def _fav_load():
    try:
        with open(_FAV_DB,"r",encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
            return {}
    except Exception:
        return {}

def _fav_save(d:dict):
    tmp = _FAV_DB + ".tmp"
    with open(tmp,"w",encoding="utf-8") as f:
        json.dump(d,f,ensure_ascii=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp,_FAV_DB)

def fav_save_preset(keyspace:str, preset:int, payload:dict, request:gr.Request|None):
    d = _fav_load()
    ipk = _ip_key(request)
    ts = int(_now().timestamp())
    per_ip = d.get(ipk, {})
    ks = f"{keyspace}:preset{int(preset)}"
    per_ip[ks] = {"ts":ts,"data":payload}
    d[ipk] = per_ip
    _fav_save(d)
    return "<span style='color:green;'>즐겨찾기 저장 완료</span>"

def fav_load_preset(keyspace:str, preset:int, request:gr.Request|None)->dict:
    d = _fav_load()
    ipk = _ip_key(request)
    ks = f"{keyspace}:preset{int(preset)}"
    obj = ((d.get(ipk) or {}).get(ks) or {}).get("data")
    return obj or {}

# =============================================================================
# 자동 계산기
# =============================================================================

def resolve_row_buff_list(mode:str)->List[Tuple[str,Dict[str,float]]]:
    if mode=="최적화":
        return [(k,v) for k,v in BUFFS_ALL.items() if k!="0벞"]
    if mode=="1벞 최적화":
        keys=["HP20%","ATK20%","DEF20%"]
        return [(k, BUFFS_ALL[k]) for k in keys if k in BUFFS_ALL]
    if mode=="0벞":
        return [("0벞",BUFFS_ALL["0벞"])]
    return [(mode, BUFFS_ALL.get(mode, {"hp":0,"atk":0,"def":0}))]

def auto_run(
    types_sel, gem_vals_sel,
    acc_packed,
    s1,m1,s2,m2,s3,m3,s4,m4,subopt,sp_skip,
    buff_mode, nerf_mode, grade_sel,
    gild_den_for_tar_unused,
    # 펜던트 자동
    pnd_opt_on, pnd_total, pnd_grade,
    # 펜던트 수동
    pnd_manual_on, pnd_manual_entries,
    dedup_type, dedup_buff, dedup_acc, dedup_ench, dedup_sp,
    topn, sort_key
)->pd.DataFrame:
    try:
        types = BASE_TYPES[:] if not types_sel else types_sel[:]
        gem_vals = [int(x) for x in (gem_vals_sel or [str(v) for v in range(34,41)])]

        # 장신구(+인첸트)
        acc_rows=[]
        for tag,use,hp,atk,df in acc_packed or []:
            if not bool(use): continue
            base = acc_row_by_label(tag)
            if not base: continue
            flags=[]
            if bool(hp): flags.append(("HP",ENCH_DICT["HP"]))
            if bool(atk): flags.append(("ATK",ENCH_DICT["ATK"]))
            if bool(df): flags.append(("DEF",ENCH_DICT["DEF"]))
            if not flags: flags = ENCH_LIST[:]
            for nm,pct in flags:
                acc_rows.append((base,nm,pct))
        if not acc_rows:
            for n in accessory_names_by_levels([19]):
                base=acc_row_by_label(n)
                if base: acc_rows.append((base,"HP",ENCH_DICT["HP"]))

        # 정령: 미사용이면 빈 리스트 → 효과 0
        spirit = [] if bool(sp_skip) else [
            {"slot":1,"stat":s1,"type":m1},
            {"slot":2,"stat":s2,"type":m2},
            {"slot":3,"stat":s3,"type":m3},
            {"slot":4,"stat":s4,"type":m4},
            {"slot":5,"stat":subopt,"type":"부가옵"},
        ]

        # 버프 후보
        buff_items = resolve_row_buff_list(buff_mode)

        # 펜던트 자동/수동 후보
        pnd_auto_list = enumerate_pendants(pnd_total if pnd_opt_on else 0, pnd_grade)
        pnd_manual_cands: List[Dict[str,float]] = []
        if pnd_manual_on:
            for ent in pnd_manual_entries or []:
                if not ent.get("use"): continue
                lines=[(ent.get("l1s"), ent.get("l1v",0)),
                       (ent.get("l2s"), ent.get("l2v",0)),
                       (ent.get("l3s"), ent.get("l3v",0))]
                accp={"hp":0,"atk":0,"def":0}
                for st,v in lines:
                    st2={"체력":"hp","공격력":"atk","방어력":"def"}.get(st)
                    v=max(0,min(6,int(v or 0)))
                    if st2: accp[st2]+=v
                if accp["hp"]+accp["atk"]+accp["def"]>0:
                    pnd_manual_cands.append({k:accp[k]/100.0 for k in accp})

        rows=[]
        for typ in types:
            base_manual = None if "(진각)" in typ else _get_base_stats_cached(strip_jingak(typ))
            jingak = "(진각)" in typ

            for gv in gem_vals:
                for dist in GEM_DISTS:
                    if sum(dist)!=5: continue
                    gem_sums = (dist[0]*gv, dist[1]*gv, dist[2]*gv)

                    for (acc_row,ench_name,ench_pct) in acc_rows:
                        pendants = (pnd_manual_cands if (pnd_manual_on and pnd_manual_cands) else pnd_auto_list)

                        for buf_label,_pct in buff_items:
                            for pnd_pct in pendants:
                                bib, final = compute_final_stat_and_bib(
                                    typ, ench_pct, acc_row, gem_sums,
                                    spirit, pnd_pct, buf_label, nerf_mode,
                                    grade_sel, base_stat_manual=base_manual, jingak=jingak
                                )
                                M = get_plain_M(2, strip_jingak(typ), buf_label if buf_label in TWO_BUFFS else "HP40%")
                                TAR = tar_percent(bib, M)

                                rows.append({
                                    "타입":typ, "버프":buf_label, "너프":nerf_mode,
                                    "펜던트": pendant_label(pnd_pct),
                                    "젬수치":gv, "젬분배":f"{dist[0]}/{dist[1]}/{dist[2]}",
                                    "장신구":acc_label(acc_row), "인첸트":ench_name,
                                    "정령": ("미사용" if not spirit else format_spirit_label(spirit)),
                                    "비벨":int(bib), "TAR%":TAR,
                                    "HP":final["hp"], "ATK":final["atk"], "DEF":final["def"],
                                })

        if not rows: return pd.DataFrame()
        df=pd.DataFrame(rows)
        if sort_key=="TAR%":
            df=df.sort_values(["TAR%","비벨"],ascending=False)
        else:
            df=df.sort_values(["비벨","TAR%"],ascending=False)

        keys=[]
        if dedup_type: keys.append("타입")
        if dedup_buff: keys.append("버프")
        if dedup_acc: keys.append("장신구")
        if dedup_ench: keys.append("인첸트")
        if dedup_sp: keys.append("정령")
        if keys: df=df.drop_duplicates(subset=keys, keep="first")

        return df.reset_index(drop=True).head(int(topn))
    except Exception as e:
        return pd.DataFrame({"오류":[str(e)]})

# =============================================================================
# 길드전 최적화 (젬은 try_place에서 '현재 풀'로 배치)
# =============================================================================

NUM_DECKS = 3

def _approx_bib_upper_for_slot(typ, ench_name, acc_row, dist, sp, pnd, buf, nerf, grade_sel)->int:
    """상계 계산용: 남은 풀과 무관하게 '모두 40젬'이라고 가정하여 bib 상한 추정"""
    gv = 40.0
    sums = (dist[0]*gv, dist[1]*gv, dist[2]*gv)
    base_manual = None if "(진각)" in typ else _get_base_stats_cached(strip_jingak(typ))
    jingak = "(진각)" in typ
    pct = ENCH_DICT.get(ench_name, {"hp":0,"atk":0,"def":0})
    bib,_ = compute_final_stat_and_bib(
        typ, pct, acc_row, sums, sp["spec"], pnd,
        buf, nerf, grade_sel, base_stat_manual=base_manual, jingak=jingak
    )
    return int(bib)

def guild_optimize(
    week_typ, week_buff,   # 분모 기준(DB2)
    slot_used_list, slot_typ_list, slot_buff_list, slot_grade_list, slot_nerf_list,
    pendant_inv_rows,  # 펜던트 인벤
    acc_df, gem_df, spirits_rows,
    top_k=80, min_tar_cut=0.0,
    full_search:bool=False,
    goal:str="비벨우선"
):
    used_slots = [i+1 for i,u in enumerate(slot_used_list) if bool(u)]
    if len(used_slots) < NUM_DECKS:
        return pd.DataFrame({"오류":[f"드래곤 슬롯 최소 {NUM_DECKS}개 사용 체크 필요"]}), pd.DataFrame({"평균 비벨":[0],"평균TAR":[0]}), {"choice":[]}

    M = get_plain_M(2, week_typ, week_buff)

    # --------- 인벤 파싱(동치 합치기 없이 그대로) ----------
    def parse_acc(df:pd.DataFrame)->List[dict]:
        base = accmod.df_acc.copy()
        key={}
        for _,r in base.iterrows():
            key[f'{int(r.get("lv",19))} {str(r.get("이름","")).strip()}'] = r.to_dict()
        out=[]
        if df is None or df.empty: return out
        for _,r in df.iterrows():
            if not bool(r.get("사용",False)): continue
            tag = str(r.get("장신구","")).strip()
            acc_row = key.get(tag)
            if not acc_row: continue
            allowed=set()
            if bool(r.get("HP",True)): allowed.add("HP")
            if bool(r.get("ATK",True)): allowed.add("ATK")
            if bool(r.get("DEF",True)): allowed.add("DEF")
            out.append({"key":f'{tag} {str(r.get("인스턴스","#1"))}','acc':acc_row, 'allowed': (allowed or {"HP","ATK","DEF"})})
        return out

    acc_pool = parse_acc(acc_df)
    if not acc_pool:
        return pd.DataFrame({"오류":["장신구 인벤 사용 체크 필요"]}), pd.DataFrame({"평균 비벨":[0],"평균TAR":[0]}), {"choice":[]}

    gem_pool = gem_inventory_to_pool(gem_df)
    if sum(gem_pool.values()) < 5*NUM_DECKS:
        return pd.DataFrame({"오류":[f"젬 최소 {5*NUM_DECKS}개 필요"]}), pd.DataFrame({"평균 비벨":[0],"평균TAR":[0]}), {"choice":[]}

    def make_spirit_pool(rows:List[dict])->List[dict]:
        pool=[]
        for i,r in enumerate(rows, start=1):
            if not r.get("사용",False): continue
            obj = make_spirit_obj(r)
            bind = r.get("귀속","정령 귀속X")
            bind_slot=None
            if isinstance(bind,str) and bind.endswith("번드래곤"):
                try: bind_slot=int(bind.replace("번드래곤",""))
                except Exception: bind_slot=None
            pool.append({"id":f"SP{i}","bind":bind_slot,"spec":obj})
        return pool

    spirit_pool = make_spirit_pool(spirits_rows)
    if len(spirit_pool) < NUM_DECKS:
        return pd.DataFrame({"오류":[f"정령 최소 {NUM_DECKS}개 사용 체크 필요"]}), pd.DataFrame({"평균 비벨":[0],"평균TAR":[0]}), {"choice":[]}

    # 펜던트(입력 그대로)
    pendant_pool=[]
    for i,pr in enumerate(pendant_inv_rows, start=1):
        if not pr.get("use",False): continue
        lines=[(pr.get("l1s"), pr.get("l1v",0)),
               (pr.get("l2s"), pr.get("l2v",0)),
               (pr.get("l3s"), pr.get("l3v",0))]
        accp={"hp":0,"atk":0,"def":0}
        for st,v in lines:
            st2={"체력":"hp","공격력":"atk","방어력":"def"}.get(st)
            v=max(0,min(6,int(v or 0)))
            if st2: accp[st2]+=v
        if accp["hp"]+accp["atk"]+accp["def"]>0:
            pendant_pool.append({"id":f"PND{i}", "pct":{k:accp[k]/100.0 for k in accp}})
    pendant_pool_with_none = [None] + pendant_pool

    feasible_patterns = _feasible_dists_by_count(gem_pool)

    # --------- 슬롯별 후보(젬은 확정하지 않음! dist만) ----------
    cand_by_slot={}
    for i in used_slots:
        typ = slot_typ_list[i-1]
        buf = slot_buff_list[i-1]
        nerf = slot_nerf_list[i-1]
        grade_sel = slot_grade_list[i-1]

        cands=[]
        for dist in feasible_patterns:
            for acc in acc_pool:
                for sp in spirit_pool:
                    if sp.get("bind") not in (None,i): 
                        continue
                    for pn in pendant_pool_with_none:
                        pnd_pct = None if pn is None else pn["pct"]

                        # 인첸트 최적 1개 선별(슬롯 상수)
                        best = None; best_bib=-1
                        for nm,pct in ENCH_LIST:
                            if nm not in (acc["allowed"] if acc["allowed"] else {"HP","ATK","DEF"}): 
                                continue
                            # 상한용 근사치(40젬 가정)를 바탕으로 선택
                            est = _approx_bib_upper_for_slot(typ, nm, acc["acc"], dist, sp, pnd_pct, buf, nerf, grade_sel)
                            if est>best_bib:
                                best_bib=est; best=(nm,pct)
                        ench_name, ench_pct = best if best else ("HP",ENCH_DICT["HP"])

                        cands.append({
                            "slot":i,"typ":typ,"buff":buf,"nerf":nerf,"grade":grade_sel,
                            "acc_key":acc["key"],"acc_row":acc["acc"],"allowed":acc["allowed"],
                            "sp":sp, "ench":ench_name, "dist":dist,
                            "pnd":pnd_pct, "pnd_id": (None if pn is None else pn["id"]),
                            "ub": best_bib  # 상계(40젬 가정) — 가지치기에 사용
                        })
        if not cands: 
            continue

        if full_search:
            cand_by_slot[i]=cands
        else:
            # 컷 적용: ub로 대략 거르고, 상위 top_k만
            cands.sort(key=lambda x:x["ub"], reverse=True)
            cand_by_slot[i]=cands[:top_k if top_k else None]

    valid=[i for i in used_slots if cand_by_slot.get(i)]
    if len(valid) < NUM_DECKS:
        return pd.DataFrame({"오류":["유효 후보 부족"]}), pd.DataFrame({"평균 비벨":[0],"평균TAR":[0]}), {"choice":[]}

    # BnB 상계: 슬롯별 최대 ub
    slot_lists=[cand_by_slot[i] for i in valid]
    slot_ub=[(slot_lists[k][0]["ub"] if slot_lists[k] else 0) for k in range(len(slot_lists))]

    best_choice=None
    best_bib_sum=0
    best_score=None

    def score_tuple(sel:List[dict])->Tuple[float,float]:
        tars = sum(c["T"] for c in sel)/NUM_DECKS
        bibs = sum(c["bib"] for c in sel)/NUM_DECKS
        if goal=="TAR우선":   return (tars, bibs)
        return (bibs, tars)   # 비벨우선

    def try_place(idx:int, sel:List[dict], pool:Dict[Tuple[str,int],int], used_acc:Set[str], used_sp:Set[str], used_pn:Set[str], cur_bib_sum:int):
        nonlocal best_choice, best_bib_sum, best_score
        if len(sel)==NUM_DECKS:
            sc=score_tuple(sel)
            if (best_score is None) or (sc>best_score):
                best_score=sc; best_choice=list(sel); best_bib_sum=sum(c["bib"] for c in sel)
            return

        remain = NUM_DECKS - len(sel)
        ub = cur_bib_sum + sum(slot_ub[idx: idx+remain])
        if ub <= best_bib_sum:
            return

        for c in slot_lists[idx]:
            if c["acc_key"] in used_acc: continue
            if c["sp"]["id"] in used_sp: continue
            if c["pnd_id"] is not None and c["pnd_id"] in used_pn: continue

            # ★ 지금 남은 전역 풀로 실제 젬 배치
            ok, use_map, sums, used_str = allocate_gems_for_dist(pool, c["dist"])
            if not ok: 
                continue

            # 실제 bib/T 계산
            base_manual = None if "(진각)" in c["typ"] else _get_base_stats_cached(strip_jingak(c["typ"]))
            jingak = "(진각)" in c["typ"]
            bib, final = compute_final_stat_and_bib(
                c["typ"], ENCH_DICT.get(c["ench"], {"hp":0,"atk":0,"def":0}),
                c["acc_row"], sums, c["sp"]["spec"], c["pnd"],
                c["buff"], c["nerf"], c["grade"],
                base_stat_manual=base_manual, jingak=jingak
            )
            T = tar_percent(bib, get_plain_M(2, week_typ, week_buff))

            c_run = dict(c)
            c_run.update({
                "use_map":use_map, "sums":sums, "used":used_str,
                "bib":int(bib), "T":float(T), "final":final
            })

            # TAR 컷은 실제 값으로 검사(전수조사 OFF일 때만 의미)
            if (not full_search) and (float(T) < float(min_tar_cut)):
                continue

            # 소모
            for k,v in use_map.items(): pool[k]-=v
            used_acc.add(c["acc_key"]); used_sp.add(c["sp"]["id"])
            if c["pnd_id"] is not None: used_pn.add(c["pnd_id"])
            sel.append(c_run)

            try_place(idx+1, sel, pool, used_acc, used_sp, used_pn, cur_bib_sum + int(bib))

            # 롤백
            sel.pop()
            if c["pnd_id"] is not None: used_pn.remove(c["pnd_id"])
            used_sp.remove(c["sp"]["id"]); used_acc.remove(c["acc_key"])
            for k,v in use_map.items(): pool[k]+=v

    try_place(0, [], dict(gem_pool), set(), set(), set(), 0)

    if not best_choice:
        return pd.DataFrame({"오류":["조합 없음"]}), pd.DataFrame({"평균 비벨":[0],"평균TAR":[0]}), {"choice":[]}

    rows=[]
    for i,c in enumerate(best_choice, start=1):
        used = _compact_used_gems(c["used"])
        rows.append({
            "순번":i,"타입":c["typ"],"버프":c["buff"],"너프":c["nerf"], "등급":c["grade"],
            "펜던트": pendant_label(c["pnd"]),
            "젬분배":f"{c['dist'][0]}/{c['dist'][1]}/{c['dist'][2]}",
            "젬":used,
            "장신구":c["acc_key"].rsplit(" ",1)[0], "인첸트":c["ench"],
            "정령":format_spirit_label(c["sp"]["spec"]),
            "비벨":c["bib"], "TAR":round(c["T"],2),
            "HP":c["final"]["hp"], "ATK":c["final"]["atk"], "DEF":c["final"]["def"]
        })
    df=pd.DataFrame(rows)
    avg=pd.DataFrame({"평균 비벨":[int(sum(r["비벨"] for r in rows)/len(rows))],
                      "평균TAR":[round(sum(r["TAR"] for r in rows),2)/len(rows)]})
    return df, avg, {"choice":best_choice}

def view_base_from_ctx(ctx):
    choice=(ctx or {}).get("choice",[])
    if not choice: return pd.DataFrame({"오류":["먼저 길드전 최적화를 실행하세요."]})
    rows=[]
    for i,c in enumerate(choice, start=1):
        rows.append({"순번":i,"타입":c["typ"],"버프":c["buff"],"너프":c["nerf"],
                     "HP":c["final"]["hp"],"ATK":c["final"]["atk"],"DEF":c["final"]["def"]})
    return pd.DataFrame(rows)

# =============================================================================
# CSS / JS
# =============================================================================

CSS = """
footer, #footer, .theme-toggle {display:none !important;}
.gradio-container {max-width: 1220px !important; margin: 0 auto !important;}
#guild_table_df table td:nth-child(1), #guild_table_df table th:nth-child(1){ width:44px !important; text-align:center; }
"""

FOCUS_KILL_JS = """
<script>
(function(){
  var T=200, until=0;
  function now(){return Date.now();}
  function isTab(el){ if(!el) return false; var t=(el.tagName||'').toLowerCase(); return t==='summary'||(el.closest && el.closest('summary')); }
  document.addEventListener('pointerdown', e=>{ if(isTab(e.target)) until=now()+T;}, true);
  document.addEventListener('focusout', e=>{ if(now()<until) e.stopPropagation(); }, true);
})();
</script>
"""

# =============================================================================
# UI
# =============================================================================

def build_ui():
    with gr.Blocks(theme=gr.themes.Soft(), analytics_enabled=False, css=CSS) as demo:
        gr.HTML("<h2>TAR 정령 자동 계산기 / 길드전 최적화</h2>")
        visits_html = gr.HTML()
        gr.HTML(FOCUS_KILL_JS)

        with gr.Tabs():
            # -------------------- 자동 계산기 --------------------
            with gr.Tab("자동 계산기"):
                with gr.Row():
                    # 좌1
                    with gr.Column(scale=1, min_width=260):
                        types = gr.CheckboxGroup(label="타입", choices=TYPES_ALL, value=BASE_TYPES)
                        buff = gr.Dropdown(label="버프", choices=["최적화","1벞 최적화","0벞"]+ALL_BUFF_CHOICES, value="최적화")
                        nerf = gr.Dropdown(label="너프", choices=ALL_BUFF_CHOICES, value="0벞")
                        grade_sel = gr.Dropdown(label="등급", choices=["7.0","8.0","9.0"], value="9.0")

                        gr.Markdown("### 펜던트(자동)")
                        pnd_opt_on = gr.Checkbox(label="펜던트 자동 최적화 사용", value=True)
                        pnd_total = gr.Slider(0,18, value=18, step=1, label="펜던트 총합(%)")
                        pnd_grade = gr.Dropdown(label="펜던트 등급", choices=["별","달","태양"], value="태양",
                                                info="별=1줄(최대6), 달=2줄(최대12), 태양=3줄(최대18)")

                    # 좌2
                    with gr.Column(scale=1, min_width=260):
                        gemvals = gr.CheckboxGroup(label="젬 수치", choices=[str(v) for v in range(34,41)], value=["37"])

                        gr.Markdown("### 장신구")
                        acc_controls=[]
                        try:
                            base_df = accmod.df_acc.copy().sort_values(["lv","이름"])
                        except Exception:
                            base_df = pd.DataFrame([{"lv":19,"이름":"황보"}])
                        for lv in sorted(base_df["lv"].astype(int).unique()):
                            with gr.Accordion(f"{lv} 레벨", open=(lv==19)):
                                sub = base_df[base_df["lv"].astype(int)==lv]
                                for name in sub["이름"].astype(str).tolist():
                                    label=f"{lv} {name}"
                                    with gr.Row():
                                        use=gr.Checkbox(label=f"{label} 사용", value=(lv==19))
                                    with gr.Row():
                                        hp = gr.Checkbox(label="HP", value=True)
                                        atk= gr.Checkbox(label="ATK", value=True)
                                        df = gr.Checkbox(label="DEF", value=True)
                                    acc_controls.append((label,use,hp,atk,df))

                    # 좌3
                    with gr.Column(scale=1, min_width=260):
                        stat_choices=["체력","공격력","방어력"]; mode_choices=["+","%"]
                        with gr.Row():
                            s1=gr.Dropdown(stat_choices, value="체력", label="1옵 스탯")
                            m1=gr.Dropdown(mode_choices, value="%", label="1옵 모드")
                        with gr.Row():
                            s2=gr.Dropdown(stat_choices, value="공격력", label="2옵 스탯")
                            m2=gr.Dropdown(mode_choices, value="%", label="2옵 모드")
                        with gr.Row():
                            s3=gr.Dropdown(stat_choices, value="방어력", label="3옵 스탯")
                            m3=gr.Dropdown(mode_choices, value="%", label="3옵 모드")
                        with gr.Row():
                            s4=gr.Dropdown(stat_choices, value="방어력", label="4옵 스탯")
                            m4=gr.Dropdown(mode_choices, value="+", label="4옵 모드")
                        subopt=gr.Dropdown(["체력40","공격력10","방어력10"], value="체력40", label="부가옵")
                        sp_skip=gr.Checkbox(label="정령 미사용", value=False)

                    # 좌4
                    with gr.Column(scale=1, min_width=260):
                        topn = gr.Slider(1, 3000, value=200, step=1, label="상위 N개")
                        sort_key = gr.Radio(["비벨","TAR%"], value="TAR%", label="정렬")

                        gr.Markdown("### 중복 제거")
                        dedup_type = gr.Checkbox(label="타입", value=False)
                        dedup_buff = gr.Checkbox(label="버프", value=False)
                        dedup_acc  = gr.Checkbox(label="장신구", value=False)
                        dedup_ench = gr.Checkbox(label="인첸트", value=False)
                        dedup_sp   = gr.Checkbox(label="정령", value=False)

                # --- 펜던트(수동) ---
                gr.Markdown("### 펜던트(수동)")
                pnd_manual_on = gr.Checkbox(label="펜던트 수동 사용", value=False)
                pnd_manual_entries=[]
                for i in range(1,6+1):
                    with gr.Accordion(f"수동 펜던트 #{i}", open=False):
                        u = gr.Checkbox(label="사용", value=False)
                        with gr.Row():
                            l1s = gr.Dropdown(["체력","공격력","방어력"], value="체력", label="라인1 스탯")
                            l1v = gr.Slider(0,6,value=0,step=1,label="라인1 %")
                        with gr.Row():
                            l2s = gr.Dropdown(["체력","공격력","방어력"], value="공격력", label="라인2 스탯")
                            l2v = gr.Slider(0,6,value=0,step=1,label="라인2 %")
                        with gr.Row():
                            l3s = gr.Dropdown(["체력","공격력","방어력"], value="방어력", label="라인3 스탯")
                            l3v = gr.Slider(0,6,value=0,step=1,label="라인3 %")
                        pnd_manual_entries.append((u,l1s,l1v,l2s,l2v,l3s,l3v))

                run_btn = gr.Button("계산하기", variant="primary")
                reset_btn = gr.Button("리셋", variant="secondary")
                table = gr.Dataframe(label="결과", interactive=False, wrap=True)

                # ---------- 자동 계산기 프리셋 ----------
                gr.Markdown("### 즐겨찾기 프리셋 (자동 계산기)")
                auto_preset = gr.Radio(["1","2","3","4"], value="1", label="프리셋 선택")
                fav_msg = gr.HTML("")
                fav_save_btn = gr.Button("프리셋 저장")
                fav_load_btn = gr.Button("프리셋 불러오기")
                # 저장 확인 다이얼로그
                auto_confirm = gr.HTML(visible=False)
                auto_yes = gr.Button("예", visible=False)
                auto_no  = gr.Button("아니오", visible=False)

                auto_flat_inputs = [ctrl for g in acc_controls for ctrl in (g[1],g[2],g[3],g[4])]
                pnd_manual_flat = [ctrl for e in pnd_manual_entries for ctrl in e]

                def _collect_acc(*vals, acc_snapshot=acc_controls):
                    out=[]; it=iter(vals)
                    for (label,_u,_h,_a,_d) in acc_snapshot:
                        use=bool(next(it)); hp=bool(next(it)); atk=bool(next(it)); df=bool(next(it))
                        out.append((label,use,hp,atk,df))
                    return out

                def _collect_pnd_manual(*vals, snapshot=pnd_manual_entries):
                    out=[]; it=iter(vals)
                    for _ in snapshot:
                        u=bool(next(it)); l1s=next(it); l1v=int(next(it) or 0); l2s=next(it); l2v=int(next(it) or 0); l3s=next(it); l3v=int(next(it) or 0)
                        out.append({"use":u,"l1s":l1s,"l1v":l1v,"l2s":l2s,"l2v":l2v,"l3s":l3s,"l3v":l3v})
                    return out

                def _run_wrap(*vals, acc_snapshot=acc_controls, pnd_snapshot=pnd_manual_entries):
                    idx=0
                    types_sel = vals[idx]; idx+=1
                    gem_vals_sel = vals[idx]; idx+=1
                    acc_vals = vals[idx: idx+len(acc_snapshot)*4]; idx+=len(acc_snapshot)*4
                    acc_packed = _collect_acc(*acc_vals, acc_snapshot=acc_snapshot)
                    s1v,m1v,s2v,m2v,s3v,m3v,s4v,m4v,subv,spv = vals[idx:idx+10]; idx+=10
                    buff_mode = vals[idx]; idx+=1
                    nerf_mode = vals[idx]; idx+=1
                    grade_sel = vals[idx]; idx+=1
                    # 펜던트 자동
                    p_on, p_total, p_grade = vals[idx:idx+3]; idx+=3
                    # 수동 펜던트
                    pnd_man_on = vals[idx]; idx+=1
                    pnd_man_vals = vals[idx: idx+len(pnd_snapshot)*7]; idx+=len(pnd_snapshot)*7
                    pnd_manual = _collect_pnd_manual(*pnd_man_vals, snapshot=pnd_snapshot)
                    d_t,d_b,d_a,d_e,d_s = vals[idx:idx+5]; idx+=5
                    topnv,sortv = vals[idx:idx+2]; idx+=2

                    return auto_run(
                        types_sel, gem_vals_sel, acc_packed,
                        s1v,m1v,s2v,m2v,s3v,m3v,s4v,m4v,subv,spv,
                        buff_mode, nerf_mode, grade_sel,
                        False,
                        p_on, p_total, p_grade,
                        pnd_man_on, pnd_manual,
                        d_t,d_b,d_a,d_e,d_s,
                        topnv,sortv
                    )

                run_btn.click(
                    fn=_run_wrap,
                    inputs=[types, gemvals] + auto_flat_inputs + [
                        s1,m1,s2,m2,s3,m3,s4,m4,subopt,sp_skip,
                        buff, nerf, grade_sel,
                        pnd_opt_on, pnd_total, pnd_grade,
                        pnd_manual_on] + pnd_manual_flat + [
                        dedup_type, dedup_buff, dedup_acc, dedup_ench, dedup_sp,
                        topn, sort_key
                    ],
                    outputs=[table]
                )
                reset_btn.click(lambda: pd.DataFrame(), None, [table])

                # ---- 자동 프리셋 저장/불러오기 ----
                def _auto_save_preset(preset, *vals, request:gr.Request|None=None, acc_snapshot=acc_controls, pnd_snapshot=pnd_manual_entries):
                    try:
                        payload={"vals":list(vals)}
                        msg = fav_save_preset("auto", int(preset), payload, request)
                        return msg, gr.update(visible=False), gr.update(visible=False)
                    except Exception as e:
                        return f"<span style='color:red;'>저장 실패: {e}</span>", gr.update(visible=False), gr.update(visible=False)

                def _auto_prepare_confirm(preset):
                    return (
                        f"<b>정말로 {preset}번 프리셋에 저장하시겠습니까?</b>",
                        gr.update(visible=True), gr.update(visible=True)
                    )

                fav_save_btn.click(
                    fn=_auto_prepare_confirm,
                    inputs=[auto_preset],
                    outputs=[auto_confirm, auto_yes, auto_no]
                )

                auto_yes.click(
                    fn=_auto_save_preset,
                    inputs=[auto_preset, types, gemvals] + auto_flat_inputs + [
                        s1,m1,s2,m2,s3,m3,s4,m4,subopt,sp_skip,
                        buff, nerf, grade_sel,
                        pnd_opt_on, pnd_total, pnd_grade,
                        pnd_manual_on] + pnd_manual_flat + [
                        dedup_type, dedup_buff, dedup_acc, dedup_ench, dedup_sp,
                        topn, sort_key
                    ],
                    outputs=[fav_msg, auto_yes, auto_no]
                )
                auto_no.click(lambda: ("<span>취소됨</span>", gr.update(visible=False), gr.update(visible=False)),
                              None, [fav_msg, auto_yes, auto_no])

                def _auto_load_preset(preset, request:gr.Request|None=None, acc_snapshot=acc_controls, pnd_snapshot=pnd_manual_entries):
                    data = fav_load_preset("auto", int(preset), request)
                    vals = data.get("vals", [])
                    need = (2 + len(acc_snapshot)*4 + 10 + 3 + 3 + 1 + len(pnd_snapshot)*7 + 5 + 2)
                    if not vals:
                        return [gr.update()] * need + ["<span>해당 프리셋 저장본 없음</span>"]
                    vals = (vals + [gr.update()]*(need-len(vals)))[:need]
                    return vals + ["<span style='color:blue;'>불러옴</span>"]

                fav_load_btn.click(
                    fn=_auto_load_preset, inputs=[auto_preset],
                    outputs=[types, gemvals] + auto_flat_inputs + [
                        s1,m1,s2,m2,s3,m3,s4,m4,subopt,sp_skip,
                        buff, nerf, grade_sel,
                        pnd_opt_on, pnd_total, pnd_grade,
                        pnd_manual_on] + pnd_manual_flat + [
                        dedup_type, dedup_buff, dedup_acc, dedup_ench, dedup_sp,
                        topn, sort_key,
                        fav_msg
                    ]
                )

            # -------------------- 길드전 셋팅 최적화 --------------------
            with gr.Tab("길드전 셋팅 최적화"):
                gr.Markdown("### 이번 주 기준 (분모: DB2)")
                with gr.Row():
                    week_typ = gr.Dropdown(label="타입", choices=BASE_TYPES, value="체")
                    week_buff= gr.Dropdown(label="버프(2벞)", choices=TWO_BUFFS, value="HP40%")

                gr.Markdown("### 드래곤 슬롯")
                slot_used=[]; slot_typ=[]; slot_buff=[]; slot_grade=[]; slot_nerf=[]
                for i in range(1,7):
                    with gr.Accordion(f"드래곤 {i}", open=(i<=3)):
                        with gr.Row():
                            used = gr.Checkbox(label="사용", value=(i<=3)); slot_used.append(used)
                            s_type= gr.Dropdown(label="타입", choices=BASE_TYPES, value="체"); slot_typ.append(s_type)
                            s_buff= gr.Dropdown(label="버프", choices=ALL_BUFF_CHOICES, value="HP40%"); slot_buff.append(s_buff)
                            s_nerf= gr.Dropdown(label="너프", choices=ALL_BUFF_CHOICES, value="0벞"); slot_nerf.append(s_nerf)
                            s_grade=gr.Dropdown(label="등급", choices=["7.0","8.0","9.0"], value="7.0"); slot_grade.append(s_grade)

                gr.Markdown("### 장신구 인벤")
                acc_controls_g=[]
                try:
                    base_df = accmod.df_acc.copy().sort_values(["lv","이름"])
                except Exception:
                    base_df = pd.DataFrame([{"lv":19,"이름":"황보"}])
                for lv in sorted(base_df["lv"].astype(int).unique()):
                    with gr.Accordion(f"{lv} 레벨", open=(lv==19)):
                        sub = base_df[base_df["lv"].astype(int)==lv]
                        for name in sub["이름"].astype(str).tolist():
                            label=f"{lv} {name}"
                            with gr.Accordion(f"{label} (인스턴스)", open=False):
                                for inst in (1,2,3):
                                    with gr.Row():
                                        use= gr.Checkbox(label=f"{label} #{inst} 사용", value=False)
                                        hp = gr.Checkbox(label="HP", value=True)
                                        atk= gr.Checkbox(label="ATK", value=True)
                                        df = gr.Checkbox(label="DEF", value=True)
                                    acc_controls_g.append((label,inst,use,hp,atk,df))

                def _collect_acc_df(*vals, acc_snapshot=acc_controls_g):
                    rows=[]; it=iter(vals)
                    for (label,inst,_u,_h,_a,_d) in acc_snapshot:
                        use=bool(next(it)); hp=bool(next(it)); atk=bool(next(it)); df=bool(next(it))
                        rows.append({"장신구":label,"인스턴스":f"#{inst}","사용":use,"HP":hp,"ATK":atk,"DEF":df})
                    return pd.DataFrame(rows)

                gr.Markdown("### 펜던트 인벤")
                pend_controls=[]
                for i in range(1,7):
                    with gr.Accordion(f"펜던트 #{i}", open=(i<=3)):
                        u=gr.Checkbox(label="사용", value=(i<=3))
                        with gr.Row():
                            l1s=gr.Dropdown(["체력","공격력","방어력"], value="체력", label="라인1 스탯"); l1v=gr.Slider(0,6,value=0,step=1,label="라인1 %")
                        with gr.Row():
                            l2s=gr.Dropdown(["체력","공격력","방어력"], value="공격력", label="라인2 스탯"); l2v=gr.Slider(0,6,value=0,step=1,label="라인2 %")
                        with gr.Row():
                            l3s=gr.Dropdown(["체력","공격력","방어력"], value="방어력", label="라인3 스탯"); l3v=gr.Slider(0,6,value=0,step=1,label="라인3 %")
                        pend_controls.append((u,l1s,l1v,l2s,l2v,l3s,l3v))

                def _collect_pendant_inv(*vals, snapshot=pend_controls):
                    out=[]; it=iter(vals)
                    for _ in snapshot:
                        u=bool(next(it)); l1s=next(it); l1v=int(next(it) or 0); l2s=next(it); l2v=int(next(it) or 0); l3s=next(it); l3v=int(next(it) or 0)
                        out.append({"use":u,"l1s":l1s,"l1v":l1v,"l2s":l2s,"l2v":l2v,"l3s":l3s,"l3v":l3v})
                    return out

                gr.Markdown("### 젬 인벤")
                gem_df = gr.Dataframe(value=gem_inventory_default(), interactive=True, wrap=True)

                gr.Markdown("### 정령")
                spirits=[]
                spirit_binds=["정령 귀속X"]+[f"{i}번드래곤" for i in range(1,7)]
                for i in range(1,7):
                    d=spirit_default_row()
                    with gr.Accordion(f"정령 {i}", open=(i<=3)):
                        use=gr.Checkbox(label="사용", value=(i<=3))   # 1~3 기본 사용 체크
                        bind=gr.Dropdown(label="정령 귀속", choices=spirit_binds, value="정령 귀속X")
                        s1_=gr.Dropdown(["체력","공격력","방어력"], value=d["1옵 스탯"], label="1옵 스탯")
                        m1_=gr.Dropdown(["+","%"], value=d["1옵 모드"], label="1옵 모드")
                        s2_=gr.Dropdown(["체력","공격력","방어력"], value=d["2옵 스탯"], label="2옵 스탯")
                        m2_=gr.Dropdown(["+","%"], value=d["2옵 모드"], label="2옵 모드")
                        s3_=gr.Dropdown(["체력","공격력","방어력"], value=d["3옵 스탯"], label="3옵 스탯")
                        m3_=gr.Dropdown(["+","%"], value=d["3옵 모드"], label="3옵 모드")
                        s4_=gr.Dropdown(["체력","공격력","방어력"], value=d["4옵 스탯"], label="4옵 스탯")
                        m4_=gr.Dropdown(["+","%"], value=d["4옵 모드"], label="4옵 모드")
                        sub_=gr.Dropdown(["체력40","공격력10","방어력10"], value=d["부가옵"], label="부가옵")
                        spirits.append((use,bind,s1_,m1_,s2_,m2_,s3_,m3_,s4_,m4_,sub_))

                # 정확도/속도 컨트롤
                with gr.Row():
                    full_search = gr.Checkbox(label="전수조사(정확도 100%)", value=False, info="켜면 슬롯별 상한/컷을 무시하고 모든 후보를 탐색합니다.")
                    top_k = gr.Slider(0, 400, value=80, step=1, label="슬롯별 후보 상한(0=무제한, 전수조사 OFF에서만 적용)")
                min_tar = gr.Slider(0, 999, value=0, step=1, label="후보 TAR 하한(%, 전수조사 OFF에서만 적용)")
                goal = gr.Radio(["비벨우선","TAR우선"], value="비벨우선", label="목표")  # 혼합 제거

                run2 = gr.Button("최적화 시작", variant="primary")
                guild_table = gr.Dataframe(label="길드전 최적 세팅(3덱)", interactive=False, wrap=True, elem_id="guild_table_df")
                guild_avg = gr.Dataframe(label="평균", interactive=False, wrap=True)
                ctx_state = gr.State({})

                flatten_acc_g=[]
                for (_,_,use,hp,atk,df) in acc_controls_g:
                    flatten_acc_g.extend([use,hp,atk,df])

                flatten_sp=[]
                for s in spirits:
                    flatten_sp.extend(list(s))

                flatten_pends=[]
                for p in pend_controls:
                    flatten_pends.extend(list(p))

                def _guild_run(week_typ, week_buff, *vals, acc_snapshot=acc_controls_g, pend_snapshot=pend_controls):
                    idx=0
                    used=[bool(vals[idx+i]) for i in range(6)]; idx+=6
                    styp=[vals[idx+i] for i in range(6)]; idx+=6
                    sbuf=[vals[idx+i] for i in range(6)]; idx+=6
                    sner=[vals[idx+i] for i in range(6)]; idx+=6
                    sgra=[vals[idx+i] for i in range(6)]; idx+=6

                    pend_vals = vals[idx: idx+len(pend_snapshot)*7]; idx+=len(pend_snapshot)*7
                    pnd_rows = _collect_pendant_inv(*pend_vals, snapshot=pend_snapshot)

                    gem_df_val=vals[idx]; idx+=1

                    acc_vals = vals[idx: idx+len(acc_snapshot)*4]; idx+=len(acc_snapshot)*4
                    acc_df_val = _collect_acc_df(*acc_vals, acc_snapshot=acc_snapshot)

                    spr_rows=[]
                    for _ in range(6):
                        use=bool(vals[idx]); idx+=1
                        bind=vals[idx]; idx+=1
                        s1v=vals[idx]; idx+=1; m1v=vals[idx]; idx+=1
                        s2v=vals[idx]; idx+=1; m2v=vals[idx]; idx+=1
                        s3v=vals[idx]; idx+=1; m3v=vals[idx]; idx+=1
                        s4v=vals[idx]; idx+=1; m4v=vals[idx]; idx+=1
                        subv=vals[idx]; idx+=1
                        spr_rows.append({"사용":use,"귀속":bind,"1옵 스탯":s1v,"1옵 모드":m1v,"2옵 스탯":s2v,"2옵 모드":m2v,"3옵 스탯":s3v,"3옵 모드":m3v,"4옵 스탯":s4v,"4옵 모드":m4v,"부가옵":subv})

                    full = bool(vals[idx]); idx+=1
                    top=int(vals[idx]); idx+=1
                    mt=float(vals[idx]); idx+=1
                    goalv=vals[idx]; idx+=1

                    df, avg, ctx = guild_optimize(
                        week_typ, week_buff,
                        used, styp, sbuf, sgra, sner,
                        pnd_rows,
                        acc_df_val, gem_df_val, spr_rows,
                        top_k=0 if full else top,
                        min_tar_cut=0.0 if full else mt,
                        full_search=full,
                        goal=goalv
                    )
                    return df, avg, ctx

                run2.click(
                    fn=_guild_run,
                    inputs=[week_typ, week_buff] +
                           slot_used + slot_typ + slot_buff + slot_nerf + slot_grade +
                           flatten_pends + [gem_df] +
                           flatten_acc_g + flatten_sp +
                           [full_search, top_k, min_tar, goal],
                    outputs=[guild_table, guild_avg, ctx_state]
                )

                view_btn = gr.Button("선택 덱 기본 스탯 보기", variant="secondary")
                base_view = gr.Dataframe(label="기본 스탯", interactive=False, wrap=True)
                view_btn.click(fn=view_base_from_ctx, inputs=[ctx_state], outputs=[base_view])

                # ---------- 길드전 프리셋 ----------
                gr.Markdown("### 즐겨찾기 프리셋 (길드전)")
                guild_preset = gr.Radio(["1","2","3","4"], value="1", label="프리셋 선택")
                fav_msg_g = gr.HTML("")
                fav_save_g = gr.Button("프리셋 저장")
                fav_load_g = gr.Button("프리셋 불러오기")
                # 저장 확인 다이얼로그
                guild_confirm = gr.HTML(visible=False)
                guild_yes = gr.Button("예", visible=False)
                guild_no  = gr.Button("아니오", visible=False)

                def _fav_save_g_preset(preset, week_typ, week_buff, *vals, request:gr.Request|None=None, pend_snapshot=pend_controls, acc_snapshot=acc_controls_g):
                    try:
                        vals = list(vals)
                        # 구성: slots(6*5) -> pend(6*7) -> [gem_df] -> acc(4*Na) -> spirits(66) -> 정확도컨트롤(4)
                        gem_idx = 6*5 + len(pend_snapshot)*7
                        gem_df_val = vals[gem_idx]
                        if isinstance(gem_df_val, pd.DataFrame):
                            vals[gem_idx] = {"__gem_records__": gem_df_val.to_dict("records")}
                        payload=[week_typ, week_buff] + vals
                        msg=fav_save_preset("guild", int(preset), {"vals":payload}, request)
                        return msg, gr.update(visible=False), gr.update(visible=False)
                    except Exception as e:
                        return f"<span style='color:red;'>저장 실패: {e}</span>", gr.update(visible=False), gr.update(visible=False)

                def _guild_prepare_confirm(preset):
                    return (
                        f"<b>정말로 {preset}번 프리셋에 저장하시겠습니까?</b>",
                        gr.update(visible=True), gr.update(visible=True)
                    )

                fav_save_g.click(
                    fn=_guild_prepare_confirm,
                    inputs=[guild_preset],
                    outputs=[guild_confirm, guild_yes, guild_no]
                )

                guild_yes.click(
                    fn=_fav_save_g_preset,
                    inputs=[guild_preset, week_typ, week_buff] +
                           slot_used + slot_typ + slot_buff + slot_nerf + slot_grade +
                           flatten_pends + [gem_df] +
                           flatten_acc_g + flatten_sp +
                           [full_search, top_k, min_tar, goal],
                    outputs=[fav_msg_g, guild_yes, guild_no]
                )
                guild_no.click(lambda: ("<span>취소됨</span>", gr.update(visible=False), gr.update(visible=False)),
                               None, [fav_msg_g, guild_yes, guild_no])

                def _fav_load_g_preset(preset, request:gr.Request|None=None, acc_snapshot=acc_controls_g, pend_snapshot=pend_controls):
                    data = fav_load_preset("guild", int(preset), request)
                    vals = data.get("vals", [])
                    if not vals:
                        need = 2 + 6*5 + len(pend_snapshot)*7 + 1 + len(acc_snapshot)*4 + 6*11 + 4
                        return [gr.update()]*need + ["<span>해당 프리셋 저장본 없음</span>"]
                    vals = list(vals)
                    gem_pos = 2 + 6*5 + len(pend_snapshot)*7
                    if gem_pos < len(vals):
                        obj = vals[gem_pos]
                        if isinstance(obj, dict) and "__gem_records__" in obj:
                            vals[gem_pos] = pd.DataFrame(obj["__gem_records__"])
                    need = 2 + 6*5 + len(pend_snapshot)*7 + 1 + len(acc_snapshot)*4 + 6*11 + 4
                    vals = (vals + [gr.update()]*(need-len(vals)))[:need]
                    return vals + ["<span style='color:blue;'>불러옴</span>"]

                fav_load_g.click(
                    fn=_fav_load_g_preset, inputs=[guild_preset],
                    outputs=[week_typ, week_buff] +
                            slot_used + slot_typ + slot_buff + slot_nerf + slot_grade +
                            flatten_pends + [gem_df] +
                            flatten_acc_g + flatten_sp +
                            [full_search, top_k, min_tar, goal, fav_msg_g]
                )

        # 방문자 카운트
        def _visit(request: gr.Request):
            return register_unique_visit(request)
        demo.load(_visit, inputs=None, outputs=visits_html)

    return demo

# =============================================================================
# Main
# =============================================================================

# 맨 아래 launch 부분만 수정
import os
port = int(os.environ.get("PORT", "7860"))
demo = build_ui()
demo.queue().launch(server_name="0.0.0.0", server_port=port, show_api=False, share=False)


