# -*- coding: utf-8 -*-
import sys
from itertools import product, permutations
from PyQt5.QtWidgets import (
    QApplication, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QListWidget, QCheckBox, QLineEdit,
    QGroupBox, QTableView, QHeaderView, QMessageBox, QProgressBar, QSpinBox
, QDialog, QPlainTextEdit, QDialogButtonBox, QGridLayout)
from PyQt5.QtCore import Qt, QSortFilterProxyModel, QThread, pyqtSignal
from PyQt5.QtGui import QStandardItemModel, QStandardItem


# ---- helper: robust accessory label ----
def _acc_label(row):
    if row is None:
        return ""
    if isinstance(row, dict):
        lvl = row.get("level", row.get("lv", row.get("레벨", 0)))
        name = row.get("이름", row.get("name", ""))
        try: lvl = int(lvl)
        except Exception: pass
        return f"{lvl} {name}".strip() if (name or lvl) else str(row)
    try:
        lvl = getattr(row, "level", getattr(row, "lv", getattr(row, "레벨", 0)))
        name = getattr(row, "이름", getattr(row, "name", ""))
        try: lvl = int(lvl)
        except Exception: pass
        if name or lvl:
            return f"{lvl} {name}".strip()
    except Exception:
        pass
    if isinstance(row, (list, tuple)) and len(row) >= 2:
        try: lvl = int(row[0])
        except Exception: lvl = row[0]
        return f"{lvl} {row[1]}".strip()
    return str(row)
class AllFilterProxy(QSortFilterProxyModel):
    """Column-wise exact-match filter via per-column combo boxes."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.str_filters = {}  # {col_index: selected_text or ""}
        self.setSortRole(Qt.EditRole)

    def set_str_filter(self, col, val):
        self.str_filters[col] = val or ""
        self.invalidateFilter()

    def clear_filters(self):
        self.str_filters.clear()
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row, source_parent):
        model = self.sourceModel()
        # check exact matches for selected values (ignore empty selections)
        for col, val in self.str_filters.items():
            if not val:
                continue
            idx = model.index(source_row, col, source_parent)
            cell = "" if not idx.isValid() else str(idx.data())
            if cell != val:
                return False
        return True


    def lessThan(self, left, right):
        col = left.column()
        role = self.sortRole()
        lv = left.data(role)
        rv = right.data(role)

        def to_num(x):
            if isinstance(x, (int, float)):
                return float(x)
            s = str(x).strip()
            if s.endswith('%'):
                s = s[:-1]
            s = s.replace(',', '')
            try:
                return float(s)
            except Exception:
                import re
                m = re.search(r'-?\d+(?:\.\d+)?', s)
                return float(m.group(0)) if m else float('nan')

        ln = to_num(lv)
        rn = to_num(rv)

        if ln != ln and rn != rn:
            return False
        if ln != ln:
            return True
        if rn != rn:
            return False
        return ln < rn

# ---- calc deps ----
from calc.type import get_base_stats
from calc.awakening import get_awakening_stat
from calc.spirit import spirit_breakdown
from calc.collection import apply_collection
from calc.potion import apply_potion
import calc.accessory as accmod
import grail_cols
from calc.buff import BUFFS_ALL, ALL_BUFFS, ONE_BUFFS
from calc.gem import GEM_DISTS
BASE_TYPES = ["체","공","방","체공","체방","공방"]

# 인첸트 상수 (누락 복구)
ENCH_KR = {"HP": "체력 인첸트", "ATK": "공격력 인첸트", "DEF": "방어력 인첸트"}
ENCH_LIST = [
    ("HP",  {"hp":0.21,"atk":0.0,"def":0.0}),
    ("ATK", {"hp":0.0,"atk":0.21,"def":0.0}),
    ("DEF", {"hp":0.0,"atk":0.0,"def":0.21}),
]
TYPES = [
    "체","공","방","체공","체방","공방",
    "(진각)체","(진각)공","(진각)방","(진각)체공","(진각)체방","(진각)공방"
]

def _two_buff_pairs():
    def pct(hp=0.0, atk=0.0, df=0.0):
        return {"hp": float(hp), "atk": float(atk), "def": float(df)}
    return [
        ("HP40%", pct(0.40,0,0)),
        ("ATK40%", pct(0,0.40,0)),
        ("DEF40%", pct(0,0,0.40)),
        ("HP+ATK20%", pct(0.20,0.20,0)),
        ("HP+DEF20%", pct(0.20,0,0.20)),
        ("ATK+DEF20%", pct(0,0.20,0.20)),
    ]


def _filter_two_buff_candidates_by_buflbl(buf_lbl, tb_pairs):
    """
    최종 규칙:
    - 2벞(예: "HP+ATK20%") → 해당 2벞만.
    - 1벞 40%("ATK40%","DEF40%","HP40%") → 해당 40%만.
    - 1벞 20%:
        * ATK20% → {"ATK40%","HP+ATK20%","ATK+DEF20%"}
        * DEF20% → {"DEF40%","HP+DEF20%","ATK+DEF20%"}
        * HP20%  → {"HP40%", "HP+ATK20%", "HP+DEF20%"}
    - 그 외 → 원본 유지.
    """
    try:
        label = None if isinstance(buf_lbl, dict) else str(buf_lbl)
    except Exception:
        label = None

    if not tb_pairs:
        return tb_pairs

    # 표준 후보 집합
    names = [lab for (lab, _pct) in tb_pairs if isinstance(lab, str)]
    name_set = set(names)

    # 2벞 라벨은 해당 페어만
    if label and "+" in label:
        for pair_key in ("HP+ATK", "HP+DEF", "ATK+DEF"):
            if pair_key in label:
                pick = f"{pair_key}20%"
                if pick in name_set:
                    return [(lab, pct) for (lab, pct) in tb_pairs if lab == pick]
                break

    # 1벞 40% 라벨은 동일 40%만
    if label in ("ATK40%", "DEF40%", "HP40%"):
        if label in name_set:
            return [(lab, pct) for (lab, pct) in tb_pairs if lab == label]

    # 1벞 20% → 해당 40% + 관련 2벞
    mapping_keep_20 = {
        "ATK20%": {"ATK40%", "HP+ATK20%", "ATK+DEF20%"},
        "DEF20%": {"DEF40%", "HP+DEF20%", "ATK+DEF20%"},
        "HP20%" : {"HP40%",  "HP+ATK20%", "HP+DEF20%"},
    }
    if label in mapping_keep_20:
        keep = mapping_keep_20[label]
        return [(lab, pct) for (lab, pct) in tb_pairs if lab in keep]

    return tb_pairs
# ----- 정령 전 조합 (M 탐색 및 입력 확장용) -----
def gen_spirit_combis(mode):
    stats = ["체력","공격력","방어력"]
    combis = []
    if mode == "단일플":
        for flat in stats:
            per_stats = [s for s in stats if s != flat]
            for p1, p2 in permutations(per_stats, 2):
                combis.append([
                    {"stat":flat,"type":"+","slot":1},
                    {"stat":p1,  "type":"%","slot":2},
                    {"stat":p2,  "type":"%","slot":3},
                    {"stat":flat,"type":"%","slot":4},
                    {"stat":"체력40","type":"부가옵","slot":5},
                ])
    elif mode == "올퍼":
        for s1,s2,s3,s4 in product(stats, repeat=4):
            combis.append([
                {"stat":s1,"type":"%","slot":1},
                {"stat":s2,"type":"%","slot":2},
                {"stat":s3,"type":"%","slot":3},
                {"stat":s4,"type":"%","slot":4},
                {"stat":"체력40","type":"부가옵","slot":5},
            ])
    elif mode == "유막":
        for flat in stats:
            per_stats = [s for s in stats if s != flat]
            for p1, p2 in permutations(per_stats, 2):
                combis.append([
                    {"stat":flat,"type":"+","slot":1},
                    {"stat":p1,  "type":"%","slot":2},
                    {"stat":p2,  "type":"%","slot":3},
                    {"stat":flat,"type":"+","slot":4},
                    {"stat":"체력40","type":"부가옵","slot":5},
                ])
    elif mode == "막플":
        # 모든 경우의 수: slot1~3은 % 아무 조합(중복 허용), slot4는 + 아무 스탯 허용
        for s1,s2,s3 in product(stats, repeat=3):
            for flat in stats:
                combis.append([
                    {"stat":s1,"type":"%","slot":1},
                    {"stat":s2,"type":"%","slot":2},
                    {"stat":s3,"type":"%","slot":3},
                    {"stat":flat,"type":"+","slot":4},
                    {"stat":"체력40","type":"부가옵","slot":5},
                ])
    elif mode == "극막":
        # 모든 경우의 수: slot1~3은 % 3종 서로 다른 조합, slot4는 + 아무 스탯 허용
        for s1,s2,s3 in permutations(stats, 3):
            for flat in stats:
                combis.append([
                    {"stat":s1,"type":"%","slot":1},
                    {"stat":s2,"type":"%","slot":2},
                    {"stat":s3,"type":"%","slot":3},
                    {"stat":flat,"type":"+","slot":4},
                    {"stat":"체력40","type":"부가옵","slot":5},
                ])
    return combis

def _all_spirit_combis_for_max():
    """Stub in SPEC build: not used; keep for compatibility."""
    return []

def _spirit_combis_for_max_spec():
    """막플(81) + 극막(18) = 99 × 부가옵3 = 297 조합"""
    stats = ["체력","공격력","방어력"]
    base = []
    # 막플: % 슬롯 3개는 중복 허용, + 슬롯은 아무 스탯
    for s1 in stats:
        for s2 in stats:
            for s3 in stats:
                for flat in stats:
                    base.append([
                        {"stat":s1,"type":"%","slot":1},
                        {"stat":s2,"type":"%","slot":2},
                        {"stat":s3,"type":"%","slot":3},
                        {"stat":flat,"type":"+","slot":4},
                        {"stat":"체력40","type":"부가옵","slot":5},
                    ])
    # 극막: % 슬롯 3개 서로 다른 3종(순열), + 슬롯은 아무 스탯
    for s1, s2, s3 in permutations(stats, 3):
        for flat in stats:
            base.append([
                {"stat":s1,"type":"%","slot":1},
                {"stat":s2,"type":"%","slot":2},
                {"stat":s3,"type":"%","slot":3},
                {"stat":flat,"type":"+","slot":4},
                {"stat":"체력40","type":"부가옵","slot":5},
            ])
    # 부가옵 3종 확장
    expanded = []
    for comb in base:
        for sub in ("체력40","공격력10","방어력10"):
            c = [dict(x) for x in comb]
            c[-1]["stat"] = sub
            expanded.append(c)
    return expanded

    base = []
    # Only 막플, 극막 전수조사
    for m in ("막플","극막"):
        base += gen_spirit_combis(m)
    expanded = []
    for comb in base:
        for sub in ("체력40","공격력10","방어력10"):
            c = [dict(x) for x in comb]
            c[-1]["stat"] = sub
            expanded.append(c)
    return expanded

def _compute_bibel_core(typ, ench_pct, acc_row, gemv, dist, buf_lbl, spirit_opts, base_stat_manual=None):
    # base stats + awakening (수동입력 시 각성 미적용)
    if base_stat_manual is None:
        bs = get_base_stats(typ)
        aw = get_awakening_stat(typ)
    else:
        bs = dict(base_stat_manual)
        aw = {"hp":0,"atk":0,"def":0}
    C1 = {k: bs[k] + aw[k] for k in ("hp","atk","def")}

    # gem
    C2 = {"hp": gemv * 4 * dist[0], "atk": gemv * dist[1], "def": gemv * dist[2]}
    C3 = {k: C1[k] + C2[k] for k in C1}

    # potion
    pot = apply_potion({})
    C4 = {k: C3[k] + pot.get(k,0) for k in C3}

    # buff (원칙: 기본능력치 bs 기준 가산)
    buf_pct = buf_lbl if isinstance(buf_lbl, dict) else BUFFS_ALL.get(buf_lbl, {"hp":0,"atk":0,"def":0})
    buff_add = {k: int(bs[k] * buf_pct[k]) for k in bs}

    # accessory + enchant
    acc_pct = {"hp": acc_row["hp%"]/100, "atk": acc_row["atk%"]/100, "def": acc_row["def%"]/100}
    st1 = {k: int(C4[k] * (1 + acc_pct[k] + ench_pct.get(k, 0.0))) for k in C4}

    # spirit
    pct7, flat8, sub9 = spirit_breakdown(spirit_opts)
    st2 = {k: int(st1[k] * (1 + pct7[k])) + int(flat8[k] * (1 + pct7[k])) for k in st1}

    # collection + sub + buff add
    coll = apply_collection({"hp":0,"atk":0,"def":0})
    final = {
        "hp": st2["hp"] + coll["hp"] + sub9["hp"] + buff_add["hp"],
        "atk": st2["atk"] + coll["atk"] + sub9["atk"] + buff_add["atk"],
        "def": st2["def"] + coll["def"] + sub9["def"] + buff_add["def"]
    }
    return final["hp"] * final["atk"] * final["def"], final

_max_cache = {}
_max_pair_cache = {}
M_FIXED_BUFFS_MODE = '사용안함'
M_FIXED_BUFF_KEYS = []
M_FIXED_TYPE_MODE = '사용안함'  # '사용안함' | '모든 타입' | specific type in TYPES

def _strip_jingak(t: str):
    return t.replace('(진각)','') if isinstance(t,str) else t


def _ench_variants_for_max():
    """Three enchant variants (HP/ATK/DEF 21%)."""
    return [
        ("HP", {"hp": 0.21, "atk": 0.0, "def": 0.0}),
        ("ATK", {"hp": 0.0, "atk": 0.21, "def": 0.0}),
        ("DEF", {"hp": 0.0, "atk": 0.0, "def": 0.21}),
    ]




_SPEC_MAX_CACHE = {}

def get_abs_max_bibel_SPEC(typ, buf_lbl):
    """
    스펙 고정 최대 비벨:
      - 버프: buf_lbl 고정 (타 버프 탐색 금지)
      - gemv = 40 고정, GEM_DISTS 전부 탐색
      - 장신구/인첸트: 전부 탐색
      - 정령: 막플 ∪ 극막 전수조사
      - 각성 제외(기본 스펙): base_stat_manual=get_base_stats(_strip_jingak(typ))
    반환: (M, label, best_meta)
    """
    # 버프 고정
    if isinstance(buf_lbl, dict):
        fixed_lab = "(행 버프)"
        fixed_pct = buf_lbl
    else:
        fixed_lab = str(buf_lbl)
        fixed_pct = BUFFS_ALL.get(fixed_lab, {"hp":0.0,"atk":0.0,"def":0.0})

    # 장신구: 레벨 19만 사용
    try:
        import calc.accessory as accmod
        acc_rows_all = accmod.df_acc.to_dict("records")
        acc_rows = [r for r in acc_rows_all if str(r.get('lv', r.get('레벨',''))) == '19']
    except Exception:
        acc_rows = []

    # 인첸트 전체
    try:
        ENCH_LIST2 = _ench_variants_for_max()
    except Exception:
        ENCH_LIST2 = [
            ("HP",  {"hp":0.21,"atk":0.0,"def":0.0}),
            ("ATK", {"hp":0.0,"atk":0.21,"def":0.0}),
            ("DEF", {"hp":0.0,"atk":0.0,"def":0.21}),
        ]

    # 젬 분배 전체, gemv=40 고정
    try:
        from calc.gem import GEM_DISTS as _GEM_DISTS
    except Exception:
        _GEM_DISTS = [(4,1,1)]
    gemv = 40

    # 정령: 막플 ∪ 극막
    spirit_all = _spirit_combis_for_max_spec()

    # 탐색
    M = float("-inf")
    best_meta = {}
    for dist in _GEM_DISTS:
        for acc_row in (acc_rows or [{}]):
            for _ench_name, ench_pct in ENCH_LIST2:
                for comb in spirit_all:
                    try:
                        res = _compute_bibel_core(
                            typ, ench_pct, acc_row, gemv, dist,
                            fixed_pct, comb,
                            base_stat_manual=get_base_stats(_strip_jingak(typ))
                        )
                        bib = res[0] if isinstance(res, tuple) else res
                    except Exception:
                        bib = 0.0
                    if bib > M:  # float 비교, 동점 유지
                        M = float(bib)
                        best_meta = {
                            "타입": _strip_jingak(typ),
                            "젬분배": f"{dist[0]}/{dist[1]}/{dist[2]}",
                                                "장신구": f'{int((acc_row or {}).get("lv", (acc_row or {}).get("레벨", (acc_row or {}).get("level", 0))))} {(acc_row or {}).get("이름", "")}',
                            "인첸트": _ench_name,
                            "정령": (
                                comb[0]['stat']+comb[0]['type'],
                                comb[1]['stat']+comb[1]['type'],
                                comb[2]['stat']+comb[2]['type'],
                                comb[3]['stat']+comb[3]['type'],
                                comb[4]['stat'],
                            ),
                        }

    if M == float("-inf"):
        M = 0.0
    return M, fixed_lab, best_meta

def get_abs_max_bibel_SPEC_cached(typ, buf_lbl):
    """Cache M per (typ, buf)."""
    if isinstance(buf_lbl, dict):
        key = (typ, "(행 버프)", tuple(sorted(buf_lbl.items())))
    else:
        key = (typ, str(buf_lbl))
    val = _SPEC_MAX_CACHE.get(key)
    if val is None:
        val = get_abs_max_bibel_SPEC(typ, buf_lbl)
        _SPEC_MAX_CACHE[key] = val
    return val

def get_abs_max_bibel(typ, buf_lbl, *, gemv=None, base_stat_manual=None):
    """
    Compute absolute max bibel M for the given (typ, buf) row,
    honoring global fixed overrides:
      - M_FIXED_BUFFS_MODE / M_FIXED_BUFF_KEYS (list of dicts)
      - M_FIXED_TYPE_MODE ("사용안함" | "모든 타입" | type string)
    Cache key includes the resolved type set and fixed buff signature.
    Robust to empty/None accessory sources.
    """
        # Resolve buff list for M
    if M_FIXED_BUFFS_MODE == '2벞 모두':
        # 행의 '최대 버프' 라벨(예: 'HP40%')만을 대상으로 M을 계산한다.
        if isinstance(buf_lbl, dict):
            m_buffs_pairs = [("(행 버프)", buf_lbl)]
        else:
            v = BUFFS_ALL.get(str(buf_lbl))
            if not isinstance(v, dict):
                # synthesize minimal mapping for known labels
                _hp=_atk=_def=0.0
                s = str(buf_lbl)
                if s == "HP20%": _hp=0.20
                elif s == "ATK20%": _atk=0.20
                elif s == "DEF20%": _def=0.20
                elif s == "HP40%": _hp=0.40
                elif s == "ATK40%": _atk=0.40
                elif s == "DEF40%": _def=0.40
                elif s == "HP+ATK20%": _hp=_atk=0.20
                elif s == "HP+DEF20%": _hp=_def=0.20
                elif s == "ATK+DEF20%": _atk=_def=0.20
                v = {"hp":_hp, "atk":_atk, "def":_def}
            m_buffs_pairs = [(str(buf_lbl), v)]
    elif M_FIXED_BUFFS_MODE not in ('사용안함', None, '') and M_FIXED_BUFF_KEYS:
        # 전역 고정값을 사용
        m_buffs_pairs = list(M_FIXED_BUFF_KEYS)  # list of (label, dict)
    else:
        # 전역 고정 미사용: 행 버프 기준으로 M 계산
        if isinstance(buf_lbl, dict):
            m_buffs_pairs = [("(행 버프)", buf_lbl)]
        else:
            v = BUFFS_ALL.get(buf_lbl, {"hp":0,"atk":0,"def":0})
            m_buffs_pairs = [(str(buf_lbl), v)]

    # normalize m_buffs_pairs entries to (label, dict)
    _norm = []
    for b in m_buffs_pairs:
        if isinstance(b, tuple) and len(b) == 2 and isinstance(b[1], dict):
            _norm.append(b)
        elif isinstance(b, dict):
            _norm.append(("(행 버프)", b))
        else:
            _norm.append((str(b), BUFFS_ALL.get(str(b), {"hp":0,"atk":0,"def":0})))
    m_buffs_pairs = _norm

    # Resolve type list for M
    if M_FIXED_TYPE_MODE == '모든 타입':
        m_types = list(TYPES)
    elif M_FIXED_TYPE_MODE not in ('사용안함', None, ''):
        m_types = [M_FIXED_TYPE_MODE]
    else:
        m_types = [typ]

    # Build cache key
    def buff_sig(b):
        # b is (label, dict)
        lab, dic = b
        return ('LBL%', str(lab), float(dic.get('hp',0.0)), float(dic.get('atk',0.0)), float(dic.get('def',0.0)))
    key = (tuple(m_types), tuple(buff_sig(b) for b in m_buffs_pairs))
    if key in _max_cache:
        return _max_cache[key]

    # Prepare search space (defensive)
    try:
        acc_rows_19 = _get_acc_rows_level19()
        if not acc_rows_19:
            raise ValueError("empty acc_rows_19")
    except Exception:
        try:
            import calc.accessory as accmod
            acc_rows_all = accmod.df_acc.to_dict("records")
            acc_rows_19 = [r for r in acc_rows_all if str(r.get('lv', r.get('레벨',''))) == '19'] or acc_rows_all
        except Exception:
            acc_rows_19 = []  # last-resort: empty => M will be 0

    # Enchants
    try:
        ENCH_LIST2 = _ench_variants_for_max()
    except Exception:
        ENCH_LIST2 = [
            ("HP", {"hp": 0.21, "atk": 0.0, "def": 0.0}),
            ("ATK", {"hp": 0.0, "atk": 0.21, "def": 0.0}),
            ("DEF", {"hp": 0.0, "atk": 0.0, "def": 0.21}),
        ]

    # Spirits
    try:
        spirit_all = _all_spirit_combis_for_max()
    except Exception:
        spirit_all = []

    # GEM
    try:
        from calc.gem import GEM_DISTS as _GEM_DISTS
    except Exception:
        _GEM_DISTS = [(4,1,1)]  # minimal fallback
    gemv = 40 if gemv is None else gemv

    # Search
    M = 0
    best_label = None
    best_meta = {}
    for typ2 in m_types:
        for dist in _GEM_DISTS:
            for acc_row in acc_rows_19 or [{}]:
                for _ench_name, ench_pct in ENCH_LIST2:
                    iter_sp = (spirit_all if spirit_all else [None])
                    for comb in iter_sp:
                        for lab, mbl in m_buffs_pairs:
                            try:
                                res = _compute_bibel_core(_strip_jingak(typ), ench_pct, acc_row or {"hp%":0,"atk%":0,"def%":0}, gemv, dist, mbl, comb, base_stat_manual=get_base_stats(_strip_jingak(typ)))
                                bib = res[0] if isinstance(res, tuple) else res
                            except Exception:
                                try:
                                    bib, _ = _compute_bibel_core(_strip_jingak(typ), ench_pct, acc_row or {"hp%":0,"atk%":0,"def%":0}, gemv, dist, mbl, comb, base_stat_manual=get_base_stats(_strip_jingak(typ)))
                                except Exception:
                                    bib = 0
                            if bib > M:
                                M = float(bib)
                                best_label = lab
                                best_meta = {
                                    '타입': _strip_jingak(typ),
                                    '젬분배': f"{dist[0]}/{dist[1]}/{dist[2]}",
                                    '장신구': f"{int((acc_row or {}).get('레벨', (acc_row or {}).get('level',0)))} {(acc_row or {}).get('이름','')}",
                                    '인첸트': _ench_name,
                                    '정령': (comb[0]['stat']+comb[0]['type'] if comb else None,
                                             comb[1]['stat']+comb[1]['type'] if comb else None,
                                             comb[2]['stat']+comb[2]['type'] if comb else None,
                                             comb[3]['stat']+comb[3]['type'] if comb else None,
                                             comb[4]['stat'] if comb else None),
                                }
    _max_cache[key] = (M, best_label, best_meta)
    return M, best_label, best_meta

def get_abs_max_bibel_for_pair(typ, pair_label, pair_pct, *, gemv=None, base_stat_manual=None):
    key = (M_FIXED_TYPE_MODE, typ, pair_label)
    if key in _max_pair_cache:
        return _max_pair_cache[key]
    """
    Compute M using a single fixed buff pair (label, pct-dict), respecting M_FIXED_TYPE_MODE but
    ignoring M_FIXED_BUFFS_MODE globals. Returns (M, pair_label, best_meta).
    """
    # Resolve type list
    if M_FIXED_TYPE_MODE == '모든 타입':
        m_types = list(TYPES)
    elif M_FIXED_TYPE_MODE not in ('사용안함', None, ''):
        m_types = [M_FIXED_TYPE_MODE]
    else:
        m_types = [typ]

    # Accessory rows
    try:
        acc_rows_19 = _get_acc_rows_level19()
        if not acc_rows_19:
            raise ValueError("empty acc_rows_19")
    except Exception:
        try:
            import calc.accessory as accmod
            acc_rows_all = accmod.df_acc.to_dict("records")
            acc_rows_19 = [r for r in acc_rows_all if str(r.get('lv', r.get('레벨',''))) == '19'] or acc_rows_all
        except Exception:
            acc_rows_19 = []

    # Enchants
    try:
        ENCH_LIST2 = _ench_variants_for_max()
    except Exception:
        ENCH_LIST2 = [
            ("HP", {"hp": 0.21, "atk": 0.0, "def": 0.0}),
            ("ATK", {"hp": 0.0, "atk": 0.21, "def": 0.0}),
            ("DEF", {"hp": 0.0, "atk": 0.0, "def": 0.21}),
        ]

    # Spirits
    try:
        spirit_all = _all_spirit_combis_for_max()
    except Exception:
        spirit_all = []

    # GEM
    try:
        from calc.gem import GEM_DISTS as _GEM_DISTS
    except Exception:
        _GEM_DISTS = [(4,1,1)]
    gemv = 40 if gemv is None else gemv

    M = 0
    best_meta = {}
    for typ2 in m_types:
        for dist in _GEM_DISTS:
            for acc_row in acc_rows_19 or [{}]:
                for _ench_name, ench_pct in ENCH_LIST2:
                    iter_sp = (spirit_all if spirit_all else [None])
                    for comb in iter_sp:
                        try:
                            res = _compute_bibel_core(_strip_jingak(typ), ench_pct, acc_row or {"hp%":0,"atk%":0,"def%":0}, gemv, dist, pair_pct, comb, base_stat_manual=get_base_stats(_strip_jingak(typ)))
                            bib = res[0] if isinstance(res, tuple) else res
                        except Exception:
                            try:
                                bib, _ = _compute_bibel_core(_strip_jingak(typ), ench_pct, acc_row or {"hp%":0,"atk%":0,"def%":0}, gemv, dist, pair_pct, comb, base_stat_manual=get_base_stats(_strip_jingak(typ)))
                            except Exception:
                                bib = 0
                        if bib > M:
                            M = float(bib)
                            best_meta = {
                                '타입': _strip_jingak(typ),
                                '젬분배': f"{dist[0]}/{dist[1]}/{dist[2]}",
                                    '장신구': f"{int((acc_row or {}).get('레벨', (acc_row or {}).get('level',0)))} {(acc_row or {}).get('이름','')}",
                                '인첸트': _ench_name,
                                '정령': (comb[0]['stat']+comb[0]['type'] if comb else None,
                                         comb[1]['stat']+comb[1]['type'] if comb else None,
                                         comb[2]['stat']+comb[2]['type'] if comb else None,
                                         comb[3]['stat']+comb[3]['type'] if comb else None,
                                         comb[4]['stat'] if comb else None),
                            }
    _max_pair_cache[key] = (M, pair_label, best_meta)
    return _max_pair_cache[key]

# === Max-BIBEL search helpers (defined before use) ===
def _get_acc_rows_level19():
    """Return accessory rows filtered to level 19; fallback to all when empty."""
    try:
        import calc.accessory as accmod
        rows = accmod.df_acc.to_dict("records")
        rows19 = [r for r in rows if str(r.get('lv', r.get('레벨',''))) == '19']
        return rows19 or rows
    except Exception:
        return []

import grail_cols
    
def tar_percent(B, M):
    try:
        B = float(B); M = float(M)
    except:
        return 0.0
    if M <= 0: return 0.0
    p = 200.0 * (B / M) - 100.0
    if p < 0: p = 0.0
    if p > 100: p = 100.0
    return round(p, 4)

# ----- Worker -----

# ---- Forced SPEC M helper (respects M_FIXED_TYPE_MODE and M_FIXED_BUFFS_MODE) ----
def _forced_SPEC_MAX(typ, current_buf_lbl):
    """Return (M, label, meta) computed with SPEC search, but:
    - If M_FIXED_TYPE_MODE is set: compute max over that type selection.
    - If M_FIXED_BUFFS_MODE is set (or '2벞 모두'): use those labels instead of the row's buf.
    """
    # Resolve labels
    # '2벞 모두' 모드에서는 행 라벨별로 M을 계산해야 하므로, *현재 행의 라벨만* 사용한다.
    if M_FIXED_BUFFS_MODE == '2벞 모두':
        labels = [current_buf_lbl]
    elif M_FIXED_BUFFS_MODE not in ('사용안함', None, '') and isinstance(M_FIXED_BUFF_KEYS, list) and M_FIXED_BUFF_KEYS:
        labels = [M_FIXED_BUFF_KEYS[0][0]]
    else:
        labels = [current_buf_lbl]

    # Resolve types
    if M_FIXED_TYPE_MODE == '모든 타입':
        tlist = list(TYPES)
    elif M_FIXED_TYPE_MODE not in ('사용안함', None, ''):
        tlist = [M_FIXED_TYPE_MODE]
    else:
        tlist = [typ]

    best_M = float("-inf")
    best_lab = None
    best_meta = {}
    for t in tlist:
        for lab in labels:
            try:
                M, _lab, meta = get_abs_max_bibel_SPEC_cached(t, lab)
            except Exception:
                M, _lab, meta = 0.0, lab, {}
            if float(M) > float(best_M):
                best_M = float(M)
                best_lab = lab
                best_meta = meta
    if best_M == float("-inf"):
        best_M = 0.0
    return best_M, best_lab, best_meta
# ---- /Forced SPEC M helper ----

class CalcWorker(QThread):
    progress = pyqtSignal(int)
    result = pyqtSignal(list)
    error = pyqtSignal(str)
    def __init__(self, params):
        super().__init__()
        self.params = params
    def run(self):
        try:
            types = self.params["types"]
            acc_rows = self.params["acc_rows"]
            ench_list = self.params["ench_list"]
            spirit_slots = self.params["spirit_slots"]
            buff_list = self.params["buff_list"]
            gemv = self.params["gemv"]
            base_stat_manual = self.params["base_stat_manual"]
            results = []
            # === 2-buff TOP1 mode (trigger: fix_max_buff == '2벞 전체(top1)') ===
            _fix = None
            try:
                _fix = self.params.get("fix_max_buff")
            except Exception:
                _fix = None
            if _fix == "2벞 전체(top1)":
                import itertools

                def _pairs6():
                    def pct(hp=0.0, atk=0.0, df=0.0):
                        return {"hp": float(hp), "atk": float(atk), "def": float(df)}
                    return [
                        ("HP40%",      pct(0.40,0.00,0.00)),
                        ("ATK40%",     pct(0.00,0.40,0.00)),
                        ("DEF40%",     pct(0.00,0.00,0.40)),
                        ("HP+ATK20%",  pct(0.20,0.20,0.00)),
                        ("HP+DEF20%",  pct(0.20,0.00,0.20)),
                        ("ATK+DEF20%", pct(0.00,0.20,0.20)),
                    ]
                P = _pairs6()

                try:
                    gem_dists = self.params.get("gem_dists") or self.params.get("GEM_DISTS")
                except Exception:
                    gem_dists = None
                if not gem_dists:
                    try:
                        from calc.gem import GEM_DISTS as _GEM_DISTS
                        gem_dists = _GEM_DISTS
                    except Exception:
                        gem_dists = [(4,1,1)]
                _acc_rows = acc_rows if 'acc_rows' in locals() else []
                _ench_list = ench_list if 'ench_list' in locals() else [("HP", {"hp":0.21,"atk":0,"def":0}), ("ATK", {"hp":0,"atk":0.21,"def":0}), ("DEF", {"hp":0,"atk":0,"def":0.21})]
                _gemv = gemv if 'gemv' in locals() else 40
                spirit_combis = list(itertools.product(*spirit_slots)) if spirit_slots else []
                results = []
                for typ in types:
                    best_M = -1.0; best_lab = None; best_pack = None
                    for (lab, pct) in P:
                        M = -1.0; best = None
                        for dist in gem_dists:
                            for acc_row in (_acc_rows or [{}]):
                                for ench_name, ench_pct in _ench_list:
                                    for comb in spirit_combis:
                                        try:
                                            bib, final = _compute_bibel_core(typ, ench_pct, acc_row, _gemv, dist, pct, comb)
                                        except Exception:
                                            continue
                                        if bib > M:
                                            M = float(bib)
                                            best = (dist, acc_row, ench_name, ench_pct, comb, final, int(bib))
                        if best is not None and M > best_M:
                            best_M = M; best_lab = lab; best_pack = best
                    if best_pack is None:
                        continue
                    dist, acc_row, ench_name, ench_pct, comb, final, bib_int = best_pack
                    row = {
                        "타입": typ,
                        "젬분배": f"{dist[0]}/{dist[1]}/{dist[2]}",
                        "장신구": _acc_label(acc_row),
                        "인첸트": ench_name,
                        "버프": best_lab,
                        "HP": final["hp"], "ATK": final["atk"], "DEF": final["def"],
                        "이벨": final["hp"] + 4*final["atk"] + 4*final["def"],
                        "비벨": int(bib_int),
                        "최대 버프": best_lab,
                        "TAR": "100.0000%",
                    }
                    try:
                        row["1옵"] = f"{comb[0]['stat']}{comb[0]['type']}"
                        row["2옵"] = f"{comb[1]['stat']}{comb[1]['type']}"
                        row["3옵"] = f"{comb[2]['stat']}{comb[2]['type']}"
                        row["4옵"] = f"{comb[3]['stat']}{comb[3]['type']}"
                        row["부가옵"] = f"{comb[4]['stat']}" if (comb and len(comb)>4 and comb[4]) else ""
                    except Exception:
                        pass
                    results.append(row)

                self.progress.emit(len(results))
                self.result.emit(results)
                return

            processed = 0
            spirit_combis = list(product(*spirit_slots)) if spirit_slots else []
            for typ in types:
                for buf_lbl, _buf_pct in buff_list:
                    # Two branches:
                    # A) 2벞 모두: for each of the 6 fixed labels compute M_i and yield 6 rows per base case
                    # B) 일반: existing behavior (single M winner)
                    tb_pairs = None
                    if M_FIXED_BUFFS_MODE == '2벞 모두' or (isinstance(M_FIXED_BUFF_KEYS, list) and len(M_FIXED_BUFF_KEYS)==6 and all(isinstance(x, tuple) for x in M_FIXED_BUFF_KEYS)):
                        tb_pairs = (M_FIXED_BUFF_KEYS if M_FIXED_BUFFS_MODE=='2벞 모두' and M_FIXED_BUFF_KEYS else _two_buff_pairs())
                    tb_pairs = _filter_two_buff_candidates_by_buflbl(buf_lbl, tb_pairs)
                    for dist in GEM_DISTS:
                        for acc_row in acc_rows:
                            for ench_name, ench_pct in ench_list:
                                for comb in spirit_combis:
                                    # compute base bibel for this row
                                    bib, final = _compute_bibel_core(typ, ench_pct, acc_row, gemv, dist, buf_lbl, comb)
                                    if tb_pairs:
                                        # produce 6 rows: each M computed with that fixed label only
                                        for lab, pct in tb_pairs:
                                            M_val, _lab2, _meta = _forced_SPEC_MAX(typ, lab)
                                            row = {
                                                "타입": typ,
                                                "젬분배": f"{dist[0]}/{dist[1]}/{dist[2]}",
                                                "장신구": f'{int((acc_row or {}).get("lv", (acc_row or {}).get("레벨", (acc_row or {}).get("level", 0))))} {(acc_row or {}).get("이름", "")}',
                                                "인첸트": ench_name,
                                                "버프": buf_lbl,
                                                "HP": final["hp"],
                                                "ATK": final["atk"],
                                                "DEF": final["def"],
                                                "이벨": final["hp"] + 4*final["atk"] + 4*final["def"],
                                                "비벨": int(bib),
                                                "최대 버프": lab,
                                                "_M_META": _meta,
                                                "1옵": f'{comb[0]["stat"]}{comb[0]["type"]}',
                                                "2옵": f'{comb[1]["stat"]}{comb[1]["type"]}',
                                                "3옵": f'{comb[2]["stat"]}{comb[2]["type"]}',
                                                "4옵": f'{comb[3]["stat"]}{comb[3]["type"]}',
                                                "부가옵": f'{comb[4]["stat"]}',
                                                "TAR%": tar_percent(bib, M_val),
                                            }
                                            results.append(row)
                                    else:
                                        # single-winner branch (SPEC)
                                        M, winner, _meta = _forced_SPEC_MAX(typ, buf_lbl)
                                        row = {
                                            "타입": typ,
                                            "젬분배": f"{dist[0]}/{dist[1]}/{dist[2]}",
                                            "장신구": f'{int((acc_row or {}).get("lv", (acc_row or {}).get("레벨", (acc_row or {}).get("level", 0))))} {(acc_row or {}).get("이름", "")}',
                                            "인첸트": ench_name,
                                            "버프": buf_lbl,
                                            "HP": final["hp"],
                                            "ATK": final["atk"],
                                            "DEF": final["def"],
                                            "이벨": final["hp"] + 4*final["atk"] + 4*final["def"],
                                            "비벨": int(bib),
                                            "최대 버프": winner if winner else "(행 버프)",
                                            "_M_META": _meta,
                                            "1옵": f'{comb[0]["stat"]}{comb[0]["type"]}',
                                            "2옵": f'{comb[1]["stat"]}{comb[1]["type"]}',
                                            "3옵": f'{comb[2]["stat"]}{comb[2]["type"]}',
                                            "4옵": f'{comb[3]["stat"]}{comb[3]["type"]}',
                                            "부가옵": f'{comb[4]["stat"]}',
                                            "TAR%": tar_percent(bib, M),
                                        }
                                        results.append(row)
                                    processed += 1
                                    if processed % 200 == 0:
                                        self.progress.emit(processed)
            self.progress.emit(processed)
            self.result.emit(results)
        except Exception as e:
            import traceback
            self.error.emit(traceback.format_exc())

# ----- Main UI -----
class App(QWidget):
    def _resolve_mfix_label(self, label: str):
            """Map fixed-buff label to a pct dict. Prefer BUFFS_ALL if the label exists,
            otherwise synthesize from the label semantics."""
            if label in BUFFS_ALL:
                v = BUFFS_ALL[label]
                if isinstance(v, dict):
                    return v
            # synthesize
            m = {"hp":0.0,"atk":0.0,"def":0.0}
            if label == "HP20%": m["hp"]=0.20
            elif label == "ATK20%": m["atk"]=0.20
            elif label == "DEF20%": m["def"]=0.20
            elif label == "HP40%": m["hp"]=0.40
            elif label == "ATK40%": m["atk"]=0.40
            elif label == "DEF40%": m["def"]=0.40
            elif label == "HP+ATK20%": m["hp"]=0.20; m["atk"]=0.20
            else:
                return None
            return m

    def _fixed_two_buff_six(self):
            """
            Return the exact 6 two-stat buffs for M calc.
            Strategy:
            1) Prefer labels in BUFFS_ALL matching any of these pair names (common naming):
               ["공+방","공+체","방+체","공%+방%","공%+체%","방%+체%"]
               and also variants without '+' like ["공방","공체","방체","공방%","공체%","방체%"].
            2) If labels not found, synthesize 6 dicts by *combining* best available
               two-stat percent buffs detected from BUFFS_ALL (exactly two non-zero entries).
            The function returns a list of *dicts* so downstream can use them directly.
            """
            want_names = [
                "공+방","공+체","방+체",
                "공%+방%","공%+체%","방%+체%",
                "공방","공체","방체","공방%","공체%","방체%"
            ]
            picked = []
            # 1) pick by label
            for name in want_names:
                if name in BUFFS_ALL and BUFFS_ALL[name] not in picked:
                    picked.append(BUFFS_ALL[name])
                if len(picked) >= 6:
                    return picked[:6]

            # 2) fallback: discover two-nonzero buffs and choose representatives per pair
            def nz2(v): 
                return int(bool(v.get("hp",0))) + int(bool(v.get("atk",0))) + int(bool(v.get("def",0))) == 2
            # group by which stats are non-zero
            groups = {("hp","atk"):[], ("hp","def"):[], ("atk","def"):[]}
            for lab, buff in BUFFS_ALL.items():
                if not isinstance(buff, dict): 
                    continue
                if nz2(buff):
                    key = tuple(sorted([s for s in ("hp","atk","def") if buff.get(s,0)]))
                    if key in groups:
                        groups[key].append(buff)
            # choose the "strongest" per group twice: one for percent-ish (higher sum), one for flat-ish (lower sum) fallback
            out = []
            for key in [("atk","def"),("atk","hp"),("def","hp")]:
                lst = groups.get(key, [])
                if lst:
                    lst = sorted(lst, key=lambda d: (d.get(key[0],0)+d.get(key[1],0)), reverse=True)
                    out.append(lst[0])
                    if len(lst) > 1:
                        out.append(lst[-1])
                    else:
                        out.append(lst[0])
            return out[:6]

    def _two_buff_labels(self, limit=6):
            """Pick labels in BUFFS_ALL whose pct dict has exactly two non-zero entries."""
            picks = []
            for k, v in BUFFS_ALL.items():
                try:
                    nz = sum(1 for x in (v.get("hp",0), v.get("atk",0), v.get("def",0)) if x)
                    if nz == 2:
                        picks.append(k)
                except Exception:
                    continue
            # 안정적으로 정렬
            picks = sorted(set(picks))[:limit]
            return picks


        # ---- helpers for buff container shapes ----
    def _normalize_buff_container(self, obj):
            """Return list of (label, pct_dict) from:
            - dict like {label: pct_dict}
            - iterable of labels
            - iterable of (label, pct_dict)
            """
            items = []
            if isinstance(obj, dict):
                for k, v in obj.items():
                    items.append((k, v))
                return items
            try:
                for it in obj:
                    if isinstance(it, tuple) and len(it) == 2 and isinstance(it[1], dict):
                        items.append((it[0], it[1]))
                    else:
                        # treat as label
                        items.append((it, BUFFS_ALL.get(it, BUFFS_ALL.get("0벞", {"hp":0,"atk":0,"def":0}))))
            except Exception:
                pass
            return items
            try:
                for k in obj:
                    if isinstance(k, tuple) and len(k) == 2 and isinstance(k[1], dict):
                        items.append((k[0], k[1]))
                    else:
                        # treat as label
                        items.append((k, BUFFS_ALL.get(k, BUFFS_ALL.get("0벞", {"hp":0,"atk":0,"def":0}))))
            except Exception:
                pass
            return items


        # --- safe wrapper to avoid AttributeError during early init ---
    def _safe_update_combo_count(self):
        try:
            if hasattr(self, "update_combo_count"):
                self.update_combo_count()
        except Exception:
            pass

    def __init__(self):
            super().__init__()
            self._ui_ready = False
            self.setWindowTitle("TAR 계산기")
            self.resize(1280, 900)
            self.tabs = QTabWidget(self)
            root = QVBoxLayout(self)
            root.addWidget(self.tabs)
            self._build_input_tab()
            self._build_all_tab()
            self._build_save_tab()
            self.worker = None

        # 입력 탭
    def _build_input_tab(self):
            tab = QWidget()
            lay = QVBoxLayout(tab)
            lay.setContentsMargins(8,8,8,8)
            lay.setSpacing(6)

            # 타입
            g_type = QGroupBox("타입")
            l_type = QHBoxLayout(g_type)
            l_type.setContentsMargins(6,6,6,6)
            l_type.setSpacing(4)
            self.type_cb = QComboBox(); self.type_cb.addItem("최적화"); self.type_cb.addItem("진각 제외 최적화"); self.type_cb.addItems(TYPES)
            # 기본값: 진각 제외 최적화
            self.type_cb.setCurrentText("진각 제외 최적화")
            self.type_cb.currentIndexChanged.connect(self.update_combo_count)
            self.manual_chk = QCheckBox("수동입력")
            self.hp_in = QLineEdit(); self.hp_in.setPlaceholderText("HP")
            self.atk_in = QLineEdit(); self.atk_in.setPlaceholderText("ATK")
            self.def_in = QLineEdit(); self.def_in.setPlaceholderText("DEF")
            for w in (self.hp_in, self.atk_in, self.def_in): w.setFixedWidth(100)
            l_type.addWidget(self.type_cb); l_type.addWidget(self.manual_chk)
            l_type.addWidget(self.hp_in); l_type.addWidget(self.atk_in); l_type.addWidget(self.def_in)
            def on_manual(v):
                flag = (v == Qt.Checked)
                self.type_cb.setDisabled(flag)
                for w in (self.hp_in, self.atk_in, self.def_in):
                    w.setEnabled(flag)
                self.update_combo_count()
            self.manual_chk.stateChanged.connect(on_manual)  # defer first call until end

            # ---- (L2) 장신구 ----
            g_acc = QGroupBox("장신구")
            l_acc = QVBoxLayout(g_acc)
            l_acc.setContentsMargins(6, 6, 6, 6)
            l_acc.setSpacing(4)

            row = QHBoxLayout()
            row.setSpacing(6)

            self.acc_level_cb = QComboBox()
            self.acc_level_cb.addItems(["19", "18", "17", "16", "0", "5", "8"])
            self.acc_level_cb.setFixedWidth(80)
            self.acc_level_cb.setSizeAdjustPolicy(QComboBox.AdjustToContents)

            self.acc_name_cb = QComboBox()
            self.acc_name_cb.setFixedWidth(220)
            self.acc_name_cb.setSizeAdjustPolicy(QComboBox.AdjustToContents)

            self.acc_add_btn = QPushButton("추가")
            self.acc_del_btn = QPushButton("삭제")

            row.addWidget(QLabel("레벨:"))
            row.addWidget(self.acc_level_cb)
            row.addWidget(QLabel("이름:"))
            row.addWidget(self.acc_name_cb)
            row.addWidget(self.acc_add_btn)
            row.addWidget(self.acc_del_btn)
            row.addStretch(1)

            l_acc.addLayout(row)

            self.acc_list = QListWidget()
            self.acc_list.setFixedHeight(110)
            l_acc.addWidget(QLabel("선택 목록:"))
            l_acc.addWidget(self.acc_list)

            def fill_acc_names():
                lvl = int(self.acc_level_cb.currentText())
                names = list(accmod.df_acc[accmod.df_acc['lv']==lvl]['이름'])
                self.acc_name_cb.blockSignals(True)
                self.acc_name_cb.clear()
                self.acc_name_cb.addItem("최적화")
                for n in names:
                    self.acc_name_cb.addItem(f"{lvl} {n}")
                self.acc_name_cb.blockSignals(False)
                self.update_combo_count()
            self.acc_level_cb.currentTextChanged.connect(lambda _: fill_acc_names())
            fill_acc_names()

            def on_acc_add():
                name = self.acc_name_cb.currentText()
                if name == "최적화": return
                for i in range(self.acc_list.count()):
                    if self.acc_list.item(i).text() == name: return
                self.acc_list.addItem(name)
                self.update_combo_count()
            def on_acc_del():
                removed = False
                for it in self.acc_list.selectedItems():
                    self.acc_list.takeItem(self.acc_list.row(it))
                    removed = True
                if removed: self.update_combo_count()
            self.acc_add_btn.clicked.connect(on_acc_add)
            self.acc_del_btn.clicked.connect(on_acc_del)

            # 인첸트
            g_ench = QGroupBox("인첸트")
            l_ench = QHBoxLayout(g_ench)
            l_ench.setContentsMargins(6,6,6,6)
            l_ench.setSpacing(4)
            self.ench_cb = QComboBox(); self.ench_cb.addItem("최적화"); self.ench_cb.addItems(list(ENCH_KR.values()))
            self.ench_cb.currentIndexChanged.connect(self.update_combo_count)
            l_ench.addWidget(self.ench_cb)
            # 정령
            g_sp = QGroupBox("정령")
            l_sp = QVBoxLayout(g_sp)
            l_sp.setContentsMargins(6,6,6,6)
            l_sp.setSpacing(2)
            # --- compact spirit layout (QGridLayout, flush-left) ---
            grid = QGridLayout()
            grid.setContentsMargins(0,0,0,0)
            grid.setHorizontalSpacing(6)
            grid.setVerticalSpacing(4)
            grid.setContentsMargins(0,0,0,0)
            grid.setHorizontalSpacing(6)
            grid.setVerticalSpacing(4)

            self.sp_stat = []; self.sp_mode = []
            stat_choices = ["체력", "공격력", "방어력"]
            mode_choices = ["+", "%"]
            for i in range(4):
                s = QComboBox(); s.addItems(stat_choices); s.setSizeAdjustPolicy(QComboBox.AdjustToContents); s.setMinimumWidth(120)
                m = QComboBox(); m.addItems(mode_choices); m.setSizeAdjustPolicy(QComboBox.AdjustToContents); m.setMinimumWidth(80)
                s.currentIndexChanged.connect(self.update_combo_count)
                m.currentIndexChanged.connect(self.update_combo_count)
                self.sp_stat.append(s); self.sp_mode.append(m)

            # Labels (fixed narrow widths)
            lbls = [QLabel("1옵:"), QLabel("2옵:"), QLabel("3옵:"), QLabel("4옵:"), QLabel("부가옵:")]
            for idx, lb in enumerate(lbls):
                # 1~4옵 라벨 28, 부가옵 라벨 48
                lb.setFixedWidth(44 if idx < 4 else 70)

            # Row 0: 1옵 (col0~2), 2옵 (col3~5)
            grid.addWidget(lbls[0], 0, 0)
            grid.addWidget(self.sp_stat[0], 0, 1)
            grid.addWidget(self.sp_mode[0], 0, 2)
            grid.addWidget(lbls[1], 0, 3)
            grid.addWidget(self.sp_stat[1], 0, 4)
            grid.addWidget(self.sp_mode[1], 0, 5)

            # Row 1: 3옵, 4옵
            grid.addWidget(lbls[2], 1, 0)
            grid.addWidget(self.sp_stat[2], 1, 1)
            grid.addWidget(self.sp_mode[2], 1, 2)
            grid.addWidget(lbls[3], 1, 3)
            grid.addWidget(self.sp_stat[3], 1, 4)
            grid.addWidget(self.sp_mode[3], 1, 5)

            # Row 2: 부가옵
            self.sub_cb = QComboBox(); self.sub_cb.addItems(["체력40","공격력10","방어력10"]); self.sub_cb.setSizeAdjustPolicy(QComboBox.AdjustToContents); self.sub_cb.setMinimumWidth(130)
            self.sub_cb.currentIndexChanged.connect(self.update_combo_count)
            grid.addWidget(lbls[4], 2, 0)
            grid.addWidget(self.sub_cb, 2, 1)

            # Column stretch to keep left-packed and avoid centering
            for c in range(0, 6):
                grid.setColumnStretch(c, 0)
            grid.setColumnStretch(6, 1)  # trailing space column

            # Skip checkbox under grid
            self.sp_skip = QCheckBox("정령 미사용")
            def on_sp_skip(v):
                disabled = v == Qt.Checked
                for s in self.sp_stat: s.setDisabled(disabled)
                for m in self.sp_mode: m.setDisabled(disabled)
                self.sub_cb.setDisabled(disabled)
                self.update_combo_count()
            self.sp_skip.stateChanged.connect(on_sp_skip)

            l_sp.addLayout(grid)
            l_sp.addWidget(self.sp_skip)
            # --- /compact spirit layout ---
            # 젬
            g_gem = QGroupBox("젬")
            l_gem = QHBoxLayout(g_gem)
            l_gem.setContentsMargins(6,6,6,6)
            l_gem.setSpacing(4)
            l_gem.setContentsMargins(6,6,6,6)
            l_gem.setSpacing(4)
            self.gem_cb = QComboBox(); self.gem_cb.addItems(["37","38","39","40"]); self.gem_cb.setFixedWidth(90)
            self.gem_cb.setMinimumWidth(80)
            try:
                self.gem_cb.setSizeAdjustPolicy(QComboBox.AdjustToContents)
            except Exception:
                pass
            self.gem_cb.currentIndexChanged.connect(self.update_combo_count)
            l_gem.addWidget(QLabel("젬 수치:")); l_gem.addWidget(self.gem_cb)
            # 버프
            g_buff = QGroupBox("버프")
            l_buff = QVBoxLayout(g_buff)
            l_buff.setContentsMargins(6,6,6,6)
            l_buff.setSpacing(4)
            rowb = QHBoxLayout()
            self.buff_cb = QComboBox()
            self.buff_cb.addItem("최적화", "ALL-ALIAS")  # 최적화 = 모든 버프 다 돈다
            self.buff_cb.addItem("1벞 최적화", "ONE")
            self.buff_cb.addItem("0벞", "0벞")
            for key in BUFFS_ALL.keys():
                if key != "0벞": self.buff_cb.addItem(key, key)
            self.buff_cb.currentIndexChanged.connect(self.update_combo_count)
            self.buff_add_btn = QPushButton("추가"); self.buff_add_btn.setVisible(False); self.buff_add_btn.setEnabled(False); self.buff_del_btn = QPushButton("삭제"); self.buff_del_btn.setVisible(False); self.buff_del_btn.setEnabled(False)
            rowb.addWidget(self.buff_cb); rowb.addWidget(self.buff_add_btn); rowb.addWidget(self.buff_del_btn)
            l_buff.addLayout(rowb)
            self.compare_list = QListWidget(); self.compare_list.setFixedHeight(100)
            #        l_buff.addWidget(QLabel("비교 조합:")); l_buff.addWidget(self.compare_list)
            # 최대 비벨 고정 타입
            g_mfix_type = QGroupBox("최대 비벨 타입 고정")
            l_mfix_type = QHBoxLayout(g_mfix_type)
            l_mfix_type.setContentsMargins(6,6,6,6)
            l_mfix_type.setSpacing(4)
            self.mfix_type_cb = QComboBox()
            g_mfix_type.setToolTip("TAR% = 200*(B/M)-100 의 분모 M을 계산할 때의 타입 기준을 고정합니다.\n사용안함: 현재 행의 타입으로 M 계산\n모든 타입: 모든 타입 중 최대 비벨을 M으로 사용\n특정 타입: 그 타입 조건에서의 최대 비벨을 M으로 사용")
            self.mfix_type_cb.setToolTip("TAR% 분모(M) 계산 시 타입 고정 옵션")
            # 옵션: 사용안함 / 모든 타입 / 개별 타입들
            self.mfix_type_cb.addItem("사용안함")
            self.mfix_type_cb.addItem("모든 타입")
            for t in TYPES:
                self.mfix_type_cb.addItem(t)
            l_mfix_type.addWidget(self.mfix_type_cb)
            # 최대 비벨 고정 버프 설정
            g_mfix = QGroupBox("최대 비벨 버프 고정")
            l_mfix = QHBoxLayout(g_mfix)
            l_mfix.setContentsMargins(6,6,6,6)
            l_mfix.setSpacing(4)
            self.mfix_cb = QComboBox()
            g_mfix.setToolTip("TAR% 분모 M을 계산할 때 버프 기준을 고정합니다.\n사용안함: 현재 행의 버프로 M 계산\n목록에서 선택: 해당 버프(또는 조합) 조건에서의 최대 비벨을 M으로 사용")
            self.mfix_cb.setToolTip("TAR% 분모(M) 계산 시 버프 고정 옵션")
            self.mfix_cb.addItems(["사용안함","HP20%","ATK20%","DEF20%","HP40%","ATK40%","DEF40%","HP+ATK20%", "HP+DEF20%", "ATK+DEF20%"])
            self.mfix_cb.currentIndexChanged.connect(self.update_combo_count)
            l_mfix.addWidget(self.mfix_cb)
            self.buff_fix_2buff_both = QCheckBox('2벞 모두')
            self.buff_fix_2buff_both.setChecked(True)
            self.buff_fix_2buff_both.setToolTip('6개 두-스탯 버프(HP+ATK/HP+DEF/ATK+DEF의 20/40) 전수 최댓값으로 M 고정')
            self.buff_fix_2buff_both.stateChanged.connect(self.update_combo_count)

            l_mfix.addWidget(self.buff_fix_2buff_both)
#            lay.addWidget(QLabel("(TAR%는 위 고정 기준으로 계산한 최대 비벨 M을 분모로 사용)"))

            # 중복 필터
            g_dup = QGroupBox("중복 필터")
            l_dup = QVBoxLayout(g_dup)
            l_dup.setContentsMargins(6,6,6,6)
            l_dup.setSpacing(4)
            # 위에서 아래로 체크박스 나열: 타입, 버프, 장신구, 정령, 인첸트
            self.dup_type_chk = QCheckBox("타입")
            self.dup_buff_chk = QCheckBox("버프")
            self.dup_acc_chk = QCheckBox("장신구")
            self.dup_spirit_chk = QCheckBox("정령")
            self.dup_ench_chk = QCheckBox("인첸트")
            for w in (self.dup_type_chk, self.dup_buff_chk, self.dup_acc_chk, self.dup_spirit_chk, self.dup_ench_chk):
                l_dup.addWidget(w)
            # 우선순위(최대 기준): 이벨/비벨/TAR%
            rowp = QHBoxLayout()
            rowp.addWidget(QLabel("우선순위:"))
            self.dup_metric_cb = QComboBox()
            # 기본값: 비벨. 열 이름과 정확히 일치시킨다.
            self.dup_metric_cb.addItems(["비벨","이벨","TAR%"])
            rowp.addWidget(self.dup_metric_cb)
            rowp.addStretch(1)
            l_dup.addLayout(rowp)
            lay.addWidget(g_dup)


            def on_buff_add():
                t = self.type_cb.currentText()
                b = self.buff_cb.currentText()
                txt = f"{t} | {b}"
                for i in range(self.compare_list.count()):
                    if self.compare_list.item(i).text() == txt: return
                self.compare_list.addItem(txt)
                self.update_combo_count()
            def on_buff_del():
                removed = False
                for it in self.compare_list.selectedItems():
                    self.compare_list.takeItem(self.compare_list.row(it))
                    removed = True
                if removed: self.update_combo_count()
            # (removed) self.buff_add_btn.clicked.connect(...)
# (removed) self.buff_del_btn.clicked.connect(...)
# 조합 수 / 진행 / 실행
            self.combo_count_label = QLabel("조합 수: 0")
            self.progress = QProgressBar(); self.progress.setRange(0,0); self.progress.setVisible(False)
            self.calc_btn = QPushButton("최적화")
            # --- compact row layout to use horizontal space (ORDERED as requested) ---
            # 1행: 타입 · 버프 · 젬 · 인첸트
            row1 = QHBoxLayout(); row1.setContentsMargins(0,0,0,0); row1.setSpacing(6)
            row1.addWidget(g_type); row1.addWidget(g_buff); row1.addWidget(g_gem); row1.addWidget(g_ench); row1.addStretch(1)
            lay.addLayout(row1)

            # 2행: 최대 비벨(타입 고정) · 버프 고정 · 성배 모드
            row2 = QHBoxLayout(); row2.setContentsMargins(0,0,0,0); row2.setSpacing(6)
            row2.addWidget(g_mfix_type); row2.addWidget(g_mfix); row2.addStretch(1)
            lay.addLayout(row2)

            # 이후: 정령 → 장신구 → 중복 필터
            lay.addWidget(g_sp)
            lay.addWidget(g_acc)
            lay.addWidget(g_dup)
            # --- /compact row layout ---
            lay.addWidget(self.combo_count_label)
            lay.addWidget(self.progress)
            lay.addWidget(self.calc_btn)
            self.calc_btn.clicked.connect(self.handle_calculate)

            self.tabs.addTab(tab, "입력")
            self._ui_ready = True
            on_manual(self.manual_chk.checkState())
            self.update_combo_count()

        # 모든 경우 탭
    
    def _row_to_text(self, row: int) -> str:
            headers = [self.all_model.headerData(i, Qt.Horizontal) for i in range(self.all_model.columnCount())]
            vals = []
            for i,h in enumerate(headers):
                idx = self.all_model.index(row, i)
                vals.append(f"{h}: {self.all_model.data(idx, Qt.DisplayRole)}")
            return "\n".join(vals)

    def _save_row_to_text(self, row: int) -> str:
            headers = [self.save_model.headerData(i, Qt.Horizontal) for i in range(self.save_model.columnCount())]
            vals = []
            for i,h in enumerate(headers):
                idx = self.save_model.index(row, i)
                vals.append(f"{h}: {self.save_model.data(idx, Qt.DisplayRole)}")
            return "\n".join(vals)

    
    def _show_row_popup(self, index):
            try:
                src_index = self.all_proxy.mapToSource(index)
            except Exception:
                src_index = index
            row = src_index.row()
            col = src_index.column()

            headers = [self.all_model.headerData(i, Qt.Horizontal) for i in range(self.all_model.columnCount())]
            h_clicked = headers[col] if 0 <= col < len(headers) else ""

            def get(h):
                try:
                    ci = headers.index(h)
                    return self.all_model.data(self.all_model.index(row, ci), Qt.DisplayRole)
                except Exception:
                    return ""

            # 메타 로드
            meta = {}
            try:
                ci0 = headers.index('타입'); idx0 = self.all_model.index(row, ci0)
                m = self.all_model.data(idx0, Qt.UserRole+1)
                if isinstance(m, dict): 
                    meta = m
            except Exception:
                pass

            # 정령 문자열 조립 함수
            def spirit_tuple_to_str(tup):
                try:
                    a,b,c,d,e = tup
                    return f"{a} / {b} / {c} / {d} / {e}"
                except Exception:
                    return str(tup)

            def current_spirit_str():
                return f"{get('1옵')} / {get('2옵')} / {get('3옵')} / {get('4옵')} / {get('부가옵')}"

            # 분기: 비벨최대값 칼럼 더블클릭 시 = 최대 셋팅, 그 외 = 현재 셋팅
            if False:
                title = "비벨 최대값 셋팅"
                body = [
                    f"타입: {meta.get('타입', get('타입'))}",
                    f"버프: {get('최대 버프')}",
                    f"장신구: {meta.get('장신구', get('장신구'))}",
                    f"젬배치: {meta.get('젬분배', get('젬분배'))}",
                    f"정령: {spirit_tuple_to_str(meta.get('정령', (get('1옵'), get('2옵'), get('3옵'), get('4옵'), get('부가옵'))))}",
                ]
            else:
                title = "현재 셋팅"
                body = [
                    f"타입: {get('타입')}",
                    f"버프: {get('버프')}",
                    f"장신구: {get('장신구')}",
                    f"젬배치: {get('젬분배')}",
                    f"정령: {current_spirit_str()}",
                ]

            text = "※ " + title + "\n" + "\n".join(body)

            dlg = QDialog(self); dlg.setWindowTitle("설정 상세")
            v = QVBoxLayout(dlg)
            txt = QPlainTextEdit(); txt.setPlainText(text); txt.setReadOnly(True); v.addWidget(txt)

            btns = QDialogButtonBox()
            btn_copy = QPushButton("복사")
            btn_record = QPushButton("기록")
            btn_close = QPushButton("닫기")
            btns.addButton(btn_copy, QDialogButtonBox.ActionRole)
            btns.addButton(btn_record, QDialogButtonBox.ActionRole)
            btns.addButton(btn_close, QDialogButtonBox.RejectRole)
            v.addWidget(btns)

            btn_copy.clicked.connect(lambda: QApplication.clipboard().setText(text))
            # on record: copy the selected ALL row into SAVE model and switch tab
            try:
                src_row = row
            except Exception:
                src_row = None
            def _do_record():
                try:
                    if src_row is not None:
                        self._append_to_save_from_all(src_row)
                        # switch to 기록 탭
                        self.tabs.setCurrentIndex(self.tabs.count()-1)
                finally:
                    dlg.reject()
            btn_record.clicked.connect(_do_record)
            btn_close.clicked.connect(dlg.reject)

            dlg.resize(520, 420)
            dlg.exec_()

    def _show_save_popup(self, index):
            row = index.row()
            text = self._save_row_to_text(row)
            dlg = QDialog(self); dlg.setWindowTitle("기록 상세")
            v = QVBoxLayout(dlg)
            txt = QPlainTextEdit(); txt.setPlainText(text); txt.setReadOnly(True); v.addWidget(txt)

            btns = QDialogButtonBox()
            btn_copy = QPushButton("복사")
            btn_delete = QPushButton("삭제")
            btn_close = QPushButton("닫기")
            btns.addButton(btn_copy, QDialogButtonBox.ActionRole)
            btns.addButton(btn_delete, QDialogButtonBox.ActionRole)
            btns.addButton(btn_close, QDialogButtonBox.RejectRole)
            v.addWidget(btns)

            btn_copy.clicked.connect(lambda: QApplication.clipboard().setText(text))
            def _do_delete():
                try:
                    self.save_model.removeRow(row)
                finally:
                    dlg.reject()
            btn_delete.clicked.connect(_do_delete)
            btn_close.clicked.connect(dlg.reject)

            dlg.resize(520, 420)
            dlg.exec_()

    def _build_all_tab(self):

        tab = QWidget()
        lay = QVBoxLayout(tab)
        headers = ["타입","젬분배","장신구","인첸트","버프","HP","ATK","DEF","TAR","최대 버프","이벨","비벨","1옵","2옵","3옵","4옵","부가옵"]
        self.all_model = QStandardItemModel(0, len(headers))
        self.all_model.setHorizontalHeaderLabels(headers)
        self.all_proxy = AllFilterProxy()
        self.all_proxy.setSourceModel(self.all_model)
        self.all_proxy.setSortRole(Qt.EditRole)
        self.all_proxy.setFilterKeyColumn(-1)
        self.all_view = QTableView()
        self.all_view.setModel(self.all_proxy)
        self.all_view.setSortingEnabled(True)
        self.all_view.doubleClicked.connect(self._show_row_popup)
        self.all_view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # --- Chance boxes (column filters) ---
        self.filter_boxes = []  # list of (col_index, QComboBox)
        filter_row = QHBoxLayout()
        for col in range(self.all_model.columnCount()):
            cb = QComboBox()
            cb.setMinimumWidth(110)
            cb.addItem("전체")  # default shows all
            self.filter_boxes.append((col, cb))
            filter_row.addWidget(cb)
        lay.addLayout(filter_row)

        lay.addWidget(self.all_view)
        self.tabs.addTab(tab, "모든 경우")

    
    def _append_to_save_from_all(self, src_row: int):
            """Copy a row from all_model into save_model, preserving Display/Edit roles and meta."""
            if not hasattr(self, "save_model") or self.save_model is None:
                return
            headers = [self.all_model.headerData(i, Qt.Horizontal) for i in range(self.all_model.columnCount())]
            items = []
            for col in range(self.all_model.columnCount()):
                sidx = self.all_model.index(src_row, col)
                it = QStandardItem()
                # Preserve DisplayRole
                it.setData(self.all_model.data(sidx, Qt.DisplayRole), Qt.DisplayRole)
                # Preserve EditRole for numeric sorting
                it.setData(self.all_model.data(sidx, Qt.EditRole), Qt.EditRole)
                # Also keep visible text to avoid view quirks
                val = self.all_model.data(sidx, Qt.DisplayRole)
                it.setText("" if val is None else str(val))
                items.append(it)
            self.save_model.appendRow(items)
            # copy meta into first cell
            try:
                meta = self.all_model.data(self.all_model.index(src_row, 0), Qt.UserRole+1)
                if isinstance(meta, dict):
                    self.save_model.item(self.save_model.rowCount()-1, 0).setData(meta, Qt.UserRole+1)
            except Exception:
                pass
    def _build_save_tab(self):
            tab = QWidget()
            lay = QVBoxLayout(tab)
            try:
                headers = [self.all_model.headerData(i, Qt.Horizontal) for i in range(self.all_model.columnCount())]
            except Exception:
                headers = ["타입","젬분배","장신구","인첸트","버프","HP","ATK","DEF","TAR","최대 버프","이벨","비벨","1옵","2옵","3옵","4옵","부가옵"]
            self.save_model = QStandardItemModel(0, len(headers))
            self.save_model.setHorizontalHeaderLabels(headers)
            self.save_view = QTableView()
            self.save_view.setModel(self.save_model)
            self.save_view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            self.save_view.doubleClicked.connect(self._show_save_popup)
            lay.addWidget(self.save_view)
            self.tabs.addTab(tab, "기록")

        # ----- helpers to read UI -----
    
    def _build_spirit_slots_from_ui(self):
        if self.sp_skip.isChecked():
            return [[{"stat":"없음","type":"없음","slot":1}],
                    [{"stat":"없음","type":"없음","slot":2}],
                    [{"stat":"없음","type":"없음","slot":3}],
                    [{"stat":"없음","type":"없음","slot":4}],
                    [{"stat":"없음","type":"부가옵","slot":5}]]
        slots = []
        for i in range(4):
            s = self.sp_stat[i].currentText()
            m = self.sp_mode[i].currentText()
            cand = [{"stat": s, "type": m, "slot": i+1}]
            slots.append(cand)
        sub = self.sub_cb.currentText()
        slots.append([{"stat": sub, "type": "부가옵", "slot": 5}])
        return slots



    def _resolve_types(self):
            if self.manual_chk.isChecked():
                return ["(수동)"]
            t = self.type_cb.currentText()
            if t == "최적화":
                return TYPES
            if t == "진각 제외 최적화":
                return BASE_TYPES
            return [t]

    def _resolve_acc_rows(self):
            if self.acc_list.count() > 0:
                want = set(self.acc_list.item(i).text() for i in range(self.acc_list.count()))
                rows = []
                for _, r in accmod.df_acc.iterrows():
                    key = f'{int(r["lv"])} {r["이름"]}'
                    if key in want:
                        rows.append(r.to_dict())
                return rows
            lvl = int(self.acc_level_cb.currentText())
            return accmod.df_acc[accmod.df_acc['lv']==lvl].to_dict("records")

    def _resolve_ench_list(self):
            if self.ench_cb.currentText() == "최적화":
                return ENCH_LIST
            name = None
            for k,v in ENCH_KR.items():
                if v == self.ench_cb.currentText():
                    name = k; break
            if name is None: name = "HP"
            for nm, pct in ENCH_LIST:
                if nm == name: return [(nm, pct)]
            return [ENCH_LIST[0]]

    def _resolve_buffs(self):
            txt = self.buff_cb.currentText()
            if txt == "최적화":
                return self._normalize_buff_container(ALL_BUFFS)
            if txt == "1벞 최적화":
                return self._normalize_buff_container(ONE_BUFFS)
            return [(txt, BUFFS_ALL.get(txt, BUFFS_ALL.get("0벞", {"hp":0,"atk":0,"def":0})))]

    def _count_spirit_cases(self):
        if self.sp_skip.isChecked():
            return 1
        # 정령 최적화 옵션 제거됨: 항상 1가지 케이스
        return 1


    
    
    def _apply_dup_filters(self, results):
            """
            중복 필터(체크 방식):
            - 체크박스에 체크된 차원들로 그룹핑 키를 만든다.
              예) [타입, 버프] 체크 -> (타입, 버프)별로 묶음
            - 선택한 우선순위(이벨/비벨/TAR%) 기준으로 그룹마다 **최대 1개(최댓값)**만 남긴다.
            - 아무것도 체크 안 하면 필터링 안 함(모두 표시).
            """
            if not results:
                return results
    
            metric = self.dup_metric_cb.currentText() if hasattr(self, "dup_metric_cb") else "비벨"
            metric_key = metric  # "이벨","비벨","TAR%"와 동일
    
            # 어떤 차원이 체크되었는지 수집
            dims = []
            if hasattr(self, "dup_type_chk") and self.dup_type_chk.isChecked():
                dims.append(("타입", lambda r: r.get("타입")))
            if hasattr(self, "dup_buff_chk") and self.dup_buff_chk.isChecked():
                dims.append(("버프", lambda r: (r.get("버프"), r.get("최대 버프"))))
            if hasattr(self, "dup_acc_chk") and self.dup_acc_chk.isChecked():
                dims.append(("장신구", lambda r: r.get("장신구")))
            if hasattr(self, "dup_spirit_chk") and self.dup_spirit_chk.isChecked():
                dims.append(("정령", lambda r: (r.get("1옵"), r.get("2옵"), r.get("3옵"), r.get("4옵"), r.get("부가옵"))))
            if hasattr(self, "dup_ench_chk") and self.dup_ench_chk.isChecked():
                dims.append(("인첸트", lambda r: r.get("인첸트")))
    
            # 체크가 하나도 없으면 그대로 반환
            if not dims:
                return results
    
            def to_float(x):
                try:
                    return float(x)
                except Exception:
                    return 0.0
    
            # 그룹키 생성 함수
            def make_key(r):
                keys = []
                for _name, fn in dims:
                    try:
                        keys.append(fn(r))
                    except Exception:
                        keys.append(None)
                return tuple(keys)
    
            best = {}
            for r in results:
                k = make_key(r)
                v = to_float(r.get(metric_key, 0))
                prev = best.get(k)
                if (prev is None) or (v > to_float(prev.get(metric_key, 0))):
                    best[k] = r
    
            return list(best.values())
    
    

    def update_combo_count(self):
            if not getattr(self, '_ui_ready', False):
                return
            # 타입
            t_txt = self.type_cb.currentText()
            if self.manual_chk.isChecked():
                t_cnt = 1
            elif t_txt == "최적화":
                t_cnt = len(TYPES)
            elif t_txt == "진각 제외 최적화":
                t_cnt = len(BASE_TYPES)
            else:
                t_cnt = 1
            # 장신구
            if self.acc_list.count() > 0:
                acc_cnt = self.acc_list.count()
            else:
                lvl = int(self.acc_level_cb.currentText())
                acc_cnt = int((accmod.df_acc['lv']==lvl).sum())
            # 인첸트
            e_cnt = 3 if self.ench_cb.currentText()=="최적화" else 1
            # 정령
            s_cnt = self._count_spirit_cases()
            # 젬분배 (고정된 GEM_DISTS 전 탐색)
            g_cnt = len(GEM_DISTS)
            # 버프
            txt = self.buff_cb.currentText()
            if txt == "최적화":
                b_cnt = len(self._normalize_buff_container(ALL_BUFFS))
            elif txt == "1벞 최적화":
                b_cnt = len(self._normalize_buff_container(ONE_BUFFS))
            else:
                b_cnt = 1
            total = t_cnt * acc_cnt * e_cnt * s_cnt * g_cnt * b_cnt
            self.combo_count_label.setText(f"조합 수: {total:,}")

    def handle_calculate(self):
            try:
                types = self._resolve_types()
                acc_rows = self._resolve_acc_rows()
                ench_list = self._resolve_ench_list()
                spirit_slots = self._build_spirit_slots_from_ui()
                buff_list = self._resolve_buffs()
                gemv = int(self.gem_cb.currentText())
                base_stat_manual = None
                if self.manual_chk.isChecked():
                    def to_int(s):
                        try:
                            return int(str(s).strip())
                        except Exception:
                            return 0
                    base_stat_manual = {"hp": to_int(self.hp_in.text()), "atk": to_int(self.atk_in.text()), "def": to_int(self.def_in.text())}
                # set global M-fixed mode
                global M_FIXED_BUFFS_MODE, M_FIXED_BUFF_KEYS, M_FIXED_TYPE_MODE
                M_FIXED_TYPE_MODE = self.mfix_type_cb.currentText()
                cb2 = getattr(self, 'buff_fix_2buff_both', None)
                if cb2 and cb2.isChecked():
                    M_FIXED_BUFFS_MODE = '2벞 모두'
                    M_FIXED_BUFF_KEYS = _two_buff_pairs()
                else:
                    M_FIXED_BUFFS_MODE = self.mfix_cb.currentText()
                    if M_FIXED_BUFFS_MODE in ('사용안함', None, ''):
                        M_FIXED_BUFF_KEYS = []
                    else:
                        v = self._resolve_mfix_label(M_FIXED_BUFFS_MODE)
                        M_FIXED_BUFF_KEYS = [(M_FIXED_BUFFS_MODE, v)]

            

                params = {
                    "types": types,
                    "acc_rows": acc_rows,
                    "ench_list": ench_list,
                    "spirit_slots": spirit_slots,
                    "buff_list": buff_list,
                    "gemv": gemv,
                    "base_stat_manual": base_stat_manual,
                }
# run
                self.all_model.removeRows(0, self.all_model.rowCount())
                # configure progress as percent (filled bar)
                # pre-count combinations to set maximum
                t_cnt = len(types)
                b_cnt = len(buff_list)
                s_cnt = 1
                for slot in (spirit_slots or []):
                    s_cnt *= (len(slot) if slot else 1)
                    for slot in spirit_slots:
                        s_cnt *= max(1, len(slot))
                g_cnt = len(GEM_DISTS)
                a_cnt = len(acc_rows)
                e_cnt = len(ench_list)
                total = t_cnt * b_cnt * s_cnt * g_cnt * a_cnt * e_cnt
                self.progress.setRange(0, max(1, total))
                self.progress.setFormat("진행: %p%")
                self.progress.setValue(0)
                self.progress.setVisible(True); self.calc_btn.setEnabled(False)
                self.worker = CalcWorker(params)
                self.worker.progress.connect(lambda n: self.progress.setValue(int(n)))
                self.worker.result.connect(self._show_results)
                self.worker.error.connect(self._show_error)
                self.worker.start()
            except Exception as e:
                self._show_error(str(e))


    
    def _populate_filter_boxes(self):
        """Fill each column filter combo with unique values from the model."""
        try:
            model = self.all_model
            rows = model.rowCount()
            cols = model.columnCount()
            uniques = [set() for _ in range(cols)]
            for r in range(rows):
                for c in range(cols):
                    idx = model.index(r, c)
                    val = "" if not idx.isValid() else str(idx.data())
                    if val:
                        uniques[c].add(val)
            for c, cb in self.filter_boxes:
                prev = cb.currentText() if cb.count() else "전체"
                cb.blockSignals(True)
                cb.clear()
                cb.addItem("전체")
                for v in sorted(uniques[c]):
                    cb.addItem(v)
                if prev and prev != "전체":
                    i = cb.findText(prev)
                    if i >= 0:
                        cb.setCurrentIndex(i)
                cb.blockSignals(False)
                try:
                    cb.currentIndexChanged.disconnect()
                except Exception:
                    pass
                cb.currentIndexChanged.connect(lambda _i, col=c, combo=cb: self._on_filter_changed(col, combo))
        except Exception as e:
            print("populate_filter_boxes error:", e)

    def _on_filter_changed(self, col, combo):
        text = combo.currentText()
        if text == "전체":
            text = ""
        self.all_proxy.set_str_filter(col, text)
    
    def _show_results(self, results):
            self.progress.setVisible(False); self.calc_btn.setEnabled(True)
            # 중복 필터 적용
            results = self._apply_dup_filters(results)
            headers = [self.all_model.headerData(i, Qt.Horizontal) for i in range(self.all_model.columnCount())]
            for rec in results:
                headers = [self.all_model.headerData(i, Qt.Horizontal) for i in range(self.all_model.columnCount())]
                items = []
                for h in headers:
                    if h == "TAR":
                        # 숫자 정렬은 EditRole을, 화면 표시는 DisplayRole을 사용
                        try:
                            tnum = float(rec.get("TAR%", rec.get("TAR", 0)) or 0.0)
                        except Exception:
                            try:
                                tnum = float(str(rec.get("TAR%", rec.get("TAR", 0))).replace('%',''))
                            except Exception:
                                tnum = 0.0
                        it = QStandardItem()
                        it.setData(f"{tnum:.4f}%", Qt.DisplayRole)
                        it.setData(tnum, Qt.EditRole)
                        it.setText(f"{tnum:.4f}%")
                    elif h in ("HP","ATK","DEF","이벨","비벨"):
                        v = rec.get(h, rec.get(0) if h == "비벨최대값" else 0)
                        try:
                            val = int(v)
                        except Exception:
                            try:
                                val = int(float(str(v).strip()))
                            except Exception:
                                val = 0
                        it = QStandardItem()
                        it.setData(val, Qt.DisplayRole)
                        it.setData(val, Qt.EditRole)
                        it.setText(str(val))
                    else:
                        it = QStandardItem(str(rec.get(h, "")))
                    items.append(it)
                self.all_model.appendRow(items)

                # 메타 정보는 첫 칸(UserRole+1)에 저장
                try:
                    if isinstance(rec.get('_M_META'), dict):
                        self.all_model.item(self.all_model.rowCount()-1, 0).setData(rec.get('_M_META'), Qt.UserRole+1)
                except Exception:
                    pass

            # 필터 갱신 및 탭 전환
            self._populate_filter_boxes()
            self.tabs.setCurrentIndex(self.tabs.indexOf(self.all_view.parentWidget()))


    def _show_error(self, msg):
        self.progress.setVisible(False); self.calc_btn.setEnabled(True)
        QMessageBox.critical(self, "에러", msg)

def main():
    app = QApplication(sys.argv)
    w = App()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()