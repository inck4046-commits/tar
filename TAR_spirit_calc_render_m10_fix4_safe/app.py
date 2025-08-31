
from __future__ import annotations
import os, itertools, time
from functools import lru_cache
from typing import List, Tuple, Dict, Any, Optional

import gradio as gr
import pandas as pd

def _empty_df():
    import pandas as pd
    return pd.DataFrame()

def _safe7(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        st = f"ERROR: {e}\n" + tb[-2000:]
        return st, _empty_df(), 0, 0, 0, 0, "0초"

def _safe8(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        st = f"ERROR: {e}\n" + tb[-2000:]
        return st, _empty_df(), b"", 0, 0, 0, 0, "0초"


from calc.type import get_base_stats
from calc.awakening import get_awakening_stat
from calc.buff import BUFFS_ALL, TAR_BUFFS
from calc.collection import apply_collection
from calc.potion import apply_potion
from calc.spirit import spirit_breakdown
from calc.gem import GEM_DISTS
import calc.accessory as accmod

TITLE = "TAR 정령 자동 계산기"

BASE_TYPES = ["체","공","방","체공","체방","공방"]
ALL_TYPES  = BASE_TYPES + ["(진각)체","(진각)공","(진각)방","(진각)체공","(진각)체방","(진각)공방"]
ENCHANTS = {"HP":{"hp":0.21,"atk":0.0,"def":0.0},
            "ATK":{"hp":0.0,"atk":0.21,"def":0.0},
            "DEF":{"hp":0.0,"atk":0.0,"def":0.21}}
SPIRIT_STATS = ["체력","공격력","방어력"]
SUB_OPTS = ["체력40","공격력10","방어력10"]
GEM_VALUES = [37,38,39,40]
TWO_BUFF6 = [lab for (lab,_) in TAR_BUFFS]

def _safe(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        st = "ERROR: " + str(e) + "\n" + tb[-2000:]
        empty = pd.DataFrame()
        return st, empty, 0, 0, 0, 0, "0초"

def _strip_jingak(t: str) -> str:
    return t.replace("(진각)","") if isinstance(t, str) else t

def _acc_rows(levels: List[int] | None) -> List[Dict[str, Any]]:
    df = accmod.df_acc
    if not levels:
        return df.to_dict("records")
    return df[df["lv"].isin(list(map(int, levels)))].to_dict("records")

def _prepare_spirit(s1_stat,s1_mode,s2_stat,s2_mode,s3_stat,s3_mode,s4_stat,s4_mode,sub_stat):
    return [
        {"stat": s1_stat, "type": s1_mode, "slot": 1},
        {"stat": s2_stat, "type": s2_mode, "slot": 2},
        {"stat": s3_stat, "type": s3_mode, "slot": 3},
        {"stat": s4_stat, "type": s4_mode, "slot": 4},
        {"stat": sub_stat, "type": "부가옵", "slot": 5},
    ]

def _calc_core(typ: str, ench_pct: Dict[str,float], acc_row: Dict[str, Any],
               gemv: int, dist: Tuple[int,int,int], buf_lbl: str | Dict[str,float],
               spirit_opts: List[Dict[str, Any]]):
    # Base + Awakening
    bs = get_base_stats(typ)
    aw = get_awakening_stat(typ)
    C1 = {k: bs[k] + aw.get(k,0) for k in ("hp","atk","def")}
    # Gem
    C2 = {"hp": gemv*4*dist[0], "atk": gemv*dist[1], "def": gemv*dist[2]}
    C3 = {k: C1[k] + C2[k] for k in C1}
    # Potion
    pot = apply_potion({})
    C4 = {k: C3[k] + pot.get(k,0) for k in C3}
    # Buff add (based on *base* only)
    buf_pct = buf_lbl if isinstance(buf_lbl, dict) else BUFFS_ALL.get(str(buf_lbl), {"hp":0,"atk":0,"def":0})
    buff_add = {k: int(bs[k] * buf_pct.get(k,0.0)) for k in bs}
    # Accessory + Enchant
    acc_pct = {"hp": float((acc_row or {}).get("hp%",0))/100.0,
               "atk": float((acc_row or {}).get("atk%",0))/100.0,
               "def": float((acc_row or {}).get("def%",0))/100.0}
    st1 = {k: int(C4[k] * (1 + acc_pct[k] + ench_pct.get(k,0.0))) for k in C4}
    # Spirit breakdown
    pct7, flat8, sub9 = spirit_breakdown(spirit_opts)
    st2 = {k: int(st1[k] * (1 + pct7[k])) + int(flat8[k] * (1 + pct7[k])) for k in st1}
    coll = apply_collection({"hp":0,"atk":0,"def":0})
    final = {"hp": st2["hp"] + coll["hp"] + sub9["hp"] + buff_add["hp"],
             "atk": st2["atk"] + coll["atk"] + sub9["atk"] + buff_add["atk"],
             "def": st2["def"] + coll["def"] + sub9["def"] + buff_add["def"]}
    bibel = int(final["hp"] * final["atk"] * final["def"])
    ivel  = int((final["hp"]//4 + final["atk"] + final["def"]) * 4)
    return bibel, ivel, final

# ---------- TAR 분모(최대 비벨) ----------
@lru_cache(maxsize=512)
def _spec_max(typ: str, buf_label: str) -> Tuple[int, Dict[str, Any]]:
    gemv = 40
    stats = ["체력","공격력","방어력"]
    base_sp = []
    for s1 in stats:
        for s2 in stats:
            for s3 in stats:
                for flat in stats:
                    base_sp.append([
                        {"stat":s1,"type":"%","slot":1},
                        {"stat":s2,"type":"%","slot":2},
                        {"stat":s3,"type":"%","slot":3},
                        {"stat":flat,"type":"+","slot":4},
                        {"stat":"체력40","type":"부가옵","slot":5},
                    ])
    # also distinct permutations
    from itertools import permutations
    for s1,s2,s3 in permutations(stats,3):
        for flat in stats:
            base_sp.append([
                {"stat":s1,"type":"%","slot":1},
                {"stat":s2,"type":"%","slot":2},
                {"stat":s3,"type":"%","slot":3},
                {"stat":flat,"type":"+","slot":4},
                {"stat":"체력40","type":"부가옵","slot":5},
            ])
    expanded = []
    for comb in base_sp:
        for sub in SUB_OPTS:
            cc = [dict(x) for x in comb]; cc[-1]["stat"] = sub; expanded.append(cc)

    acc_rows = accmod.df_acc.to_dict("records")
    best_M = 0; best_meta = {}
    for dist in GEM_DISTS:
        for acc in acc_rows:
            for ench_name, ench_pct in ENCHANTS.items():
                for comb in expanded:
                    bib, ivel, final = _calc_core(typ, ench_pct, acc, gemv, dist, buf_label, comb)
                    if bib > best_M:
                        best_M = int(bib)
                        best_meta = {"젬분배":f"{dist[0]}/{dist[1]}/{dist[2]}",
                                     "장신구":f"{int(acc.get('lv',0))} {acc.get('이름','')}",
                                     "인첸트":ench_name}
    return best_M, best_meta

@lru_cache(maxsize=1024)
def _spec_max_fixed(typ: str, row_buf: str, mfix_type: str, mfix_buff: str, two_buff_all: bool):
    # type candidates
    if mfix_type == "모든 타입":
        types = tuple(ALL_TYPES)
    elif mfix_type and mfix_type != "사용안함":
        types = (mfix_type,)
    else:
        types = (typ,)
    # buff candidates
    if two_buff_all:
        buffs = tuple(TWO_BUFF6)
    elif mfix_buff and mfix_buff != "사용안함":
        buffs = (mfix_buff,)
    else:
        buffs = (row_buf,)
    best_M = 0.0; best_lab = None
    for t in types:
        for b in buffs:
            M, _ = _spec_max(t, b)
            if float(M) > float(best_M):
                best_M = float(M); best_lab = b
    return float(best_M), (best_lab or (buffs[0] if buffs else row_buf))

def _apply_dup_filters_df(df: pd.DataFrame, by_type: bool, by_buff: bool, by_acc: bool, by_spirit: bool, by_ench: bool, metric_key: str) -> pd.DataFrame:
    if df.empty or not any([by_type, by_buff, by_acc, by_spirit, by_ench]):
        return df
    for col in ["타입","버프","최대 버프","장신구","인첸트","1옵","2옵","3옵","4옵","부가옵","비벨","이벨","TAR%"]:
        if col not in df.columns:
            if col == "TAR%": df[col] = 0.0
            else: df[col] = ""
    keys = []
    for _, r in df.iterrows():
        k = []
        if by_type:   k.append(r["타입"])
        if by_buff:   k.append( (r["버프"], r.get("최대 버프","")) )
        if by_acc:    k.append(r["장신구"])
        if by_ench:   k.append(r["인첸트"])
        if by_spirit: k.append( (r.get("1옵",""), r.get("2옵",""), r.get("3옵",""), r.get("4옵",""), r.get("부가옵","")) )
        keys.append(tuple(k))
    df = df.copy(); df["_key"] = keys
    if metric_key == "TAR%":
        metric = pd.to_numeric(df["TAR%"], errors="coerce").fillna(0.0)
    elif metric_key == "이벨":
        metric = pd.to_numeric(df["이벨"], errors="coerce").fillna(0.0)
    else:
        metric = pd.to_numeric(df["비벨"], errors="coerce").fillna(0.0)
    df["_metric"] = metric
    idx = df.sort_values("_metric", ascending=False).groupby("_key", as_index=False).head(1).index
    out = df.loc[idx].drop(columns=["_key","_metric"]).reset_index(drop=True)
    return out

def _fmt_seconds(sec: float) -> str:
    if sec < 1: return f"{sec*1000:.0f} ms"
    m, s = divmod(int(sec+0.5), 60)
    h, m = divmod(m, 60)
    if h: return f"{h}시간 {m}분 {s}초"
    if m: return f"{m}분 {s}초"
    return f"{s}초"

def _estimate_runtime(types, buffs, gem_values, parsed_dists, enchs, rows, spirit, need_tar, mfix_type, mfix_buff, mfix_two, max_cases:int):
    raw_total = len(types) * len(buffs) * len(gem_values) * len(parsed_dists) * len(enchs) * len(rows)
    gen_target = raw_total if (not max_cases or max_cases<=0) else min(raw_total, int(max_cases))
    if gen_target == 0:
        return raw_total, 0, "0초"
    # sample
    sample_n = min(200, gen_target)
    done = 0
    t0 = time.perf_counter()
    for typ in types:
        for buf in buffs:
            for gemv in gem_values:
                for dist in parsed_dists[:2]:
                    for ench in enchs:
                        for acc in rows[:1]:
                            _ = _calc_core(typ, ENCHANTS[ench], acc, int(gemv), dist, buf, spirit)
                            if need_tar:
                                _ = _spec_max_fixed(typ, buf, mfix_type, mfix_buff, bool(mfix_two))
                            done += 1
                            if done >= sample_n: break
                        if done >= sample_n: break
                    if done >= sample_n: break
                if done >= sample_n: break
            if done >= sample_n: break
        if done >= sample_n: break
    t1 = time.perf_counter()
    per_row = max(1e-6, (t1 - t0) / max(1, done))
    total_sec = per_row * gen_target * 1.15
    return raw_total, gen_target, _fmt_seconds(total_sec)

def _mobile_trim_df(df: pd.DataFrame, simple: bool) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame): return df
    if not simple: return df
    keep = [c for c in ["타입","버프","최대 버프","장신구","인첸트","젬수치","젬분배","비벨","이벨","TAR%"] if c in df.columns]
    if not keep: return df
    return df[keep]

def run_search(
    types: List[str], buffs: List[str], gem_values: List[int], dists: List[str],
    enchs: List[str], acc_levels: List[int], acc_names: List[str],
    s1_stat: str, s1_mode: str, s2_stat: str, s2_mode: str,
    s3_stat: str, s3_mode: str, s4_stat: str, s4_mode: str, sub_stat: str,
    do_tar: bool, topk: int, sort_key: str, all_cases: bool, max_cases: int,
    dup_type: bool, dup_buff: bool, dup_acc: bool, dup_spirit: bool, dup_ench: bool, dup_metric: str,
    mfix_type: str, mfix_buff: str, mfix_two_buff: bool, mobile_simple: bool=False
):
    if not types or not buffs or not gem_values or not dists or not enchs or not acc_levels:
        return "선택 항목 부족", pd.DataFrame(), 0, 0, 0, 0, "0초"

    # parse dists
    parsed_dists = []
    for t in dists:
        try:
            i,j,k = [int(x.strip()) for x in t.split("/")[:3]]
            parsed_dists.append((i,j,k))
        except: pass
    if not parsed_dists: return "젬 분배 형식 오류", pd.DataFrame(), 0, 0, 0, 0, "0초"

    rows = _acc_rows(acc_levels)
    if acc_names:
        filt = set([n for n in acc_names if n])
        if filt:
            rows = [r for r in rows if str(r.get("이름","")) in filt]
    if not rows:
        rows = _acc_rows(acc_levels)

    spirit = _prepare_spirit(s1_stat,s1_mode,s2_stat,s2_mode,s3_stat,s3_mode,s4_stat,s4_mode,sub_stat)

    need_tar = bool(do_tar or (sort_key == "TAR%") or (dup_metric == "TAR%"))

    raw_total, gen_target, eta_text = _estimate_runtime(types, buffs, gem_values, parsed_dists, enchs, rows, spirit, need_tar, mfix_type, mfix_buff, mfix_two_buff, 0 if (all_cases or (not max_cases or max_cases<=0)) else int(max_cases))

    spirit_strs = { f"{i}옵": f"{spirit[i-1]['stat']}{spirit[i-1]['type']}" for i in [1,2,3,4] }
    sub_str = sub_stat

    results = []
    count = 0
    for typ, buf, gemv, dist, ench in itertools.product(types, buffs, gem_values, parsed_dists, enchs):
        for acc in rows:
            bib, ivel, final = _calc_core(typ, ENCHANTS[ench], acc, int(gemv), dist, buf, spirit)
            row = {"타입": typ, "버프": buf, "젬수치": int(gemv), "젬분배": f"{dist[0]}/{dist[1]}/{dist[2]}",
                   "인첸트": ench, "장신구": f"{int(acc.get('lv',0))} {acc.get('이름','')}",
                   "HP": final["hp"], "ATK": final["atk"], "DEF": final["def"], "이벨": ivel, "비벨": int(bib),
                   "1옵": spirit_strs["1옵"], "2옵": spirit_strs["2옵"], "3옵": spirit_strs["3옵"], "4옵": spirit_strs["4옵"], "부가옵": sub_str}
            # 최대 버프 라벨(분모와는 독립)
            try:
                Mlab = _spec_max_fixed(typ, buf, mfix_type, mfix_buff, bool(mfix_two_buff))[1]
            except Exception:
                Mlab = buf
            row["최대 버프"] = Mlab
            results.append(row)
            count += 1
            if (not all_cases) and (max_cases and max_cases>0) and count >= int(max_cases): break
        if (not all_cases) and (max_cases and max_cases>0) and count >= int(max_cases): break

    if not results: return "결과 없음", pd.DataFrame(), raw_total, 0, 0, 0, eta_text
    df0 = pd.DataFrame(results)
    generated_n = len(df0)

    # TAR%
    if need_tar:
        Ms = []
        Mlab = []
        for _, r in df0.iterrows():
            M, used_lab = _spec_max_fixed(r["타입"], r["버프"], mfix_type, mfix_buff, bool(mfix_two_buff))
            Ms.append(M if M else 0.0); Mlab.append(used_lab)
        df0["TAR%"] = [
            round(max(0.0, min(100.0, 200.0*(float(b)/float(m))-100.0)), 4) if float(m) > 0 else 0.0
            for b,m in zip(df0["비벨"].tolist(), Ms)
        ]
        df0["(M기준 버프)"] = Mlab
        df0["(M기준 타입)"] = mfix_type if (mfix_type and mfix_type != "사용안함") else df0["타입"]
    else:
        df0["TAR%"] = "(생략)"

    # dedup
    df1 = _apply_dup_filters_df(df0, dup_type, dup_buff, dup_acc, dup_spirit, dup_ench, dup_metric)
    after_dedup_n = len(df1)

    # sort
    if sort_key == "TAR%":
        df1 = df1.sort_values(["TAR%","비벨","이벨"], ascending=[False,False,False])
    else:
        df1 = df1.sort_values(["비벨","이벨","HP","ATK","DEF"], ascending=[False,False,False,False,False])

    # topN
    if topk and topk > 0:
        df = df1.head(int(topk)).reset_index(drop=True)
    else:
        df = df1.reset_index(drop=True)

    df = _mobile_trim_df(df, mobile_simple)
    final_n = len(df)

    status = f"OK | 원시 조합 {raw_total:,} / 생성 {generated_n:,} / 중복 후 {after_dedup_n:,} / 표시 {final_n:,}"
    return status, df, int(raw_total), int(generated_n), int(after_dedup_n), int(final_n), eta_text

def build_ui():
    custom_css = """
    a[href^='https://www.gradio.app']{display:none !important;}
    button[aria-label='Open settings'], button[aria-label='Settings']{display:none !important;}
    a:has(svg[data-testid='rocket']){display:none !important;}
    footer, #footer {display:none !important;}
    @media (max-width: 768px){
      .gradio-container { --body-text-size: 14px; --button-large-padding: 12px 14px; }
      .gradio-container .wrap { overflow-x: auto !important; }
      .gradio-container label, .gradio-container button { font-size: 14px; }
    }
    """
    with gr.Blocks(title=TITLE, theme=gr.themes.Soft(), css=custom_css) as demo:
        with gr.Row():
            ms_types = gr.CheckboxGroup(choices=ALL_TYPES, value=BASE_TYPES, label="타입 (다중 선택) — 기본 '진각 제외' 전체")

        with gr.Row():
            ms_buffs = gr.CheckboxGroup(choices=list(BUFFS_ALL.keys()), value=list(BUFFS_ALL.keys()), label="버프 (다중)")
            ms_gemv  = gr.CheckboxGroup(choices=GEM_VALUES, value=[37], label="젬 수치 (다중)")

        dist_choices = [f"{i}/{j}/{k}" for (i,j,k) in GEM_DISTS]
        with gr.Row():
            ms_dists = gr.CheckboxGroup(choices=dist_choices, value=dist_choices, label="젬 분배 (다중, 합계=5)")
            ms_ench  = gr.CheckboxGroup(choices=list(ENCHANTS.keys()), value=list(ENCHANTS.keys()), label="인첸트 (다중)")

        # Accessory defaults
        df = accmod.df_acc
        levels_sorted = sorted(list(set(map(int, df["lv"]))), reverse=True)
        default_level = 19 if 19 in levels_sorted else (levels_sorted[0] if levels_sorted else 19)
        names19 = sorted(set(df[df["lv"]==default_level]["이름"])) if not df.empty else []
        with gr.Row():
            ms_acc_levels = gr.CheckboxGroup(choices=levels_sorted, value=[default_level], label="장신구 레벨 (다중) — 기본 19만")
            ms_acc_names  = gr.CheckboxGroup(choices=names19, value=[], label=f"장신구 이름 (레벨 {default_level}, 비우면 해당 레벨 전체)")
        def update_acc_names(levels: List[int]):
            d = accmod.df_acc
            if not levels:
                names = list(d["이름"])
            else:
                names = list(d[d["lv"].isin(list(map(int, levels)))]["이름"])
            names = sorted(set(names))
            return gr.CheckboxGroup(choices=names, value=[])
        ms_acc_levels.change(fn=update_acc_names, inputs=ms_acc_levels, outputs=ms_acc_names)

        # Spirits
        with gr.Row():
            s1_stat = gr.Dropdown(SPIRIT_STATS, value="체력", label="1옵 스탯")
            s1_mode = gr.Dropdown(["%","+"], value="%", label="1옵 타입")
            s2_stat = gr.Dropdown(SPIRIT_STATS, value="공격력", label="2옵 스탯")
            s2_mode = gr.Dropdown(["%","+"], value="%", label="2옵 타입")
        with gr.Row():
            s3_stat = gr.Dropdown(SPIRIT_STATS, value="방어력", label="3옵 스탯")
            s3_mode = gr.Dropdown(["%","+"], value="%", label="3옵 타입")
            s4_stat = gr.Dropdown(SPIRIT_STATS, value="체력", label="4옵 스탯")
            s4_mode = gr.Dropdown(["%","+"], value="+", label="4옵 타입")
        sub_stat = gr.Dropdown(SUB_OPTS, value="체력40", label="부가옵")

        # TAR denominator fix
        with gr.Accordion("TAR 분모 고정 (최대 비벨 타입/버프 고정)", open=False):
            with gr.Row():
                mfix_type = gr.Dropdown(choices=["사용안함","모든 타입"]+ALL_TYPES, value="사용안함", label="타입 고정")
                mfix_buff = gr.Dropdown(choices=["사용안함","HP20%","ATK20%","DEF20%","HP40%","ATK40%","DEF40%","HP+ATK20%","HP+DEF20%","ATK+DEF20%"], value="사용안함", label="버프 고정")
                mfix_two  = gr.Checkbox(value=False, label="2벞 모두(6종)")

        with gr.Tabs():
            with gr.Tab("최적화"):
                with gr.Row():
                    do_tar = gr.Checkbox(value=False, label="TAR% 계산 (느림)")
                    sort_key = gr.Radio(choices=["비벨", "TAR%"], value="비벨", label="정렬 기준")
                    topk = gr.Slider(1, 10000, value=200, step=1, label="상위 N개만 표시 (최대 10000)")
                    max_cases = gr.Number(value=0, precision=0, label="최대 조합 수 제한 (0=무제한)")
                with gr.Row():
                    mobile_simple = gr.Checkbox(value=False, label="모바일 간단 보기(핵심 컬럼만)")

                with gr.Accordion("중복 필터 (ui13 규칙)", open=False):
                    with gr.Row():
                        dup_type   = gr.Checkbox(value=False, label="타입")
                        dup_buff   = gr.Checkbox(value=False, label="버프(최대 버프 포함)")
                        dup_acc    = gr.Checkbox(value=False, label="장신구")
                        dup_spirit = gr.Checkbox(value=False, label="정령(1~4옵+부가옵)")
                        dup_ench   = gr.Checkbox(value=False, label="인첸트")
                        dup_metric = gr.Radio(choices=["비벨","이벨","TAR%"], value="비벨", label="우선순위")

                run_btn = gr.Button("최적화 실행")
                status = gr.Textbox(label="상태")

                with gr.Row():
                    cnt_raw = gr.Number(label="원시 조합 수", interactive=False)
                    cnt_gen = gr.Number(label="생성 행 수(제한 반영)", interactive=False)
                    cnt_ded = gr.Number(label="중복 후 행 수", interactive=False)
                    cnt_fin = gr.Number(label="표시 행 수", interactive=False)
                eta_box = gr.Textbox(label="예상 시간(대략)", interactive=False)

                table  = gr.Dataframe(label="결과", wrap=True, interactive=False, row_count=(1,"dynamic"))

                def run_opt_wrapper(*args):
                    return run_search(*args)

                run_btn.click(
                    lambda *a: _safe7(run_opt_wrapper, *a),
                    inputs=[ms_types, ms_buffs, ms_gemv, ms_dists, ms_ench, ms_acc_levels, ms_acc_names,
                            s1_stat,s1_mode,s2_stat,s2_mode,s3_stat,s3_mode,s4_stat,s4_mode,sub_stat,
                            do_tar, topk, sort_key, gr.State(False), max_cases,
                            dup_type, dup_buff, dup_acc, dup_spirit, dup_ench, dup_metric,
                            mfix_type, mfix_buff, mfix_two, mobile_simple],
                    outputs=[status, table, cnt_raw, cnt_gen, cnt_ded, cnt_fin, eta_box],
                )

            with gr.Tab("모든 경우"):
                all_cases_state = gr.State(True)
                max_cases_state = gr.State(0)
                with gr.Row():
                    do_tar2 = gr.Checkbox(value=False, label="TAR% 계산 (느림)")
                    sort_key2 = gr.Radio(choices=["비벨","TAR%"], value="비벨", label="정렬 기준")
                    topk2 = gr.Slider(1, 10000, value=300, step=1, label="표시 상한 N")
                    mobile_simple2 = gr.Checkbox(value=False, label="모바일 간단 보기(핵심 컬럼만)")
                with gr.Accordion("중복 필터 (ui13 규칙)", open=False):
                    with gr.Row():
                        dup_type2   = gr.Checkbox(value=False, label="타입")
                        dup_buff2   = gr.Checkbox(value=False, label="버프(최대 버프 포함)")
                        dup_acc2    = gr.Checkbox(value=False, label="장신구")
                        dup_spirit2 = gr.Checkbox(value=False, label="정령(1~4옵+부가옵)")
                        dup_ench2   = gr.Checkbox(value=False, label="인첸트")
                        dup_metric2 = gr.Radio(choices=["비벨","이벨","TAR%"], value="비벨", label="우선순위")

                run_btn2 = gr.Button("모든 경우 실행")
                status2 = gr.Textbox(label="상태(표시분)")

                with gr.Row():
                    cnt_raw2 = gr.Number(label="원시 조합 수", interactive=False)
                    cnt_gen2 = gr.Number(label="생성 행 수(제한 반영)", interactive=False)
                    cnt_ded2 = gr.Number(label="중복 후 행 수", interactive=False)
                    cnt_fin2 = gr.Number(label="표시 행 수", interactive=False)
                eta_box2 = gr.Textbox(label="예상 시간(대략)", interactive=False)

                table2  = gr.Dataframe(label="표시 결과", wrap=True, interactive=False, row_count=(1,"dynamic"))
                csv_btn = gr.DownloadButton(label="CSV 다운로드(전체)", file_name="all_cases.csv")

                def run_all_and_csv(types, buffs, gem_values, dists, enchs, acc_levels, acc_names,
                                    s1_stat,s1_mode,s2_stat,s2_mode,s3_stat,s3_mode,s4_stat,s4_mode,sub_stat,
                                    do_tar2, topk2, sort_key2, _all_cases_state, _max_cases_state,
                                    dup_type2, dup_buff2, dup_acc2, dup_spirit2, dup_ench2, dup_metric2,
                                    mfix_type, mfix_buff, mfix_two, mobile_simple2):
                    st, df, rawC, genC, dedC, finC, eta = run_search(
                        types, buffs, gem_values, dists, enchs, acc_levels, acc_names,
                        s1_stat,s1_mode,s2_stat,s2_mode,s3_stat,s3_mode,s4_stat,s4_mode,sub_stat,
                        do_tar2, 100000000, sort_key2, True, 0,
                        dup_type2, dup_buff2, dup_acc2, dup_spirit2, dup_ench2, dup_metric2,
                        mfix_type, mfix_buff, mfix_two, mobile_simple2
                    )
                    showN = int(topk2)
                    df_show = df.head(showN).reset_index(drop=True) if isinstance(df, pd.DataFrame) else df
                    # CSV bytes (no temp files)
                    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
                    return st, df_show, csv_bytes, int(rawC), int(genC), int(dedC), int(len(df_show)), eta

                run_btn2.click(
                    lambda *a: _safe8(run_all_and_csv, *a),
                    inputs=[ms_types, ms_buffs, ms_gemv, ms_dists, ms_ench, ms_acc_levels, ms_acc_names,
                            s1_stat,s1_mode,s2_stat,s2_mode,s3_stat,s3_mode,s4_stat,s4_mode,sub_stat,
                            do_tar2, topk2, sort_key2, all_cases_state, max_cases_state,
                            dup_type2, dup_buff2, dup_acc2, dup_spirit2, dup_ench2, dup_metric2,
                            mfix_type, mfix_buff, mfix_two, mobile_simple2],
                    outputs=[status2, table2, csv_btn, cnt_raw2, cnt_gen2, cnt_ded2, cnt_fin2, eta_box2],
                )

    return demo

if __name__ == "__main__":
    demo = build_ui()
    port = int(os.getenv("PORT", "7860"))
    demo.queue(concurrency_count=1).launch(server_name="0.0.0.0", server_port=port, show_api=False)
