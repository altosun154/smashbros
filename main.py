# app.py — Smash Bracket (Regular = balanced; Teams forbids same-team) + full bracket view

import streamlit as st

import random

from dataclasses import dataclass

from typing import List, Optional, Tuple, Dict

import pandas as pd

import math

import os



st.set_page_config(page_title="Smash Bracket", page_icon="🎮", layout="wide")

st.markdown("""

<style>

.match-box { border:1px solid #ddd; border-radius:10px; padding:6px 8px; margin:6px 0;

  font-size:14px; line-height:1.25; background:#fff; }

.round-title { font-weight:700; margin-bottom:8px; }

.name-line { display:flex; align-items:center; gap:6px; }

.name-line img { vertical-align:middle; }

.tbd { opacity:0.6; font-style:italic; }

.legend-badge { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:6px; vertical-align:middle; }

.small { font-size:13px; }

</style>

""", unsafe_allow_html=True)



st.title("🎮 Smash Bracket — Regular & Teams")



# ---------------------------- Data types ----------------------------

@dataclass(frozen=True)

class Entry:

    player: str

    character: str



# ---------------------------- Power-of-two helpers ----------------------------

def next_power_of_two(n: int) -> int:

    if n <= 1:

        return 1

    return 1 << (n - 1).bit_length()



def byes_needed(n: int) -> int:

    return max(0, next_power_of_two(n) - n)



# ---------------------------- Icons & colors ----------------------------

ICON_DIR = os.path.join(os.path.dirname(__file__), "images")



def get_character_icon_path(char_name: str) -> Optional[str]:

    if not char_name:

        return None

    fname = f"{char_name.title().replace(' ', '_')}.png"

    path = os.path.join(ICON_DIR, fname)

    return path if os.path.exists(path) else None



TEAM_COLOR_FALLBACKS = [

    "#E91E63", "#3F51B5", "#009688", "#FF9800", "#9C27B0",

    "#4CAF50", "#2196F3", "#FF5722", "#795548", "#607D8B"

]

PLAYER_FALLBACKS = [

    "#FF6F61", "#6B5B95", "#88B04B", "#F7CAC9", "#92A8D1",

    "#955251", "#B565A7", "#009B77", "#DD4124", "#45B8AC"

]



def render_name_html(player: str, team_of: Dict[str, str], team_colors: Dict[str, str], player_colors: Dict[str, str]) -> str:

    t = team_of.get(player, "")

    if t and team_colors.get(t):

        color = team_colors[t]

    else:

        color = player_colors.setdefault(player, PLAYER_FALLBACKS[len(player_colors) % len(PLAYER_FALLBACKS)])

    safe_player = player.replace("<", "&lt;").replace(">", "&gt;")

    return f"<span style='color:{color};font-weight:600'>{safe_player}</span>"



def render_entry_line(e: Optional[Entry], team_of: Dict[str, str], team_colors: Dict[str, str], player_colors: Dict[str, str]) -> str:

    if e is None:

        return "<div class='name-line tbd'>TBD</div>"

    if e.character.upper() == "BYE":

        return "<div class='name-line tbd'>BYE</div>"

    icon = get_character_icon_path(e.character)

    name_html = render_name_html(e.player, team_of, team_colors, player_colors)

    char_safe = e.character.replace("<", "&lt;").replace(">", "&gt;")

    if icon:

        return f"<div class='name-line'><img src='file://{icon}' width='24'/> <b>{char_safe}</b> ({name_html})</div>"

    else:

        return f"<div class='name-line'><b>{char_safe}</b> ({name_html})</div>"



def entry_to_label(e: Optional[Entry]) -> str:

    if e is None: return ""

    return f"{e.player} — {e.character}"



# ---------------------------- Balanced generator (Regular core) ----------------------------

def pick_from_lowest_tally(cands: List[Entry], tally: Dict[str, int], exclude_player: Optional[str] = None) -> Optional[Entry]:

    pool = [e for e in cands if e.player != exclude_player]

    if not pool:

        return None

    m = min(tally.get(e.player, 0) for e in pool)

    lowest = [e for e in pool if tally.get(e.player, 0) == m]

    return random.choice(lowest)



def generate_bracket_balanced(

    entries: List[Entry],

    *,

    forbid_same_team: bool = False,

    team_of: Optional[Dict[str, str]] = None

) -> List[Tuple[Entry, Entry]]:

    """

    Balanced-random pairing:

      - no self-match,

      - optional: forbid same-team,

      - fills BYEs to next power of two,

      - uses per-player tallies for fairness.

    """

    team_of = team_of or {}

    base = [e for e in entries if e.player != "SYSTEM"]

    need = byes_needed(len(base))



    bag = base.copy()

    random.shuffle(bag)

    tally: Dict[str, int] = {}

    pairs: List[Tuple[Entry, Entry]] = []



    # Use some BYEs first if needed

    while need > 0 and bag:

        a = pick_from_lowest_tally(bag, tally)

        bag.remove(a)

        pairs.append((a, Entry("SYSTEM", "BYE")))

        tally[a.player] = tally.get(a.player, 0) + 1

        need -= 1



    def pick_opponent(a: Entry, pool: List[Entry]) -> Optional[Entry]:

        pool2 = [x for x in pool if x.player != a.player]

        if forbid_same_team:

            ta = team_of.get(a.player, "")

            if ta:

                pool2 = [x for x in pool2 if team_of.get(x.player, "") != ta]

        if not pool2:

            return None

        m = min(tally.get(x.player, 0) for x in pool2)

        lowest = [x for x in pool2 if tally.get(x.player, 0) == m]

        return random.choice(lowest)



    while len(bag) >= 2:

        a = pick_from_lowest_tally(bag, tally)

        bag.remove(a)

        b = pick_opponent(a, bag)

        if b is None:

            # try turn this into a BYE if adding one still gets us to power-of-two

            if byes_needed(len(bag)+1) > 0:

                pairs.append((a, Entry("SYSTEM", "BYE")))

                tally[a.player] += 1 if a.player in tally else 1

            else:

                bag.append(a)

                random.shuffle(bag)

                if len(bag) == 1:

                    break

            continue

        bag.remove(b)

        pairs.append((a, b))

        tally[a.player] = tally.get(a.player, 0) + 1

        tally[b.player] = tally.get(b.player, 0) + 1



    if bag:  # odd leftover

        pairs.append((bag[0], Entry("SYSTEM", "BYE")))

    return pairs



def generate_bracket_regular(entries: List[Entry]) -> List[Tuple[Entry, Entry]]:

    # Regular is the balanced generator (what "everything/groups" did)

    return generate_bracket_balanced(entries)



def generate_bracket_teams(entries: List[Entry], team_of: Dict[str, str]) -> List[Tuple[Entry, Entry]]:

    # Same as regular but forbids same-team R1

    return generate_bracket_balanced(entries, forbid_same_team=True, team_of=team_of)



# ---------------------------- Sidebar ----------------------------

with st.sidebar:

    st.header("Rule Set")

    rule = st.selectbox(

        "Choose mode",

        options=["regular", "teams"],

        index=0,

        help=(

            "regular: balanced random (no self-matches), fills BYEs to next power of 2.\n"

            "teams: regular + forbids same-team matches in round 1 (names colored by team)."

        )

    )



    st.divider()

    st.header("Players")

    default_players = "You\nFriend1\nFriend2"

    players_multiline = st.text_area(

        "Enter player names (one per line)",

        value=st.session_state.get("players_multiline", default_players),

        height=140,

        help="These names populate the Player dropdown."

    )

    players = [p.strip() for p in players_multiline.splitlines() if p.strip()]

    st.session_state["players_multiline"] = players_multiline



    # Teams UI only in Teams mode

    team_of: Dict[str, str] = {}

    team_colors: Dict[str, str] = {}

    if rule == "teams":

        st.divider()

        st.header("Teams & Colors")

        team_names_input = st.text_input(

            "Team labels (comma separated)",

            value="Red, Blue",

            help="Example: Red, Blue, Green"

        )

        team_labels = [t.strip() for t in team_names_input.split(",") if t.strip()]

        if not team_labels:

            team_labels = ["Team A", "Team B"]



        st.caption("Pick a color for each team:")

        for i, t in enumerate(team_labels):

            default = TEAM_COLOR_FALLBACKS[i % len(TEAM_COLOR_FALLBACKS)]

            team_colors[t] = st.color_picker(f"{t} color", value=default, key=f"team_color_{t}")



        st.caption("Assign each player to a team:")

        for p in players:

            team_of[p] = st.selectbox(f"{p}", options=["(none)"] + team_labels, key=f"team_{p}")

        team_of = {p: (t if t != "(none)" else "") for p, t in team_of.items()}



    st.divider()

    st.header("Characters per player")

    chars_per_person = st.number_input("How many per player?", min_value=1, max_value=50, value=2, step=1)



    st.divider()

    st.subheader("Build / Fill")

    build_clicked = st.button("⚙️ Auto-Create/Reset Entries", use_container_width=True)

    shuffle_within_player = st.checkbox("Shuffle names when auto-filling", value=True)

    auto_fill_clicked = st.button("🎲 Auto-fill Characters (Character 1..k)", use_container_width=True)



    st.divider()

    st.header("General")

    clean_rows = st.checkbox("Remove empty rows", value=True)



# ---------------------------- Table helpers ----------------------------

def build_entries_df(players: List[str], k: int) -> pd.DataFrame:

    rows = []

    for _ in range(k):

        for p in players:

            rows.append({"Player": p, "Character": ""})

    return pd.DataFrame(rows)



def auto_fill_characters(df: pd.DataFrame, players: List[str], k: int, shuffle_each: bool) -> pd.DataFrame:

    out = df.copy()

    for p in players:

        idxs = list(out.index[out["Player"] == p])

        labels = [f"Character {i+1}" for i in range(len(idxs))]

        if shuffle_each:

            random.shuffle(labels)

        for row_i, label in zip(idxs, labels):

            out.at[row_i, "Character"] = label

    return out



def df_to_entries(df: pd.DataFrame, clean_rows_flag: bool) -> List[Entry]:

    entries: List[Entry] = []

    for _, row in df.iterrows():

        pl = str(row.get("Player", "")).strip()

        ch = str(row.get("Character", "")).strip()

        if clean_rows_flag and (not pl or not ch):

            continue

        if pl and ch:

            entries.append(Entry(player=pl, character=ch))

    return entries



# ---------------------------- State & editor ----------------------------

if "table_df" not in st.session_state:

    st.session_state.table_df = pd.DataFrame([

        {"Player": "You", "Character": "Mario"},

        {"Player": "You", "Character": "Link"},

        {"Player": "Friend1", "Character": "Kirby"},

        {"Player": "Friend1", "Character": "Fox"},

        {"Player": "Friend2", "Character": "Samus"},

    ])



if build_clicked:

    if not players:

        st.warning("Add at least one player in the sidebar before building entries.")

    else:

        st.session_state.table_df = build_entries_df(players, int(chars_per_person))



if auto_fill_clicked:

    if not players:

        st.warning("Add players first.")

    else:

        st.session_state.table_df = auto_fill_characters(

            st.session_state.table_df, players, int(chars_per_person), shuffle_within_player

        )



if players:

    st.session_state.table_df["Player"] = st.session_state.table_df["Player"].apply(

        lambda p: p if p in players else (players[0] if p == "" else p)

    )



st.subheader("Entries")

table_df = st.data_editor(

    st.session_state.table_df,

    num_rows="dynamic",

    use_container_width=True,

    column_config={

        "Player": st.column_config.SelectboxColumn("Player", options=players if players else [], required=True),

        "Character": st.column_config.TextColumn(required=True),

    },

    key="table_editor",

)

entries = df_to_entries(table_df, clean_rows_flag=clean_rows)



# ---------------------------- Rounds building & rendering ----------------------------

def compute_rounds_pairs(r1_pairs: List[Tuple[Entry, Entry]], winners_map: Dict[int, str]) -> List[List[Tuple[Optional[Entry], Optional[Entry]]]]:

    rounds: List[List[Tuple[Optional[Entry], Optional[Entry]]]] = []

    rounds.append([(a, b) for (a, b) in r1_pairs])



    total_real = sum(1 for (a, b) in r1_pairs for e in (a, b) if e and e.player != "SYSTEM")

    target = next_power_of_two(total_real)

    num_rounds = int(math.log2(target)) if target >= 2 else 1



    prev = rounds[0]



    def winner_of_pair(pair_index: int, pairs_list: List[Tuple[Optional[Entry], Optional[Entry]]]) -> Optional[Entry]:

        if pair_index >= len(pairs_list): return None

        a, b = pairs_list[pair_index]

        if a is None and b is None: return None

        if a is None: return b if (b and b.character.upper() != "BYE") else None

        if b is None: return a if (a and a.character.upper() != "BYE") else None

        if a.character.upper() == "BYE" and b.character.upper() != "BYE": return b

        if b.character.upper() == "BYE" and a.character.upper() != "BYE": return a



        # Only R1 has explicit selections

        label_a, label_b = entry_to_label(a), entry_to_label(b)

        sel = winners_map.get(pair_index + 1, "")

        if sel == label_a: return a

        if sel == label_b: return b

        return None



    for _ in range(1, num_rounds):

        nxt: List[Tuple[Optional[Entry], Optional[Entry]]] = []

        for i in range(0, len(prev), 2):

            w1 = winner_of_pair(i, prev)

            w2 = winner_of_pair(i + 1, prev)

            nxt.append((w1, w2))

        rounds.append(nxt)

        prev = nxt

    return rounds



def render_bracket_grid(all_rounds: List[List[Tuple[Optional[Entry], Optional[Entry]]]], team_of: Dict[str, str], team_colors: Dict[str, str]):

    cols = st.columns(len(all_rounds))

    if team_colors:

        legend = "  ".join([f"<span class='legend-badge' style='background:{c}'></span>{t}" for t, c in team_colors.items()])

        st.markdown(f"<div class='small'><b>Legend:</b> {legend}</div>", unsafe_allow_html=True)



    for round_idx, round_pairs in enumerate(all_rounds):

        with cols[round_idx]:

            st.markdown(f"<div class='round-title'>Round {round_idx+1}</div>", unsafe_allow_html=True)

            player_colors: Dict[str, str] = {}

            for pair in round_pairs:

                a, b = pair

                st.markdown("<div class='match-box'>", unsafe_allow_html=True)

                st.markdown(render_entry_line(a, team_of, team_colors, player_colors), unsafe_allow_html=True)

                st.markdown(render_entry_line(b, team_of, team_colors, player_colors), unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)



def r1_winner_controls(r1_pairs: List[Tuple[Entry, Entry]]):

    if "r1_winners" not in st.session_state:

        st.session_state.r1_winners = {}

    st.write("### Pick Round 1 Winners")

    for i, (a, b) in enumerate(r1_pairs, start=1):

        label_a = entry_to_label(a)

        label_b = entry_to_label(b)

        prev = st.session_state.r1_winners.get(i, "")

        if prev == label_a: idx = 0

        elif prev == label_b: idx = 1

        else: idx = 2

        choice = st.radio(

            f"Match {i}",

            options=[label_a, label_b, "(undecided)"],

            index=idx,

            key=f"winner_{i}",

            horizontal=True,

        )

        st.session_state.r1_winners[i] = choice if choice != "(undecided)" else ""



# ---------------------------- Generate & show ----------------------------

st.divider()

col_gen, col_clear = st.columns([2, 1])



with col_gen:

    if st.button("🎲 Generate Bracket", type="primary"):

        if len(entries) < 2:

            st.error("Add at least 2 entries (characters).")

        else:

            if rule == "regular":

                bracket = generate_bracket_regular(entries)

            else:  # teams

                bracket = generate_bracket_teams(entries, team_of)



            if not bracket:

                st.error("Couldn't build a valid round-1 bracket with those constraints.")

            else:

                total_real = len([e for e in entries if e.player != "SYSTEM"])

                target = next_power_of_two(total_real)

                need = target - total_real

                st.success(f"Entries: {total_real} → Target: {target} (BYEs: {need}) — Mode: {rule}")



                st.session_state["last_bracket"] = [(a, b) for (a, b) in bracket]

                st.session_state["last_rule"] = rule

                st.session_state["last_team_of"] = team_of if rule == "teams" else {}

                st.session_state["last_team_colors"] = team_colors if rule == "teams" else {}



# Persist & render compact full bracket

if "last_bracket" in st.session_state and st.session_state["last_bracket"]:

    r1_pairs = st.session_state["last_bracket"]

    if st.session_state.get("last_rule") == "teams":

        st.info("Bracket view (all rounds) — Teams mode")

    else:

        st.info("Bracket view (all rounds) — Regular mode")



    r1_winner_controls(r1_pairs)

    rounds = compute_rounds_pairs(r1_pairs, st.session_state.get("r1_winners", {}))

    render_bracket_grid(rounds, st.session_state.get("last_team_of", {}), st.session_state.get("last_team_colors", {}))



with col_clear:

    if st.button("🧹 Clear Table"):

        st.session_state.table_df = pd.DataFrame(columns=["Player", "Character"])

        st.session_state.pop("last_bracket", None)

        st.session_state.pop("r1_winners", None)

        st.rerun()



st.caption("Regular uses balanced randomization; Teams forbids same-team R1. Add an 'images/' folder with character PNGs to show icons.")
