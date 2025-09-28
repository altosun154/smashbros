# app.py — Smash Bracket (no slots) + Teams with colored names + R1 winners + power-of-two BYEs
import streamlit as st
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import pandas as pd
import math
import os

st.set_page_config(page_title="Smash Bracket (Teams & Colors)", page_icon="🎮", layout="wide")
st.title("🎮 Smash Bracket — Round 1 Generator (No Slots)")

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

# ---------------------------- Pairing (regular / teams) via backtracking ----------------------------
def recursive_pairing(entries: List[Entry], forbid_same_team: bool, team_of: Dict[str, str]) -> Optional[List[Tuple[Entry, Entry]]]:
    """
    Perfect matching with constraints:
      - Forbid same-player match (always).
      - If forbid_same_team: forbid same-team (when both players have a team label and it matches).
      - BYE can face anyone.
    """
    n = len(entries)
    if n == 0:
        return []
    a = entries[0]
    for i in range(1, n):
        b = entries[i]

        # BYE can face anyone
        if "BYE" in (a.character.upper(), b.character.upper()):
            remaining = entries[1:i] + entries[i+1:]
            rest = recursive_pairing(remaining, forbid_same_team, team_of)
            if rest is not None:
                return [(a, b)] + rest
            continue

        # Base rule: no self
        if a.player == b.player:
            continue

        # Teams rule
        if forbid_same_team:
            ta = team_of.get(a.player, "")
            tb = team_of.get(b.player, "")
            if ta and tb and ta == tb:
                continue

        remaining = entries[1:i] + entries[i+1:]
        rest = recursive_pairing(remaining, forbid_same_team, team_of)
        if rest is not None:
            return [(a, b)] + rest
    return None

def generate_bracket_regular_or_teams(raw_entries: List[Entry], forbid_same_team: bool, team_of: Dict[str, str]):
    entries = raw_entries.copy()
    # Fill to next power of 2 with BYEs
    need = byes_needed(len(entries))
    for _ in range(need):
        entries.append(Entry(player="SYSTEM", character="BYE"))

    # Try multiple shuffles to find a valid matching
    for _ in range(700):
        random.shuffle(entries)
        result = recursive_pairing(entries, forbid_same_team, team_of)
        if result is not None:
            # Randomize within-pair order for fairness
            out: List[Tuple[Entry, Entry]] = []
            for a, b in result:
                if random.random() < 0.5:
                    out.append((b, a))
                else:
                    out.append((a, b))
            return out
    return None

# ---------------------------- Balanced-random "Groups" (no slots) with exact BYEs ----------------------------
def players_from_entries(entries: List[Entry]) -> List[str]:
    seen = []
    for e in entries:
        if e.player != "SYSTEM" and e.player not in seen:
            seen.append(e.player)
    random.shuffle(seen)  # one-time mix
    return seen

def pick_from_lowest_tally(cands: List[Entry], tally: Dict[str, int], exclude_player: Optional[str] = None) -> Optional[Entry]:
    pool = [e for e in cands if e.player != exclude_player]
    if not pool:
        return None
    m = min(tally.get(e.player, 0) for e in pool)
    lowest = [e for e in pool if tally.get(e.player, 0) == m]
    return random.choice(lowest)

def generate_bracket_groups(entries: List[Entry]) -> List[Tuple[Entry, Entry]]:
    """
    Balanced-random pairing across the entire pool (no slots), with exact BYEs to reach next power of 2.
    - Keep a per-player tally of how many R1 pairings they've been assigned (just for balancing).
    - Greedily pick two from the *current* lowest-tally pool; if odd leftover, it gets a BYE.
    """
    base = [e for e in entries if e.player != "SYSTEM"]
    need = byes_needed(len(base))

    bag = base.copy()
    random.shuffle(bag)
    tally: Dict[str, int] = {}
    pairs: List[Tuple[Entry, Entry]] = []

    # First, use BYEs to consume some items if needed (spread roughly evenly)
    # Strategy: periodically drop a BYE on a lowest-tally pick until BYEs are used up.
    while need > 0 and bag:
        a = pick_from_lowest_tally(bag, tally)
        bag.remove(a)
        pairs.append((a, Entry("SYSTEM", "BYE")))
        tally[a.player] = tally.get(a.player, 0) + 1
        need -= 1

    # Pair the rest
    while len(bag) >= 2:
        a = pick_from_lowest_tally(bag, tally)
        bag.remove(a)
        b = pick_from_lowest_tally(bag, tally, exclude_player=a.player)
        if b is None:
            # no valid opponent now -> give BYE if any left, else push back and try later
            if need > 0:
                pairs.append((a, Entry("SYSTEM", "BYE")))
                tally[a.player] = tally.get(a.player, 0) + 1
                need -= 1
            else:
                # Put a back and reshuffle a bit to escape deadlock
                bag.append(a)
                random.shuffle(bag)
                # if still impossible (rare), break to avoid infinite loop
                if len(bag) == 1:
                    break
                continue
        else:
            bag.remove(b)
            pairs.append((a, b))
            tally[a.player] = tally.get(a.player, 0) + 1
            tally[b.player] = tally.get(b.player, 0) + 1

    # If an odd one remains, it must get a BYE (and we should still be at exact power-of-two)
    if bag:
        pairs.append((bag[0], Entry("SYSTEM", "BYE")))

    return pairs

# ---------------------------- Icon + color helpers ----------------------------
ICON_DIR = os.path.join(os.path.dirname(__file__), "images")

def get_character_icon_path(char_name: str) -> Optional[str]:
    if not char_name:
        return None
    fname = f"{char_name.title().replace(' ', '_')}.png"
    path = os.path.join(ICON_DIR, fname)
    return path if os.path.exists(path) else None

# Default palettes
TEAM_COLOR_FALLBACKS = [
    "#E91E63", "#3F51B5", "#009688", "#FF9800", "#9C27B0",
    "#4CAF50", "#2196F3", "#FF5722", "#795548", "#607D8B"
]
PLAYER_FALLBACKS = [
    "#FF6F61", "#6B5B95", "#88B04B", "#F7CAC9", "#92A8D1",
    "#955251", "#B565A7", "#009B77", "#DD4124", "#45B8AC"
]

def render_name(player: str, team_of: Dict[str, str], team_colors: Dict[str, str], player_colors: Dict[str, str]) -> str:
    t = team_of.get(player, "")
    color = ""
    if t and team_colors.get(t):
        color = team_colors[t]
    else:
        color = player_colors.setdefault(player, PLAYER_FALLBACKS[len(player_colors) % len(PLAYER_FALLBACKS)])
    safe_player = player.replace("<", "&lt;").replace(">", "&gt;")
    return f"<span style='color:{color};font-weight:600'>{safe_player}</span>"

def render_entry(e: Entry, team_of: Dict[str, str], team_colors: Dict[str, str], player_colors: Dict[str, str]) -> str:
    if e.character.upper() == "BYE":
        return "<span style='opacity:0.7;font-style:italic'>BYE</span>"
    icon = get_character_icon_path(e.character)
    name_html = render_name(e.player, team_of, team_colors, player_colors)
    char_safe = e.character.replace("<", "&lt;").replace(">", "&gt;")
    if icon:
        # Use HTML <img> tag to show local file
        return f"<img src='file://{icon}' width='36' style='vertical-align:middle;margin-right:8px'/> <b>{char_safe}</b> ({name_html})"
    else:
        return f"<b>{char_safe}</b> ({name_html})"

# ---------------------------- Sidebar (rule-first, adaptive) ----------------------------
with st.sidebar:
    st.header("Rule Set")
    rule = st.selectbox(
        "Choose mode first",
        options=["regular", "groups", "teams"],
        index=0,
        help=(
            "regular: random bracket (no self-match) + BYEs to power-of-2.\n"
            "groups: balanced-random using per-player tallies (no slots) + BYEs.\n"
            "teams: like regular but also forbids same-team in R1; choose team colors."
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

    # Teams-only UI
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

        # Color pickers per team
        st.caption("Pick a color for each team:")
        for i, t in enumerate(team_labels):
            default = TEAM_COLOR_FALLBACKS[i % len(TEAM_COLOR_FALLBACKS)]
            team_colors[t] = st.color_picker(f"{t} color", value=default, key=f"team_color_{t}")

        st.caption("Assign each player to a team:")
        for p in players:
            team_of[p] = st.selectbox(f"{p}", options=["(none)"] + team_labels, key=f"team_{p}")
        # Normalize "(none)"
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
    st.caption("Tip: If one player owns more than half of all entries, valid constraints may be impossible.")

# ---------------------------- Table helpers (no slots) ----------------------------
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

# ---------------------------- State: entries table ----------------------------
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

# Normalize Player to known list (optional; keep as-is if you want free text)
if players:
    st.session_state.table_df["Player"] = st.session_state.table_df["Player"].apply(
        lambda p: p if p in players else (players[0] if p == "" else p)
    )

# ---------------------------- Editor ----------------------------
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

# ---------------------------- Bracket + Winners UI ----------------------------
st.divider()
col_gen, col_clear = st.columns([2, 1])

def show_bracket_and_winner_ui(bracket: List[Tuple[Entry, Entry]], rule_label: str, team_of: Dict[str, str], team_colors: Dict[str, str]):
    total = len([e for e in entries if e.player != "SYSTEM"])
    target = next_power_of_two(total)
    need = target - total
    st.success(f"Bracket generated — Entries: {total} → Target: {target} (BYEs: {need}) — Rule: {rule_label}")

    # init state
    if "r1_winners" not in st.session_state:
        st.session_state.r1_winners = {}

    st.write("### Round 1 Matches")
    player_colors: Dict[str, str] = {}  # fallback per-player colors for non-teams/none
    for i, (a, b) in enumerate(bracket, start=1):
        st.markdown(f"**Match {i}**")

        # Render rows with icons + colored names
        a_html = render_entry(a, team_of, team_colors, player_colors)
        b_html = render_entry(b, team_of, team_colors, player_colors)
        st.markdown(a_html, unsafe_allow_html=True)
        st.markdown("vs")
        st.markdown(b_html, unsafe_allow_html=True)

        # Winner radio
        label_a = f"{a.player} — {a.character}"
        label_b = f"{b.player} — {b.character}"
        default_value = st.session_state.r1_winners.get(i, "")
        idx = 2  # (undecided)
        if default_value == label_a: idx = 0
        elif default_value == label_b: idx = 1

        choice = st.radio(
            "Pick winner:",
            options=[label_a, label_b, "(undecided)"],
            index=idx,
            key=f"winner_{i}",
            horizontal=True,
        )
        st.session_state.r1_winners[i] = choice if choice != "(undecided)" else ""
        st.write("---")

    if st.button("💾 Save Round 1 Winners"):
        winners = [w for w in st.session_state.r1_winners.values() if w]
        st.success(f"Saved {len(winners)} winner selections.")
        if winners:
            st.markdown("**Round 1 Winners:**")
            for w in winners:
                st.write("• ", w)

with col_gen:
    if st.button("🎲 Generate Bracket", type="primary"):
        if len(entries) < 2:
            st.error("Add at least 2 entries (characters).")
        else:
            if rule == "regular":
                bracket = generate_bracket_regular_or_teams(entries, forbid_same_team=False, team_of={})
            elif rule == "groups":
                bracket = generate_bracket_groups(entries)
            else:  # teams
                bracket = generate_bracket_regular_or_teams(entries, forbid_same_team=True, team_of=team_of)

            if bracket is None:
                st.error("Couldn't build a valid round-1 bracket with those constraints. Try balancing counts or team assignments.")
            else:
                st.session_state["last_bracket"] = [(a, b) for (a, b) in bracket]
                st.session_state["last_rule"] = rule
                st.session_state["last_team_of"] = team_of
                st.session_state["last_team_colors"] = team_colors
                show_bracket_and_winner_ui(bracket, rule, team_of, team_colors)

# Persist view on rerun
if "last_bracket" in st.session_state and st.session_state["last_bracket"]:
    st.info("Showing last generated bracket:")
    show_bracket_and_winner_ui(
        st.session_state["last_bracket"],
        st.session_state.get("last_rule", "regular"),
        st.session_state.get("last_team_of", {}),
        st.session_state.get("last_team_colors", {}),
    )

with col_clear:
    if st.button("🧹 Clear Table"):
        st.session_state.table_df = pd.DataFrame(columns=["Player", "Character"])
        st.session_state.pop("last_bracket", None)
        st.session_state.pop("r1_winners", None)
        st.rerun()

st.caption("Names are colored by team (in Teams mode) or by unique player color if no team set. BYEs expand to the next power-of-two bracket.")
