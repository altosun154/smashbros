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
/* CSS for Match Boxes and Bracket */
.match-box { border: 1px solid #ddd; border-radius: 10px; padding: 6px 8px; margin: 6px 0;
  font-size: 14px; line-height: 1.25; background: #fff; }
.round-title { font-weight: 700; margin-bottom: 8px; }
.name-line { display: flex; align-items: center; gap: 6px; }
.name-line img { vertical-align: middle; }
.tbd { opacity: 0.6; font-style: italic; }
.legend-badge { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; vertical-align: middle; }
.small { font-size: 13px; }

/* CSS for Round Robin Leaderboard */
.leaderboard-container {
    padding: 10px;
    border-radius: 10px;
    background-color: #f0f2f6;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------- GLOBAL STATE & NAVIGATION SETUP ----------------------------
if "page" not in st.session_state:
    st.session_state.page = "Bracket Generator"

if "player_colors" not in st.session_state:
    st.session_state.player_colors = {}

if "rr_results" not in st.session_state:
    st.session_state.rr_results = {}

if "rr_records" not in st.session_state:
    st.session_state.rr_records = {}

# Primary list of players (default value)
if "players_multiline" not in st.session_state:
    st.session_state.players_multiline = "You\nFriend1\nFriend2"

# NEW: skill inputs storage
if "skill_bias_enabled" not in st.session_state:
    st.session_state.skill_bias_enabled = False
if "ranked_players_input" not in st.session_state:
    st.session_state.ranked_players_input = ""
if "char_strengths" not in st.session_state:
    st.session_state.char_strengths = {}  # {player: [char best->worst]}

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
        color = st.session_state.player_colors.setdefault(
            player,
            PLAYER_FALLBACKS[len(st.session_state.player_colors) % len(PLAYER_FALLBACKS)]
        )
    safe_player = player.replace("<", "&lt;").replace(">", "&gt;")
    return f"<span style='color:{color};font-weight:600'>{safe_player}</span>"

def render_entry_line(e: Optional[Entry], team_of: Dict[str, str], team_colors: Dict[str, str], player_colors: Dict[str, str]) -> str:
    if e is None:
        return "<div class='name-line tbd'>TBD</div>"
    if e.character.upper() == "BYE":
        return "<div class='name-line tbd'>BYE</div>"
    icon = get_character_icon_path(e.character)
    name_html = render_name_html(e.player, team_of, team_colors, st.session_state.player_colors)
    char_safe = e.character.replace("<", "&lt;").replace(">", "&gt;")
    if icon:
        return f"<div class='name-line'><img src='file://{icon}' width='24'/> <b>{char_safe}</b> ({name_html})</div>"
    else:
        return f"<div class='name-line'><b>{char_safe}</b> ({name_html})</div>"

def entry_to_label(e: Optional[Entry]) -> str:
    if e is None:
        return ""
    return f"{e.player} — {e.character}"

# ---------------------------- Skill-bias helpers (NEW) ----------------------------
def parse_rank_map(players: List[str], ranked_players_input: str) -> Dict[str, int]:
    """
    ranked_players_input: names one per line, best -> worst
    returns {player: rank_index} where 0 is best.
    Players not listed are placed after listed players.
    """
    order = [x.strip() for x in ranked_players_input.splitlines() if x.strip()]
    rank_map = {p: i for i, p in enumerate(order)}
    # Ensure all players exist with a rank
    bottom_start = len(rank_map)
    for p in players:
        if p not in rank_map:
            rank_map[p] = bottom_start
            bottom_start += 1
    return rank_map

def normalize_rank(rank: int, max_rank: int) -> float:
    if max_rank <= 0:
        return 0.0
    return rank / max_rank  # 0 best, 1 worst

def char_rank_for_entry(player: str, character: str, char_strengths: Dict[str, List[str]]) -> int:
    """
    char_strengths[player] is list best->worst (index 0 strongest).
    If not found, treat as middle-ish (not always worst).
    """
    lst = char_strengths.get(player, [])
    if not lst:
        return 1  # neutral default
    try:
        return lst.index(character)
    except ValueError:
        # not listed: treat as middle
        return max(0, len(lst) // 2)

def entry_power_score(
    e: Entry,
    rank_map: Dict[str, int],
    char_strengths: Dict[str, List[str]],
) -> float:
    """
    Higher score = stronger "entry".
    Designed so:
      - best players + best chars => high
      - best players + worst chars => medium
      - worst players + best chars => medium
      - worst players + worst chars => low
    """
    max_rank = max(rank_map.values()) if rank_map else 0
    player_rank_norm = normalize_rank(rank_map.get(e.player, max_rank), max_rank)  # 0..1
    # player_strength_norm: 1 best, 0 worst
    player_strength_norm = 1.0 - player_rank_norm

    lst = char_strengths.get(e.player, [])
    if lst:
        cr = char_rank_for_entry(e.player, e.character, char_strengths)
        max_cr = max(1, len(lst) - 1)
        char_rank_norm = cr / max_cr  # 0 best, 1 worst
        char_strength_norm = 1.0 - char_rank_norm
    else:
        # neutral if no char list
        char_strength_norm = 0.5

    # weight player a bit more than character
    return (player_strength_norm * 2.0) + (char_strength_norm * 1.0)

# ---------------------------- Balanced generator (Regular core) ----------------------------
def pick_from_lowest_tally(cands: List[Entry], tally: Dict[str, int], exclude_player: Optional[str] = None) -> Optional[Entry]:
    pool = [e for e in cands if e.player != exclude_player]
    if not pool:
        return None
    m = min(tally.get(e.player, 0) for e in pool)
    lowest = [e for e in pool if tally.get(e.player, 0) == m]
    return random.choice(lowest)

def matchup_key(a_player: str, b_player: str) -> Tuple[str, str]:
    return tuple(sorted((a_player, b_player)))

def generate_bracket_balanced(
    entries: List[Entry],
    *,
    forbid_same_team: bool = False,
    team_of: Optional[Dict[str, str]] = None,
    skill_bias: bool = False,
    ranked_players_input: str = "",
    char_strengths: Optional[Dict[str, List[str]]] = None,
) -> List[Tuple[Entry, Entry]]:
    """
    Balanced-random pairing:
      - no self-match,
      - optional: forbid same-team,
      - fills BYEs to next power of two,
      - uses per-player tallies for fairness,
      - NEW: within-this-bracket opponent diversity (avoid same two players too often),
      - NEW: optional skill-bias to pair similar "entry power" together.
    """
    team_of = team_of or {}
    char_strengths = char_strengths or {}

    base = [e for e in entries if e.player != "SYSTEM"]
    need = byes_needed(len(base))

    # LOCAL ONLY (resets every Generate Bracket):
    matchup_counts: Dict[Tuple[str, str], int] = {}

    # Skill rank map only matters if enabled
    all_players = sorted({e.player for e in base if e.player != "SYSTEM"})
    rank_map = parse_rank_map(all_players, ranked_players_input) if skill_bias else {}

    bag = base.copy()
    random.shuffle(bag)
    tally: Dict[str, int] = {}
    pairs: List[Tuple[Entry, Entry]] = []

    def add_match(a: Entry, b: Entry):
        pairs.append((a, b))
        tally[a.player] = tally.get(a.player, 0) + 1
        if b.player != "SYSTEM":
            tally[b.player] = tally.get(b.player, 0) + 1

        # Only track real player-vs-player matchups
        if b.player != "SYSTEM":
            k = matchup_key(a.player, b.player)
            matchup_counts[k] = matchup_counts.get(k, 0) + 1

    # Use some BYEs first if needed
    while need > 0 and bag:
        a = pick_from_lowest_tally(bag, tally)
        bag.remove(a)
        add_match(a, Entry("SYSTEM", "BYE"))
        need -= 1

    def pick_opponent(a: Entry, pool: List[Entry]) -> Optional[Entry]:
        pool2 = [x for x in pool if x.player != a.player]

        if forbid_same_team:
            ta = team_of.get(a.player, "")
            if ta:
                pool2 = [x for x in pool2 if team_of.get(x.player, "") != ta]

        if not pool2:
            return None

        # Score candidates:
        # 1) minimize how often these two PLAYERS have already faced (within THIS bracket)
        # 2) if skill_bias enabled: minimize difference in entry power (strong-vs-strong, weak-vs-weak)
        # 3) then minimize player appearance tally (keeps overall balance)
        scored: List[Tuple[Tuple, Entry]] = []
        a_power = entry_power_score(a, rank_map, char_strengths) if skill_bias else 0.0

        for b in pool2:
            mu = matchup_counts.get(matchup_key(a.player, b.player), 0)

            if skill_bias:
                b_power = entry_power_score(b, rank_map, char_strengths)
                power_gap = abs(a_power - b_power)
            else:
                power_gap = 0.0

            # Lower tuple = better
            score = (mu, power_gap, tally.get(b.player, 0), random.random())
            scored.append((score, b))

        scored.sort(key=lambda x: x[0])
        best = scored[0][0]
        # Randomize among tied best-ish candidates (within a tiny epsilon for power_gap)
        eps = 1e-9
        candidates = [b for (s, b) in scored if s[0] == best[0] and abs(s[1] - best[1]) <= eps and s[2] == best[2]]
        return random.choice(candidates) if candidates else scored[0][1]

    while len(bag) >= 2:
        a = pick_from_lowest_tally(bag, tally)
        bag.remove(a)

        b = pick_opponent(a, bag)
        if b is None:
            # try turn this into a BYE if adding one still gets us to power-of-two
            if byes_needed(len(bag) + 1) > 0:
                add_match(a, Entry("SYSTEM", "BYE"))
            else:
                bag.append(a)
                random.shuffle(bag)
                if len(bag) == 1:
                    break
            continue

        bag.remove(b)
        add_match(a, b)

    if bag:  # odd leftover
        add_match(bag[0], Entry("SYSTEM", "BYE"))

    return pairs

def generate_bracket_regular(entries: List[Entry]) -> List[Tuple[Entry, Entry]]:
    # Regular is balanced generator + optional skill bias
    return generate_bracket_balanced(
        entries,
        skill_bias=st.session_state.get("skill_bias_enabled", False),
        ranked_players_input=st.session_state.get("ranked_players_input", ""),
        char_strengths=st.session_state.get("char_strengths", {}),
    )

def generate_bracket_teams(entries: List[Entry], team_of: Dict[str, str]) -> List[Tuple[Entry, Entry]]:
    # Teams is regular + forbids same-team R1
    return generate_bracket_balanced(
        entries,
        forbid_same_team=True,
        team_of=team_of,
        skill_bias=st.session_state.get("skill_bias_enabled", False),
        ranked_players_input=st.session_state.get("ranked_players_input", ""),
        char_strengths=st.session_state.get("char_strengths", {}),
    )

# ---------------------------- ROUND ROBIN LOGIC ----------------------------
def generate_round_robin_schedule(players: List[str]) -> List[Tuple[str, str]]:
    """Generates a list of all unique match-ups (Player A vs Player B)."""
    current_players = players.copy()
    if len(current_players) % 2 != 0:
        current_players = current_players + ['BYE']

    n = len(current_players)
    rounds = n - 1

    schedule_key = tuple(sorted(players))
    if "rr_schedule" not in st.session_state or st.session_state["rr_schedule"].get("players") != schedule_key:
        matchups = []
        p = current_players.copy()

        for _ in range(rounds):
            half = n // 2
            for i in range(half):
                p1 = p[i]
                p2 = p[n - 1 - i]
                if p1 != 'BYE' and p2 != 'BYE':
                    matchups.append((p1, p2))
            p.insert(1, p.pop())

        st.session_state["rr_schedule"] = {"players": schedule_key, "matches": matchups}
        st.session_state["rr_results"] = {}
        st.session_state["rr_records"] = {player: {"Wins": 0, "Losses": 0} for player in players if player != 'BYE'}

    return st.session_state["rr_schedule"]["matches"]

def update_round_robin_records():
    players_in_state_raw = st.session_state.get("players_multiline", "").splitlines()
    players_in_state = [p.strip() for p in players_in_state_raw if p.strip() and p.strip() != 'BYE']

    records = {player: {"Wins": 0, "Losses": 0} for player in players_in_state}

    for match_id, winner in st.session_state.rr_results.items():
        if winner == "(Undecided)":
            continue

        p1, p2 = match_id.split('|')
        if p1 in players_in_state and p2 in players_in_state:
            loser = p2 if winner == p1 else p1
            if winner in records:
                records[winner]["Wins"] += 1
            if loser in records:
                records[loser]["Losses"] += 1

    st.session_state.rr_records = records

def show_round_robin_page(players: List[str]):
    st.subheader("Round Robin Match Results Input")
    clean_players = [p for p in players if p != 'BYE']

    if len(clean_players) < 2:
        st.error("Please enter at least two players in the sidebar to generate a Round Robin tournament.")
        return

    schedule = generate_round_robin_schedule(clean_players)
    st.info(f"Total Matches to Play: **{len(schedule)}**")
    update_round_robin_records()

    cols = st.columns(3)

    for i, (p1, p2) in enumerate(schedule, start=1):
        match_id = f"{p1}|{p2}"

        p1_color = st.session_state.player_colors.setdefault(p1, PLAYER_FALLBACKS[len(st.session_state.player_colors) % len(PLAYER_FALLBACKS)])
        p2_color = st.session_state.player_colors.setdefault(p2, PLAYER_FALLBACKS[len(st.session_state.player_colors) % len(PLAYER_FALLBACKS)])

        p1_html = f'<span style="color:{p1_color}; font-weight: bold;">{p1}</span>'
        p2_html = f'<span style="color:{p2_color}; font-weight: bold;">{p2}</span>'

        default_winner = st.session_state.rr_results.get(match_id, "(Undecided)")
        options = [p1, p2, "(Undecided)"]

        try:
            default_index = options.index(default_winner)
        except ValueError:
            default_index = 2

        with cols[i % len(cols)]:
            st.markdown(f"**Match {i}:** {p1_html} vs {p2_html}", unsafe_allow_html=True)
            winner = st.radio(
                f"Winner (Match {i})",
                options=options,
                index=default_index,
                key=f"rr_winner_{match_id}",
                horizontal=True,
                label_visibility="collapsed"
            )
            st.session_state.rr_results[match_id] = winner

    st.markdown("---")
    st.subheader("🏆 Tournament Leaderboard")

    records_df = pd.DataFrame.from_dict(st.session_state.rr_records, orient='index')

    if not records_df.empty:
        records_df.reset_index(names=['Player'], inplace=True)
        records_df["Win Rate"] = records_df.apply(
            lambda row: row['Wins'] / (row['Wins'] + row['Losses']) if (row['Wins'] + row['Losses']) > 0 else 0,
            axis=1
        )
        records_df.sort_values(by=['Wins', 'Losses', 'Player'], ascending=[False, True, True], inplace=True)
        records_df.index = records_df.index + 1

        st.dataframe(
            records_df,
            use_container_width=True,
            column_config={
                "Player": st.column_config.Column("Player", width="small"),
                "Wins": st.column_config.Column("Wins", width="small"),
                "Losses": st.column_config.Column("Losses", width="small"),
                "Win Rate": st.column_config.ProgressColumn("Win Rate", format="%.1f", width="small", min_value=0, max_value=1),
            }
        )
    else:
        st.info("No records to display. Please enter match results.")

    st.markdown("---")
    if st.button("🔄 Reset All Round Robin Records"):
        st.session_state["rr_results"] = {}
        current_players = [p.strip() for p in st.session_state.get("players_multiline", "").splitlines() if p.strip() != 'BYE']
        st.session_state["rr_records"] = {player: {"Wins": 0, "Losses": 0} for player in current_players}
        st.session_state.pop("rr_schedule", None)
        st.rerun()

# ---------------------------- Sidebar ----------------------------
with st.sidebar:
    st.header("App Navigation")
    selected_page = st.radio(
        "Switch View",
        options=["Bracket Generator", "Round Robin"],
        index=["Bracket Generator", "Round Robin"].index(st.session_state.page),
        key="page_radio"
    )
    st.session_state.page = selected_page

    st.divider()

    default_players = st.session_state.players_multiline

    if st.session_state.page == "Bracket Generator":
        st.header("Rule Set")
        rule = st.selectbox(
            "Choose mode",
            options=["regular", "teams"],
            index=0,
            key="rule_select",
            help=(
                "regular: balanced random (no self-matches), fills BYEs to next power of 2.\n"
                "teams: regular + forbids same-team matches in round 1 (names colored by team)."
            )
        )

        st.divider()
        st.header("Players")
        players_multiline_input = st.text_area(
            "Enter player names (one per line)",
            value=default_players,
            height=140,
            key="players_multiline",
            help="These names populate the Player dropdown."
        )
        players = [p.strip() for p in players_multiline_input.splitlines() if p.strip()]

        # NEW: Skill bias controls
        st.divider()
        st.header("Skill Bias (Optional)")
        st.session_state.skill_bias_enabled = st.checkbox(
            "Bias matchups using player + character rankings",
            value=st.session_state.skill_bias_enabled,
            help="If enabled, the generator tries to pair entries with similar strength: top-top, mid-mid, bottom-bottom."
        )

        if st.session_state.skill_bias_enabled:
            st.session_state.ranked_players_input = st.text_area(
                "Player ranking (best → worst), one per line",
                value=st.session_state.ranked_players_input,
                height=120,
                help="Example:\nYou\nFriend2\nFriend1\n(Unlisted players will be treated as worst.)"
            )

            st.caption("Optional: rank each player's characters best → worst (comma separated).")
            char_strengths_temp: Dict[str, List[str]] = {}
            for p in players:
                default_val = ", ".join(st.session_state.char_strengths.get(p, []))
                s = st.text_input(
                    f"{p} characters (best→worst)",
                    value=default_val,
                    key=f"char_strength_{p}",
                    help="Example: Mario, Link, Kirby"
                )
                char_strengths_temp[p] = [x.strip() for x in s.split(",") if x.strip()]
            st.session_state.char_strengths = char_strengths_temp

        # Teams UI only in Teams mode
        team_of: Dict[str, str] = {}
        team_colors: Dict[str, str] = {}
        if rule == "teams":
            st.divider()
            st.header("Teams & Colors")
            team_names_input = st.text_input(
                "Team labels (comma separated)",
                value="Red, Blue",
                key="team_names_input",
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
        chars_per_person = st.number_input("How many per player?", min_value=1, max_value=50, value=2, step=1, key="chars_per_person")

        st.divider()
        st.subheader("Build / Fill")
        build_clicked = st.button("⚙️ Auto-Create/Reset Entries", use_container_width=True)
        shuffle_within_player = st.checkbox("Shuffle names when auto-filling", value=True)
        auto_fill_clicked = st.button("🎲 Auto-fill Characters (Character 1..k)", use_container_width=True)

        st.divider()
        st.header("General")
        clean_rows = st.checkbox("Remove empty rows", value=True)

    else:
        st.header("Players")
        players_multiline_input = st.text_area(
            "Enter player names (one per line)",
            value=default_players,
            height=140,
            key="players_multiline",
            help="These names define the participants for Round Robin."
        )
        players = [p.strip() for p in players_multiline_input.splitlines() if p.strip()]
        rule, team_of, team_colors, chars_per_person, build_clicked, shuffle_within_player, auto_fill_clicked, clean_rows = "regular", {}, {}, 1, False, True, False, True

    st.session_state.players_list = players

# ---------------------------- MAIN CONTENT FLOW ----------------------------
if st.session_state.page == "Bracket Generator":
    st.title("🎮 Smash Bracket — Regular & Teams")
else:
    st.title("🗂️ Round Robin Scheduler & Leaderboard")

players = st.session_state.players_list

if st.session_state.page == "Round Robin":
    show_round_robin_page(players)

else:
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
            if pair_index >= len(pairs_list):
                return None
            a, b = pairs_list[pair_index]
            if a is None and b is None:
                return None
            if a is None:
                return b if (b and b.character.upper() != "BYE") else None
            if b is None:
                return a if (a and a.character.upper() != "BYE") else None
            if a.character.upper() == "BYE" and b.character.upper() != "BYE":
                return b
            if b.character.upper() == "BYE" and a.character.upper() != "BYE":
                return a

            label_a, label_b = entry_to_label(a), entry_to_label(b)
            sel = winners_map.get(pair_index + 1, "")
            if sel == label_a:
                return a
            if sel == label_b:
                return b
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
            legend = "  ".join([f"<span class='legend-badge' style='background:{c}'></span>{t}" for t, c in team_colors.items()])
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
            if prev == label_a:
                idx = 0
            elif prev == label_b:
                idx = 1
            else:
                idx = 2
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
                else:
                    bracket = generate_bracket_teams(entries, team_of)

                if not bracket:
                    st.error("Couldn't build a valid round-1 bracket with those constraints.")
                else:
                    total_real = len([e for e in entries if e.player != "SYSTEM"])
                    target = next_power_of_two(total_real)
                    need = target - total_real
                    mode_extra = " + skill bias" if st.session_state.get("skill_bias_enabled", False) else ""
                    st.success(f"Entries: {total_real} → Target: {target} (BYEs: {need}) — Mode: {rule}{mode_extra}")

                    st.session_state["last_bracket"] = [(a, b) for (a, b) in bracket]
                    st.session_state["last_rule"] = rule
                    st.session_state["last_team_of"] = team_of if rule == "teams" else {}
                    st.session_state["last_team_colors"] = team_colors if rule == "teams" else {}

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
