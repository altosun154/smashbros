import streamlit as st
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Set
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

# ---------------------------- GLOBAL STATE ----------------------------
if "page" not in st.session_state:
    st.session_state.page = "Bracket Generator"

if "player_colors" not in st.session_state:
    st.session_state.player_colors = {}

if "rr_results" not in st.session_state:
    st.session_state.rr_results = {}

if "rr_records" not in st.session_state:
    st.session_state.rr_records = {}

if "players_multiline" not in st.session_state:
    st.session_state.players_multiline = "You\nFriend1\nFriend2"

# NEW: player ordering state for hierarchical system
if "player_order_drawn" not in st.session_state:
    st.session_state.player_order_drawn = []
if "player_order_final" not in st.session_state:
    st.session_state.player_order_final = []

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

def render_name_html(player: str, team_of: Dict[str, str], team_colors: Dict[str, str]) -> str:
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

def render_entry_line(e: Optional[Entry], team_of: Dict[str, str], team_colors: Dict[str, str]) -> str:
    if e is None:
        return "<div class='name-line tbd'>TBD</div>"
    if e.character.upper() == "BYE":
        return "<div class='name-line tbd'>BYE</div>"
    icon = get_character_icon_path(e.character)
    name_html = render_name_html(e.player, team_of, team_colors)
    char_safe = e.character.replace("<", "&lt;").replace(">", "&gt;")
    if icon:
        return f"<div class='name-line'><img src='file://{icon}' width='24'/> <b>{char_safe}</b> ({name_html})</div>"
    return f"<div class='name-line'><b>{char_safe}</b> ({name_html})</div>"

def entry_to_label(e: Optional[Entry]) -> str:
    if e is None:
        return ""
    return f"{e.player} — {e.character}"

# ---------------------------- Hierarchical Weighted Tournament System ----------------------------
def split_half(seq: List) -> Tuple[List, List]:
    """
    Split into top half and bottom half.
    If odd, top half gets the extra element.
    """
    mid = (len(seq) + 1) // 2
    return seq[:mid], seq[mid:]

def build_player_character_map(entries: List[Entry], df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Character ordering = the order in the table editor (df row order).
    If df not usable, fallback to entries insertion order.
    """
    order_map: Dict[str, List[str]] = {}
    if df is not None and "Player" in df.columns and "Character" in df.columns:
        for _, row in df.iterrows():
            p = str(row.get("Player", "")).strip()
            c = str(row.get("Character", "")).strip()
            if not p or not c:
                continue
            order_map.setdefault(p, []).append(c)
    else:
        for e in entries:
            order_map.setdefault(e.player, []).append(e.character)
    # remove dupes while preserving order (in case user repeats)
    for p, chars in order_map.items():
        seen = set()
        deduped = []
        for c in chars:
            if c in seen:
                continue
            seen.add(c)
            deduped.append(c)
        order_map[p] = deduped
    return order_map

def categorize_entries_ABC(
    entries: List[Entry],
    player_order_final: List[str],
    df_table: pd.DataFrame
) -> Dict[Entry, str]:
    """
    Phase 2:
      - split players into top half / bottom half based on final order
      - split each player's characters into top half / bottom half
      - Category A: top chars of top-half players
      - Category B: bottom chars of top-half players + top chars of bottom-half players
      - Category C: bottom chars of bottom-half players
    """
    player_chars = build_player_character_map(entries, df_table)

    # keep only players that actually exist in entries
    present_players = [p for p in player_order_final if p in player_chars]
    if not present_players:
        present_players = sorted(player_chars.keys())

    top_players, bottom_players = split_half(present_players)
    top_set = set(top_players)
    bottom_set = set(bottom_players)

    cat: Dict[Entry, str] = {}
    for e in entries:
        chars = player_chars.get(e.player, [])
        top_chars, bottom_chars = split_half(chars)
        if e.player in top_set:
            if e.character in top_chars:
                cat[e] = "A"
            else:
                cat[e] = "B"
        elif e.player in bottom_set:
            if e.character in top_chars:
                cat[e] = "B"
            else:
                cat[e] = "C"
        else:
            # if player wasn't in the ordering for some reason, treat as bottom group
            if e.character in top_chars:
                cat[e] = "B"
            else:
                cat[e] = "C"
    return cat

def weighted_pick(
    candidates: List[Entry],
    weights: List[float]
) -> Optional[Entry]:
    if not candidates:
        return None
    return random.choices(candidates, weights=weights, k=1)[0]

def generate_bracket_hierarchical_weighted(
    entries: List[Entry],
    *,
    team_mode: bool = False,
    team_of: Optional[Dict[str, str]] = None,
    df_table: Optional[pd.DataFrame] = None,
    player_order_final: Optional[List[str]] = None
) -> List[Tuple[Entry, Entry]]:
    """
    Implements the user's Project Spec:
      Phase 1: player order provided via UI (already random + optional manual override)
      Phase 2: auto categorize entries into A/B/C
      Phase 3: pairing with:
        - first pick: pure random from remaining pool
        - second pick: weighted 40/20/20 by category relative to first pick
      Phase 4: remove both entries immediately, repeat until bracket filled
    Constraints kept:
      - no self-match (same player vs same player)
      - teams mode: forbid same-team matchups in round 1
      - BYEs fill to next power of two; BYE always paired with a real entry (no BYE vs BYE)
    """
    team_of = team_of or {}
    base = [e for e in entries if e.player != "SYSTEM" and e.character.strip()]
    if len(base) < 2:
        return []

    # BYE handling
    need_byes = byes_needed(len(base))
    remaining: List[Entry] = base.copy()
    random.shuffle(remaining)

    pairs: List[Tuple[Entry, Entry]] = []

    # Phase 2 categories for real entries (BYEs not categorized)
    final_order = player_order_final or sorted({e.player for e in base})
    cat_map = categorize_entries_ABC(base, final_order, df_table if df_table is not None else pd.DataFrame())

    # Pre-assign BYEs: pair BYE with random real entries
    bye_entry = Entry("SYSTEM", "BYE")
    while need_byes > 0 and remaining:
        a = remaining.pop()  # random-ish because list was shuffled
        pairs.append((a, bye_entry))
        need_byes -= 1

    # Now do the weighted pairing among remaining real entries
    def allowed(a: Entry, b: Entry) -> bool:
        if a.player == b.player:
            return False
        if team_mode:
            ta = team_of.get(a.player, "")
            tb = team_of.get(b.player, "")
            if ta and tb and ta == tb:
                return False
        return True

    # weights: same=0.4, other two cats=0.2 each (re-normalized over available candidates)
    while len(remaining) >= 2:
        first = random.choice(remaining)  # Phase 3: First Pick is pure random
        remaining.remove(first)

        first_cat = cat_map.get(first, "B")

        candidates = [x for x in remaining if allowed(first, x)]
        if not candidates:
            # Can't find an opponent under constraints: put first back and reshuffle,
            # and if it still can't work, break out.
            remaining.append(first)
            random.shuffle(remaining)
            # try a different first next loop; but if stuck, we eventually stop
            stuck_check = 0
            for _ in range(8):
                test_first = random.choice(remaining)
                test_candidates = [x for x in remaining if x != test_first and allowed(test_first, x)]
                if test_candidates:
                    stuck_check = 1
                    break
            if not stuck_check:
                break
            continue

        weights = []
        for c in candidates:
            c_cat = cat_map.get(c, "B")
            if c_cat == first_cat:
                weights.append(0.4)
            else:
                weights.append(0.2)

        second = weighted_pick(candidates, weights)
        if second is None:
            remaining.append(first)
            random.shuffle(remaining)
            continue

        # Phase 4: removal (permanent for this bracket generation)
        remaining.remove(second)
        pairs.append((first, second))

    # If odd leftover (can happen if constraints block matching), give it a BYE if that still makes sense
    if remaining:
        pairs.append((remaining[0], bye_entry))

    # Ensure bracket size matches power-of-two target (pairs count = target/2)
    # If we ended early due to constraints, we may be short. We'll just return what we have.
    return pairs

def generate_bracket_regular(entries: List[Entry], df_table: pd.DataFrame, player_order_final: List[str]) -> List[Tuple[Entry, Entry]]:
    return generate_bracket_hierarchical_weighted(
        entries,
        team_mode=False,
        team_of={},
        df_table=df_table,
        player_order_final=player_order_final
    )

def generate_bracket_teams(entries: List[Entry], team_of: Dict[str, str], df_table: pd.DataFrame, player_order_final: List[str]) -> List[Tuple[Entry, Entry]]:
    return generate_bracket_hierarchical_weighted(
        entries,
        team_mode=True,
        team_of=team_of,
        df_table=df_table,
        player_order_final=player_order_final
    )

# ---------------------------- ROUND ROBIN ----------------------------
def generate_round_robin_schedule(players: List[str]) -> List[Tuple[str, str]]:
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
                "regular: hierarchical weighted system (A/B/C tiering + weighted pairing), fills BYEs to next power of 2.\n"
                "teams: same logic, but forbids same-team matches in round 1."
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

        # ---- Phase 1 UI: Initial Draw + Manual Override ----
        st.divider()
        st.header("Phase 1: Player Order")

        # if players changed, reset ordering to avoid stale names
        if set(st.session_state.player_order_drawn) != set(players):
            st.session_state.player_order_drawn = []
            st.session_state.player_order_final = []

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🎲 Normal Start (Random Draw)", use_container_width=True):
                order = players.copy()
                random.shuffle(order)
                st.session_state.player_order_drawn = order
                st.session_state.player_order_final = order.copy()

        with col_b:
            if st.button("↩️ Use Drawn Order", use_container_width=True):
                if st.session_state.player_order_drawn:
                    st.session_state.player_order_final = st.session_state.player_order_drawn.copy()

        drawn = st.session_state.player_order_drawn or players
        final_default = st.session_state.player_order_final or drawn

        st.caption("Manual Override: re-rank players (Rank 1 = strongest/top).")
        # Manual ranking UI via unique selectboxes per rank
        remaining_for_pick = final_default.copy()
        new_final: List[str] = []
        for i in range(len(players)):
            default_choice = final_default[i] if i < len(final_default) else (remaining_for_pick[0] if remaining_for_pick else "")
            options = [p for p in players if p not in new_final]
            if default_choice not in options and options:
                default_choice = options[0]
            pick = st.selectbox(
                f"Rank {i+1}",
                options=options if options else [default_choice],
                index=(options.index(default_choice) if default_choice in options else 0),
                key=f"rank_pick_{i}"
            )
            new_final.append(pick)

        st.session_state.player_order_final = new_final

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
        entries_local: List[Entry] = []
        for _, row in df.iterrows():
            pl = str(row.get("Player", "")).strip()
            ch = str(row.get("Character", "")).strip()
            if clean_rows_flag and (not pl or not ch):
                continue
            if pl and ch:
                entries_local.append(Entry(player=pl, character=ch))
        return entries_local

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
                for pair in round_pairs:
                    a, b = pair
                    st.markdown("<div class='match-box'>", unsafe_allow_html=True)
                    st.markdown(render_entry_line(a, team_of, team_colors), unsafe_allow_html=True)
                    st.markdown(render_entry_line(b, team_of, team_colors), unsafe_allow_html=True)
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
                final_order = st.session_state.player_order_final or players
                if rule == "regular":
                    bracket = generate_bracket_regular(entries, table_df, final_order)
                else:
                    bracket = generate_bracket_teams(entries, team_of, table_df, final_order)

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

    st.caption("Regular/Teams now use: random player draw → optional manual reorder → A/B/C tiering → weighted pairing (40/20/20) with removal. Add an 'images/' folder with character PNGs to show icons.")
