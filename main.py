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
        # MODIFIED: Use and update persistent session state color for individual player coloring
        # Ensure the fallback selection is based on the current size of the *persistent* color dictionary
        color = st.session_state.player_colors.setdefault(player, PLAYER_FALLBACKS[len(st.session_state.player_colors) % len(PLAYER_FALLBACKS)])
    safe_player = player.replace("<", "&lt;").replace(">", "&gt;")
    return f"<span style='color:{color};font-weight:600'>{safe_player}</span>"

def render_entry_line(e: Optional[Entry], team_of: Dict[str, str], team_colors: Dict[str, str], player_colors: Dict[str, str]) -> str:
    if e is None:
        return "<div class='name-line tbd'>TBD</div>"
    if e.character.upper() == "BYE":
        return "<div class='name-line tbd'>BYE</div>"
    icon = get_character_icon_path(e.character)
    # MODIFIED: Pass the session state player_colors dictionary for consistent color lookup
    name_html = render_name_html(e.player, team_of, team_colors, st.session_state.player_colors)
    char_safe = e.character.replace("<", "&lt;").replace(">", "&gt;")
    if icon:
        return f"<div class='name-line'><img src='file://{icon}' width='24'/> <b>{char_safe}</b> ({name_html})</div>"
    else:
        return f"<div class='name-line'><b>{char_safe}</b> ({name_html})</div>"

def entry_to_label(e: Optional[Entry]) -> str:
    if e is None: return ""
    return f"{e.player} — {e.character}"

# ---------------------------- Bracket Generator Functions ----------------------------
def pick_from_lowest_tally(cands: List[Entry], tally: Dict[str, int], exclude_player: Optional[str] = None) -> Optional[Entry]:
    pool = [e for e in cands if e.player != exclude_player]
    if not pool:
        return None
    m = min(tally.get(e.player, 0) for e in pool)
    lowest = [e for e in pool if tally.get(e.player, 0) == m]
    return random.choice(lowest)

def generate_bracket_constrained_core(
    entries: List[Entry], 
    max_repeats: int = 2
) -> List[Tuple[Entry, Entry]]:
    """
    Core function to generate pairings while attempting to limit 
    the number of times any two players match (based on player names). 
    It pairs sequentially and adds a single BYE if needed, but does NOT pad to power of two.
    """
    # Filter out entries that are already BYE (though they shouldn't be here)
    base_entries = [e for e in entries if e.character.upper() != "BYE"]
    
    # 1. Create a working list and shuffle
    bag = base_entries.copy()
    random.shuffle(bag)

    match_counts: Dict[Tuple[str, str], int] = {}  # Tracks matchups between player NAMES
    bracket: List[Tuple[Entry, Entry]] = []

    def key(a: str, b: str) -> Tuple[str, str]:
        # Always sort the player names to ensure consistency
        return tuple(sorted([a, b]))

    def has_played_too_much(p1_name: str, p2_name: str) -> bool:
        return match_counts.get(key(p1_name, p2_name), 0) >= max_repeats

    def record(p1_name: str, p2_name: str):
        k = key(p1_name, p2_name)
        match_counts[k] = match_counts.get(k, 0) + 1

    i = 0
    while i < len(bag) - 1:
        e1 = bag[i]
        e2 = bag[i + 1]
        
        # If these two players have matched too many times, try swapping
        if has_played_too_much(e1.player, e2.player):
            
            # Find a valid swap partner for the second entry (e2)
            for j in range(i + 2, len(bag)):
                e_swap = bag[j]
                
                # Check if e1 vs e_swap is OK
                if not has_played_too_much(e1.player, e_swap.player):
                    # Perform the swap
                    bag[i + 1], bag[j] = bag[j], bag[i + 1]
                    e2 = bag[i + 1]  # Update e2 to the new partner
                    break
            
        # Add the final pairing
        bracket.append((e1, e2))
        record(e1.player, e2.player)
        i += 2

    # Handle odd leftover entry by assigning a single BYE
    if len(bag) % 2 != 0:
        last_entry = bag[-1]
        bracket.append((last_entry, Entry("SYSTEM", "BYE")))
        
    return bracket

def generate_bracket_balanced(
    entries: List[Entry],
    *,
    forbid_same_team: bool = False,
    team_of: Optional[Dict[str, str]] = None
) -> List[Tuple[Entry, Entry]]:
    """
    Balanced-random pairing (Used by Teams mode):
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

def generate_bracket_regular(entries: List[Entry], max_repeats: int) -> List[Tuple[Entry, Entry]]:
    """
    Regular mode uses the constrained pairing logic (user-defined max_repeats) 
    and then ensures the initial pairing includes BYEs to pad to the next power of two.
    """
    base_entries = [e for e in entries if e.player != "SYSTEM"]
    initial_players_count = len(base_entries)
    need_total_byes = byes_needed(initial_players_count)
    
    # 1. Run the core constrained logic to get initial sequential pairings
    r1_pairs_raw = generate_bracket_constrained_core(base_entries, max_repeats=max_repeats)
    
    # 2. Determine which players already got a BYE from the core logic (if odd players)
    already_bye_players = set()
    final_pairs = []

    # If the last pair is a BYE (from the odd number handling), extract it and count it
    if r1_pairs_raw and r1_pairs_raw[-1][1] == Entry("SYSTEM", "BYE"):
        last_bye_pair = r1_pairs_raw.pop()
        already_bye_players.add(last_bye_pair[0].player)
        final_pairs.append(last_bye_pair)
        need_total_byes -= 1
    
    # All non-BYE pairs from the constrained core
    non_bye_pairs = [p for p in r1_pairs_raw if p[1].character.upper() != "BYE"]
    
    # Get all entries that still need a BYE to reach the power of two total
    all_players = [e for e in base_entries]
    
    # Identify players already fully paired (in non-BYE matches)
    players_in_non_bye_pairs = {e.player for pair in non_bye_pairs for e in pair}
    
    # Candidates for the remaining BYEs (those not already in a BYE or a non-BYE match)
    bye_cands = [
        e for e in all_players 
        if e.player not in already_bye_players and e.player not in players_in_non_bye_pairs
    ]

    random.shuffle(bye_cands) # Randomize who gets the remaining BYEs
    
    # 3. Add the required number of additional BYEs
    while need_total_byes > 0 and bye_cands:
        bye_entry = bye_cands.pop(0)
        final_pairs.append((bye_entry, Entry("SYSTEM", "BYE")))
        need_total_byes -= 1
        
    # 4. Reconstruct the final R1 pairings: BYE matches first, then non-BYE matches
    r1_pairs = [p for p in final_pairs if p[1].character.upper() == "BYE"]
    r1_pairs.extend(non_bye_pairs)
    
    # Final shuffle to randomize the order of the actual matches
    random.shuffle(r1_pairs)

    return r1_pairs

def generate_bracket_teams(entries: List[Entry], team_of: Dict[str, str]) -> List[Tuple[Entry, Entry]]:
    # Same as regular but forbids same-team R1
    return generate_bracket_balanced(entries, forbid_same_team=True, team_of=team_of)

# ---------------------------- ROUND ROBIN LOGIC (NEW) ----------------------------

def generate_round_robin_schedule(players: List[str]) -> List[Tuple[str, str]]:
    """Generates a list of all unique match-ups (Player A vs Player B)."""
    matches = []
    current_players = players.copy()
    if len(current_players) % 2 != 0:
        current_players = current_players + ['BYE']
    
    n = len(current_players)
    rounds = n - 1 
    
    # Check if schedule exists in state and is valid for current players
    schedule_key = tuple(sorted(players))
    if "rr_schedule" not in st.session_state or st.session_state["rr_schedule"].get("players") != schedule_key:
        
        # Implementation of the circle method for scheduling
        matchups = []
        p = current_players.copy()
        
        for _ in range(rounds):
            half = n // 2
            for i in range(half):
                p1 = p[i]
                p2 = p[n - 1 - i]
                if p1 != 'BYE' and p2 != 'BYE':
                    matchups.append((p1, p2))
            # Rotate all players except the first
            p.insert(1, p.pop())
            
        # Store and initialize results/records
        st.session_state["rr_schedule"] = {
            "players": schedule_key,
            "matches": matchups,
        }
        st.session_state["rr_results"] = {}
        st.session_state["rr_records"] = {player: {"Wins": 0, "Losses": 0} for player in players if player != 'BYE'}
        
    return st.session_state["rr_schedule"]["matches"]

def update_round_robin_records():
    """Recalculates records based on rr_results."""
    # Ensure rr_records is initialized for all current players
    # FIX: Use the primary players_multiline key as the source of truth
    players_in_state_raw = st.session_state.get("players_multiline", "").splitlines()
    players_in_state = [p.strip() for p in players_in_state_raw if p.strip() and p.strip() != 'BYE']
    
    records = {player: {"Wins": 0, "Losses": 0} for player in players_in_state}
    
    for match_id, winner in st.session_state.rr_results.items():
        if winner == "(Undecided)":
            continue
            
        # Match ID format: Player A|Player B
        p1, p2 = match_id.split('|')
        
        # Only process if both players are currently in the list (handles player removal)
        if p1 in players_in_state and p2 in players_in_state:
            loser = p2 if winner == p1 else p1
            
            if winner in records:
                records[winner]["Wins"] += 1
            if loser in records:
                records[loser]["Losses"] += 1
            
    st.session_state.rr_records = records


def show_round_robin_page(players: List[str]):
    st.subheader("Round Robin Match Results Input")
    
    # Filter out BYE if present in the player list used for UI (shouldn't be, but safe check)
    clean_players = [p for p in players if p != 'BYE']
    
    if len(clean_players) < 2:
        st.error("Please enter at least two players in the sidebar to generate a Round Robin tournament.")
        return

    # 1. Generate/Get Schedule
    schedule = generate_round_robin_schedule(clean_players)

    st.info(f"Total Matches to Play: **{len(schedule)}**")
    
    # Recalculate records first
    update_round_robin_records()
    
    # Use st.columns(3) for match inputs
    cols = st.columns(3)
    
    for i, (p1, p2) in enumerate(schedule, start=1):
        match_id = f"{p1}|{p2}"
        
        # Determine the color-coded labels for the title
        # Ensure player colors are set if they haven't been in Bracket Generator mode yet
        p1_color = st.session_state.player_colors.setdefault(p1, PLAYER_FALLBACKS[len(st.session_state.player_colors) % len(PLAYER_FALLBACKS)])
        p2_color = st.session_state.player_colors.setdefault(p2, PLAYER_FALLBACKS[len(st.session_state.player_colors) % len(PLAYER_FALLBACKS)])
        
        p1_html = f'<span style="color:{p1_color}; font-weight: bold;">{p1}</span>'
        p2_html = f'<span style="color:{p2_color}; font-weight: bold;">{p2}</span>'
        
        # Use existing winner or default to (Undecided)
        default_winner = st.session_state.rr_results.get(match_id, "(Undecided)")
        options = [p1, p2, "(Undecided)"]
        
        try:
            default_index = options.index(default_winner)
        except ValueError:
            default_index = 2

        with cols[i % len(cols)]:
            # Render the match title with colors (the fix for the HTML issue)
            st.markdown(f"**Match {i}:** {p1_html} vs {p2_html}", unsafe_allow_html=True)
            
            # Use plain names for the radio options
            winner = st.radio(
                f"Winner (Match {i})",
                options=options,
                index=default_index,
                key=f"rr_winner_{match_id}",
                horizontal=True,
                label_visibility="collapsed"
            )
            
            # Update results if a choice was made
            st.session_state.rr_results[match_id] = winner
            
    # 2. Leaderboard Display
    st.markdown("---")
    st.subheader("🏆 Tournament Leaderboard")
    
    records_df = pd.DataFrame.from_dict(st.session_state.rr_records, orient='index')
    
    if not records_df.empty:
        records_df.reset_index(names=['Player'], inplace=True)
        
        records_df["Win Rate"] = records_df.apply(lambda row: row['Wins'] / (row['Wins'] + row['Losses']) if (row['Wins'] + row['Losses']) > 0 else 0, axis=1)
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
        # Re-initialize records based on current players
        current_players = [p.strip() for p in st.session_state.get("players_multiline", "").splitlines() if p.strip() != 'BYE']
        st.session_state["rr_records"] = {player: {"Wins": 0, "Losses": 0} for player in current_players}
        st.session_state.pop("rr_schedule", None)
        st.rerun()

# ---------------------------- Sidebar ----------------------------
with st.sidebar:
    st.header("App Navigation")
    # NEW: Control which page is shown
    selected_page = st.radio(
        "Switch View", 
        options=["Bracket Generator", "Round Robin"], 
        index=["Bracket Generator", "Round Robin"].index(st.session_state.page), # Use current state to set index
        key="page_radio" # Use a key for the radio button
    )
    # Immediately update session state page based on user interaction
    st.session_state.page = selected_page
    
    st.divider()
    
    # --- Shared Player List Handling ---
    default_players = st.session_state.players_multiline # Use the persistent value

    if st.session_state.page == "Bracket Generator":
        st.header("Rule Set")
        rule = st.selectbox(
            "Choose mode",
            options=["regular", "teams"],
            index=0,
            key="rule_select", # Added key for state management
            help=(
                "regular: attempts to limit player repeats, fills BYEs to next power of 2.\n"
                "teams: regular + forbids same-team matches in round 1 (names colored by team)."
            )
        )

        # --- NEW INPUT FOR R1 CONSTRAINT ---
        # Initialize max_repeats for the Bracket Generator scope
        max_repeats = 2 
        
        if rule == "regular":
            max_repeats = st.number_input(
                "Max times any two players can match (R1)", 
                min_value=1, max_value=10, value=2, step=1, 
                key="max_repeats_regular_input",
                help="The algorithm attempts to swap players to keep R1 pairings below this limit. Set to 1 for maximum variation."
            )
            
        # --- END NEW INPUT ---
        
        st.divider()
        st.header("Players")
        # Use primary key 'players_multiline'
        players_multiline_input = st.text_area(
            "Enter player names (one per line)",
            value=default_players,
            height=140,
            key="players_multiline", 
            help="These names populate the Player dropdown."
        )
        players = [p.strip() for p in players_multiline_input.splitlines() if p.strip()]

        # Teams UI only in Teams mode
        team_of: Dict[str, str] = {}
        team_colors: Dict[str, str] = {}
        if rule == "teams":
            st.divider()
            st.header("Teams & Colors")
            team_names_input = st.text_input(
                "Team labels (comma separated)",
                value="Red, Blue",
                key="team_names_input", # Added key for state management
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
        
        # Bracket-specific controls
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
    
    else: # st.session_state.page == "Round Robin"
        st.header("Players")
        # Use primary key 'players_multiline' here too. The state is synchronized across the two view inputs.
        players_multiline_input = st.text_area(
            "Enter player names (one per line)",
            value=default_players,
            height=140,
            key="players_multiline", 
            help="These names define the participants for Round Robin."
        )
        players = [p.strip() for p in players_multiline_input.splitlines() if p.strip()]
        
        # Initialize default values needed by the Bracket logic if it runs next
        rule, team_of, team_colors, chars_per_person, build_clicked, shuffle_within_player, auto_fill_clicked, clean_rows = "regular", {}, {}, 1, False, True, False, True
        max_repeats = 2 # Added default for when not in Bracket Generator view
        
    # Final list used by main script body
    st.session_state.players_list = players


# ---------------------------- MAIN CONTENT FLOW ----------------------------
# The title is now conditional based on the selected page (moved outside of the sidebar)
if st.session_state.page == "Bracket Generator":
    st.title("🎮 Smash Bracket — Regular & Teams")
else:
    st.title("🗂️ Round Robin Scheduler & Leaderboard")
    
# Use st.session_state.players_list for consistent access outside the sidebar
players = st.session_state.players_list


if st.session_state.page == "Round Robin":
    show_round_robin_page(players)

else: # Bracket Generator Content 
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

    # --- START OF BRACKET GENERATOR VISIBLE CONTENT ---
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
    def compute_rounds_pairs(r1_pairs: List[Tuple[Optional[Entry], Optional[Entry]]], winners_map: Dict[int, str]) -> List[List[Tuple[Optional[Entry], Optional[Entry]]]]:
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
                    bracket = generate_bracket_regular(entries, max_repeats=max_repeats)
                # Corrected: Use '#' for Python comments
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

    # Corrected: Use '#' for Python comments
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
    # --- END OF BRACKET GENERATOR VISIBLE CONTENT ---
