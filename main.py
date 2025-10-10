import streamlit as st
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import pandas as pd
import math
import os

st.set_page_config(page_title="Smash Bracket", page_icon="🎮", layout="wide")

# --- Custom CSS for Bracket Visualization ---
st.markdown("""
<style>
.match-box { 
    border: 1px solid #333; 
    border-radius: 8px; 
    padding: 8px; 
    margin: 8px 0;
    font-size: 14px; 
    line-height: 1.3; 
    background: #1e1e1e; /* Darker background for matches */
    color: #f0f0f0; /* Light text */
    box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3); 
}
.round-title { 
    font-weight: 800; 
    margin-bottom: 12px; 
    font-size: 1.1em;
    color: #4CAF50; /* Green title */
    border-bottom: 2px solid #222;
    padding-bottom: 4px;
}
.name-line { 
    display: flex; 
    align-items: center; 
    gap: 8px; 
    padding: 2px 0;
}
.tbd { 
    opacity: 0.5; 
    font-style: italic; 
    color: #aaa;
}
.legend-badge { 
    display: inline-block; 
    width: 12px; 
    height: 12px; 
    border-radius: 50%; 
    margin-right: 6px; 
    vertical-align: middle; 
    box-shadow: 0 0 4px rgba(0, 0, 0, 0.5);
}
.stRadio > label { 
    padding: 0 5px !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🥊 Smash Bracket Generator — Single & Team")
st.caption("Auto-generates balanced single-elimination brackets with team support and R1 winner tracking.")

# ---------------------------- Data types ----------------------------
@dataclass(frozen=True)
class Entry:
    player: str
    character: str

# ---------------------------- Power-of-two helpers ----------------------------
def next_power_of_two(n: int) -> int:
    if n <= 1:
        return 1
    # Check if n is already a power of 2
    if (n & (n - 1) == 0):
        return n
    return 1 << n.bit_length()

def byes_needed(n: int) -> int:
    return max(0, next_power_of_two(n) - n)

# ---------------------------- Icons & colors ----------------------------
# NOTE: Removed file-based icon loading as it requires a specific file structure,
# which isn't available in this execution environment. Keeping the functions for structure.

def get_character_icon_path(char_name: str) -> Optional[str]:
    # Placeholder for local path check (removed os.path.exists check)
    return None 

TEAM_COLOR_FALLBACKS = [
    "#FF4B4B", "#3F81F2", "#34D399", "#FBBF24", "#A855F7",
    "#059669", "#7C3AED", "#FF8000", "#737373", "#14B8A6"
]
PLAYER_FALLBACKS = [
    "#F87171", "#60A5FA", "#4ADE80", "#FCD34D", "#C4B5FD",
    "#10B981", "#A78BFA", "#F97316", "#A1A1AA", "#2DD4BF"
]
PLAYER_COLORS: Dict[str, str] = {} # Persistent color tracking

def render_name_html(player: str, team_of: Dict[str, str], team_colors: Dict[str, str]) -> str:
    global PLAYER_COLORS
    t = team_of.get(player, "")
    if t and team_colors.get(t):
        # Use team color if available
        color = team_colors[t]
    else:
        # Assign or retrieve persistent player color
        if player not in PLAYER_COLORS:
            PLAYER_COLORS[player] = PLAYER_FALLBACKS[len(PLAYER_COLORS) % len(PLAYER_FALLBACKS)]
        color = PLAYER_COLORS[player]
        
    safe_player = player.replace("<", "&lt;").replace(">", "&gt;")
    return f"<span style='color:{color};font-weight:600'>{safe_player}</span>"

def render_entry_line(e: Optional[Entry], team_of: Dict[str, str], team_colors: Dict[str, str]) -> str:
    if e is None:
        return "<div class='name-line tbd'>TBD</div>"
    if e.character.upper() == "BYE":
        return "<div class='name-line tbd'>SYSTEM BYE (Auto-Win)</div>"
    
    icon = get_character_icon_path(e.character) # Will return None
    name_html = render_name_html(e.player, team_of, team_colors)
    char_safe = e.character.replace("<", "&lt;").replace(">", "&gt;")

    # Use a generic controller icon/emoji placeholder since local file paths don't work
    icon_html = "🎮"
    
    return f"<div class='name-line'>{icon_html} <b>{char_safe}</b> ({name_html})</div>"

def entry_to_label(e: Optional[Entry]) -> str:
    if e is None: return ""
    return f"{e.player} — {e.character}"

# ---------------------------- Balanced generator (Regular core) ----------------------------
def pick_from_lowest_tally(cands: List[Entry], tally: Dict[str, int], exclude_player: Optional[str] = None) -> Optional[Entry]:
    pool = [e for e in cands if e.player != exclude_player and e.character.upper() != "BYE"]
    if not pool:
        return None
    
    # Calculate player tallies for fairness
    m = min(tally.get(e.player, 0) for e in pool)
    lowest = [e for e in pool if tally.get(e.player, 0) == m]
    
    return random.choice(lowest)

def generate_bracket_balanced(
    entries: List[Entry],
    *,
    forbid_same_team: bool = False,
    team_of: Optional[Dict[str, str]] = None
) -> List[Tuple[Entry, Entry]]:
    
    team_of = team_of or {}
    base = [e for e in entries if e.player != "SYSTEM"]
    
    bag = base.copy()
    random.shuffle(bag)
    tally: Dict[str, int] = {p: 0 for p in set(e.player for e in base)}
    pairs: List[Tuple[Entry, Entry]] = []
    
    # Calculate necessary BYEs and place them first
    needed_byes = byes_needed(len(base))
    
    # 1. Assign initial BYEs to entries with the lowest current tally
    for _ in range(needed_byes):
        if not bag: break
        a = pick_from_lowest_tally(bag, tally)
        if not a: break
        bag.remove(a)
        pairs.append((a, Entry("SYSTEM", "BYE")))
        tally[a.player] = tally.get(a.player, 0) + 1

    def pick_opponent(a: Entry, pool: List[Entry]) -> Optional[Entry]:
        pool2 = [x for x in pool if x.player != a.player and x.character.upper() != "BYE"]
        
        if forbid_same_team:
            ta = team_of.get(a.player, "")
            if ta:
                pool2 = [x for x in pool2 if team_of.get(x.player, "") != ta]
        
        if not pool2:
            return None
            
        m = min(tally.get(x.player, 0) for x in pool2)
        lowest = [x for x in pool2 if tally.get(x.player, 0) == m]
        
        return random.choice(lowest)

    # 2. Pair the remaining entries
    while len(bag) >= 2:
        # Pick the next player based on the lowest number of previous matches
        a = pick_from_lowest_tally(bag, tally)
        if not a: break
        bag.remove(a)
        
        # Pick opponent B, avoiding same player/same team (if restricted)
        b = pick_opponent(a, bag)
        
        if b is None:
            # If no suitable opponent, we're likely constrained by teams/players.
            # Put A back and try again with shuffled list if possible, otherwise it's an unpairable leftover.
            bag.append(a)
            random.shuffle(bag) 
            if len(bag) == 1:
                break
            continue
            
        bag.remove(b)
        pairs.append((a, b))
        tally[a.player] += 1
        tally[b.player] += 1

    # 3. Handle odd leftovers (should only happen if an odd number of real players remain, which should be caught by BYE logic)
    if bag: 
        pairs.append((bag[0], Entry("SYSTEM", "BYE")))
        
    return pairs

def generate_bracket_regular(entries: List[Entry]) -> List[Tuple[Entry, Entry]]:
    return generate_bracket_balanced(entries)

def generate_bracket_teams(entries: List[Entry], team_of: Dict[str, str]) -> List[Tuple[Entry, Entry]]:
    return generate_bracket_balanced(entries, forbid_same_team=True, team_of=team_of)

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
            if out.at[row_i, "Character"] == "": # Only fill empty spots
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

# ---------------------------- Bracket Logic ----------------------------
def compute_rounds_pairs(r1_pairs: List[Tuple[Entry, Entry]], winners_map: Dict[int, str]) -> List[List[Tuple[Optional[Entry], Optional[Entry]]]]:
    rounds: List[List[Tuple[Optional[Entry], Optional[Entry]]]] = []
    
    # R1 pairs are fixed
    rounds.append([(a, b) for (a, b) in r1_pairs])

    # Determine total rounds needed based on number of players
    total_real = sum(1 for (a, b) in r1_pairs for e in (a, b) if e and e.player != "SYSTEM")
    target = next_power_of_two(total_real)
    num_rounds = int(math.log2(target)) if target >= 2 else 1

    prev = rounds[0]

    def winner_of_pair(pair_index: int, pairs_list: List[Tuple[Optional[Entry], Optional[Entry]]]) -> Optional[Entry]:
        if pair_index >= len(pairs_list): return None
        a, b = pairs_list[pair_index]
        if a is None and b is None: return None
        
        # Handle BYE matches first (auto-win)
        if a and a.character.upper() == "BYE": return b if (b and b.character.upper() != "BYE") else None
        if b and b.character.upper() == "BYE": return a if (a and a.character.upper() != "BYE") else None
        
        # Check for explicit winner selection (only for R1 matches)
        label_a, label_b = entry_to_label(a), entry_to_label(b)
        # Note: The winner map key needs to be managed carefully for subsequent rounds
        # For simplicity and aligning with original code, R1 is controlled by explicit selection,
        # and subsequent rounds rely on the completion of the prior round's match-ups.
        
        if round_idx == 0: # Only check user selection for R1
            sel = winners_map.get(pair_index + 1, "")
            if sel == label_a: return a
            if sel == label_b: return b
            return None
        else: # For R2+ matches, the 'winner' is just the entry that progressed
             # Since we only control R1, the 'winners' for R2+ are based on how R1 resolved
             # This simple logic assumes the pairs are already resolved from the prior round calculation
             return a if a else b

    for round_idx in range(1, num_rounds):
        nxt: List[Tuple[Optional[Entry], Optional[Entry]]] = []
        for i in range(0, len(prev), 2):
            # Calculate winners based on the prior round's outcome
            w1 = prev[i][0] if prev[i][0] and prev[i][0].character.upper() != "BYE" else prev[i][1]
            w2 = prev[i+1][0] if prev[i+1][0] and prev[i+1][0].character.upper() != "BYE" else prev[i+1][1]
            
            # This logic needs to correctly look up the *winner* from the previous round
            # We'll re-implement the winner calculation based on the prior round's results
            
            w1 = winner_of_pair(i, prev)
            w2 = winner_of_pair(i + 1, prev)
            
            # For Rounds 2 and up, the "pair" is composed of winners from the previous round's pairs
            nxt.append((w1, w2))
            
        rounds.append(nxt)
        prev = nxt
    return rounds


def render_bracket_grid(all_rounds: List[List[Tuple[Optional[Entry], Optional[Entry]]]], team_of: Dict[str, str], team_colors: Dict[str, str]):
    cols = st.columns(len(all_rounds))
    
    # Team Legend
    if team_colors and any(team_of.values()):
        legend = " ".join([
            f"<span class='legend-badge' style='background:{c}'></span>{t}" 
            for t, c in team_colors.items() if t in team_of.values()
        ])
        st.markdown(f"**Teams:** {legend}", unsafe_allow_html=True)

    for round_idx, round_pairs in enumerate(all_rounds):
        with cols[round_idx]:
            round_name = "Finals" if round_idx == len(all_rounds) - 1 else f"Round {round_idx+1}"
            st.markdown(f"<div class='round-title'>{round_name}</div>", unsafe_allow_html=True)
            
            # Use persistent PLAYER_COLORS dict
            global PLAYER_COLORS
            
            for pair_idx, pair in enumerate(round_pairs):
                a, b = pair
                
                # Check if this match is resolved (i.e., we know the winner for the next round)
                match_is_resolved = (
                    (round_idx == 0 and st.session_state.get("r1_winners", {}).get(pair_idx + 1)) or 
                    (round_idx > 0 and (a is not None and b is not None))
                )

                box_style = "match-box"
                if match_is_resolved and round_idx == 0:
                    box_style += " border-green-500" # Highlight completed R1 matches

                st.markdown(f"<div class='{box_style}'>", unsafe_allow_html=True)
                st.markdown(render_entry_line(a, team_of, team_colors), unsafe_allow_html=True)
                st.markdown(render_entry_line(b, team_of, team_colors), unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

    # Display Champion if determined
    if all_rounds and all_rounds[-1]:
        champion_pair = all_rounds[-1][0]
        champion = champion_pair[0] if champion_pair[0] else champion_pair[1]
        
        if champion and champion.character.upper() != "BYE":
            st.balloons()
            st.subheader("🏆 Champion!")
            st.markdown(render_entry_line(champion, team_of, team_colors), unsafe_allow_html=True)

def r1_winner_controls(r1_pairs: List[Tuple[Entry, Entry]]):
    if "r1_winners" not in st.session_state:
        st.session_state.r1_winners = {}
        
    st.markdown("### ➡️ Select Round 1 Winners")
    
    # Check if there are matches that need resolution
    real_matches = [(a, b) for a, b in r1_pairs if a.character.upper() != "BYE" and b.character.upper() != "BYE"]
    if not real_matches:
        st.info("All Round 1 matches are BYEs. The next round is automatically populated.")
        return

    st.columns(1) # Ensure winner controls are in a single column
    
    for i, (a, b) in enumerate(r1_pairs, start=1):
        if a.character.upper() == "BYE" or b.character.upper() == "BYE":
            continue # Skip BYE matches in manual selection
            
        label_a = entry_to_label(a)
        label_b = entry_to_label(b)
        
        # Determine current selection index
        prev = st.session_state.r1_winners.get(i, "")
        if prev == label_a: idx = 0
        elif prev == label_b: idx = 1
        else: idx = 2
        
        # Radio button for winner selection
        st.radio(
            f"Match {i}: {label_a} vs {label_b}",
            options=[label_a, label_b, "(Undecided)"],
            index=idx,
            key=f"winner_{i}",
            horizontal=True,
            label_visibility="collapsed"
        )
        
        # Update session state after radio button interaction
        choice = st.session_state[f"winner_{i}"]
        st.session_state.r1_winners[i] = choice if choice != "(Undecided)" else ""


# ---------------------------- Sidebar & Setup ----------------------------
with st.sidebar:
    st.header("⚙️ Bracket Setup")
    
    # --- Rule Set ---
    rule = st.selectbox(
        "1. Choose Mode",
        options=["regular", "teams"],
        index=0,
        help="Regular: Random balanced pairing. Teams: Forbids same-team matches in R1."
    )

    # --- Players ---
    st.divider()
    st.header("👤 Players")
    default_players = "You\nFriend1\nFriend2"
    players_multiline = st.text_area(
        "2. Enter player names (one per line)",
        value=st.session_state.get("players_multiline", default_players),
        height=140,
        key="players_input"
    )
    players = [p.strip() for p in players_multiline.splitlines() if p.strip()]
    st.session_state["players_multiline"] = players_multiline
    
    # --- Teams UI only in Teams mode ---
    team_of: Dict[str, str] = {}
    team_colors: Dict[str, str] = {}
    if rule == "teams":
        st.divider()
        st.header("🚩 Teams & Colors")
        
        team_names_input = st.text_input(
            "3a. Team labels (comma separated)",
            value=st.session_state.get("team_names_input", "Red, Blue"),
            key="team_names_input_key"
        )
        st.session_state["team_names_input"] = team_names_input
        team_labels = [t.strip() for t in team_names_input.split(",") if t.strip()]
        if not team_labels:
            team_labels = ["Team A", "Team B"]

        st.caption("3b. Pick a color for each team:")
        for i, t in enumerate(team_labels):
            default = TEAM_COLOR_FALLBACKS[i % len(TEAM_COLOR_FALLBACKS)]
            team_colors[t] = st.color_picker(f"{t} color", value=default, key=f"team_color_{t}")

        st.caption("3c. Assign each player to a team:")
        for p in players:
            team_of[p] = st.selectbox(f"Team for {p}", options=["(none)"] + team_labels, key=f"team_{p}", label_visibility="collapsed")
        team_of = {p: (t if t != "(none)" else "") for p, t in team_of.items()}

    # --- Entry Builder ---
    st.divider()
    st.header("📝 Entry Builder")
    chars_per_person = st.number_input("4. Characters per player?", min_value=1, max_value=50, value=2, step=1)
    
    build_clicked = st.button("⚙️ Rebuild Entry Table", help="Wipes the table and builds rows based on current Players and Char/Player count.", use_container_width=True)
    shuffle_within_player = st.checkbox("Shuffle 'Character X' labels when auto-filling", value=True)
    auto_fill_clicked = st.button("🎲 Auto-fill Empty Character Slots", help="Fills empty character slots with 'Character 1', 'Character 2', etc.", use_container_width=True)

    # --- General ---
    st.divider()
    clean_rows = st.checkbox("Remove entries with empty Player/Character fields", value=True)


# ---------------------------- Table State & Editor ----------------------------
if "table_df" not in st.session_state:
    st.session_state.table_df = pd.DataFrame([
        {"Player": "You", "Character": "Mario"},
        {"Player": "You", "Character": "Link"},
        {"Player": "Friend1", "Character": "Kirby"},
        {"Player": "Friend1", "Character": "Fox"},
        {"Player": "Friend2", "Character": "Samus"},
    ])

# Logic for sidebar buttons
if build_clicked:
    if not players:
        st.warning("Add at least one player in the sidebar before building entries.")
    else:
        st.session_state.table_df = build_entries_df(players, int(chars_per_person))
        st.session_state.pop("last_bracket", None)
        st.session_state.pop("r1_winners", None)
        st.rerun()

if auto_fill_clicked:
    if not players:
        st.warning("Add players first.")
    else:
        st.session_state.table_df = auto_fill_characters(
            st.session_state.table_df, players, int(chars_per_person), shuffle_within_player
        )

# Ensure 'Player' column uses selected players
if players:
    st.session_state.table_df["Player"] = st.session_state.table_df["Player"].apply(
        lambda p: p if p in players else (players[0] if players and p == "" else p)
    )

st.subheader("📚 Character Entries List")
st.caption(f"Total Entries: {len(st.session_state.table_df)}")

# Data Editor
table_df = st.data_editor(
    st.session_state.table_df,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Player": st.column_config.SelectboxColumn("Player", options=players if players else [], required=True),
        "Character": st.column_config.TextColumn(required=True, help="Enter the name of the character."),
    },
    key="table_editor",
)
entries = df_to_entries(table_df, clean_rows_flag=clean_rows)

# ---------------------------- Generate & show ----------------------------
st.divider()

with st.form("bracket_form", clear_on_submit=False):
    col_gen, col_info, col_clear = st.columns([3, 3, 1])
    
    with col_gen:
        generate_button = st.form_submit_button("🎲 Generate New Bracket", type="primary", use_container_width=True)

    with col_clear:
        clear_button = st.form_submit_button("🧹 Clear All", use_container_width=True)

    if clear_button:
        st.session_state.table_df = pd.DataFrame(columns=["Player", "Character"])
        st.session_state.pop("last_bracket", None)
        st.session_state.pop("r1_winners", None)
        st.rerun()

    if generate_button:
        # Reset winners when generating a new bracket
        st.session_state.pop("r1_winners", None) 
        
        if len(entries) < 2:
            st.error("Add at least 2 valid entries (characters) before generating the bracket.")
        else:
            try:
                if rule == "regular":
                    bracket = generate_bracket_regular(entries)
                else:  # teams
                    bracket = generate_bracket_teams(entries, team_of)

                if not bracket:
                    st.error("Couldn't build a valid round-1 bracket with those constraints. Check if Team mode has enough cross-team players.")
                else:
                    total_real = len(entries)
                    target = next_power_of_two(total_real)
                    need = byes_needed(total_real)
                    
                    st.success(f"Entries: {total_real} → Target: {target} (BYEs: {need}) — Mode: {rule.upper()} Bracket Ready!")

                    st.session_state["last_bracket"] = [(a, b) for (a, b) in bracket]
                    st.session_state["last_rule"] = rule
                    st.session_state["last_team_of"] = team_of if rule == "teams" else {}
                    st.session_state["last_team_colors"] = team_colors if rule == "teams" else {}
                    
            except Exception as e:
                st.error(f"An unexpected error occurred during bracket generation: {e}")
                
# Persist & render compact full bracket
if "last_bracket" in st.session_state and st.session_state["last_bracket"]:
    r1_pairs = st.session_state["last_bracket"]
    
    st.divider()
    
    # R1 Winner Controls
    r1_winner_controls(r1_pairs)
    
    st.divider()
    
    st.subheader("📊 Full Tournament Bracket")
    
    # Compute subsequent rounds based on R1 winners
    rounds = compute_rounds_pairs(r1_pairs, st.session_state.get("r1_winners", {}))
    
    # Render the full bracket grid
    render_bracket_grid(
        rounds, 
        st.session_state.get("last_team_of", {}), 
        st.session_state.get("last_team_colors", {})
    )

st.caption("Tip: Use the sidebar to set players and characters, then click 'Generate New Bracket'.")
