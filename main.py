# app.py — Smash Bracket (No Self-Match) with Sidebar Players + Auto Slots
import streamlit as st
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple
import pandas as pd

st.set_page_config(page_title="Smash Bracket (No Self-Match)", page_icon="🎮", layout="wide")

st.title("🎮 Smash Bracket — No Self-Match in Round 1")
st.markdown(
    """
    Build a **first-round bracket** where **no player's characters face each other**.
    Use the **sidebar** to enter player names and how many characters per person.
    The app can auto-create entry rows (and optionally auto-fill character names from a pool).
    """
)

@dataclass(frozen=True)
class Entry:
    player: str
    character: str

# ---------- Pairing Utilities ----------
def recursive_pairing(entries: List[Entry]) -> Optional[List[Tuple[Entry, Entry]]]:
    """Backtracking perfect matching that avoids same-player pairings.
    Returns list of pairs or None if impossible."""
    n = len(entries)
    if n == 0:
        return []
    a = entries[0]
    for i in range(1, n):
        b = entries[i]
        # Allow BYE to match anyone
        if a.player != b.player or "BYE" in (a.character.upper(), b.character.upper()):
            remaining = entries[1:i] + entries[i+1:]
            rest = recursive_pairing(remaining)
            if rest is not None:
                return [(a, b)] + rest
    return None

def generate_bracket(raw_entries: List[Entry], shuffle_seed: Optional[int] = None):
    entries = raw_entries.copy()
    # If odd, add a BYE that can match anyone
    if len(entries) % 2 == 1:
        entries.append(Entry(player="SYSTEM", character="BYE"))
    rng = random.Random(shuffle_seed)
    # Try multiple shuffles to find a valid matching quickly
    for _ in range(200):
        rng.shuffle(entries)
        result = recursive_pairing(entries)
        if result is not None:
            # Randomize within-pair order for fairness
            result2 = []
            for a, b in result:
                if rng.random() < 0.5:
                    result2.append((b, a))
                else:
                    result2.append((a, b))
            return result2
    return None

# ---------- Sidebar: Players, Slots, Optional Pool ----------
with st.sidebar:
    st.header("Players")
    default_players = "You\nFriend1\nFriend2"
    players_multiline = st.text_area(
        "Enter player names (one per line)",
        value=st.session_state.get("players_multiline", default_players),
        height=140,
        help="These names will appear in the Player dropdown."
    )
    players = [p.strip() for p in players_multiline.splitlines() if p.strip()]
    st.session_state["players_multiline"] = players_multiline

    st.divider()
    st.header("Characters per person")
    chars_per_person = st.number_input(
        "How many characters per player?",
        min_value=1, max_value=50, value=2, step=1
    )

    st.caption("Click the button below to auto-create the entries table.")
    auto_pool = st.text_area(
        "Optional: Character pool (comma-separated; will be assigned round-robin)",
        value="",
        placeholder="Mario, Link, Kirby, Fox, Samus, Pikachu, Jigglypuff, Ness, Captain Falcon"
    )

    seed = st.number_input("Shuffle seed (optional)", value=0,
                           help="Set for reproducible shuffles, or leave 0 to be random.")
    clean_rows = st.checkbox("Remove empty rows", value=True)
    st.caption("Tip: If one player owns more than half of all entries, a valid bracket may be impossible.")

    build_clicked = st.button("⚙️ Auto-Create/Reset Entries", use_container_width=True)

# ---------- Build/Reset Entries Table ----------
def build_entries_df(players: List[str], k: int, pool: List[str]) -> pd.DataFrame:
    rows = []
    # Round-robin through pool if provided; otherwise create blank character slots.
    pool_idx = 0
    total_slots = len(players) * k
    for i in range(k):
        for p in players:
            if pool and pool_idx < len(pool):
                ch = pool[pool_idx].strip()
                pool_idx += 1
            else:
                ch = ""  # leave empty for manual fill
            rows.append({"Player": p, "Character": ch})
    # If pool had fewer names than slots, remaining characters stay blank.
    return pd.DataFrame(rows)

# Initialize table if not present
if "table_df" not in st.session_state:
    st.session_state.table_df = pd.DataFrame([
        {"Player": "You", "Character": "Mario"},
        {"Player": "You", "Character": "Link"},
        {"Player": "Friend1", "Character": "Kirby"},
        {"Player": "Friend1", "Character": "Fox"},
        {"Player": "Friend2", "Character": "Samus"},
    ])

# If user clicked build/reset, rebuild table based on sidebar inputs
if build_clicked:
    if not players:
        st.warning("Add at least one player in the sidebar before building entries.")
    else:
        pool_list = [x for x in [s.strip() for s in auto_pool.split(",")] if x] if auto_pool.strip() else []
        st.session_state.table_df = build_entries_df(players, int(chars_per_person), pool_list)

# Ensure Player values match current players list (or blank if not) if players exist
if players:
    def normalize_player(p):
        p = str(p).strip()
        return p if p in players else (players[0] if p == "" else "")
    st.session_state.table_df["Player"] = st.session_state.table_df["Player"].map(normalize_player)

# ---------- Main: Entries Table & Quick Add ----------
st.subheader("Entries")

# Quick Add Row
with st.container():
    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        add_player = st.selectbox("Player", options=players if players else ["— add players in sidebar —"])
    with c2:
        add_char = st.text_input("Character", placeholder="e.g., Mario")
    with c3:
        if st.button("➕ Add Entry", use_container_width=True, disabled=(not players or not add_char.strip())):
            if players and add_char.strip():
                new_row = {"Player": add_player, "Character": add_char.strip()}
                st.session_state.table_df = pd.concat([st.session_state.table_df, pd.DataFrame([new_row])], ignore_index=True)

# Editable table with Player as a dropdown
table_df = st.data_editor(
    st.session_state.table_df,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Player": st.column_config.SelectboxColumn(
            "Player",
            options=players if players else [],
            required=True
        ),
        "Character": st.column_config.TextColumn(required=True),
    },
    key="table_editor",
)

# Convert DF to Entry list
def df_to_entries(df: pd.DataFrame) -> List[Entry]:
    entries = []
    for _, row in df.iterrows():
        pl = str(row.get("Player", "")).strip()
        ch = str(row.get("Character", "")).strip()
        if clean_rows and (not pl or not ch):
            continue
        if pl and ch:
            entries.append(Entry(player=pl, character=ch))
    return entries

entries = df_to_entries(table_df)

st.divider()
col_gen, col_clear = st.columns([2, 1])

with col_gen:
    if st.button("🎲 Generate Bracket", type="primary"):
        if len(entries) < 2:
            st.error("Add at least 2 entries (characters).")
        else:
            use_seed = None if seed == 0 else int(seed)
            bracket = generate_bracket(entries, shuffle_seed=use_seed)
            if bracket is None:
                st.error("Couldn't build a bracket without self-matches. Try balancing counts across players or keep an odd total to allow a BYE.")
            else:
                st.success("Bracket generated!")
                # Display results
                out_lines = []
                for i, (a, b) in enumerate(bracket, start=1):
                    out_lines.append(f"Match {i}: {a.character} ({a.player})  vs  {b.character} ({b.player})")
                st.code("\n".join(out_lines), language="text")

                # CSV download
                import io, csv
                buffer = io.StringIO()
                writer = csv.writer(buffer)
                writer.writerow(["Match", "Player A", "Character A", "Player B", "Character B"])
                for i, (a, b) in enumerate(bracket, start=1):
                    writer.writerow([i, a.player, a.character, b.player, b.character])
                st.download_button(
                    label="⬇️ Download bracket as CSV",
                    data=buffer.getvalue().encode("utf-8"),
                    file_name="smash_bracket_round1.csv",
                    mime="text/csv",
                )

with col_clear:
    if st.button("🧹 Clear Table"):
        st.session_state.table_df = pd.DataFrame(columns=["Player", "Character"])
        st.rerun()

st.divider()
st.markdown(
    """
    **Notes**
    - Use the **sidebar** to maintain player names and pick **characters per person**; click **Auto-Create/Reset Entries** to generate slots.
    - You can paste a **comma-separated character pool**; the app will assign names in round-robin order across players. Leave blank to generate empty slots.
    - If you have an **odd number** of characters, the app adds a **BYE** so no one fights themselves.
    - If one player has **more than half** the total characters, a valid no-self-match bracket may be **impossible**.
    - Use the **seed** setting for a reproducible shuffle.
    """
)
