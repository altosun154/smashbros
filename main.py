# app.py — Smash Bracket (No Self-Match) with Auto Player Slots + Auto-Fill Characters
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

    1) In the **sidebar**, enter player names and how many characters per player.  
    2) Click **Auto-Create/Reset Entries** to make the slots.  
    3) Click **🎲 Auto-fill Characters** to automatically set names like **Character 1, Character 2, ...** per player.  
    4) Press **Generate Bracket**.
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

# ---------- Sidebar: Players, Slots, Optional Pool & Auto-Fill ----------
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

    st.caption("Click **Auto-Create/Reset Entries** to generate rows for everyone.")
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

    st.divider()
    st.subheader("Auto-fill Characters")
    shuffle_within_player = st.checkbox("Shuffle numbers within each player's slots", value=True,
                                        help="If on: the order of 'Character 1..k' is randomized per player.")
    auto_fill_clicked = st.button("🎲 Auto-fill Characters", use_container_width=True)

# ---------- Builders ----------
def build_entries_df(players: List[str], k: int, pool: List[str]) -> pd.DataFrame:
    rows = []
    pool_idx = 0
    for i in range(k):
        for p in players:
            if pool and pool_idx < len(pool):
                ch = pool[pool_idx].strip()
                pool_idx += 1
            else:
                ch = ""  # leave empty for manual fill or auto-fill button
            rows.append({"Player": p, "Character": ch})
    return pd.DataFrame(rows)

def auto_fill_characters(df: pd.DataFrame, players: List[str], k: int, shuffle_each: bool) -> pd.DataFrame:
    # For each player, set that player's rows' Character to "Character 1..k"
    out = df.copy()
    for p in players:
        mask = (out["Player"] == p)
        idxs = list(out.index[mask])
        # Ensure we only number up to the count of that player's rows
        count = len(idxs)
        labels = [f"Character {i+1}" for i in range(min(k, count))]
        # If player has more than k rows (manual adds), extend numbering
        if count > k:
            labels = [f"Character {i+1}" for i in range(count)]
        if shuffle_each:
            random.shuffle(labels)
        for row_i, label in zip(idxs, labels):
            out.at[row_i, "Character"] = label
    return out

# ---------- State: Entries Table ----------
if "table_df" not in st.session_state:
    st.session_state.table_df = pd.DataFrame([
        {"Player": "You", "Character": "Mario"},
        {"Player": "You", "Character": "Link"},
        {"Player": "Friend1", "Character": "Kirby"},
        {"Player": "Friend1", "Character": "Fox"},
        {"Player": "Friend2", "Character": "Samus"},
    ])

# Build/reset from sidebar
if build_clicked:
    if not players:
        st.warning("Add at least one player in the sidebar before building entries.")
    else:
        pool_list = [x for x in [s.strip() for s in auto_pool.split(",")] if x] if auto_pool.strip() else []
        st.session_state.table_df = build_entries_df(players, int(chars_per_person), pool_list)

# Auto-fill characters button
if auto_fill_clicked:
    if not players:
        st.warning("Add players first.")
    else:
        st.session_state.table_df = auto_fill_characters(
            st.session_state.table_df, players, int(chars_per_person), shuffle_within_player
        )

# Keep Player values aligned with current players (if provided)
if players:
    def normalize_player(p):
        p = str(p).strip()
        return p if p in players else (players[0] if p == "" else "")
    st.session_state.table_df["Player"] = st.session_state.table_df["Player"].map(normalize_player)

# ---------- Main: Entries Table & Quick Add ----------
st.subheader("Entries")

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
    - Use the **sidebar** to set **players** and **characters per player**; click **Auto-Create/Reset Entries**.
    - Click **🎲 Auto-fill Characters** to auto-name each player's slots (e.g., *Character 1..k*), with optional shuffle per player.
    - You can also paste a **character pool**; the reset button assigns them round-robin across players.
    - If you have an **odd number** of characters, the app adds a **BYE** so no one fights themselves.
    - If one player has **more than half** the total characters, a valid no-self-match bracket may be **impossible**.
    - Use the **seed** setting for a reproducible shuffle.
    """
)
st.caption("Made with ❤️ for quick living-room brackets.")
