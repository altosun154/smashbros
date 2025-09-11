# app.py — Smash Bracket with Rule Sets: Regular / 1st pick / Groups (robust Slot handling)
import streamlit as st
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple
import pandas as pd

st.set_page_config(page_title="Smash Bracket maker", page_icon="🎮", layout="wide")

st.title("🎮 Smash Bros Bracket Maker")
st.markdown(
    """
    Use the **sidebar** to add players, set **characters per player**, pick a **Rule Set**, then
    **Auto-Create/Reset Entries** and **Auto-fill Characters**. Generate a round-1 bracket with constraints.
    """
)

@dataclass(frozen=True)
class Entry:
    player: str
    character: str
    slot: int  # per-player slot index (1 = first pick)

# ---------- Pairing utilities ----------
def recursive_pairing(entries: List[Entry], rule: str) -> Optional[List[Tuple[Entry, Entry]]]:
    n = len(entries)
    if n == 0:
        return []
    a = entries[0]
    for i in range(1, n):
        b = entries[i]
        # BYE can match anyone
        if "BYE" in (a.character.upper(), b.character.upper()):
            remaining = entries[1:i] + entries[i+1:]
            rest = recursive_pairing(remaining, rule)
            if rest is not None:
                return [(a, b)] + rest
            continue
        # Base rule: no same-player
        if a.player == b.player:
            continue
        # 1st pick rule: disallow Slot 1 vs Slot 1
        if rule == "first_pick" and a.slot == 1 and b.slot == 1:
            continue
        remaining = entries[1:i] + entries[i+1:]
        rest = recursive_pairing(remaining, rule)
        if rest is not None:
            return [(a, b)] + rest
    return None

def generate_bracket_regular_or_firstpick(raw_entries: List[Entry], shuffle_seed: Optional[int], rule: str):
    entries = raw_entries.copy()
    if len(entries) % 2 == 1:
        entries.append(Entry(player="SYSTEM", character="BYE", slot=0))
    rng = random.Random(shuffle_seed)
    for _ in range(300):
        rng.shuffle(entries)
        result = recursive_pairing(entries, rule)
        if result is not None:
            out = []
            for a, b in result:
                if rng.random() < 0.5:
                    out.append((b, a))
                else:
                    out.append((a, b))
            return out
    return None

def generate_bracket_groups(raw_entries: List[Entry], shuffle_seed: Optional[int]):
    entries = raw_entries.copy()
    rng = random.Random(shuffle_seed)
    by_player = {}
    for e in entries:
        by_player.setdefault(e.player, []).append(e)
    players = [p for p in by_player.keys() if p != "SYSTEM"]
    rng.shuffle(players)
    for p in players:
        by_player[p].sort(key=lambda x: x.slot)
    seed_order: List[Entry] = []
    max_k = max((len(by_player[p]) for p in players), default=0)
    for i in range(max_k):
        round_entries = []
        for p in players:
            if i < len(by_player[p]):
                round_entries.append(by_player[p][i])
        rng.shuffle(round_entries)
        seed_order.extend(round_entries)
    if len(seed_order) % 2 == 1:
        seed_order.append(Entry(player="SYSTEM", character="BYE", slot=0))
    bracket = []
    for i in range(0, len(seed_order), 2):
        a, b = seed_order[i], seed_order[i+1]
        if a.player == b.player and i + 2 < len(seed_order):
            seed_order[i+1], seed_order[i+2] = seed_order[i+2], seed_order[i+1]
            b = seed_order[i+1]
        bracket.append((a, b))
    return bracket

# ---------- Sidebar ----------
with st.sidebar:
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

    st.divider()
    st.header("Characters per player")
    chars_per_person = st.number_input("How many per player?", min_value=1, max_value=50, value=2, step=1)

    st.divider()
    st.header("Rule Set")
    rule = st.selectbox(
        "Choose rule",
        options=["regular", "first_pick", "groups"],
        index=0,
        help=(
            "regular: no self-match in round 1\n"
            "first_pick: also prevents Slot 1 vs Slot 1 in round 1\n"
            "groups: seeds by round-robin so same player's entries are spaced by #players"
        )
    )

    st.divider()
    st.subheader("Build / Fill")
    auto_pool = st.text_area(
        "Optional character pool (comma-separated) — used when building",
        value="",
        placeholder="Mario, Link, Kirby, Fox, Samus, Pikachu, Jigglypuff"
    )
    build_clicked = st.button("⚙️ Auto-Create/Reset Entries", use_container_width=True)

    shuffle_within_player = st.checkbox("Shuffle numbers within each player's slots (for auto-fill)", value=True)
    auto_fill_clicked = st.button("🎲 Auto-fill Characters (Character 1..k)", use_container_width=True)

    st.divider()
    st.header("General")
    seed = st.number_input("Shuffle seed (optional)", value=0, help="Set for reproducible results; 0 = random.")
    clean_rows = st.checkbox("Remove empty rows", value=True)
    st.caption("Tip: If one player owns more than half of all entries, a valid no-self-match bracket may be impossible.")

# ---------- Helpers (FIXED) ----------
def assign_slots(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure each player's rows have Slot = 1..n based on current order.
    Robust even if 'Slot' column is missing or malformed.
    """
    out = df.copy()
    # If Slot is missing, create it with zeros of correct length
    if "Slot" not in out.columns:
        out["Slot"] = 0
    # Coerce to numeric safely (works for Series; if somehow scalar slipped in, broadcast)
    if not isinstance(out["Slot"], pd.Series):
        out["Slot"] = pd.Series([0] * len(out), index=out.index)
    out["Slot"] = pd.to_numeric(out["Slot"], errors="coerce").fillna(0).astype(int)

    # Assign per-player 1..n in displayed order
    if "Player" not in out.columns:
        out["Player"] = ""
    for p, grp in out.groupby("Player", sort=False, dropna=False):
        idxs = list(grp.index)
        for j, idx in enumerate(idxs, start=1):
            out.at[idx, "Slot"] = j
    return out

def build_entries_df(players: List[str], k: int, pool: List[str]) -> pd.DataFrame:
    rows = []
    pool_idx = 0
    for i in range(1, k + 1):  # slot 1..k
        for p in players:
            if pool and pool_idx < len(pool):
                ch = pool[pool_idx].strip()
                pool_idx += 1
            else:
                ch = ""  # leave empty for auto-fill or manual edit
            rows.append({"Player": p, "Character": ch, "Slot": i})
    return pd.DataFrame(rows)

def auto_fill_characters(df: pd.DataFrame, players: List[str], k: int, shuffle_each: bool) -> pd.DataFrame:
    out = assign_slots(df)
    for p in players:
        mask = (out["Player"] == p)
        idxs = list(out.index[mask])
        idxs.sort(key=lambda i: int(out.at[i, "Slot"]) if str(out.at[i, "Slot"]).isdigit() else 9999)
        labels = [f"Character {i+1}" for i in range(len(idxs))]
        if shuffle_each:
            random.shuffle(labels)
        for row_i, label in zip(idxs, labels):
            out.at[row_i, "Character"] = label
    return out

def df_to_entries(df: pd.DataFrame, clean_rows_flag: bool) -> List[Entry]:
    df2 = assign_slots(df)  # <- robust now
    entries: List[Entry] = []
    for _, row in df2.iterrows():
        pl = str(row.get("Player", "")).strip()
        ch = str(row.get("Character", "")).strip()
        try:
            sl = int(row.get("Slot", 0))
        except Exception:
            sl = 0
        if clean_rows_flag and (not pl or not ch):
            continue
        if pl and ch:
            entries.append(Entry(player=pl, character=ch, slot=sl))
    return entries

# ---------- State: Entries Table ----------
if "table_df" not in st.session_state:
    st.session_state.table_df = pd.DataFrame([
        {"Player": "You", "Character": "Mario", "Slot": 1},
        {"Player": "You", "Character": "Link", "Slot": 2},
        {"Player": "Friend1", "Character": "Kirby", "Slot": 1},
        {"Player": "Friend1", "Character": "Fox", "Slot": 2},
        {"Player": "Friend2", "Character": "Samus", "Slot": 1},
    ])

# Build/reset from sidebar
if build_clicked:
    if not players:
        st.warning("Add at least one player in the sidebar before building entries.")
    else:
        pool_list = [x for x in [s.strip() for s in auto_pool.split(",")] if x] if auto_pool.strip() else []
        st.session_state.table_df = build_entries_df(players, int(chars_per_person), pool_list)

# Auto-fill characters
if auto_fill_clicked:
    if not players:
        st.warning("Add players first.")
    else:
        st.session_state.table_df = auto_fill_characters(
            st.session_state.table_df, players, int(chars_per_person), shuffle_within_player
        )

# Normalize Player values if players list exists
if players:
    def normalize_player(p):
        p = str(p).strip()
        return p if p in players else (players[0] if p == "" else "")
    st.session_state.table_df["Player"] = st.session_state.table_df["Player"].map(normalize_player)

# ---------- Entries editor ----------
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
                df = assign_slots(st.session_state.table_df)
                next_slot = 1
                existing_for_player = df[df["Player"] == add_player]
                if not existing_for_player.empty:
                    next_slot = int(existing_for_player["Slot"].max()) + 1
                new_row = {"Player": add_player, "Character": add_char.strip(), "Slot": next_slot}
                st.session_state.table_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

table_df = st.data_editor(
    st.session_state.table_df,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Player": st.column_config.SelectboxColumn("Player", options=players if players else [], required=True),
        "Character": st.column_config.TextColumn(required=True),
        "Slot": st.column_config.NumberColumn("Slot", min_value=1, step=1, help="Per-player slot (1 = first pick)"),
    },
    key="table_editor",
)

entries = df_to_entries(table_df, clean_rows_flag=clean_rows)

# ---------- Generate bracket ----------
st.divider()
col_gen, col_clear = st.columns([2, 1])

with col_gen:
    if st.button("🎲 Generate Bracket", type="primary"):
        if len(entries) < 2:
            st.error("Add at least 2 entries (characters).")
        else:
            use_seed = None if seed == 0 else int(seed)
            if rule in ("regular", "first_pick"):
                bracket = generate_bracket_regular_or_firstpick(entries, use_seed, rule)
            else:
                bracket = generate_bracket_groups(entries, use_seed)

            if bracket is None:
                st.error("Couldn't build a valid round-1 bracket with those constraints. Try balancing counts or allowing a BYE (odd total).")
            else:
                st.success(f"Bracket generated using rule: {rule}")
                out_lines = []
                for i, (a, b) in enumerate(bracket, start=1):
                    out_lines.append(f"Match {i}: {a.character} ({a.player}, Slot {a.slot})  vs  {b.character} ({b.player}, Slot {b.slot})")
                st.code("\n".join(out_lines), language="text")

                # CSV download
                import io, csv
                buffer = io.StringIO()
                writer = csv.writer(buffer)
                writer.writerow(["Match", "Player A", "Character A", "Slot A", "Player B", "Character B", "Slot B"])
                for i, (a, b) in enumerate(bracket, start=1):
                    writer.writerow([i, a.player, a.character, a.slot, b.player, b.character, b.slot])
                st.download_button(
                    label="⬇️ Download bracket as CSV",
                    data=buffer.getvalue().encode("utf-8"),
                    file_name=f"smash_bracket_round1_{rule}.csv",
                    mime="text/csv",
                )

with col_clear:
    if st.button("🧹 Clear Table"):
        st.session_state.table_df = pd.DataFrame(columns=["Player", "Character", "Slot"])
        st.rerun()

st.divider()
st.markdown(
    """
    **Rule Set details**
    - **regular**: random bracket; forbids **same-player** matches in Round 1.
    - **first_pick**: same as regular **plus** forbids **Slot 1 vs Slot 1** in Round 1.
    - **groups**: seeds by round-robin (**1..k across players**) so a player's entries are spaced by **#players** in the seed order; then pairs neighbors.
    """
)
st.caption("Made with ❤️ for quick living-room brackets.")
