# app.py — Smash Bracket with Rule Sets: regular / first_pick / groups / everything (groups-first, then targeted Slot1 fix)
import streamlit as st
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple
import pandas as pd

st.set_page_config(page_title="Smash Bracket (No Self-Match)", page_icon="🎮", layout="wide")

st.title("🎮 Smash Bracket — No Self-Match in Round 1")
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

# ---------- Pairing for 'regular' / 'first_pick' ----------
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
        # 1st pick protection
        if rule == "first_pick" and a.slot == 1 and b.slot == 1:
            continue
        remaining = entries[1:i] + entries[i+1:]
        rest = recursive_pairing(remaining, rule)
        if rest is not None:
            return [(a, b)] + rest
    return None

def generate_bracket_regular_or_firstpick(raw_entries: List[Entry], rule: str):
    entries = raw_entries.copy()
    if len(entries) % 2 == 1:
        entries.append(Entry(player="SYSTEM", character="BYE", slot=0))
    for _ in range(400):
        random.shuffle(entries)
        result = recursive_pairing(entries, rule)
        if result is not None:
            out = []
            for a, b in result:
                if random.random() < 0.5:
                    out.append((b, a))
                else:
                    out.append((a, b))
            return out
    return None

# ---------- Round-robin seeding helpers (used by groups/everything) ----------
def players_from_entries(entries: List[Entry]) -> List[str]:
    seen = []
    for e in entries:
        if e.player != "SYSTEM" and e.player not in seen:
            seen.append(e.player)
    random.shuffle(seen)  # one-time shuffle of player order for fairness
    return seen

def build_rings(entries: List[Entry]) -> Tuple[List[List[Entry]], List[str]]:
    """
    Build rings by slot with fixed player order:
      ring[0] = all Slot 1 entries in player order,
      ring[1] = all Slot 2 entries in same order, etc.
    No shuffling inside rings -> guarantees spacing by #players across the seed order.
    """
    by_player_slot = {}
    max_slot = 0
    for e in entries:
        if e.player == "SYSTEM":
            continue
        by_player_slot.setdefault((e.player, e.slot), []).append(e)
        max_slot = max(max_slot, e.slot)

    p_order = players_from_entries(entries)
    rings: List[List[Entry]] = []
    for s in range(1, max_slot + 1):
        ring = []
        for p in p_order:
            es = by_player_slot.get((p, s), [])
            if es:
                ring.append(es[0])
        rings.append(ring)
    return rings, p_order

def pair_within_ring(ring: List[Entry], carry: Optional[Entry]) -> Tuple[List[Tuple[Entry, Entry]], Optional[Entry]]:
    pairs: List[Tuple[Entry, Entry]] = []
    items = ring.copy()

    if carry is not None:
        first = items.pop(0) if items else None
        if first is None:
            return pairs, carry
        if carry.player == first.player and carry.player != "SYSTEM":
            if items:
                first, items[0] = items[0], first
        pairs.append((carry, first))
        carry = None

    i = 0
    while i + 1 < len(items):
        a, b = items[i], items[i+1]
        if a.player == b.player and i + 2 < len(items):
            items[i+1], items[i+2] = items[i+2], items[i+1]
            b = items[i+1]
        pairs.append((a, b))
        i += 2

    leftover = items[i] if i < len(items) else None
    if leftover is not None:
        carry = leftover

    return pairs, carry

def generate_bracket_groups(entries: List[Entry]) -> List[Tuple[Entry, Entry]]:
    rings, _ = build_rings(entries)
    pairs: List[Tuple[Entry, Entry]] = []
    carry: Optional[Entry] = None
    for ring in rings:
        ring_pairs, carry = pair_within_ring(ring, carry)
        pairs.extend(ring_pairs)
    if carry is not None:
        pairs.append((carry, Entry(player="SYSTEM", character="BYE", slot=0)))
    return pairs

# ---------- EVERYTHING: groups first, then targeted Slot-1 fix ----------
def fix_first_pick_conflicts(bracket: List[Tuple[Entry, Entry]]) -> List[Tuple[Entry, Entry]]:
    """
    For any match where both sides are Slot 1 (different players),
    swap one Slot-1 with a non-Slot-1 from the SAME player, but only if that other match's opponent is not Slot-1.
    Keep trying up to a few passes.
    """
    def is_slot1(e: Entry) -> bool:
        return e.slot == 1 and e.player != "SYSTEM"

    def safe_swap_ok(a_entry: Entry, a_opp: Entry, b_entry: Entry, b_opp: Entry) -> bool:
        # After swapping a_entry with b_entry, ensure no same-player clashes are created.
        return (a_entry.player != b_opp.player) and (b_entry.player != a_opp.player)

    for _ in range(5):  # a few cleanup passes
        changed = False
        # Build quick index for a player's appearances
        appearances = {}  # player -> list of (match_idx, side, entry, opp)
        for idx, (x, y) in enumerate(bracket):
            appearances.setdefault(x.player, []).append((idx, "A", x, y))
            appearances.setdefault(y.player, []).append((idx, "B", y, x))

        # Find conflicts
        conflicts = [i for i, (x, y) in enumerate(bracket) if is_slot1(x) and is_slot1(y) and x.player != y.player]
        if not conflicts:
            break

        for i in conflicts:
            a, b = bracket[i]  # both slot 1
            # try to fix by swapping 'a' (slot1) with another of a.player
            fixed_here = False
            for (j, side, entry, opp) in appearances.get(a.player, []):
                if j == i:
                    continue
                # need a non-slot1 entry and its opponent not slot1
                if entry.slot != 1 and not is_slot1(opp):
                    # check safe to swap: 'entry' would face b. 'a' would face 'opp'
                    if safe_swap_ok(entry, opp, a, b):
                        # do the swap
                        if side == "A":
                            # swap bracket[i].A with bracket[j].A
                            bracket[i] = (entry, b)
                            bracket[j] = (a, bracket[j][1])
                        else:
                            bracket[i] = (entry, b)
                            bracket[j] = (bracket[j][0], a)
                        changed = True
                        fixed_here = True
                        break
            if fixed_here:
                continue
            # try fixing by swapping 'b' instead
            for (j, side, entry, opp) in appearances.get(b.player, []):
                if j == i:
                    continue
                if entry.slot != 1 and not is_slot1(opp):
                    if safe_swap_ok(entry, opp, b, a):
                        if side == "A":
                            bracket[i] = (a, entry)
                            bracket[j] = (b, bracket[j][1])
                        else:
                            bracket[i] = (a, entry)
                            bracket[j] = (bracket[j][0], b)
                        changed = True
                        fixed_here = True
                        break
            # if neither side could be fixed, leave as-is for this pass

        if not changed:
            break
    return bracket

def generate_bracket_everything(entries: List[Entry]) -> List[Tuple[Entry, Entry]]:
    """
    1) Build bracket with **Groups** (keeps spacing by #players).
    2) Post-process: for any Slot1 vs Slot1 match, swap that Slot1 with a non-Slot1 from the same player
       whose opponent is also not Slot1. This avoids Slot1 vs Slot1 without breaking spacing or causing self-matches.
    """
    base = generate_bracket_groups(entries)
    fixed = fix_first_pick_conflicts(base)
    return fixed

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
        options=["regular", "first_pick", "groups", "everything"],
        index=0,
        help=(
            "regular: no self-match in round 1\n"
            "first_pick: regular + forbids Slot 1 vs Slot 1\n"
            "groups: fixed round-robin seeding; a player's entries are spaced by #players\n"
            "everything: groups first, then swap away any Slot1-vs-Slot1 by trading with a same-player non-Slot1 whose opponent isn't Slot1"
        )
    )

    st.divider()
    st.subheader("Build / Fill")
    build_clicked = st.button("⚙️ Auto-Create/Reset Entries", use_container_width=True)

    shuffle_within_player = st.checkbox("Shuffle numbers within each player's slots (for auto-fill)", value=True)
    auto_fill_clicked = st.button("🎲 Auto-fill Characters (Character 1..k)", use_container_width=True)

    st.divider()
    st.header("General")
    clean_rows = st.checkbox("Remove empty rows", value=True)
    st.caption("Tip: If one player owns more than half of all entries, a valid no-self-match bracket may be impossible.")

# ---------- Helpers (robust Slot handling) ----------
def assign_slots(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Slot" not in out.columns:
        out["Slot"] = 0
    if not isinstance(out["Slot"], pd.Series):
        out["Slot"] = pd.Series([0] * len(out), index=out.index)
    out["Slot"] = pd.to_numeric(out["Slot"], errors="coerce").fillna(0).astype(int)
    if "Player" not in out.columns:
        out["Player"] = ""
    for p, grp in out.groupby("Player", sort=False, dropna=False):
        idxs = list(grp.index)
        for j, idx in enumerate(idxs, start=1):
            out.at[idx, "Slot"] = j
    return out

def build_entries_df(players: List[str], k: int) -> pd.DataFrame:
    rows = []
    for i in range(1, k + 1):  # slot 1..k
        for p in players:
            rows.append({"Player": p, "Character": "", "Slot": i})
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
    df2 = assign_slots(df)
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
        st.session_state.table_df = build_entries_df(players, int(chars_per_person))

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
            if rule == "regular":
                bracket = generate_bracket_regular_or_firstpick(entries, "regular")
            elif rule == "first_pick":
                bracket = generate_bracket_regular_or_firstpick(entries, "first_pick")
            elif rule == "groups":
                bracket = generate_bracket_groups(entries)
            else:  # everything
                bracket = generate_bracket_everything(entries)

            if bracket is None:
                st.error("Couldn't build a valid round-1 bracket with those constraints. Try balancing counts or allowing a BYE (odd total).")
            else:
                st.success(f"Bracket generated using rule: {rule}")
                out_lines = []
                for i, (a, b) in enumerate(bracket, start=1):
                    out_lines.append(
                        f"Match {i}: {a.character} ({a.player}, Slot {a.slot})  vs  {b.character} ({b.player}, Slot {b.slot})"
                    )
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
    - **first_pick**: regular **+** forbids **Slot 1 vs Slot 1** in Round 1.
    - **groups**: fixed round-robin seeding; a player's entries are spaced by **#players**; pairs within each ring.
    - **everything**: **groups first**, then if any **Slot1 vs Slot1**, swap that Slot1 with a **non-Slot1** from the same player whose opponent is **not** Slot1.
    """
)
