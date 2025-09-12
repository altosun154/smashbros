# app.py — Smash Bracket: regular / first_pick / groups / everything (more randomness, spacing preserved)
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
    """Get unique players and randomize their global order once per run."""
    seen = []
    for e in entries:
        if e.player != "SYSTEM" and e.player not in seen:
            seen.append(e.player)
    random.shuffle(seen)  # one-time global shuffle
    return seen

def build_rings(entries: List[Entry]) -> Tuple[List[List[Entry]], List[str]]:
    """
    Build rings by slot with fixed player order:
      ring[0] = all Slot 1 entries in player order,
      ring[1] = all Slot 2 entries in the SAME order, etc.
    Keeping the same order across rings guarantees spacing by #players.
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

# ---- Groups pairing with alternating parity for more variety (spacing preserved) ----
def pair_within_ring_alternating(ring: List[Entry], start_parity: int, carry: Optional[Entry]) -> Tuple[List[Tuple[Entry, Entry]], Optional[Entry]]:
    """
    Pair neighbors in ring using starting parity 0 or 1:
      parity=0 -> pairs (0,1),(2,3),...
      parity=1 -> pairs (1,2),(3,4),... (wrap the last with 0 if needed)
    Carry, if present, pairs with the first eligible element.
    """
    pairs: List[Tuple[Entry, Entry]] = []
    items = ring.copy()

    # If there's a carry from the previous ring, pair it with the first item (rotate once if needed)
    if carry is not None and items:
        if carry.player == items[0].player and carry.player != "SYSTEM" and len(items) > 1:
            # rotate one step to avoid same-player clash
            items = items[1:] + items[:1]
        pairs.append((carry, items[0]))
        items = items[1:]
        carry = None

    n = len(items)
    if n == 0:
        return pairs, carry
    if n == 1:
        # nothing to pair with -> becomes new carry
        return pairs, items[0]

    # Build index order based on parity
    idxs = list(range(n))
    if start_parity % 2 == 1:
        idxs = idxs[1:] + idxs[:1]  # shift by 1

    i = 0
    while i + 1 < len(idxs):
        a = items[idxs[i]]
        b = items[idxs[i+1]]
        if a.player == b.player and len(idxs) >= 3:
            # try swap next index to avoid same-player
            idxs[i+1], idxs[(i+2) % len(idxs)] = idxs[(i+2) % len(idxs)], idxs[i+1]
            b = items[idxs[i+1]]
        pairs.append((a, b))
        i += 2

    # Leftover if odd count
    if i < len(idxs):
        carry = items[idxs[i]]

    return pairs, carry

def generate_bracket_groups(entries: List[Entry]) -> List[Tuple[Entry, Entry]]:
    rings, _ = build_rings(entries)
    pairs: List[Tuple[Entry, Entry]] = []
    carry: Optional[Entry] = None
    # Alternate pairing parity per ring to vary who meets whom
    for ring_idx, ring in enumerate(rings):
        start_parity = random.randint(0, 1)  # 0 or 1, per ring
        ring_pairs, carry = pair_within_ring_alternating(ring, start_parity, carry)
        pairs.extend(ring_pairs)
    if carry is not None:
        pairs.append((carry, Entry(player="SYSTEM", character="BYE", slot=0)))
    return pairs

# ---- EVERYTHING: groups spacing + random cross-ring offset (no Slot1 vs Slot1) ----
def generate_bracket_everything(entries: List[Entry]) -> List[Tuple[Entry, Entry]]:
    """
    Keep the Groups seed order (rings built with same player order = spacing kept).
    Pair ring0 vs ring1, ring2 vs ring3, ... using a RANDOM offset k in [1, P-1]:
      b = ring_b[(j + k) % len(ring_b)]
    This guarantees:
      - no same-player (k != 0),
      - Slot1 (ring0) never faces Slot1 (ring1),
      - more variety across runs.
    If odd number of rings, the last ring is paired internally (with alternating parity).
    """
    rings, p_order = build_rings(entries)
    P = len(p_order)

    pairs: List[Tuple[Entry, Entry]] = []
    i = 0
    while i + 1 < len(rings):
        ring_a, ring_b = rings[i], rings[i+1]
        nA, nB = len(ring_a), len(ring_b)
        n = min(nA, nB)
        if n == 0:
            i += 2
            continue

        # choose random offset k ∈ [1, nB-1] (if nB==1, we can't offset -> pair with BYE)
        if nB > 1:
            k = random.randint(1, nB - 1)
        else:
            k = 0

        for j in range(n):
            a = ring_a[j]
            if nB > 1:
                b = ring_b[(j + k) % nB]
            else:
                b = Entry("SYSTEM", "BYE", 0)
            # Safety guard (very rare with k!=0)
            if a.player == b.player and a.player != "SYSTEM":
                # try another offset quickly
                alt = (k + 1) % max(nB, 1)
                if nB > 1:
                    b = ring_b[(j + alt) % nB]
                else:
                    b = Entry("SYSTEM", "BYE", 0)
            pairs.append((a, b))

        # any leftovers -> BYEs
        if nA > n:
            for t in range(n, nA):
                pairs.append((ring_a[t], Entry("SYSTEM", "BYE", 0)))
        if nB > n:
            for t in range(n, nB):
                pairs.append((Entry("SYSTEM", "BYE", 0), ring_b[t]))
        i += 2

    if i < len(rings):  # leftover ring -> internal pairing with alternating parity
        start_parity = random.randint(0, 1)
        tail_pairs, carry = pair_within_ring_alternating(rings[i], start_parity, carry=None)
        pairs.extend(tail_pairs)
        if carry is not None:
            pairs.append((carry, Entry(player="SYSTEM", character="BYE", slot=0)))

    return pairs

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
            "groups: keeps spacing by #players; alternating parity per ring for variety\n"
            "everything: keeps spacing; pairs ring0 vs ring1 (random offset), ring2 vs ring3, ..."
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
    - **groups**: fixed round-robin seeding (spacing by #players) with **alternating parity per ring** for variety.
    - **everything**: same spacing; **ring0 vs ring1** (random offset k∈[1,P-1]), **ring2 vs ring3**, … to avoid Slot1 vs Slot1 and add variety.
    """
)
st.caption("Made with ❤️ for quick living-room brackets.")
