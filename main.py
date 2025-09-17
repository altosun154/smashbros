# app.py — Smash Bracket: regular / first_pick / groups (balanced; power-of-2 BYEs) / everything
import streamlit as st
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import pandas as pd
import math

st.set_page_config(page_title="Smash Bracket (No Self-Match)", page_icon="🎮", layout="wide")

st.title("🎮 Smash Bracket — No Self-Match in Round 1")
st.markdown("Pick a rule set, auto-fill entries, and generate a round-1 bracket that fits a **power-of-two** bracket size (with BYEs).")

@dataclass(frozen=True)
class Entry:
    player: str
    character: str
    slot: int  # per-player slot index (1 = first pick)

# ---------- Power-of-two helpers ----------
def next_power_of_two(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()

def byes_needed(n: int) -> int:
    return max(0, next_power_of_two(n) - n)

# ---------- Regular / First-pick (backtracking) ----------
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
        # Base: no same player
        if a.player == b.player:
            continue
        # 1st pick guard
        if rule == "first_pick" and a.slot == 1 and b.slot == 1:
            continue
        remaining = entries[1:i] + entries[i+1:]
        rest = recursive_pairing(remaining, rule)
        if rest is not None:
            return [(a, b)] + rest
    return None

def generate_bracket_regular_or_firstpick(raw_entries: List[Entry], rule: str):
    entries = raw_entries.copy()
    # Add exactly the required BYEs up front
    need = byes_needed(len(entries))
    for _ in range(need):
        entries.append(Entry(player="SYSTEM", character="BYE", slot=0))
    # Find a valid matching
    for _ in range(500):
        random.shuffle(entries)
        result = recursive_pairing(entries, rule)
        if result is not None:
            # Randomize within-pair order
            out = []
            for a, b in result:
                if random.random() < 0.5:
                    out.append((b, a))
                else:
                    out.append((a, b))
            return out
    return None

# ---------- Groups / Everything (balanced-random with *exact* BYE distribution) ----------
def players_from_entries(entries: List[Entry]) -> List[str]:
    seen = []
    for e in entries:
        if e.player != "SYSTEM" and e.player not in seen:
            seen.append(e.player)
    random.shuffle(seen)  # single global shuffle for variety
    return seen

def build_rings(entries: List[Entry]) -> Tuple[List[List[Entry]], List[str]]:
    """ring[0]=all Slot1 entries (one per player), ring[1]=all Slot2, ..."""
    max_slot = 0
    by_player_slot: Dict[Tuple[str, int], Entry] = {}
    for e in entries:
        if e.player == "SYSTEM":
            continue
        by_player_slot[(e.player, e.slot)] = e
        max_slot = max(max_slot, e.slot)
    order = players_from_entries(entries)
    rings: List[List[Entry]] = []
    for s in range(1, max_slot + 1):
        ring = [by_player_slot[(p, s)] for p in order if (p, s) in by_player_slot]
        rings.append(ring)
    return rings, order

def pick_from_lowest_tally(cands: List[Entry], tally: Dict[str, int], exclude_player: Optional[str] = None) -> Optional[Entry]:
    pool = [e for e in cands if e.player != exclude_player]
    if not pool:
        return None
    m = min(tally.get(e.player, 0) for e in pool)
    lowest = [e for e in pool if tally.get(e.player, 0) == m]
    return random.choice(lowest)

def pair_ring_balanced_with_bye_quota(
    ring: List[Entry],
    tally: Dict[str, int],
    carry: Optional[Entry],
    bye_quota: int
) -> Tuple[List[Tuple[Entry, Entry]], Optional[Entry], int]:
    """
    Pair inside a ring using balanced-random tallies.
    Use up to bye_quota BYEs inside this ring.
     - If a carry exists and we still have BYEs -> pair carry with BYE.
     - Otherwise pair carry against a lowest-tally opponent (not same player).
     - For remaining items, repeatedly:
         * pick 'a' from lowest-tally at random,
         * if bye_quota>0 -> (a, BYE) and consume one,
         * else pick 'b' from lowest-tally (not same player).
    Returns (pairs, new_carry, bye_used_left_for_this_ring).
    """
    pairs: List[Tuple[Entry, Entry]] = []
    items = ring.copy()

    # 1) Handle carry first
    if carry is not None:
        if bye_quota > 0:
            pairs.append((carry, Entry("SYSTEM", "BYE", 0)))
            tally[carry.player] = tally.get(carry.player, 0) + 1
            bye_quota -= 1
            carry = None
        else:
            opp = pick_from_lowest_tally(items, tally, exclude_player=carry.player)
            if opp is not None:
                items.remove(opp)
                pairs.append((carry, opp))
                tally[carry.player] = tally.get(carry.player, 0) + 1
                tally[opp.player] = tally.get(opp.player, 0) + 1
                carry = None
            else:
                # no opponent available now -> keep carry
                return pairs, carry, bye_quota

    # 2) Pair remaining entries
    while items:
        a = pick_from_lowest_tally(items, tally)
        if a is None:
            break
        items.remove(a)

        # Prefer to spend a BYE here if quota remains
        if bye_quota > 0:
            pairs.append((a, Entry("SYSTEM", "BYE", 0)))
            tally[a.player] = tally.get(a.player, 0) + 1
            bye_quota -= 1
            continue

        b = pick_from_lowest_tally(items, tally, exclude_player=a.player)
        if b is None:
            # Can't find valid opponent now -> make 'a' the new carry
            carry = a
            break
        items.remove(b)
        pairs.append((a, b))
        tally[a.player] = tally.get(a.player, 0) + 1
        tally[b.player] = tally.get(b.player, 0) + 1

    # If one leftover without quota, becomes carry
    if items:
        carry = items[0]
    return pairs, carry, bye_quota

def generate_bracket_groups(entries: List[Entry]) -> List[Tuple[Entry, Entry]]:
    """
    Balanced-random groups with *exact* BYE count to reach the next power of 2.
    BYEs are distributed proportionally across rings to keep things fair.
    """
    base_entries = [e for e in entries if e.player != "SYSTEM"]
    need = byes_needed(len(base_entries))
    rings, _ = build_rings(base_entries)

    # Proportional BYE allocation per ring
    ring_sizes = [len(r) for r in rings]
    total_slots = sum(ring_sizes)
    # Guard against zero division
    if total_slots == 0:
        return []

    # First pass: proportional allocation
    quotas = [ (need * sz) / total_slots for sz in ring_sizes ]
    quotas = [int(math.floor(q)) for q in quotas]
    used = sum(quotas)
    # Distribute leftovers (largest fractional opportunity: by ring size)
    leftover = need - used
    order = sorted(range(len(rings)), key=lambda i: ring_sizes[i], reverse=True)
    j = 0
    while leftover > 0 and order:
        idx = order[j % len(order)]
        # don't exceed ring size
        if quotas[idx] < ring_sizes[idx]:
            quotas[idx] += 1
            leftover -= 1
        j += 1

    pairs: List[Tuple[Entry, Entry]] = []
    carry: Optional[Entry] = None
    tally: Dict[str, int] = {}

    for r_idx, ring in enumerate(rings):
        ring_pairs, carry, unused_quota = pair_ring_balanced_with_bye_quota(
            ring, tally, carry, quotas[r_idx]
        )
        pairs.extend(ring_pairs)
        # If we couldn't spend all quota inside this ring (e.g., too few items), push it forward
        if unused_quota > 0 and r_idx + 1 < len(quotas):
            quotas[r_idx + 1] += unused_quota

    # After all rings, if carry remains, it must take a BYE (consume from any remaining quota).
    leftover_quota = sum(quotas[len(rings):]) if len(quotas) > len(rings) else 0
    if carry is not None:
        pairs.append((carry, Entry("SYSTEM", "BYE", 0)))

    return pairs

# ---------- EVERYTHING: groups first, then slot-1 conflict fix ----------
def fix_first_pick_conflicts(bracket: List[Tuple[Entry, Entry]]) -> List[Tuple[Entry, Entry]]:
    def is_slot1(e: Entry) -> bool:
        return e.slot == 1 and e.player != "SYSTEM"
    def safe_swap_ok(a_entry: Entry, a_opp: Entry, b_entry: Entry, b_opp: Entry) -> bool:
        return (a_entry.player != b_opp.player) and (b_entry.player != a_opp.player)

    for _ in range(6):
        changed = False
        app: Dict[str, List[Tuple[int, str, Entry, Entry]]] = {}
        for idx, (x, y) in enumerate(bracket):
            app.setdefault(x.player, []).append((idx, "A", x, y))
            app.setdefault(y.player, []).append((idx, "B", y, x))
        conflicts = [i for i, (x, y) in enumerate(bracket) if is_slot1(x) and is_slot1(y) and x.player != y.player]
        if not conflicts:
            break
        for i in conflicts:
            a, b = bracket[i]
            fixed = False
            for (j, side, entry, opp) in app.get(a.player, []):
                if j == i: continue
                if entry.slot != 1 and not is_slot1(opp) and safe_swap_ok(entry, opp, a, b):
                    if side == "A":
                        bracket[i] = (entry, b)
                        bracket[j] = (a, bracket[j][1])
                    else:
                        bracket[i] = (entry, b)
                        bracket[j] = (bracket[j][0], a)
                    changed = True; fixed = True; break
            if fixed: continue
            for (j, side, entry, opp) in app.get(b.player, []):
                if j == i: continue
                if entry.slot != 1 and not is_slot1(opp) and safe_swap_ok(entry, opp, b, a):
                    if side == "A":
                        bracket[i] = (a, entry)
                        bracket[j] = (b, bracket[j][1])
                    else:
                        bracket[i] = (a, entry)
                        bracket[j] = (bracket[j][0], b)
                    changed = True; fixed = True; break
        if not changed:
            break
    return bracket

def generate_bracket_everything(entries: List[Entry]) -> List[Tuple[Entry, Entry]]:
    base = generate_bracket_groups(entries)
    return fix_first_pick_conflicts(base)

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
            "groups: balanced-random per ring using tallies; BYEs distributed to reach next power of 2\n"
            "everything: groups first, then swap away Slot1 vs Slot1 when possible"
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
                st.error("Couldn't build a valid round-1 bracket with those constraints.")
            else:
                # Show target size / BYEs summary
                total = len(entries)
                target = next_power_of_two(total)
                need = target - total
                st.success(f"Bracket generated — Total entries: {total} → Target: {target}, BYEs: {need}")
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

st.caption("BYEs are allocated so the round fits the next power-of-two bracket (16, 32, 64, …).")
