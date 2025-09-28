# app.py — Smash Bracket w/ Teams mode, R1 winner marking, and rule-first sidebar
import streamlit as st
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import pandas as pd
import math

st.set_page_config(page_title="Smash Bracket (No Self-Match)", page_icon="🎮", layout="wide")
st.title("🎮 Smash Bracket — Round 1 Generator")

# ---------------------------- Data types ----------------------------
@dataclass(frozen=True)
class Entry:
    player: str
    character: str
    slot: int  # 1 = first pick

# ---------------------------- Power-of-two helpers ----------------------------
def next_power_of_two(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()

def byes_needed(n: int) -> int:
    return max(0, next_power_of_two(n) - n)

# ---------------------------- Backtracking for regular / first_pick / teams ----------------------------
def recursive_pairing(entries: List[Entry], rule: str, team_of: Optional[Dict[str, str]] = None) -> Optional[List[Tuple[Entry, Entry]]]:
    """
    Backtracking perfect matching avoiding forbidden pairings.
    Rules:
      - Always forbid same-player R1.
      - first_pick: forbid Slot1 vs Slot1 (across different players).
      - teams: forbid same-team (if both players have a team label and it matches).
      - BYE can face anyone.
    """
    n = len(entries)
    if n == 0:
        return []
    a = entries[0]
    for i in range(1, n):
        b = entries[i]

        # BYE can match anyone
        if "BYE" in (a.character.upper(), b.character.upper()):
            remaining = entries[1:i] + entries[i+1:]
            rest = recursive_pairing(remaining, rule, team_of)
            if rest is not None:
                return [(a, b)] + rest
            continue

        # base: no self
        if a.player == b.player:
            continue

        # first_pick protection
        if rule == "first_pick" and a.slot == 1 and b.slot == 1:
            continue

        # teams protection
        if rule == "teams" and team_of is not None:
            ta = team_of.get(a.player, "")
            tb = team_of.get(b.player, "")
            if ta and tb and ta == tb:
                continue

        remaining = entries[1:i] + entries[i+1:]
        rest = recursive_pairing(remaining, rule, team_of)
        if rest is not None:
            return [(a, b)] + rest
    return None

def generate_bracket_regular_first_teams(raw_entries: List[Entry], rule: str, team_of: Optional[Dict[str, str]] = None):
    """
    Handles 'regular', 'first_pick', and 'teams' modes using backtracking.
    Fills BYEs to reach next power of two.
    """
    entries = raw_entries.copy()
    need = byes_needed(len(entries))
    for _ in range(need):
        entries.append(Entry(player="SYSTEM", character="BYE", slot=0))

    for _ in range(600):
        random.shuffle(entries)
        result = recursive_pairing(entries, rule, team_of)
        if result is not None:
            # Randomize within-pair order
            out: List[Tuple[Entry, Entry]] = []
            for a, b in result:
                if random.random() < 0.5:
                    out.append((b, a))
                else:
                    out.append((a, b))
            return out
    return None

# ---------------------------- Balanced-random Groups + BYEs ----------------------------
def players_from_entries(entries: List[Entry]) -> List[str]:
    seen = []
    for e in entries:
        if e.player != "SYSTEM" and e.player not in seen:
            seen.append(e.player)
    random.shuffle(seen)  # one-time mix
    return seen

def build_rings(entries: List[Entry]) -> Tuple[List[List[Entry]], List[str]]:
    """ring[0]=Slot1 entries (one per player), ring[1]=Slot2, ... using a fixed player order."""
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
    pairs: List[Tuple[Entry, Entry]] = []
    items = ring.copy()

    # carry
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
                return pairs, carry, bye_quota

    # remaining
    while items:
        a = pick_from_lowest_tally(items, tally)
        if a is None:
            break
        items.remove(a)

        if bye_quota > 0:
            pairs.append((a, Entry("SYSTEM", "BYE", 0)))
            tally[a.player] = tally.get(a.player, 0) + 1
            bye_quota -= 1
            continue

        b = pick_from_lowest_tally(items, tally, exclude_player=a.player)
        if b is None:
            # leave 'a' as carry for next ring
            carry = a
            break
        items.remove(b)
        pairs.append((a, b))
        tally[a.player] = tally.get(a.player, 0) + 1
        tally[b.player] = tally.get(b.player, 0) + 1

    if items:
        carry = items[0]
    return pairs, carry, bye_quota

def generate_bracket_groups(entries: List[Entry]) -> List[Tuple[Entry, Entry]]:
    """Balanced-random groups with exact BYEs to reach next power of two."""
    base_entries = [e for e in entries if e.player != "SYSTEM"]
    need = byes_needed(len(base_entries))
    if not base_entries:
        return []

    rings, _ = build_rings(base_entries)
    ring_sizes = [len(r) for r in rings]
    total_slots = sum(ring_sizes)

    # proportional BYE allocation per ring
    quotas = [int(math.floor(need * (sz / total_slots))) for sz in ring_sizes]
    distributed = sum(quotas)
    leftover = need - distributed
    # hand out leftover to largest rings first
    order = sorted(range(len(rings)), key=lambda i: ring_sizes[i], reverse=True)
    k = 0
    while leftover > 0 and order:
        idx = order[k % len(order)]
        if quotas[idx] < ring_sizes[idx]:
            quotas[idx] += 1
            leftover -= 1
        k += 1

    pairs: List[Tuple[Entry, Entry]] = []
    carry: Optional[Entry] = None
    tally: Dict[str, int] = {}
    for r_idx, ring in enumerate(rings):
        ring_pairs, carry, unused = pair_ring_balanced_with_bye_quota(ring, tally, carry, quotas[r_idx])
        pairs.extend(ring_pairs)
        # push unused quota forward
        if unused > 0 and r_idx + 1 < len(quotas):
            quotas[r_idx + 1] += unused
    if carry is not None:
        pairs.append((carry, Entry("SYSTEM", "BYE", 0)))
    return pairs

# ---------------------------- EVERYTHING = Groups then Slot1 fixer ----------------------------
def fix_first_pick_conflicts(bracket: List[Tuple[Entry, Entry]]) -> List[Tuple[Entry, Entry]]:
    def is_slot1(e: Entry) -> bool:
        return e.slot == 1 and e.player != "SYSTEM"
    def safe_swap_ok(a_entry: Entry, a_opp: Entry, b_entry: Entry, b_opp: Entry) -> bool:
        return (a_entry.player != b_opp.player) and (b_entry.player != a_opp.player)

    for _ in range(6):
        changed = False
        # where does each player appear?
        where: Dict[str, List[Tuple[int, str, Entry, Entry]]] = {}
        for idx, (x, y) in enumerate(bracket):
            where.setdefault(x.player, []).append((idx, "A", x, y))
            where.setdefault(y.player, []).append((idx, "B", y, x))

        conflicts = [i for i, (x, y) in enumerate(bracket) if is_slot1(x) and is_slot1(y) and x.player != y.player]
        if not conflicts:
            break

        for i in conflicts:
            a, b = bracket[i]
            fixed = False
            # try swap a's slot1 with a non-slot1 from same player whose opp isn't slot1
            for (j, side, entry, opp) in where.get(a.player, []):
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
            # try swap b's slot1
            for (j, side, entry, opp) in where.get(b.player, []):
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

# ---------------------------- Sidebar (rule first, adaptive options) ----------------------------
with st.sidebar:
    st.header("Rule Set")
    rule = st.selectbox(
        "Choose the mode first",
        options=["regular", "first_pick", "groups", "everything", "teams"],
        index=0,
        help=(
            "regular: No self-matches in R1.\n"
            "first_pick: regular + forbids Slot1 vs Slot1.\n"
            "groups: balanced-random by slot rings; spreads BYEs to reach next power of 2.\n"
            "everything: groups first, then fixes any Slot1 vs Slot1.\n"
            "teams: No self-matches + no same-team matches in R1 (power-of-2 BYEs)."
        )
    )

    st.divider()
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

    # Teams-only UI
    team_of: Dict[str, str] = {}
    if rule == "teams":
        st.divider()
        st.header("Teams (Teams Mode)")
        team_names_input = st.text_input(
            "Team labels (comma separated)", value="Red, Blue",
            help="Add labels like: Red, Blue, Green. Then assign each player below."
        )
        team_labels = [t.strip() for t in team_names_input.split(",") if t.strip()]
        if not team_labels:
            team_labels = ["Team A", "Team B"]

        st.caption("Assign each player to a team:")
        for p in players:
            team_of[p] = st.selectbox(f"{p}", options=["(none)"] + team_labels, key=f"team_{p}")
        # normalize "(none)" -> ""
        team_of = {p: (t if t != "(none)" else "") for p, t in team_of.items()}

    st.divider()
    st.subheader("Build / Fill")
    build_clicked = st.button("⚙️ Auto-Create/Reset Entries", use_container_width=True)
    shuffle_within_player = st.checkbox("Shuffle numbers within each player's slots (for auto-fill)", value=True)
    auto_fill_clicked = st.button("🎲 Auto-fill Characters (Character 1..k)", use_container_width=True)

    st.divider()
    st.header("General")
    clean_rows = st.checkbox("Remove empty rows", value=True)
    st.caption("Note: If one player owns more than half of all entries, valid constraints may be impossible.")

# ---------------------------- Table helpers ----------------------------
def assign_slots(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Slot" not in out.columns:
        out["Slot"] = 0
    out["Slot"] = pd.to_numeric(out["Slot"], errors="coerce").fillna(0).astype(int)
    if "Player" not in out.columns:
        out["Player"] = ""
    # re-number per player by appearance order
    for p, grp in out.groupby("Player", sort=False, dropna=False):
        idxs = list(grp.index)
        for j, idx in enumerate(idxs, start=1):
            out.at[idx, "Slot"] = j
    return out

def build_entries_df(players: List[str], k: int) -> pd.DataFrame:
    rows = []
    for i in range(1, k + 1):
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
        sl = int(pd.to_numeric(row.get("Slot", 0), errors="coerce") or 0)
        if clean_rows_flag and (not pl or not ch):
            continue
        if pl and ch:
            entries.append(Entry(player=pl, character=ch, slot=sl))
    return entries

# ---------------------------- State: entries table ----------------------------
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

# Normalize Player values
if players:
    def normalize_player(p):
        p = str(p).strip()
        return p if p in players else (players[0] if p == "" else "")
    st.session_state.table_df["Player"] = st.session_state.table_df["Player"].map(normalize_player)

# ---------------------------- Editor ----------------------------
st.subheader("Entries")
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

# ---------------------------- Generate bracket + R1 winner UI ----------------------------
st.divider()
col_gen, col_clear = st.columns([2, 1])

def show_bracket_and_winner_ui(bracket: List[Tuple[Entry, Entry]], rule_label: str):
    total = len(entries)
    target = next_power_of_two(total)
    need = target - total
    st.success(f"Bracket generated — Entries: {total} → Target: {target} (BYEs: {need}) — Rule: {rule_label}")

    # Initialize winner state
    if "r1_winners" not in st.session_state:
        st.session_state.r1_winners = {}

    # Render matches with winner picks
    for i, (a, b) in enumerate(bracket, start=1):
        st.markdown(f"**Match {i}**")
        # labels
        label_a = f"{a.character} ({a.player}, Slot {a.slot})"
        label_b = f"{b.character} ({b.player}, Slot {b.slot})"

        # Preselect if previously set
        default_value = st.session_state.r1_winners.get(i, "")
        choice = st.radio(
            "Pick winner:",
            options=[label_a, label_b, "(undecided)"],
            index= [0,1,2][ (0 if default_value==label_a else 1 if default_value==label_b else 2) ],
            key=f"winner_{i}",
            horizontal=True
        )
        # Store selection (avoid "(undecided)")
        st.session_state.r1_winners[i] = choice if choice != "(undecided)" else ""

        # Show BYE note
        if a.character.upper() == "BYE" or b.character.upper() == "BYE":
            st.caption("This match includes a BYE (auto-advance).")

        st.write("---")

    # Summary
    if st.button("💾 Save Round 1 Winners"):
        winners = [w for w in st.session_state.r1_winners.values() if w]
        st.success(f"Saved {len(winners)} winner selections.")
        if winners:
            st.markdown("**Round 1 Winners:**")
            for w in winners:
                st.write("• ", w)

with col_gen:
    if st.button("🎲 Generate Bracket", type="primary"):
        if len(entries) < 2:
            st.error("Add at least 2 entries (characters).")
        else:
            if rule == "regular":
                bracket = generate_bracket_regular_first_teams(entries, "regular")
            elif rule == "first_pick":
                bracket = generate_bracket_regular_first_teams(entries, "first_pick")
            elif rule == "groups":
                bracket = generate_bracket_groups(entries)
            elif rule == "everything":
                bracket = generate_bracket_everything(entries)
            else:  # teams
                # build with team rule (no same team in R1). If team_of empty, it behaves like regular.
                bracket = generate_bracket_regular_first_teams(entries, "teams", team_of)

            if bracket is None:
                st.error("Couldn't build a valid round-1 bracket with those constraints. Try balancing counts or team assignments.")
            else:
                # store in session so winners persist if you re-render UI
                st.session_state["last_bracket"] = [(a, b) for (a, b) in bracket]
                show_bracket_and_winner_ui(bracket, rule)

# If we have a stored bracket (e.g., after rerun), show it again with the winner UI
if "last_bracket" in st.session_state and st.session_state["last_bracket"]:
    st.info("Showing last generated bracket:")
    show_bracket_and_winner_ui(st.session_state["last_bracket"], rule)

with col_clear:
    if st.button("🧹 Clear Table"):
        st.session_state.table_df = pd.DataFrame(columns=["Player", "Character", "Slot"])
        st.session_state.pop("last_bracket", None)
        st.session_state.pop("r1_winners", None)
        st.rerun()

st.caption("Tip: In **Teams** mode, round 1 forbids both self-matches and same-team matches. Other modes ignore teams.")
