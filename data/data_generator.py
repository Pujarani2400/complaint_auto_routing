"""
data/data_generator.py
=======================
Hybrid municipal complaint dataset generator.
Source taxonomy: NYC 311 Service Request complaint types & descriptors.
Maps to 5 municipal officers with realistic citizen-language narratives.
Target: 800-1500 complaints | Schema: complaint_text, officer, priority, eta_days
"""

import csv
import random
import os

random.seed(42)

# ─────────────────────────────────────────────
# LOCATIONS — injected into narrative templates
# ─────────────────────────────────────────────
LOCATIONS = [
    "near the main market", "on Elm Street", "in Sector 4", "near the bus depot",
    "on Oak Avenue", "near the railway crossing", "in the residential colony",
    "on 5th and Broadway", "near the school", "on Park Lane", "in Ward 7",
    "near the hospital", "on Cedar Road", "in the downtown area",
    "near the vegetable market", "on Maple Street", "in the old city area",
    "near the flyover", "on MG Road", "in Block C of the housing society",
    "near the playground", "on the highway service road", "in Sector 12",
    "near the temple", "on Station Road"
]

DURATIONS = [
    "since yesterday", "for the past 3 days", "for over a week",
    "since last Monday", "for the past 2 days", "since this morning",
    "for 10 days now", "since last night", "for the past month",
    "for the last 4 days", "since the rain last week", "for several days",
]

CITIZEN_VOICES = [
    "Residents are complaining that", "I would like to report that",
    "Please look into this urgently —", "We have noticed that",
    "This is a serious issue —", "Kindly send someone to fix this,",
    "I am writing to inform you that", "Multiple families are affected as",
    "Our locality has been suffering because", "Please take immediate action —",
    "This problem has been ignored for too long —", "I wanted to bring to your attention that",
    "The situation is getting worse —", "Several complaints have been raised about this —",
    "As a resident, I urge you to address the fact that",
]

# ─────────────────────────────────────────────────────────────────────
# NYC 311 TAXONOMY → NARRATIVE TEMPLATES PER OFFICER
# Each tuple: (template_string, priority_weight, eta_range)
# priority_weight: 'H'=High, 'M'=Medium, 'L'=Low
# ─────────────────────────────────────────────────────────────────────

WATER_TEMPLATES = [
    # (text_template, priority, eta_days_range)
    ("water main broke {loc} causing the entire block to lose water supply {dur}", "High", (1, 3)),
    ("there is a major pipe burst {loc} and water is gushing onto the road {dur}", "High", (1, 2)),
    ("no water supply in our area {loc} {dur} — taps are completely dry", "High", (1, 3)),
    ("water pressure is very low {loc} — water barely reaches upper floors {dur}", "Medium", (3, 5)),
    ("dirty brown water is coming from the taps {loc} {dur} — not safe to drink", "High", (1, 3)),
    ("water is leaking from the main pipeline {loc} and flooding the road {dur}", "High", (1, 2)),
    ("the water supply has been disrupted {loc} since the pipeline repair work began {dur}", "Medium", (2, 4)),
    ("tap water tastes foul and has a strange odor {loc} {dur}", "High", (1, 3)),
    ("there is no hot water supply in the building {loc} {dur}", "Medium", (3, 5)),
    ("water pipes appear to have a crack {loc} — water seeping into the ground {dur}", "Medium", (2, 4)),
    ("drinking water supplied {loc} appears discolored with visible sediment {dur}", "High", (1, 2)),
    ("water meter at {loc} is broken and possibly causing billing errors", "Low", (4, 6)),
    ("underground water pipe leaking near {loc} — road surface getting soft and dangerous", "High", (1, 3)),
    ("intermittent water supply {loc} — water comes for only 30 minutes a day {dur}", "Medium", (3, 5)),
    ("frozen pipe suspected {loc} — no water flow in the entire building {dur}", "High", (1, 2)),
    ("the overhead water tank at {loc} is overflowing and wasting water {dur}", "Low", (4, 6)),
    ("sewage water mixing with drinking water supply {loc} — serious health risk {dur}", "High", (1, 2)),
    ("water connection was illegally disconnected {loc} — residents have no water {dur}", "High", (1, 3)),
    ("water tanker not arriving on schedule {loc} {dur} — community is suffering", "Medium", (2, 4)),
    ("pump house {loc} is not functioning properly — water supply affected {dur}", "Medium", (3, 5)),
]

ELECTRICAL_TEMPLATES = [
    ("the street light at the intersection {loc} has been out {dur} — very unsafe at night", "Medium", (4, 6)),
    ("multiple street lights are not working {loc} {dur} — area is completely dark after 8 PM", "Medium", (3, 5)),
    ("transformer {loc} is sparking and making loud noises — several houses lost power {dur}", "High", (2, 4)),
    ("power outage affecting entire neighbourhood {loc} {dur} — no electricity at all", "High", (2, 3)),
    ("street light pole knocked down {loc} — posing risk to passing traffic {dur}", "High", (1, 3)),
    ("traffic signal at {loc} is not working — causing major traffic confusion {dur}", "High", (1, 3)),
    ("dim and flickering street lights {loc} make the road hazardous at night {dur}", "Medium", (4, 7)),
    ("electric wire hanging low over the road {loc} {dur} — serious safety hazard", "High", (1, 2)),
    ("frequent power cuts happening {loc} {dur} — load shedding with no prior notice", "Medium", (3, 5)),
    ("electrical meter board at {loc} appears to have faulty wiring — sparks visible", "High", (1, 3)),
    ("signal timing at {loc} is off — green light too short causing congestion {dur}", "Medium", (5, 7)),
    ("street light is on during the day and off at night {loc} — sensor malfunction", "Low", (5, 7)),
    ("underground cable fault {loc} causing power fluctuations {dur}", "Medium", (3, 5)),
    ("electricity pole tilting dangerously {loc} {dur} — risk of falling on pedestrians", "High", (1, 2)),
    ("missing signal head at {loc} junction — drivers confused about right of way {dur}", "High", (2, 4)),
    ("high voltage buzzing sound from transformer {loc} {dur} — people are afraid", "High", (1, 3)),
    ("broken electric meter box {loc} exposed to rain — shock risk {dur}", "High", (1, 2)),
    ("no street lights installed on the new road {loc} — dangerous after dark", "Medium", (5, 7)),
    ("power supply interruption affecting hospitals and essential services {loc} {dur}", "High", (1, 2)),
    ("unauthorized electrical connection spotted at {loc} — risk of overloading", "Medium", (4, 6)),
]

ROAD_TEMPLATES = [
    ("large pothole {loc} is getting bigger — cars are swerving dangerously {dur}", "High", (5, 8)),
    ("pothole on the highway {loc} caused a vehicle to lose control {dur}", "High", (4, 7)),
    ("road has completely caved in {loc} — one lane blocked {dur}", "High", (4, 6)),
    ("multiple potholes on the main arterial road {loc} causing traffic slowdown {dur}", "Medium", (7, 10)),
    ("sidewalk {loc} is cracked and uneven — elderly person tripped there recently", "Medium", (7, 10)),
    ("manhole cover missing on {loc} — open pit is extremely dangerous for pedestrians {dur}", "High", (4, 6)),
    ("road surface badly damaged {loc} — rough patches causing damage to vehicles {dur}", "Medium", (8, 12)),
    ("guardrail on the highway {loc} is broken — vehicles at risk of going off road {dur}", "High", (4, 6)),
    ("footpath {loc} is completely blocked by debris and encroachment {dur}", "Medium", (8, 12)),
    ("road markings completely faded {loc} — causing confusion especially at night {dur}", "Low", (12, 15)),
    ("raised speed breaker {loc} is not painted — invisible at night and causing accidents", "High", (5, 7)),
    ("road collapse due to waterlogging {loc} — several vehicles got stuck {dur}", "High", (4, 6)),
    ("expressway pothole {loc} is very deep — tyres of vehicles get damaged daily", "High", (5, 7)),
    ("damaged road {loc} has not been repaired despite multiple complaints {dur}", "Medium", (10, 14)),
    ("roadside drain cover broken at {loc} — open gutter visible beside the road {dur}", "Medium", (7, 10)),
    ("highway onramp {loc} has surface cracking — dangerous for heavy vehicles {dur}", "High", (5, 7)),
    ("road divider damaged {loc} — risk of vehicles crossing into oncoming traffic {dur}", "High", (4, 6)),
    ("construction debris left on road {loc} not cleared by contractor {dur}", "Medium", (7, 10)),
    ("severe road damage near school zone {loc} — children safety at risk {dur}", "High", (4, 6)),
    ("footpath tiles broken and uneven {loc} — causing falls especially in rainy season", "Low", (12, 15)),
]

SANITATION_TEMPLATES = [
    ("garbage not collected from our residential block {loc} {dur} — health hazard", "High", (2, 4)),
    ("overflowing garbage bins {loc} {dur} — waste spilling onto the road and footpath", "High", (2, 3)),
    ("illegal dumping of construction waste noticed {loc} {dur} — blocking pedestrian path", "Medium", (3, 5)),
    ("litter basket {loc} is overflowing — no collection in days {dur}", "Medium", (3, 5)),
    ("dead animal carcass lying {loc} {dur} — causing foul smell and health concern", "High", (1, 2)),
    ("commercial establishment {loc} dumping garbage on the street after hours {dur}", "High", (2, 3)),
    ("recyclable waste not being collected separately {loc} {dur} — mixed with general waste", "Low", (4, 6)),
    ("bulk garbage items like old furniture dumped {loc} — blocking the footpath {dur}", "Medium", (3, 5)),
    ("garbage truck not arriving {loc} {dur} — 10+ days of accumulated waste on street", "High", (2, 4)),
    ("open lot {loc} being used for illegal garbage dumping {dur}", "Medium", (4, 6)),
    ("foul smell from garbage {loc} — unbearable for residents nearby {dur}", "High", (2, 3)),
    ("sweeping of road not done {loc} {dur} — leaves and garbage scattered everywhere", "Low", (4, 6)),
    ("construction waste from site {loc} not removed by contractor {dur}", "Medium", (4, 5)),
    ("alley behind houses {loc} filled with garbage — breeding ground for mosquitoes {dur}", "High", (2, 4)),
    ("litter basket cover broken at {loc} — waste exposed to rain and pests {dur}", "Medium", (4, 6)),
    ("commercial waste being mixed with residential garbage {loc} {dur}", "Medium", (3, 5)),
    ("open defecation being practiced near {loc} — very unhygienic condition {dur}", "High", (1, 3)),
    ("garbage bins at {loc} not disinfected — spreading disease risk {dur}", "High", (2, 3)),
    ("waste not collected before festival season {loc} — piling up on roads {dur}", "High", (1, 2)),
    ("civic contractor skipping collection in {loc} area {dur} — residents fed up", "Medium", (3, 5)),
]

DRAINAGE_TEMPLATES = [
    ("catch basin {loc} completely clogged — street flooded knee-deep after last night rain", "High", (1, 3)),
    ("open drain {loc} overflowing {dur} — sewage water entering homes", "High", (1, 2)),
    ("stormwater drain blocked {loc} — entire road flooded every time it rains {dur}", "High", (2, 4)),
    ("manhole overflowing with sewage {loc} {dur} — unbearable smell and hygiene risk", "High", (1, 2)),
    ("basement flooding in {loc} area due to blocked drainage {dur}", "High", (1, 3)),
    ("standing water not draining {loc} {dur} — mosquito breeding happening", "Medium", (3, 5)),
    ("drain cover missing {loc} — open sewer gutter accessible to pedestrians {dur}", "High", (1, 3)),
    ("sewage backup in residential area {loc} — water coming up from floor drains {dur}", "High", (1, 2)),
    ("pooling water after rain {loc} — drain is clearly insufficient for rainfall volume", "Medium", (3, 5)),
    ("gutter {loc} clogged with debris — overflow during monsoon causing road damage {dur}", "Medium", (3, 5)),
    ("foul odor from sewer {loc} is affecting entire neighbourhood {dur}", "Medium", (3, 5)),
    ("stormwater grate broken at {loc} — water rushing in uncontrolled manner {dur}", "High", (1, 3)),
    ("drainage canal near {loc} needs urgent desilting — overflow risk before monsoon", "Medium", (4, 6)),
    ("sewage pipe burst {loc} — raw sewage flowing on road surface {dur}", "High", (1, 2)),
    ("catch basin at {loc} requires repairing — broken cover creating fall hazard", "Medium", (4, 6)),
    ("waterlogging reported {loc} every monsoon — permanent drainage solution needed", "Medium", (5, 7)),
    ("children falling into open drainage pit {loc} — emergency cover needed {dur}", "High", (1, 2)),
    ("rain water from highway {loc} not draining — entering residential plots {dur}", "High", (2, 4)),
    ("illegal connection to main sewer near {loc} causing backup for residents {dur}", "Medium", (4, 6)),
    ("drainage infrastructure {loc} is old and collapsed — requires full replacement", "Low", (6, 7)),
]

# ───────────────────────────────────────────────────────
# PARAPHRASE AUGMENTATION — synonym maps per officer domain
# ───────────────────────────────────────────────────────
WATER_SYNONYMS = {
    "broke": ["burst", "ruptured", "cracked", "failed", "gave way"],
    "water supply": ["water connection", "water flow", "municipal water"],
    "flooding": ["waterlogging", "inundation", "overflow"],
    "leaking": ["seeping", "dripping", "flowing out"],
    "pipe": ["pipeline", "main line", "water line", "conduit"],
}

ELECTRICAL_SYNONYMS = {
    "not working": ["out of order", "broken", "malfunctioning", "non-functional", "dead"],
    "street light": ["road light", "lamp post", "street lamp", "sodium light"],
    "power outage": ["power cut", "electricity failure", "blackout", "load shedding"],
    "sparking": ["flashing", "arcing", "short-circuiting"],
    "transformer": ["substation unit", "distribution transformer"],
}

ROAD_SYNONYMS = {
    "pothole": ["crater", "deep hole", "road cavity", "road pit", "depression on road"],
    "damaged": ["broken", "deteriorated", "worn out", "badly maintained"],
    "sidewalk": ["footpath", "pavement", "walking path"],
    "road": ["street", "avenue", "lane", "highway", "road stretch"],
    "dangerous": ["hazardous", "risky", "unsafe", "life-threatening"],
}

SANITATION_SYNONYMS = {
    "garbage": ["waste", "trash", "rubbish", "refuse", "litter"],
    "not collected": ["not picked up", "uncollected", "not cleared", "left behind"],
    "overflowing": ["spillover", "brimming over", "overfull", "piled up"],
    "foul smell": ["stench", "bad odor", "unbearable smell", "putrid smell"],
    "health hazard": ["hygiene issue", "public health risk", "sanitation problem"],
}

DRAINAGE_SYNONYMS = {
    "clogged": ["blocked", "choked", "obstructed", "jammed"],
    "flooded": ["waterlogged", "inundated", "submerged", "under water"],
    "sewer": ["drain", "drainage pipe", "sewerage", "stormwater line"],
    "overflowing": ["brimming", "spilling over", "backing up", "overrunning"],
    "sewage": ["wastewater", "effluent", "grey water", "drain water"],
}

OFFICER_CONFIG = {
    "Water Officer":      {"templates": WATER_TEMPLATES,      "synonyms": WATER_SYNONYMS,      "target": 230},
    "Electrical Officer": {"templates": ELECTRICAL_TEMPLATES, "synonyms": ELECTRICAL_SYNONYMS, "target": 200},
    "Road Officer":       {"templates": ROAD_TEMPLATES,       "synonyms": ROAD_SYNONYMS,       "target": 210},
    "Sanitation Officer": {"templates": SANITATION_TEMPLATES, "synonyms": SANITATION_SYNONYMS, "target": 190},
    "Drainage Officer":   {"templates": DRAINAGE_TEMPLATES,   "synonyms": DRAINAGE_SYNONYMS,   "target": 200},
}


def apply_synonyms(text: str, synonyms: dict) -> str:
    """Randomly substitute one synonym pair in the text."""
    keys = list(synonyms.keys())
    random.shuffle(keys)
    for key in keys:
        if key in text:
            text = text.replace(key, random.choice(synonyms[key]), 1)
            break
    return text


def build_narrative(template: str, loc: str, dur: str, voice: str) -> str:
    """Combine template + location + duration + citizen voice into a full complaint."""
    text = template.format(loc=loc, dur=dur)
    # 50% chance to prepend a citizen voice prefix
    if random.random() > 0.5:
        text = voice + " " + text
    # Ensure first letter is capitalized and ends with period
    text = text.strip()
    text = text[0].upper() + text[1:]
    if not text.endswith(('.', '!', '?')):
        text += "."
    return text


def compute_eta(priority: str, eta_range: tuple) -> int:
    """Derive ETA from priority label + officer domain range (not random)."""
    low, high = eta_range
    if priority == "High":
        hi = max(low, low + 1)
        return random.randint(low, hi)
    elif priority == "Medium":
        mid_lo = low + 1
        mid_hi = max(mid_lo, high - 1)
        return random.randint(mid_lo, mid_hi)
    else:  # Low
        lo = max(low, high - 1)
        return random.randint(lo, high)


def generate_complaints() -> list:
    """Generate all complaints using NYC 311 taxonomy templates with augmentation."""
    rows = []

    for officer, config in OFFICER_CONFIG.items():
        templates = config["templates"]
        synonyms  = config["synonyms"]
        target    = config["target"]

        generated = 0
        while generated < target:
            template_text, priority, eta_range = random.choice(templates)
            loc  = random.choice(LOCATIONS)
            dur  = random.choice(DURATIONS)
            voice = random.choice(CITIZEN_VOICES)

            # Base narrative
            narrative = build_narrative(template_text, loc, dur, voice)

            # Augmentation pass: 40% chance to apply synonym substitution
            if random.random() < 0.4:
                narrative = apply_synonyms(narrative, synonyms)

            # Priority variation: occasionally bump Low→Medium (10% chance) for diversity
            if priority == "Low" and random.random() < 0.1:
                priority = "Medium"

            eta = compute_eta(priority, eta_range)
            rows.append({
                "complaint_text": narrative,
                "officer": officer,
                "priority": priority,
                "eta_days": eta,
            })
            generated += 1

    # Shuffle to avoid officer clustering
    random.shuffle(rows)
    return rows


def save_dataset(rows: list, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["complaint_text", "officer", "priority", "eta_days"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[✓] Dataset saved → {path}  ({len(rows)} rows)")


def print_stats(rows: list):
    from collections import Counter
    officers   = Counter(r["officer"]   for r in rows)
    priorities = Counter(r["priority"]  for r in rows)
    print("\n== Dataset Statistics ==========================")
    print(f"  Total complaints : {len(rows)}")
    print(f"\n  By Officer:")
    for k, v in sorted(officers.items()):
        print(f"    {k:<25} {v}")
    print(f"\n  By Priority:")
    for k, v in sorted(priorities.items()):
        print(f"    {k:<10} {v}")
    eta_vals = [r["eta_days"] for r in rows]
    print(f"\n  ETA: min={min(eta_vals)}, max={max(eta_vals)}, "
          f"avg={sum(eta_vals)/len(eta_vals):.1f} days")
    print("================================================\n")


if __name__ == "__main__":
    OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "complaints.csv")
    complaints = generate_complaints()
    print_stats(complaints)
    save_dataset(complaints, OUTPUT_PATH)
