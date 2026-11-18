import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "valorant.db")

WEAPONS_DATA = [
    (1, "Classic", "Sidearm", 0, 9.75, 12, "Medium", 78, 26, 22),
    (2, "Ghost", "Sidearm", 500, 6.75, 15, "Medium", 105, 30, 25),
    (3, "Sheriff", "Sidearm", 800, 4.0, 6, "High", 159, 55, 46),
    (4, "Stinger", "SMG", 1100, 16.0, 20, "Low", 67, 27, 22),
    (5, "Spectre", "SMG", 1600, 13.33, 30, "Medium", 78, 26, 22),
    (6, "Judge", "Shotgun", 1850, 3.5, 5, "Low", 34, 17, 14),
    (7, "Bulldog", "Rifle", 2050, 10.0, 24, "Medium", 115, 35, 30),
    (8, "Guardian", "Rifle", 2250, 5.25, 12, "High", 195, 65, 49),
    (9, "Phantom", "Rifle", 2900, 11.0, 30, "Medium", 156, 39, 33),
    (10, "Vandal", "Rifle", 2900, 9.75, 25, "Medium", 160, 40, 34),
    (11, "Outlaw", "Sniper", 2400, 2.75, 2, "High", 238, 140, 119),
    (12, "Operator", "Sniper", 4700, 0.6, 5, "High", 255, 150, 120),
]

ABILITIES_DATA = [
    (1, "Jett", "Duelist", "C", "Cloudburst", 200, 2, "Throws a projectile that expands into a brief vision-blocking cloud for 4.5s."),
    (2, "Jett", "Duelist", "Q", "Updraft", 150, 1, "Propels Jett high into the air to reach elevated angles."),
    (3, "Jett", "Duelist", "E", "Tailwind", 0, 1, "Activates a 7.5s wind window allowing an instant dash in the movement direction."),
    (4, "Jett", "Duelist", "X", "Blade Storm", 8, 1, "Equips 5 deadly throwing knives that deal 50 body and 150 head damage and recharge on kills."),
    (5, "Omen", "Controller", "C", "Shrouded Step", 100, 2, "Channels a brief shadow walk to teleport up to 15m to a marked location."),
    (6, "Omen", "Controller", "Q", "Paranoia", 250, 1, "Fires a shadow projectile through walls up to 35m that nearsights and deafens players hit."),
    (7, "Omen", "Controller", "E", "Dark Cover", 150, 2, "Casts a long-range hollow shadow sphere smoke lasting 15s with a 30s cooldown."),
    (8, "Omen", "Controller", "X", "From the Shadows", 7, 1, "Tactical map teleport to any location on the map; can be canceled during shade form."),
    (9, "Sova", "Initiator", "C", "Owl Drone", 400, 1, "Deploys a pilotable drone for 7s that fires a tracking dart revealing hit enemies twice."),
    (10, "Sova", "Initiator", "Q", "Shock Bolt", 150, 2, "Fires an explosive energy arrow that deals 1 to 75 area damage and destroys traps."),
    (11, "Sova", "Initiator", "E", "Recon Bolt", 0, 1, "Fires a sonar bolt with 40s cooldown that pulses twice to reveal enemies in line of sight."),
    (12, "Sova", "Initiator", "X", "Hunter's Fury", 8, 1, "Fires up to 3 wall-piercing energy blasts across the map dealing 80 damage and marking hit targets."),
    (13, "Cypher", "Sentinel", "C", "Trapwire", 200, 2, "Places a stealth tripwire between walls that slows by 50 percent and concusses after a 0.5s windup in Patch 9.08."),
    (14, "Cypher", "Sentinel", "Q", "Cyber Cage", 100, 2, "Tosses a remote activation zone that blocks vision for 7.25s and plays a louder audio cue when crossed."),
    (15, "Cypher", "Sentinel", "E", "Spycam", 0, 1, "Places a remote camera on surfaces to watch angles and fire a tracking dart."),
    (16, "Cypher", "Sentinel", "X", "Neural Theft", 7, 1, "Costs 7 ult points in Patch 9.08; extracts info from an enemy corpse to reveal all living enemies twice."),
]


def init_database() -> str:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE weapons (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            category TEXT NOT NULL,
            cost INTEGER NOT NULL,
            fire_rate REAL NOT NULL,
            magazine_size INTEGER NOT NULL,
            wall_penetration TEXT NOT NULL,
            head_damage_close INTEGER NOT NULL,
            body_damage_close INTEGER NOT NULL,
            leg_damage_close INTEGER NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE abilities (
            id INTEGER PRIMARY KEY,
            agent TEXT NOT NULL,
            role TEXT NOT NULL,
            slot TEXT NOT NULL,
            name TEXT NOT NULL,
            cost_or_ult_points INTEGER NOT NULL,
            max_charges INTEGER NOT NULL,
            description TEXT NOT NULL
        )
        """
    )
    cur.executemany(
        "INSERT INTO weapons VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        WEAPONS_DATA,
    )
    cur.executemany(
        "INSERT INTO abilities VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ABILITIES_DATA,
    )
    conn.commit()
    conn.close()
    return DB_PATH


if __name__ == "__main__":
    path = init_database()
    print(f"Initialized SQLite database at {path} with {len(WEAPONS_DATA)} weapons and {len(ABILITIES_DATA)} abilities.")
