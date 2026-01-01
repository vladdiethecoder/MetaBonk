export const ICON_MAP = {
  sheet: { cols: 8, rows: 8, order: "row-major", index_range: [0, 63] as const },
  variants: ["normal", "active", "alert", "disabled"] as const,
  icons: [
    { i: 0, key: "borgar", label: "Borgar (healing burger drop)", hud: "badge/count: burgers spawned or picked up" },
    { i: 1, key: "big_bonk", label: "Big Bonk proc (20× hit)", hud: "badge/count: procs this run" },

    { i: 2, key: "tome_agility", label: "Agility Tome", hud: "pill: tome level" },
    { i: 3, key: "tome_damage", label: "Damage Tome", hud: "pill: tome level" },
    { i: 4, key: "tome_cooldown", label: "Cooldown Tome", hud: "pill: tome level" },
    { i: 5, key: "tome_gold", label: "Gold Tome", hud: "pill: tome level" },
    { i: 6, key: "tome_xp", label: "XP Tome", hud: "pill: tome level" },
    { i: 7, key: "tome_luck", label: "Luck Tome", hud: "pill: tome level" },
    { i: 8, key: "tome_duration", label: "Duration Tome", hud: "pill: tome level" },
    { i: 9, key: "tome_evasion", label: "Evasion Tome", hud: "pill: tome level" },
    { i: 10, key: "tome_regen", label: "Regen/Healing Tome", hud: "pill: tome level" },
    { i: 11, key: "tome_shield", label: "Shield/Armor Tome", hud: "pill: tome level" },
    { i: 12, key: "tome_cursed", label: "Cursed Tome", hud: "flag: curse/difficulty-up enabled" },
    { i: 13, key: "tome_random", label: "Random Tome", hud: "flag: random boost active" },
    { i: 14, key: "tome_mastery", label: "Tome Mastery (total tome levels)", hud: "pill: sum of tome levels" },

    { i: 15, key: "charge_shrine", label: "Charge Shrine / Interactable", hud: "flag: shrine active / progress" },
    { i: 16, key: "beacon_zone", label: "Healing Zone / Beacon effect", hud: "flag: heal-zone active" },

    { i: 17, key: "loot_chest", label: "Chest / Loot drop", hud: "badge/count: opened" },
    { i: 18, key: "loot_goblin", label: "Loot Goblin / Runner event", hud: "flag: present / target" },
    { i: 19, key: "rarity_star", label: "Rarity / Quality", hud: "pill: rarity tier" },
    { i: 20, key: "currency_coins", label: "Coins / Gold pickup", hud: "pill: current coins" },
    { i: 21, key: "bonk_bucks", label: "Bonk Bucks (stream economy)", hud: "pill: balance / pool" },

    { i: 22, key: "projectiles", label: "Projectile Count / Multi-shot", hud: "pill: modifier" },
    { i: 23, key: "aoe", label: "AoE / Splash radius", hud: "pill: modifier" },
    { i: 24, key: "crit", label: "Crit (rate or multiplier)", hud: "pill: crit% or crit×" },
    { i: 25, key: "overcrit", label: "Overcrit / Spike event", hud: "badge/count: overcrits" },

    { i: 26, key: "bloodmark", label: "Bloodmark / Lifesteal", hud: "pill: % or stacks" },
    { i: 27, key: "poison", label: "Poison / DoT", hud: "pill: stacks/chance" },
    { i: 28, key: "burn", label: "Burn / Ignite", hud: "pill: stacks/chance" },
    { i: 29, key: "freeze", label: "Freeze / Chill", hud: "flag: chilled/frozen" },
    { i: 30, key: "shock", label: "Shock / Lightning", hud: "flag: shocked/energized" },
    { i: 31, key: "stun_knockback", label: "Stun / Knockback", hud: "flag: CC-heavy build" },

    { i: 32, key: "boss", label: "Boss present", hud: "flag: boss fight" },
    { i: 33, key: "elite", label: "Elite present", hud: "flag: elite on screen" },
    { i: 34, key: "swarm", label: "Swarm / Wave density", hud: "pill: intensity" },
    { i: 35, key: "danger", label: "Danger meter", hud: "pill: danger% / risk" },

    { i: 36, key: "health", label: "HP / Survivability", hud: "pill: current HP or %" },
    { i: 37, key: "heartbreak", label: "Death / Downed", hud: "event: run ended / KO" },

    { i: 38, key: "speed", label: "Move speed / Pace", hud: "pill: speed or pace Δ" },
    { i: 39, key: "dps", label: "DPS / Output", hud: "pill: DPS" },
    { i: 40, key: "score", label: "Score", hud: "pill: score" },
    { i: 41, key: "level", label: "Level", hud: "pill: level" },
    { i: 42, key: "time", label: "Run timer", hud: "pill: mm:ss" },

    { i: 43, key: "record_wr", label: "WR / PB pace line", hud: "flag: ahead/behind" },
    { i: 44, key: "rank_up", label: "Rank up / Overtake", hud: "event: gained rank" },
    { i: 45, key: "rank_down", label: "Rank down / Overtaken", hud: "event: lost rank" },
    { i: 46, key: "focus", label: "Focus camera", hud: "state: featured tile" },

    { i: 47, key: "replay", label: "Replay available", hud: "button/state: show replay" },
    { i: 48, key: "moment", label: "Moment captured", hud: "event: highlight card" },
    { i: 49, key: "hall_of_fame", label: "Hall of Fame", hud: "panel: best runs" },
    { i: 50, key: "wall_of_shame", label: "Wall of Shame", hud: "panel: worst runs" },

    { i: 51, key: "hype", label: "Hype state", hud: "flag: hype# / momentum" },
    { i: 52, key: "shame", label: "Shame state", hud: "flag: shame# / penalty" },

    { i: 53, key: "policy", label: "Policy/Brain (PPO)", hud: "tag: policy name/id" },
    { i: 54, key: "mutation", label: "PBT Mutation / Evolution", hud: "event: new genome/weights" },
    { i: 55, key: "planning", label: "Planner (MCTS/Deliberation)", hud: "flag: deliberative mode" },
    { i: 56, key: "training", label: "Training update / Step", hud: "pill: steps or upd/s" },

    { i: 57, key: "no_gpu_feed", label: "No GPU feed", hud: "state: feed offline" },
    { i: 58, key: "menu_stuck", label: "Stuck", hud: "state: recovery needed" },

    { i: 59, key: "warning", label: "Warning / Alert", hud: "event: alert" },
    { i: 60, key: "info", label: "Info / Legend", hud: "tooltip/legend" },
    { i: 61, key: "settings", label: "Settings", hud: "panel button" },
    { i: 62, key: "refresh", label: "Refresh / Reroll", hud: "button: refresh/reroll" },

    { i: 63, key: "unknown", label: "Unknown / Placeholder", hud: "fallback" },
  ],
} as const;

export type IconKey = (typeof ICON_MAP.icons)[number]["key"];
export type IconVariant = (typeof ICON_MAP.variants)[number];

export const ICON_INDEX_BY_KEY = Object.freeze(
  Object.fromEntries(ICON_MAP.icons.map((x) => [x.key, x.i])) as Record<IconKey, number>,
);

export function iconIndex(key: IconKey, fallback: number = 63) {
  return ICON_INDEX_BY_KEY[key] ?? fallback;
}

export function iconVariantClass(variant: IconVariant = "normal") {
  return `mb-variant-${variant}`;
}
