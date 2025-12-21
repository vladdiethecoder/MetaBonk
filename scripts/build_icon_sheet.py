from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def _fit_center(img: Image.Image, box: int) -> Image.Image:
    src = img.convert("RGBA")
    w, h = src.size
    if w <= 0 or h <= 0:
        return Image.new("RGBA", (box, box), (0, 0, 0, 0))
    scale = min(box / w, box / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = src.resize((nw, nh), Image.Resampling.LANCZOS)
    out = Image.new("RGBA", (box, box), (0, 0, 0, 0))
    ox = (box - nw) // 2
    oy = (box - nh) // 2
    out.alpha_composite(resized, (ox, oy))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a 0–63 (8×8) icon sheet for the HUD.")
    ap.add_argument("--repo-root", default=".", help="Repo root (default: .)")
    ap.add_argument("--out", default="src/frontend/src/assets/icon_sheet.png", help="Output PNG path")
    ap.add_argument("--tile", type=int, default=128, help="Tile size in pixels (default: 128)")
    ap.add_argument("--inner", type=int, default=112, help="Inner fit box per tile (default: 112)")
    args = ap.parse_args()

    root = Path(args.repo_root).resolve()
    out_path = (root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Canonical keys (0–63) from icons/icon_map.json.
    # This source mapping intentionally uses "megabonk assets" icons first; swap any path here to refine art.
    src: dict[str, str] = {
        "borgar": "megabonk assets/Items/ItemBorgor.png",
        "big_bonk": "megabonk assets/Items/ItemBonker.png",
        "tome_agility": "megabonk assets/Tomes/SpeedTome.png",
        "tome_damage": "megabonk assets/Tomes/DamageTome.png",
        "tome_cooldown": "megabonk assets/Tomes/CooldownTome.png",
        "tome_gold": "megabonk assets/Tomes/GoldenTome.png",
        "tome_xp": "megabonk assets/Tomes/XpTome.png",
        "tome_luck": "megabonk assets/Tomes/LuckTome.png",
        "tome_duration": "megabonk assets/Tomes/DurationTome.png",
        "tome_evasion": "megabonk assets/Tomes/EvasionTome.png",
        "tome_regen": "megabonk assets/Tomes/RegenerationTome.png",
        "tome_shield": "megabonk assets/Tomes/ShieldTome.png",
        "tome_cursed": "megabonk assets/Tomes/CursedTome.png",
        "tome_random": "megabonk assets/Tomes/ChaosTome.png",
        "tome_mastery": "megabonk assets/Items/ItemShatteredWisdom.png",
        "charge_shrine": "megabonk assets/Items/ItemAnvil.png",
        "beacon_zone": "megabonk assets/Items/Itembeacon.png",
        "loot_chest": "megabonk assets/Items/ItemBackpack.png",
        "loot_goblin": "megabonk assets/Items/Speed boi.png",
        "rarity_star": "megabonk assets/Items/ItemGoldenRing.png",
        "currency_coins": "megabonk assets/Items/ItemCreditCardGreen.png",
        "bonk_bucks": "megabonk assets/Items/ItemCreditCardRed.png",
        "projectiles": "megabonk assets/Tomes/QuantityTome.png",
        "aoe": "megabonk assets/Tomes/SizeTome.png",
        "crit": "megabonk assets/Tomes/PrecisionTome.png",
        "overcrit": "megabonk assets/Items/ItemOverpoweredLamp.png",
        "bloodmark": "megabonk assets/Tomes/BloodTOme.png",
        "poison": "megabonk assets/Weapons/Poison Flask.png",
        "burn": "megabonk assets/Items/ItemDragonfire.png",
        "freeze": "megabonk assets/Items/ItemIceCube.png",
        "shock": "megabonk assets/Weapons/Lightning Bolt.png",
        "stun_knockback": "megabonk assets/Tomes/KnockbackTome.png",
        "boss": "megabonk assets/Items/ItemBossBuster.png",
        "elite": "megabonk assets/Items/ItemDemonBlade.png",
        "swarm": "megabonk assets/Weapons/Chunkers.png",
        "danger": "megabonk assets/Items/Gasmask.png",
        "health": "megabonk assets/Items/ItemMedkit.png",
        "heartbreak": "megabonk assets/Items/ItemBobDead.png",
        "speed": "megabonk assets/Items/ItemRollerblades.png",
        "dps": "megabonk assets/Weapons/Sword.png",
        "score": "megabonk assets/Items/ItemGoldenGlove.png",
        "level": "megabonk assets/Items/ItemGymSauce.png",
        "time": "megabonk assets/Items/ItemTimeBracelet.png",
        "record_wr": "megabonk assets/Items/ItemGoldenShield.png",
        "rank_up": "megabonk assets/Items/ItemSpeedyCloak.png",
        "rank_down": "megabonk assets/Items/ItemMoldyCheese.png",
        "focus": "megabonk assets/Items/ItemTacticalGoggles.png",
        "replay": "megabonk assets/Items/ItemMirror.png",
        "moment": "megabonk assets/Items/ItemEchoShard.png",
        "hall_of_fame": "megabonk assets/Items/ItemGoldenShield.png",
        "wall_of_shame": "megabonk assets/Items/ItemCursedDoll.png",
        "hype": "megabonk assets/Items/IdleJuice.png",
        "shame": "megabonk assets/Items/ItemCursedDoll.png",
        "policy": "megabonk assets/Items/ItemMechanics.png",
        "mutation": "megabonk assets/Items/Unstable Transfusion.png",
        "planning": "megabonk assets/Items/GamerLens.png",
        "training": "megabonk assets/Items/ItemEnergyCore.png",
        "no_gpu_feed": "megabonk assets/Items/ItemGhost.png",
        "menu_stuck": "megabonk assets/Items/ItemCactus.png",
        "warning": "megabonk assets/Items/ToxicBarrel.png",
        "info": "megabonk assets/Items/ItemHolyBook.png",
        "settings": "megabonk assets/Items/ItemWrench.png",
        "refresh": "megabonk assets/Items/ItemElectricPlug.png",
        "unknown": "megabonk assets/Items/ItemKey.png",
    }

    # Index -> key (0–63). Keep in sync with icons/icon_map.json.
    idx_to_key = [
        "borgar",
        "big_bonk",
        "tome_agility",
        "tome_damage",
        "tome_cooldown",
        "tome_gold",
        "tome_xp",
        "tome_luck",
        "tome_duration",
        "tome_evasion",
        "tome_regen",
        "tome_shield",
        "tome_cursed",
        "tome_random",
        "tome_mastery",
        "charge_shrine",
        "beacon_zone",
        "loot_chest",
        "loot_goblin",
        "rarity_star",
        "currency_coins",
        "bonk_bucks",
        "projectiles",
        "aoe",
        "crit",
        "overcrit",
        "bloodmark",
        "poison",
        "burn",
        "freeze",
        "shock",
        "stun_knockback",
        "boss",
        "elite",
        "swarm",
        "danger",
        "health",
        "heartbreak",
        "speed",
        "dps",
        "score",
        "level",
        "time",
        "record_wr",
        "rank_up",
        "rank_down",
        "focus",
        "replay",
        "moment",
        "hall_of_fame",
        "wall_of_shame",
        "hype",
        "shame",
        "policy",
        "mutation",
        "planning",
        "training",
        "no_gpu_feed",
        "menu_stuck",
        "warning",
        "info",
        "settings",
        "refresh",
        "unknown",
    ]
    assert len(idx_to_key) == 64

    tile = int(args.tile)
    inner = int(args.inner)
    if inner > tile:
        raise SystemExit("--inner must be <= --tile")

    sheet = Image.new("RGBA", (tile * 8, tile * 8), (0, 0, 0, 0))
    for i, key in enumerate(idx_to_key):
        rel = src.get(key)
        if not rel:
            raise SystemExit(f"missing source mapping for key={key}")
        p = (root / rel).resolve()
        if not p.exists():
            raise SystemExit(f"missing source file for key={key}: {rel}")
        icon = Image.open(p).convert("RGBA")

        tile_img = Image.new("RGBA", (tile, tile), (0, 0, 0, 0))
        inner_img = _fit_center(icon, inner)
        ox = (tile - inner) // 2
        oy = (tile - inner) // 2
        tile_img.alpha_composite(inner_img, (ox, oy))

        x = (i % 8) * tile
        y = (i // 8) * tile
        sheet.alpha_composite(tile_img, (x, y))

    sheet.save(out_path)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

