"""Author + validate + save the 10-image overfit caption set.

One-off: descriptions are hand-authored by the in-session vision describer;
geometry (palette, OCR text boxes) is pulled from data/_overfit/geom.json.
Coordinates are Ideogram-schema normalized [ymin, xmin, ymax, xmax] in 0..1000.
"""

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rtie.captioning import build as B  # noqa: E402

geom = json.load(open("data/_overfit/geom.json", encoding="utf-8"))
G = {Path(k).stem: v for k, v in geom.items()}


def pal(stem, n=6):
    return G[stem]["palette"][:n]


CAPS = {}

CAPS["7709344486_t0"] = B.build(  # Steal a Brainrot (tycoon/steal)
    "A bright blocky Roblox 'steal' tycoon thumbnail showing two avatars and a giant meme 'brainrot' cat creature on a red-and-green checkered floor, with a huge green money-per-second counter across the top.",
    B.style_block(
        "loud, hyper-saturated, high-contrast, meme-driven, readable at small size, giant_reward archetype",
        "flat bright indoor lighting, strong even key light, bold cyan sky behind",
        "3d_render",
        "Roblox-inspired blocky 3D promotional render, glossy plastic materials, chunky exaggerated avatars, bold simple shapes",
        pal("7709344486_t0"),
    ),
    "A bright blocky indoor base with red storage shelves along the back wall, a vivid cyan sky beyond, and a red-and-green checkered tile floor receding toward the back.",
    [
        B.obj("A muscular blocky Roblox avatar on the left in a white tank top and blue pants, oversized arms, shocked open-mouth expression, holding a small orange block of butter.", bbox=[150, 40, 760, 330]),
        B.obj("A meme 'brainrot' creature in the center: a fluffy cat head fused onto a blocky toast/bread body, sitting on the checkered floor, the focal oddity of the image.", bbox=[300, 360, 720, 650]),
        B.obj("A normal blocky Roblox avatar on the right with brown hair, a black jacket over a blue shirt, smiling calmly with hands behind back.", bbox=[170, 650, 860, 960]),
        B.text("$997,151,167/s", "Huge white display number with a thick green outline spanning the top, a money-per-second counter, very readable at small size.", bbox=[23, 141, 199, 868], color_palette=["#FFFFFF", "#2BD44A", "#1B3342"]),
        B.text("SECRET", "Small red 'SECRET' tag with a 'Steak' label above it, floating beside the cat creature.", bbox=[230, 170, 330, 305], color_palette=["#FF2D55", "#FFFFFF"]),
    ],
)

CAPS["9584852943_t0"] = B.build(  # +1 Speed (simulator)
    "A Roblox speed-simulator thumbnail of a blocky avatar leaping across a floor of wooden crates under pink cherry-blossom trees, with several green '+1' reward popups.",
    B.style_block(
        "bright, energetic, cozy-saturated, readable at small size, progression archetype",
        "soft daylight through pink blossoms, gentle warmth on the wood",
        "3d_render",
        "Roblox-inspired blocky 3D render, matte wood and plastic materials, chunky avatar, simple stylized foliage",
        pal("9584852943_t0"),
    ),
    "A brown wooden play area built from stacked crates and plank flooring, framed by pink cherry-blossom trees and a soft pink sky in the background.",
    [
        B.obj("A blocky Roblox avatar with brown hair and a dark outfit mid-stride, running and jumping across the wooden crates toward the viewer, energetic pose.", bbox=[260, 300, 760, 560]),
        B.text("+1", "Green '+1' reward popup, upper area.", bbox=[181, 296, 326, 385], color_palette=["#5AA02C", "#FFFFFF"]),
        B.text("+1", "Green '+1' reward popup, lower left.", bbox=[498, 266, 669, 368], color_palette=["#5AA02C", "#FFFFFF"]),
    ],
)

CAPS["3310460039_t0"] = B.build(  # Barry's Prison Run (obby)
    "A cartoon Roblox obby thumbnail of a giant angry policeman leaning in from the left with a baton, a prison cell and tiny prisoner on the right, and a blue PLAY button at the bottom.",
    B.style_block(
        "punchy, saturated, comedic-menacing, readable at small size, chase_danger archetype",
        "bright even indoor lighting, yellow corridor glow",
        "3d_render",
        "Roblox-inspired blocky 3D render, glossy plastic skin, hugely exaggerated rounded proportions, bold cartoon shapes",
        pal("3310460039_t0"),
    ),
    "A yellow-and-grey prison corridor with a metal-barred cell on the right and a striped hazard wall, vanishing toward the back.",
    [
        B.obj("A giant angry cartoon policeman filling the left side: bald head, blue police cap and uniform, huge round belly, furious eyes and gritted teeth, gripping a wooden baton across his body.", bbox=[60, 0, 900, 470]),
        B.obj("A small Roblox prisoner in an orange jumpsuit standing inside the barred cell on the right, dwarfed by the policeman.", bbox=[470, 560, 720, 660]),
        B.text("MOST EVIL PRISONERS CELL", "Bold dark uppercase banner text across the very top against a wood sign.", bbox=[2, 300, 150, 900], color_palette=["#3A2A1A", "#FFFFFF"]),
        B.text("PRISON\nPLAY", "White 'PRISON' label above a blue 'PLAY' button flanked by grey arrows, centered along the bottom.", bbox=[748, 430, 895, 560], color_palette=["#3A6EA5", "#FFFFFF"]),
    ],
)

CAPS["7326934954_t0"] = B.build(  # 99 Nights in the Forest (horror/survival)
    "A Roblox survival-horror thumbnail of two small torch-carrying survivors in a fenced forest clearing at night facing a giant glowing red beast emerging from the dark trees.",
    B.style_block(
        "moody, high-contrast, ominous, readable at small size, chase_danger archetype",
        "dark night lighting with a red glow on the beast and warm torchlight on the survivors",
        "3d_render",
        "Roblox-inspired blocky 3D render, matte stylized materials, simple chunky characters, dark stylized forest",
        pal("7326934954_t0"),
    ),
    "A dark forest clearing ringed by a pointed wooden spike fence, dense black-green trees behind, and a packed-dirt ground; a thin UI item bar sits along the very bottom.",
    [
        B.obj("A huge menacing beast at center-back, a snarling red-furred wolf/boar head glowing against the darkness, looming over the clearing as the central threat.", bbox=[60, 330, 560, 650]),
        B.obj("A small blocky survivor on the left holding a lit torch, light outfit, facing the beast.", bbox=[470, 150, 820, 340]),
        B.obj("A small blocky survivor on the right holding a lit torch, mirroring the left one, facing the beast.", bbox=[470, 640, 820, 840]),
    ],
)

CAPS["4778845442_t0"] = B.build(  # Toilet Tower Defense
    "A dark cinematic Roblox tower-defense thumbnail of a towering flaming skull-headed titan wreathed in fire in a moody blue forest, with a TOILET TOWER DEFENSE logo badge in the corner.",
    B.style_block(
        "cinematic, dramatic, high-contrast, ominous, power_fantasy archetype",
        "low-key teal-blue ambient night with intense orange fire glow and god-rays through trees",
        "3d_render",
        "Roblox-inspired stylized 3D render with cinematic grading, glossy metal skull, volumetric fire and embers",
        pal("4778845442_t0"),
    ),
    "A dark blue-teal forest at night with tall silhouetted trees, beams of cold light cutting through, and drifting embers in the air.",
    [
        B.obj("A towering titan at center: a metallic flaming skull for a head atop a dark armored body, wreathed in bright orange fire it holds in its hands, dominating the frame.", bbox=[120, 300, 900, 720]),
        B.text("TOILET TOWER DEFENSE", "Small square logo badge in the lower-right with a stylized toilet icon and stacked text.", bbox=[800, 820, 930, 990], color_palette=["#FFFFFF", "#1B3342"]),
    ],
)

CAPS["2655311011_t0"] = B.build(  # Anime Dimensions Simulator
    "A Roblox anime-simulator thumbnail showing four anime hero portraits in vertical panels on a dark background with a bold ANIME DIMENSIONS title.",
    B.style_block(
        "high-contrast, vivid, anime-poster, readable at small size, collection_grid archetype",
        "dramatic rim lighting on each character against near-black panels",
        "3d_render",
        "anime-styled character render arranged as a four-panel poster, bold cel-shaded look",
        pal("2655311011_t0"),
    ),
    "A near-black background split into four vertical panels, each lit to frame a single anime character bust.",
    [
        B.obj("Far-left panel: a black-haired swordsman in a dark-and-red checkered-pattern outfit (Demon Slayer style), determined expression.", bbox=[120, 0, 820, 250]),
        B.obj("Center-left panel: a spiky blonde ninja in orange (Naruto style), facing forward.", bbox=[80, 250, 820, 500]),
        B.obj("Center-right panel: a calm white-haired character with covered eyes (Gojo style).", bbox=[80, 500, 820, 750]),
        B.obj("Far-right panel: a sharp-eyed blonde character with a confident smirk.", bbox=[120, 750, 820, 1000]),
        B.text("ANIME\nDIMENSIONS", "Bold white title stacked over the lower center, slightly distressed, readable at small size.", bbox=[736, 170, 998, 803], color_palette=["#FFFFFF", "#FF2D55"]),
    ],
)

CAPS["383310974_t0"] = B.build(  # Adopt Me!
    "A bright cute Roblox pet thumbnail of three fluffy animal characters in little outfits filling the frame, with a big ADOPT ME! title across the bottom.",
    B.style_block(
        "adorable, soft, hyper-saturated, friendly, readable at small size, pet_showcase archetype",
        "bright soft daylight with gentle shading on fluffy fur",
        "3d_render",
        "soft rounded 3D pet render, plush fluffy materials, big expressive eyes, toy-like and cute",
        pal("383310974_t0"),
    ),
    "A bright outdoor park with soft red and green foliage and simple trees, warmly blurred behind the pets.",
    [
        B.obj("Left: a fluffy brown bison/yak character with shaggy fur and small horns, wearing a little red neckerchief.", bbox=[120, 0, 860, 330]),
        B.obj("Center: a round tan beaver/groundhog character with huge dark eyes, a green scout ranger hat and red bandana, the cute focal pet.", bbox=[60, 300, 820, 680]),
        B.obj("Right: a tan dog-like character in a brown cowboy hat with a star, glancing to the side.", bbox=[80, 680, 820, 1000]),
        B.text("ADOPT ME!", "Big bold white uppercase title with a dark outline across the lower-left, very readable at small size.", bbox=[803, 9, 972, 586], color_palette=["#FFFFFF", "#1B3342"]),
    ],
)

CAPS["1686885941_t0"] = B.build(  # Brookhaven RP
    "A clean Roblox roleplay thumbnail framed through an open white window onto a modern suburban mansion, with a smiling avatar in a white dress and a red BROOKHAVEN banner.",
    B.style_block(
        "clean, bright, aspirational, readable at small size, roleplay lifestyle archetype",
        "bright midday daylight, clear blue sky, soft even shading",
        "3d_render",
        "Roblox-inspired blocky 3D render, clean matte architecture, simple bright environment",
        pal("1686885941_t0"),
    ),
    "An open white French-window frame opens onto a modern grey-and-white two-story house with a driveway and green lawn under a clear blue sky; a red ribbon banner sits across the lower-left.",
    [
        B.obj("The primary background element: a modern grey-and-white suburban mansion with a garage and large windows, centered beyond the open window.", bbox=[180, 200, 720, 760]),
        B.obj("A female Roblox avatar on the right with long brown hair in a white dress, smiling and gesturing toward the house.", bbox=[280, 720, 900, 980]),
        B.text("BROOKHAVEN", "White uppercase text on a red ribbon banner across the lower-left corner.", bbox=[803, 69, 919, 409], color_palette=["#FFFFFF", "#FF2D55"]),
        B.text("RP", "Small 'RP' badge in the lower-right.", bbox=[820, 860, 930, 980], color_palette=["#FFFFFF", "#5AA02C"]),
    ],
)

CAPS["1202096104_t0"] = B.build(  # Driving Empire
    "A sleek cinematic Roblox racing thumbnail of a dark hypercar speeding on a wet coastal road with green underglow and a bold DRIVING EMPIRE logo at the top.",
    B.style_block(
        "sleek, premium, cinematic, high-contrast, readable at small size, power status archetype",
        "moody coastal daylight with bright reflections on a wet road and neon-green underglow",
        "3d_render",
        "high-detail stylized 3D car render with motion blur and cinematic grading",
        pal("1202096104_t0"),
    ),
    "A wet reflective coastal highway streaking with motion blur toward a bright ocean horizon under a pale sky.",
    [
        B.obj("A dark brown-black hypercar (Koenigsegg-style) angled toward the viewer at speed, large rear wing, glowing neon-green underglow and taillights, dominating the lower frame.", bbox=[250, 60, 900, 980]),
        B.text("DRIVING\nEMPIRE", "Bold black-and-red logo wordmark with diagonal speed lines across the top center.", bbox=[40, 300, 220, 700], color_palette=["#1B1B1B", "#FF2D55", "#FFFFFF"]),
    ],
)

CAPS["65241_t0"] = B.build(  # Natural Disaster Survival
    "A classic bright Roblox disaster thumbnail of a small floating voxel island with a suburban house being smashed by a fiery meteor and a fiery explosion, over blue ocean.",
    B.style_block(
        "bright, punchy, classic-blocky, high-energy, readable at small size, update_event disaster archetype",
        "bright daylight with a vivid orange fireball glow against blue sky",
        "3d_render",
        "Roblox-inspired blocky 3D render, simple voxel terrain and house, bold explosion effects",
        pal("65241_t0"),
    ),
    "A bright blue sky over a calm blue ocean, filling the background behind a small floating island.",
    [
        B.obj("A small floating island with a grass top and chunky dirt/voxel underside, carrying a green-and-white suburban house at its center.", bbox=[300, 180, 900, 860]),
        B.obj("A fiery orange meteor streaking in from the upper right with a flaming tail, about to strike the house.", bbox=[120, 640, 420, 900]),
        B.obj("A bright orange explosion bursting from the left side of the house, scattering blocky debris into the air.", bbox=[120, 150, 520, 520]),
    ],
)


def main():
    Path("data/captions").mkdir(parents=True, exist_ok=True)
    ok = 0
    for stem, cap in CAPS.items():
        w = B.validate(cap)
        if w:
            print(f"[FAIL] {stem}: {w[:2]}")
        else:
            B.save(cap, f"data/captions/{stem}.json")
            ok += 1
    print(f"\n[done] {ok}/{len(CAPS)} captions validated + saved to data/captions/")


if __name__ == "__main__":
    main()
