// Generates "[Tinesh] Sun Dancer.esp" (ESL-flagged, new records only).
// args: <Skyrim.esm> <UBE_AllRace.esp> <out dir> <meshes_report.json> <version>
using System.Text.Json;
using Mutagen.Bethesda;
using Mutagen.Bethesda.Plugins;
using Mutagen.Bethesda.Plugins.Records;
using Mutagen.Bethesda.Skyrim;
using Noggog;

var skyrimPath = args[0];
var ubePath = args[1];
var outDir = args[2];
using var meshReport = JsonDocument.Parse(File.ReadAllText(args[3]));
var version = args[4];

ObjectBounds Bounds(JsonElement e)
{
    var o = e.GetProperty("obnd").EnumerateArray().Select(v => (short)v.GetInt32()).ToArray();
    return new ObjectBounds { First = new P3Int16(o[0], o[1], o[2]), Second = new P3Int16(o[3], o[4], o[5]) };
}
JsonElement Item(string key) => meshReport.RootElement.GetProperty("items").GetProperty(key);

var modKey = ModKey.FromFileName("[Tinesh] Sun Dancer.esp");
FormKey Mine(uint id) => new(modKey, id);
const SkyrimRelease release = SkyrimRelease.SkyrimSE;
const string meshDir = @"!UBE\Tinesh\SunDancer\";

using var skyrim = SkyrimMod.CreateFromBinaryOverlay(skyrimPath, release);
using var ube = SkyrimMod.CreateFromBinaryOverlay(ubePath, release);
T Find<T>(IEnumerable<T> records, string editorId) where T : IMajorRecordGetter =>
    records.FirstOrDefault(r => r.EditorID == editorId) ?? throw new InvalidOperationException($"{editorId} not found in Skyrim.esm");
IFormLinkGetter<IKeywordGetter> Kw(string editorId) => Find(skyrim.Keywords, editorId).ToLinkGetter();
FormKey Ingredient(string editorId) =>
    skyrim.MiscItems.Cast<IMajorRecordGetter>().Concat(skyrim.Ingredients).Concat(skyrim.Ingestibles).Concat(skyrim.SoulGems)
        .FirstOrDefault(r => r.EditorID == editorId)?.FormKey ?? throw new InvalidOperationException($"{editorId} not found in Skyrim.esm");

var mod = new SkyrimMod(modKey, release);
mod.ModHeader.Flags |= SkyrimModHeader.HeaderFlag.Small;
// header version 1.7, not Mutagen's default 1.71: every form ID is 0x800 or above (the classic light-plugin range), so
// the plugin loads on every SE/AE runtime without Backported Extended ESL Support (1.71 needs it before 1.6.1130)
mod.ModHeader.Stats.Version = 1.7f;
mod.ModHeader.Author = "Tinesh";
mod.ModHeader.Description = $"Sun Dancer {version}: linen, leather and gold for a dancer of the sun, made for UBE (eleven light armor pieces: the loincloth with or without its briefs, or the briefs alone; the loincloth and the arm veils on SMP physics) and the Eclipse Scythe, crafted at the forge without perks.";

// UBE races defined by UBE_AllRace.esp (beast races left out: the outfit is made for the UBE human and elf bodies)
var ubeRaces = ube.Races.Where(r => r.FormKey.ModKey == ube.ModKey)
    .Where(r => r.EditorID is { } e && !e.Contains("Khajiit", StringComparison.OrdinalIgnoreCase)
                && !e.Contains("Argonian", StringComparison.OrdinalIgnoreCase))
    .OrderBy(r => r.FormKey.ID).ToList();
if (ubeRaces.Count == 0) throw new InvalidOperationException("no UBE races found");
Console.WriteLine($"UBE races ({ubeRaces.Count}): {string.Join(", ", ubeRaces.Select(r => r.EditorID))}");

var forge = Kw("CraftingSmithingForge");
var armorTable = Kw("CraftingSmithingArmorTable");
var sharpeningWheel = Kw("CraftingSmithingSharpeningWheel");
var leather = Ingredient("Leather01");
var strips = Ingredient("LeatherStrips");
var linen = Ingredient("RuinsLinenPile01");
var gold = Ingredient("IngotGold");
var steel = Ingredient("IngotSteel");
var arcaneBlacksmith = new FormKey(ModKey.FromFileName("Skyrim.esm"), 0x05218E);
var defaultRace = Find(skyrim.Races, "DefaultRace").FormKey;
var clothingUp = Find(skyrim.SoundDescriptors, "ITMClothingUpSD").FormKey;
var clothingDown = Find(skyrim.SoundDescriptors, "ITMClothingDownSD").FormKey;

BipedObjectFlag Slot(int slot) => (BipedObjectFlag)(1u << (slot - 30));

var pieces = new[]
{
    // the top: cups, band, halter and the whole UBE body (the briefs are now worn with the loincloth or alone)
    new { Id = 0u, Key = "wrap", Edid = "Wrap", Name = "Wrap", Slot = 32, Rating = 22f, Weight = 1.6f, Value = 210u, Mesh = "wrap_1.nif",
          Sliders = true, FirstPerson = meshDir + "wrap_1st_1.nif", Sounds = "ArmorLeatherCuirass",
          Keywords = new[] { "ArmorLight", "ArmorCuirass", "ArmorMaterialLeather", "VendorItemArmor" },
          Recipe = new[] { (linen, 1), (strips, 1), (gold, 1) }, Temper = strips },
    // three choices for slot 52 (one replaces the other): the loincloth over the briefs (the original look, same
    // record as before), the loincloth alone, the briefs alone
    new { Id = 1u, Key = "loincloth", Edid = "Loincloth", Name = "Loincloth with Briefs", Slot = 52, Rating = 5f, Weight = 1.8f, Value = 150u, Mesh = "loincloth_1.nif",
          Sliders = true, FirstPerson = "", Sounds = "",
          Keywords = new[] { "ArmorLight", "ArmorMaterialLeather", "VendorItemArmor" },
          Recipe = new[] { (linen, 3), (strips, 1), (gold, 1) }, Temper = strips },
    new { Id = 10u, Key = "loinclothnobriefs", Edid = "LoinclothNoBriefs", Name = "Loincloth without Briefs", Slot = 52, Rating = 4f, Weight = 1.5f, Value = 120u, Mesh = "loinclothnobriefs_1.nif",
          Sliders = true, FirstPerson = "", Sounds = "",
          Keywords = new[] { "ArmorLight", "ArmorMaterialLeather", "VendorItemArmor" },
          Recipe = new[] { (linen, 2), (strips, 1), (gold, 1) }, Temper = strips },
    new { Id = 11u, Key = "briefs", Edid = "Briefs", Name = "Briefs", Slot = 52, Rating = 1f, Weight = 0.3f, Value = 40u, Mesh = "briefs_1.nif",
          Sliders = true, FirstPerson = "", Sounds = "",
          Keywords = new[] { "ArmorLight", "VendorItemClothing" },
          Recipe = new[] { (linen, 1), (gold, 1) }, Temper = linen },
    new { Id = 2u, Key = "sandals", Edid = "Sandals", Name = "Sandals", Slot = 37, Rating = 6f, Weight = 1f, Value = 80u, Mesh = "sandals_1.nif",
          Sliders = true, FirstPerson = "", Sounds = "ArmorLeatherBoots",
          Keywords = new[] { "ArmorLight", "ArmorBoots", "ArmorMaterialLeather", "VendorItemArmor" },
          Recipe = new[] { (leather, 1), (strips, 2) }, Temper = strips },
    // gold sun bracelets at the wrists (they replaced the forearm guards, under the same record: worn copies update)
    new { Id = 3u, Key = "bracelets", Edid = "Bracelets", Name = "Bracelets", Slot = 34, Rating = 6f, Weight = 0.3f, Value = 120u, Mesh = "bracelets_1.nif",
          Sliders = true, FirstPerson = meshDir + "bracelets_1.nif", Sounds = "",
          Keywords = new[] { "ArmorLight", "ArmorGauntlets", "VendorItemJewelry" },
          Recipe = new[] { (gold, 1) }, Temper = gold },
    new { Id = 4u, Key = "armlets", Edid = "Armlets", Name = "Armlets", Slot = 58, Rating = 3f, Weight = 0.5f, Value = 90u, Mesh = "armlets_1.nif",
          Sliders = true, FirstPerson = meshDir + "armlets_1.nif", Sounds = "",
          Keywords = new[] { "ArmorLight", "VendorItemJewelry" },
          Recipe = new[] { (gold, 1) }, Temper = gold },
    new { Id = 5u, Key = "collar", Edid = "Collar", Name = "Collar", Slot = 45, Rating = 3f, Weight = 0.6f, Value = 190u, Mesh = "collar_1.nif",
          Sliders = true, FirstPerson = "", Sounds = "",
          Keywords = new[] { "ArmorLight", "ClothingNecklace", "VendorItemJewelry" },
          Recipe = new[] { (gold, 1), (strips, 1) }, Temper = gold },
    new { Id = 6u, Key = "circlet", Edid = "Circlet", Name = "Circlet", Slot = 42, Rating = 9f, Weight = 0.3f, Value = 130u, Mesh = "circlet.nif",
          Sliders = false, FirstPerson = "", Sounds = "",
          Keywords = new[] { "ArmorLight", "ArmorHelmet", "ClothingCirclet", "VendorItemJewelry" },
          Recipe = new[] { (gold, 1) }, Temper = gold },
    // (Id 7 was the Long Skirt, removed before release; the id is left unused)
    new { Id = 8u, Key = "navel", Edid = "NavelCharm", Name = "Navel Charm", Slot = 49, Rating = 1f, Weight = 0.1f, Value = 110u, Mesh = "navel_1.nif",
          Sliders = true, FirstPerson = "", Sounds = "",
          Keywords = new[] { "ArmorLight", "VendorItemJewelry" },
          Recipe = new[] { (gold, 1) }, Temper = gold },
    // sheer chiffon veils on SMP physics, hanging from below the armlets (a separate item: worn or not)
    new { Id = 9u, Key = "veils", Edid = "ArmVeils", Name = "Arm Veils", Slot = 59, Rating = 1f, Weight = 0.2f, Value = 90u, Mesh = "veils_1.nif",
          Sliders = true, FirstPerson = "", Sounds = "",
          Keywords = new[] { "ArmorLight", "VendorItemClothing" },
          Recipe = new[] { (linen, 1), (gold, 1) }, Temper = strips },
};

ConstructibleObject Recipe(uint id, string editorId, FormKey created, IFormLinkGetter<IKeywordGetter> bench, IEnumerable<(FormKey, int)> items)
{
    var c = new ConstructibleObject(Mine(id), release)
    {
        EditorID = editorId,
        CreatedObjectCount = 1,
        Items = new ExtendedList<ContainerEntry>(items.Select(i => new ContainerEntry
        {
            Item = new ContainerItem { Item = new FormLink<IItemGetter>(i.Item1), Count = i.Item2 },
        })),
    };
    c.CreatedObject.SetTo(created);
    c.WorkbenchKeyword.SetTo(bench.FormKey);
    mod.ConstructibleObjects.Add(c);
    return c;
}

void Temper(uint id, string editorId, FormKey created, IFormLinkGetter<IKeywordGetter> bench, FormKey material, bool needsArcanePerk = true)
{
    var temper = Recipe(id, editorId, created, bench, new[] { (material, 1) });
    // the base game's rule: enchanted items need Arcane Blacksmith. The scythe comes enchanted from the forge and is
    // tempered without perks, like everything else in the set.
    if (!needsArcanePerk) return;
    temper.Conditions.Add(new ConditionFloat
    {
        CompareOperator = CompareOperator.NotEqualTo, ComparisonValue = 1, Flags = Condition.Flag.OR,
        Data = new EPTemperingItemIsEnchantedConditionData(),
    });
    var arcane = new HasPerkConditionData();
    arcane.Perk.Link.SetTo(arcaneBlacksmith);
    temper.Conditions.Add(new ConditionFloat { CompareOperator = CompareOperator.EqualTo, ComparisonValue = 1, Data = arcane });
}

foreach (var p in pieces)
{
    var flags = Slot(p.Slot);
    var addon = new ArmorAddon(Mine(0x810 + p.Id), release)
    {
        EditorID = $"SunDancer{p.Edid}AA",
        BodyTemplate = new BodyTemplate { FirstPersonFlags = flags, ArmorType = ArmorType.LightArmor },
        Priority = new GenderedItem<byte>(0, 0),
        WeightSliderEnabled = new GenderedItem<bool>(false, p.Sliders),
        WorldModel = new GenderedItem<Model?>(null, new Model { File = meshDir + p.Mesh }),
        FirstPersonModel = new GenderedItem<Model?>(null, p.FirstPerson.Length > 0 ? new Model { File = p.FirstPerson } : null),
    };
    addon.Race.SetTo(ubeRaces[0].FormKey);
    foreach (var r in ubeRaces.Skip(1))
        addon.AdditionalRaces.Add(new FormLink<IRaceGetter>(r.FormKey));
    mod.ArmorAddons.Add(addon);

    var ground = meshDir + p.Mesh.Replace("_1.nif", ".nif").Replace(".nif", "_gnd.nif");
    var armor = new Armor(Mine(0x800 + p.Id), release)
    {
        EditorID = $"SunDancer{p.Edid}",
        Name = $"Sun Dancer {p.Name}",
        ObjectBounds = Bounds(Item(p.Key).GetProperty("gnd").GetProperty("bounds")),
        BodyTemplate = new BodyTemplate { FirstPersonFlags = flags, ArmorType = ArmorType.LightArmor },
        WorldModel = new GenderedItem<ArmorModel?>(new ArmorModel { Model = new Model { File = ground } },
                                                   new ArmorModel { Model = new Model { File = ground } }),
        ArmorRating = p.Rating,
        Weight = p.Weight,
        Value = p.Value,
        Keywords = new ExtendedList<IFormLinkGetter<IKeywordGetter>>(p.Keywords.Select(Kw)),
    };
    armor.Race.SetTo(defaultRace);
    if (p.Sounds.Length > 0)
    {
        var template = Find(skyrim.Armors, p.Sounds);
        armor.PickUpSound.SetTo(template.PickUpSound.FormKeyNullable);
        armor.PutDownSound.SetTo(template.PutDownSound.FormKeyNullable);
    }
    else if (!p.Keywords.Contains("VendorItemJewelry"))
    {
        armor.PickUpSound.SetTo(clothingUp);
        armor.PutDownSound.SetTo(clothingDown);
    }
    armor.Armature.Add(addon.ToLink());
    mod.Armors.Add(armor);

    Recipe(0x820 + p.Id, $"RecipeSunDancer{p.Edid}", armor.FormKey, forge, p.Recipe);
    Temper(0x830 + p.Id, $"TemperSunDancer{p.Edid}", armor.FormKey, armorTable, p.Temper);
}

// Eclipse Scythe: a two-handed weapon on the battleaxe animations (like the scythes of the base game's mods);
// the "Scythe" name lets keyword distributors tag it as a scythe.
var source = Find(skyrim.Weapons, "SteelBattleaxe");
var bounds = Bounds(Item("scythe").GetProperty("bounds"));
var firstPerson = skyrim.Statics.First(s => s.FormKey == source.FirstPersonModel.FormKey).Duplicate(Mine(0x841));
firstPerson.EditorID = "1stPersonEclipseScythe";
firstPerson.Model = new Model { File = @"Tinesh\SunDancer\scythe.nif" };
firstPerson.ObjectBounds = bounds;
mod.Statics.Add(firstPerson);
var weapon = source.Duplicate(Mine(0x840));
weapon.EditorID = "EclipseScythe";
weapon.Name = "Eclipse Scythe";
weapon.Model = new Model { File = @"Tinesh\SunDancer\scythe.nif" };
weapon.ObjectBounds = bounds;
weapon.FirstPersonModel.SetTo(firstPerson.FormKey);
weapon.BasicStats = new WeaponBasicStats { Damage = 19, Weight = 20f, Value = 1150 };
weapon.Critical!.Damage = 9;
weapon.Data!.Reach = 1.35f;     // a longer haft and a larger blade than before
weapon.Data.Speed = 0.7f;

// Eclipse Brand: sun-fire on every hit (a golden flare on the target, a corona round the blade), the undead burn
// twice as hard, and one hit in four lets the eclipse fall on the target (a dark shroud with a gold rim) and drains its
// stamina. The effect shaders are the base game's, with the eclipse's own palettes (tools/make_fx.py).
const string fx = @"Textures\Tinesh\SunDancer\fx\";
EffectShader Shader(uint id, string editorId, string source, Action<EffectShader> tweak)
{
    var s = Find(skyrim.EffectShaders, source).Duplicate(Mine(id));
    s.EditorID = editorId;
    tweak(s);
    mod.EffectShaders.Add(s);
    return s;
}
var coronaFx = Shader(0x850, "EclipseScytheCoronaFXS", "EnchFireFXShader", s =>
{
    s.MembranePaletteTexture = fx + "GradEclipseCorona.dds";
    s.FillAlphaPulseFrequency = 0.08f;
    // a golden rim round the blade that breathes slowly (the base game's fire enchantment has no rim)
    s.EdgeEffectColor = System.Drawing.Color.FromArgb(0, 255, 178, 72);
    s.EdgeEffectFallOff = 1.2f;
    s.EdgeEffectPersistentAlphaRatio = 0.45f;
    s.EdgeEffectFullAlphaRatio = 0.75f;
    s.EdgeEffectAlphaPulseAmplitude = 0.5f;
    s.EdgeEffectAlphaPulseFrequency = 0.35f;
});
var flareFx = Shader(0x851, "EclipseSolarFlareFXS", "FireFXShader", s =>
{
    s.MembranePaletteTexture = fx + "GradEclipseFlare.dds";
    s.ParticlePaletteTexture = fx + "GradEclipseSparks.dds";
    s.EdgeEffectColor = System.Drawing.Color.FromArgb(0, 255, 176, 64);
    s.ParticleFullBirthRatio = 60;      // a little lighter than the base game's fire (90)
    s.ParticlePeristentCount = 45;
});
var umbraFx = Shader(0x852, "EclipseUmbraFXS", "DA02ArmorShadow", s =>
{
    s.MembranePaletteTexture = fx + "GradEclipseUmbra.dds";
    s.ParticlePaletteTexture = fx + "GradEclipseUmbraSmoke.dds";
    s.AmbientSound.Clear();
    s.FillAlphaFadeInTime = 0.15f;
    s.FillFullAlphaTime = 1.2f;
    s.FillFadeOutTime = 1.0f;
    s.EdgeEffectAlphaFadeInTime = 0.15f;
    s.EdgeEffectFullAlphaTime = 1.2f;
    s.EdgeEffectAlphaFadeOutTime = 1.0f;
    s.EdgeEffectColor = System.Drawing.Color.FromArgb(0, 255, 196, 96);
    s.ParticleFullBirthRatio = 24;
    s.ParticlePeristentCount = 18;
});
var fireBase = Find(skyrim.MagicEffects, "EnchFireDamageFFContact");
MagicEffect Effect(uint id, string editorId, string name, string description, Action<MagicEffect> tweak)
{
    var m = fireBase.Duplicate(Mine(id));
    m.EditorID = editorId;
    m.Name = name;
    m.Description = description;
    tweak(m);
    mod.MagicEffects.Add(m);
    return m;
}
var flare = Effect(0x853, "EclipseSolarFlareFFContact", "Solar Flare", "Sun-fire sears the target for <mag> points.", m =>
{
    m.EnchantShader.SetTo(coronaFx);
    m.HitShader.SetTo(flareFx);
});
var corona = Effect(0x854, "EclipseCoronaFFContact", "Corona", "The undead burn for <mag> more.", m =>
{
    m.EnchantShader.Clear();
    m.HitShader.SetTo(flareFx);
    m.ResistValue = ActorValue.None;    // the sun's light, not flame: no fire resistance
});
var umbra = Effect(0x855, "EclipseUmbraFFContact", "Umbra", "At times the eclipse falls on the target and drains <mag> points of its stamina.", m =>
{
    m.Archetype.ActorValue = ActorValue.Stamina;
    m.ResistValue = ActorValue.ResistMagic;
    m.EnchantShader.Clear();
    m.HitShader.SetTo(umbraFx);
    m.Keywords?.Clear();
});
ConditionFloat Undead()
{
    var data = new HasKeywordConditionData();
    data.Keyword.Link.SetTo(Kw("ActorTypeUndead").FormKey);
    return new ConditionFloat { CompareOperator = CompareOperator.EqualTo, ComparisonValue = 1, Data = data };
}
var brand = new ObjectEffect(Mine(0x856), release)
{
    EditorID = "EclipseBrandEnch",
    Name = "Eclipse Brand",
    EnchantType = ObjectEffect.EnchantTypeEnum.Enchantment,
    CastType = CastType.FireAndForget,
    TargetType = TargetType.Touch,
    Flags = ObjectEffect.Flag.NoAutoCalc,
    EnchantmentCost = 30,
    EnchantmentAmount = 30,
    Effects = new ExtendedList<Effect>
    {
        new Effect { BaseEffect = flare.ToNullableLink(), Data = new EffectData { Magnitude = 10, Area = 0, Duration = 0 } },
        new Effect
        {
            BaseEffect = corona.ToNullableLink(), Data = new EffectData { Magnitude = 10, Area = 0, Duration = 0 },
            Conditions = new ExtendedList<Condition> { Undead() },
        },
        new Effect
        {
            BaseEffect = umbra.ToNullableLink(), Data = new EffectData { Magnitude = 25, Area = 0, Duration = 0 },
            Conditions = new ExtendedList<Condition>
            {
                new ConditionFloat { CompareOperator = CompareOperator.LessThan, ComparisonValue = 25, Data = new GetRandomPercentConditionData() },
            },
        },
    },
};
mod.ObjectEffects.Add(brand);
weapon.ObjectEffect.SetTo(brand);
weapon.EnchantmentAmount = 3000;     // about 100 hits per full charge
weapon.Keywords!.Add(Kw("MagicDisallowEnchanting"));
mod.Weapons.Add(weapon);
Recipe(0x842, "RecipeEclipseScythe", weapon.FormKey, forge, new[] { (steel, 3), (gold, 1), (strips, 1), (Ingredient("FireSalts"), 1) });
Temper(0x843, "TemperEclipseScythe", weapon.FormKey, sharpeningWheel, steel, needsArcanePerk: false);

mod.ModHeader.Stats.NextFormID = 0x860;
Directory.CreateDirectory(outDir);
var outPath = Path.Combine(outDir, modKey.FileName);
mod.BeginWrite.ToPath(outPath).WithNoLoadOrder().Write();
Console.WriteLine($"WROTE {outPath} ({new FileInfo(outPath).Length} bytes)");

using var back = SkyrimMod.CreateFromBinaryOverlay(outPath, release);
Console.WriteLine($"HEADER ESL={back.ModHeader.Flags.HasFlag(SkyrimModHeader.HeaderFlag.Small)} next={back.ModHeader.Stats.NextFormID:X} records={back.ModHeader.Stats.NumRecords} masters={string.Join(",", back.ModHeader.MasterReferences.Select(m => m.Master.FileName))} author='{back.ModHeader.Author}'");
foreach (var a in back.Armors)
    Console.WriteLine($"  ARMO {a.FormKey.ID:X3} {a.EditorID}: '{a.Name?.String}' type={a.BodyTemplate?.ArmorType} slots={a.BodyTemplate?.FirstPersonFlags} AR={a.ArmorRating} W={a.Weight} V={a.Value} gnd={a.WorldModel?.Female?.Model?.File} kw={a.Keywords!.Count}");
foreach (var aa in back.ArmorAddons)
    Console.WriteLine($"  ARMA {aa.FormKey.ID:X3} {aa.EditorID}: race={aa.Race.FormKey} +{aa.AdditionalRaces.Count} female={aa.WorldModel?.Female?.File} slider={aa.WeightSliderEnabled.Female} 1st={aa.FirstPersonModel?.Female?.File}");
foreach (var w in back.Weapons)
    Console.WriteLine($"  WEAP {w.FormKey.ID:X3} {w.EditorID}: '{w.Name?.String}' anim={w.Data?.AnimationType} skill={w.Data?.Skill} dmg={w.BasicStats?.Damage} W={w.BasicStats?.Weight} V={w.BasicStats?.Value} speed={w.Data?.Speed} reach={w.Data?.Reach} crit={w.Critical?.Damage} ench={w.ObjectEffect.FormKey.ID:X3} charge={w.EnchantmentAmount} model={w.Model?.File} equip={w.EquipmentType.FormKeyNullable} kw={string.Join(",", w.Keywords!.Select(k => k.FormKey.ID.ToString("X6")))}");
foreach (var c in back.ConstructibleObjects)
    Console.WriteLine($"  COBJ {c.FormKey.ID:X3} {c.EditorID}: creates={c.CreatedObject.FormKey.ID:X3} bench={c.WorkbenchKeyword.FormKey.ID:X6} items={string.Join("+", c.Items!.Select(i => $"{i.Item.Count}x{i.Item.Item.FormKey.ID:X6}"))} conditions={c.Conditions.Count}");
foreach (var s in back.EffectShaders)
    Console.WriteLine($"  EFSH {s.FormKey.ID:X3} {s.EditorID}: palette={s.MembranePaletteTexture} particles={s.ParticlePaletteTexture} edge={s.EdgeEffectColor}");
foreach (var m in back.MagicEffects)
    Console.WriteLine($"  MGEF {m.FormKey.ID:X3} {m.EditorID}: '{m.Name?.String}' av={m.Archetype.ActorValue} resist={m.ResistValue} ench={m.EnchantShader.FormKey.ID:X3} hit={m.HitShader.FormKey.ID:X3}");
foreach (var e in back.ObjectEffects)
    Console.WriteLine($"  ENCH {e.FormKey.ID:X3} {e.EditorID}: '{e.Name?.String}' cost={e.EnchantmentCost} effects={string.Join(", ", e.Effects.Select(x => $"{x.BaseEffect.FormKey.ID:X3} mag={x.Data?.Magnitude} conds={x.Conditions.Count}"))}");
