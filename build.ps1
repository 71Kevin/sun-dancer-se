<#
.SYNOPSIS
    Builds Sun Dancer for Skyrim SE/AE (UBE) and packages it in 2K, 4K and 8K, with optional PBR add-ons for
    PG Patcher.

.DESCRIPTION
    1. Reference data (UBE body, feet, hands, head, sliders, test presets, skeleton) from the Mod Organizer 2 mods
       folder -> -Work\ref (tools\extract_refs.py, only if missing).
    2. Meshes, BodySlide projects, ground models, first-person arms, the cloth physics (SMP system files) and the
       scythe (tools\build_meshes.py, Blender); the scythe enchantment's effect palettes (tools\make_fx.py).
    3. Texture atlases painted at 8192 px, reduced to 4096 and 2048 px, vanilla and PBR sets (tools\make_textures.py).
    4. Plugin "[Tinesh] Sun Dancer.esp" (plugin\, Mutagen).
    5. Packages in -Packages:
         Sun Dancer - UBE (<res>) - <version>.7z              plugin, meshes, BodySlide files, vanilla textures
         Sun Dancer - PBR for PG Patcher (<res>) - <version>.7z  PBR textures + PBRNifPatcher JSON only
       The PBR add-on is installed on top of the main package of the same resolution; PG Patcher turns the meshes
       into TruePBR when it is run.
    With -Install the chosen resolution replaces the contents of the two Mod Organizer 2 mod folders (meta.ini is
    kept); Mod Organizer 2 must be closed.
    With -Public the BodySlide data carry only UBE's own sliders: the sliders of an add-on to the UBE slider sets
    (such as the Necoco add-on, when it is installed) are left out, since their data are not ours to redistribute.
    Release packages are always built with -Public.

.EXAMPLE
    .\build.ps1 -Public            # the release packages
    .\build.ps1 -Install           # a personal build (add-on sliders kept) installed into the Mod Organizer 2 folders
#>
param(
    [string] $Version = "1.0",
    [string] $Work = "$PSScriptRoot\cache",
    [string] $Out = "$PSScriptRoot\out",
    [string] $Packages = "$PSScriptRoot\dist",
    [string] $Game = "C:\Modding\TrueNorth\Stock Game\Data",
    [string] $Mods = "C:\Modding\TrueNorth\mods",
    [string] $ModFolder = "C:\Modding\TrueNorth\mods\[NoDelete] Tinesh - Sun Dancer (UBE)",
    [string] $PbrFolder = "C:\Modding\TrueNorth\mods\[NoDelete] Tinesh - Sun Dancer PBR (PG Patcher)",
    [string] $Blender = "C:\Modding\_tools\blender\blender-4.5.10-windows-x64\blender.exe",
    [string] $SevenZip = "C:\Modding\_tools\7z\7z.exe",
    [string] $Dotnet = "C:\Modding\_tools\dotnet\dotnet.exe",
    [ValidateSet("2K", "4K", "8K")] [string] $InstallResolution = "8K",
    [string[]] $Resolutions = @("2K", "4K", "8K"),
    [switch] $SkipMeshes,
    [switch] $SkipTextures,
    [switch] $Install,
    [switch] $Public
)
$ErrorActionPreference = "Continue"
$env:MK_MODS = $Mods
$env:MK_PUBLIC = $(if ($Public) { "1" } else { "" })
$env:DOTNET_ROOT = Split-Path $Dotnet
$env:DOTNET_CLI_TELEMETRY_OPTOUT = "1"
$resolutions = $Resolutions
$data = Join-Path $Out "data"
$textureSets = Join-Path $Out "textures"
$report = Join-Path $Out "meshes_report.json"
$ube = Join-Path $Mods "[NoDelete] UBE 2.0 - Ultimate Body Enhancer\UBE_AllRace.esp"

if (-not (Test-Path (Join-Path $Work "ref\body.pkl"))) {
    & $Blender --background --factory-startup --python "$PSScriptRoot\tools\extract_refs.py" -- (Join-Path $Work "ref") |
        Select-String -Pattern "sliders|verts|presets|Error|Traceback"
    if (-not (Test-Path (Join-Path $Work "ref\body.pkl"))) { throw "Reference extraction failed" }
}
$vanilla = Join-Path $Work "refs\vanilla"
$needed = @("meshes\armor\studded\male\body_go.nif", "meshes\weapons\steel\steelbattleaxe.nif",
    "meshes\weapons\steel\1stpersonsteelwaraxe.nif")
foreach ($bsa in @("Skyrim - Meshes0.bsa", "Skyrim - Meshes1.bsa")) {
    $missing = $needed | Where-Object { -not (Test-Path (Join-Path $vanilla $_)) }
    if ($missing) { python "$PSScriptRoot\tools\lib\bsa_get.py" "$Game\$bsa" $vanilla @missing | Out-Null }
}

if (-not $SkipMeshes) {
    foreach ($generated in @("meshes", "CalienteTools")) {
        $path = Join-Path $data $generated
        if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path -Recurse -Force }
    }
    & $Blender --background --factory-startup --python "$PSScriptRoot\tools\build_meshes.py" -- $Work $Out |
        Select-String -Pattern "public build|built|exported|first-person|done|Error|Traceback"
    if (-not (Test-Path $report)) { throw "Mesh build failed" }
}
# the enchantment's effect palettes (Eclipse Scythe): data\textures\Tinesh\SunDancer\fx
python "$PSScriptRoot\tools\make_fx.py" $Work $data $Game | Select-String -Pattern "fx palette|Error|Traceback"
if ($LASTEXITCODE) { throw "Effect palettes failed" }
if (-not $SkipTextures) {
    if (Test-Path -LiteralPath $textureSets) { Remove-Item -LiteralPath $textureSets -Recurse -Force }
    python "$PSScriptRoot\tools\make_textures.py" $Work $textureSets --sets ($resolutions -join ",") |
        Select-String -Pattern "compressed|Error|Traceback"
    if ($LASTEXITCODE) { throw "Texture build failed" }
}

& $Dotnet run -c Release --project "$PSScriptRoot\plugin" -- "$Game\Skyrim.esm" $ube $data $report $Version
if ($LASTEXITCODE) { throw "Plugin generation failed" }

New-Item -ItemType Directory -Force $Packages | Out-Null
foreach ($res in $resolutions) {
    $stage = Join-Path $Out "package\main\$res"
    if (Test-Path -LiteralPath $stage) { Remove-Item -LiteralPath $stage -Recurse -Force }
    New-Item -ItemType Directory -Force (Join-Path $stage "textures\Tinesh\SunDancer") | Out-Null
    Get-ChildItem -LiteralPath $data | ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $stage -Recurse -Force }
    Copy-Item (Join-Path $textureSets "$res\vanilla\*.dds") (Join-Path $stage "textures\Tinesh\SunDancer")
    $pbr = Join-Path $Out "package\pbr\$res"
    if (Test-Path -LiteralPath $pbr) { Remove-Item -LiteralPath $pbr -Recurse -Force }
    New-Item -ItemType Directory -Force (Join-Path $pbr "textures\pbr\Tinesh\SunDancer") | Out-Null
    Copy-Item -Recurse "$PSScriptRoot\pbr\PBRNifPatcher" $pbr
    Copy-Item (Join-Path $textureSets "$res\pbr\*.dds") (Join-Path $pbr "textures\pbr\Tinesh\SunDancer")
    foreach ($pkg in @(@($stage, "Sun Dancer - UBE ($res) - $Version.7z"),
                       @($pbr, "Sun Dancer - PBR for PG Patcher ($res) - $Version.7z"))) {
        $archive = Join-Path $Packages $pkg[1]
        if (Test-Path -LiteralPath $archive) { Remove-Item -LiteralPath $archive -Force }
        & $SevenZip a -t7z -mx=9 $archive "$($pkg[0])\*" | Out-Null
        if ($LASTEXITCODE) { throw "Packaging failed: $($pkg[1])" }
        "{0}  {1:N1} MB" -f $archive, ((Get-Item -LiteralPath $archive).Length / 1MB)
    }
}

if ($Install) {
    if (Get-Process ModOrganizer -ErrorAction SilentlyContinue) { throw "Close Mod Organizer 2 before installing." }
    foreach ($pair in @(@($ModFolder, (Join-Path $Out "package\main\$InstallResolution")),
                        @($PbrFolder, (Join-Path $Out "package\pbr\$InstallResolution")))) {
        New-Item -ItemType Directory -Force -Path $pair[0] | Out-Null
        Get-ChildItem -LiteralPath $pair[0] | Where-Object { $_.Name -ne "meta.ini" } | Remove-Item -Recurse -Force
        Get-ChildItem -LiteralPath $pair[1] | ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $pair[0] -Recurse -Force }
        Write-Host "Installed $InstallResolution to $($pair[0])"
    }
}
