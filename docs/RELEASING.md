# Release checklist

1. **Credits and permissions.** Go through [PUBLISHING.md](PUBLISHING.md): every file in the packages is original,
   derived from the game, or derived from UBE 2.0 (the body, feet and arms the wrap, sandals and first-person model
   carry, with their sliders); check UBE's current permissions and credit every source in the README and the release
   notes.
2. **Version.** Set `$Version` in `build.ps1` and add a section with the release date to `CHANGELOG.md`.
3. **Build.** Run `.\build.ps1 -Public`; the packages land in `dist\` as `Sun Dancer - UBE (<res>) - <version>.7z`
   and `Sun Dancer - PBR for PG Patcher (<res>) - <version>.7z`. `-Public` keeps only UBE's own sliders in the
   BodySlide data (the sliders of third-party add-ons to the UBE slider sets are never packaged).
4. **Check the packages.**
   - `python tools\dev\verify_packages.py <version>`: `7z t` on every archive; the outfit archives hold the plugin,
     `meshes`, `textures` and `CalienteTools` at the root (no `Data` level) and the PBR archives only
     `PBRNifPatcher` and `textures\pbr`; the three outfit packages share the same plugin, meshes and BodySlide files;
     DDS sizes and full mip chains per resolution; no add-on slider in the BodySlide data.
   - `tools\check_nif.py` on every NIF (no unweighted vertex, at most 4 bones per vertex, shapes under 65535
     vertices); `tools\clip_test.py` report reviewed (`docs\QA.md`).
   - In game, from a package installed in Mod Organizer 2: BodySlide Batch Build with Build Morphs, every item in
     third and first person, the physics of the panels and veils, the recipes, the scythe in combat, and the PBR
     add-on after a PG Patcher run.
5. **Release assets.** Copy the tested packages as `Sun-Dancer-UBE-<res>-<version>.7z` and
   `Sun-Dancer-PBR-PG-Patcher-<res>-<version>.7z` (GitHub turns spaces and brackets in asset names into dots) and
   note their SHA-256 (`Get-FileHash`).
6. **Release.** Create the release as a draft, check it, then publish it (the tag `v<version>` on `main`):

   ```powershell
   gh release create v1.0 --draft --target main --title "Sun Dancer 1.0" --notes-file notes.md `
       "dist\Sun-Dancer-UBE-2K-1.0.7z#Sun Dancer - UBE (2K) - 1.0" `
       "dist\Sun-Dancer-UBE-4K-1.0.7z#Sun Dancer - UBE (4K) - 1.0" `
       "dist\Sun-Dancer-UBE-8K-1.0.7z#Sun Dancer - UBE (8K) - 1.0" `
       "dist\Sun-Dancer-PBR-PG-Patcher-2K-1.0.7z#Sun Dancer - PBR for PG Patcher (2K) - 1.0" `
       "dist\Sun-Dancer-PBR-PG-Patcher-4K-1.0.7z#Sun Dancer - PBR for PG Patcher (4K) - 1.0" `
       "dist\Sun-Dancer-PBR-PG-Patcher-8K-1.0.7z#Sun Dancer - PBR for PG Patcher (8K) - 1.0"
   gh release edit v1.0 --draft=false
   ```

7. **README and Nexus Mods.** Point the Downloads table at the new assets
   (`https://github.com/71Kevin/sun-dancer-se/releases/download/v<version>/<file>`) and upload the same packages to
   the [Nexus Mods page](https://www.nexusmods.com/skyrimspecialedition/mods/200125) (main files and optional files).
8. **Verify.** Download every asset from the release page and compare its SHA-256 with the tested package.
