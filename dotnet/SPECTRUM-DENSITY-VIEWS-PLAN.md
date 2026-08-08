# Spectrum density: three views (handoff)

**Status: Core is done and tested; the UI is not wired.** Two summary methods exist on
`PrecursorDensityMap` and nothing calls them yet.

## What was asked for

The Spectrum density tab currently draws one thing, a heatmap. Add a view switch with three options:

1. **Heatmap** — what exists today: isolation window (y) x retention time (x), color = precursors per
   spectrum.
2. **Histogram** — how many spectra had how many precursors. x = precursors in a spectrum, y = number
   of spectra.
3. **Load over time** — x = retention time, y = **mean** precursors per spectrum at that time, with
   **dashed lines for the max and the min** at each time.

## What is already done

`dotnet/src/SkylinePrism.Core/Qc/PrecursorDensity.cs`, on the `PrecursorDensityMap` record:

```csharp
int[] PrecursorsPerSpectrumHistogram();
    // result[n] = number of spectra that resolved exactly n precursors, n = 0..MaxCount

IReadOnlyList<(double TimeMin, double Mean, double Min, double Max)> LoadOverTime();
    // one entry per RT bin, at the bin CENTER; all three are NaN when nothing was acquired then
```

Tested in `dotnet/tests/SkylinePrism.Tests/Qc/DensitySummaryViewsTests.cs` (7 tests), including the
scheduled-PRM case. **Read those tests before changing either method** — the non-obvious behaviour is
described below and they pin it.

## The one thing to understand before drawing anything

A cell in `Counts[i, j]` can mean three different things, and only two of them are spectra:

| Cell | Meaning | Counts as a spectrum? |
|---|---|---|
| 0, window firing | acquired, nothing detected | **yes** — belongs in histogram bin 0 |
| n > 0 | acquired, n precursors co-isolated | **yes** |
| window not firing at that time | no spectrum exists | **no** |

The third case only arises for scheduled acquisitions (PRM/MTM/dynamic DIA), where a window fires
during its own RT interval. Both methods already exclude it via `WasAcquired(i, j)`. Counting it would
put a huge spike on histogram bin 0 and drag every mean down — an artifact of the schedule, not the
data. `LoadOverTime()` returns NaN rather than 0 for such a time, so **the renderer must break the line
at NaN rather than plotting zero**, or a gap in the schedule will read as an idle instrument.

## Where to wire it

Everything funnels through one method, which is why this is a small change:

- `dotnet/src/SkylinePrism.App/MainWindow.Density.cs`
  - `DrawDensity()` (~line 503) — the only place that renders. Branch here on the selected view.
  - `RebinAndDraw()` (~line 483) — builds `_densityMap`; unchanged, all three views read the same map.
  - `UpdateSchemeControls()` — already shows/hides `DensityMzBinBox`. Extend it: the colormap combo
    and the m/z-bin box are heatmap-only.
  - `OnDensityPlotMouseMove` — the hover readout reads m/z and RT off the heatmap. **Guard it**, or it
    will report nonsense coordinates on the other two views.
- `dotnet/src/SkylinePrism.App/MainWindow.xaml` (~line 253-289) — add a view `ComboBox` beside
  `DensityColormapCombo`. `DensityPlot` is the shared `WpfPlot`; all three views draw into it.
- `dotnet/src/SkylinePrism.Core/Visualization/PlotRenderer.cs` — add two draw methods next to
  `DrawPrecursorDensity`. Put the drawing in Core with the others, not in the window.

## Conventions this codebase enforces

Several of these are enforced by tests that will fail the build, not by review:

- **Every `async void` handler must contain a real catch-all** at its own level.
  `UiThreadSafetyTests.EveryAsyncVoidHandler_CatchesItsOwnFailures` fails otherwise. Use
  `ReportHandlerFailure(nameof(...), ex)`. This exists because an escaping exception on a timer
  produced an error dialog every 750 ms until the tool was killed.
- **Nothing reached from a worker thread may touch a WPF control** —
  `UiThreadSafetyTests.WorkerThreadMethods_TouchNoWpfControls`, transitively. Read control values on
  the UI thread and pass them in. `Log` is safe (it marshals to the dispatcher).
- **Plot styling goes through `PlotRenderer`** — call `StyleQcPlot` and friends so fonts and axis
  weights match every other plot. `PlotFontConsistencyTests` covers this.
- **`DensityPlot.Reset()` before drawing.** The color bar attaches as an axis panel and survives
  `Plot.Clear()`, so re-drawing without a reset stacks one color bar per render. The heatmap needs it;
  the other two views should reset too so a leftover color bar does not follow them.
- **American spelling** everywhere, including identifiers and comments.

## Suggested shape

```csharp
private enum DensityView { Heatmap, Histogram, LoadOverTime }

private void DrawDensity()
{
    var map = _densityMap;
    if (map is null || map.IsEmpty) return;

    DensityPlot.Reset();
    switch (SelectedDensityView())
    {
        case DensityView.Histogram:
            PlotRenderer.DrawPrecursorLoadHistogram(DensityPlot.Plot, map);
            break;
        case DensityView.LoadOverTime:
            PlotRenderer.DrawPrecursorLoadOverTime(DensityPlot.Plot, map);
            break;
        default:
            PlotRenderer.DrawPrecursorDensity(DensityPlot.Plot, map, DensityColormap(...));
            break;
    }
    DensityPlot.Refresh();
    // status line: keep the "precursors outside the scheme's windows" warning for all three views -
    // it is how a wrong isolation scheme announces itself.
}
```

For the load-over-time plot: mean as a solid line, min and max as **dashed** (`LinePattern.Dashed`),
all three sharing one color so it reads as one band rather than three series. Consider filling between
min and max at low alpha. Break every series wherever `Mean` is NaN.

## Verification

- `dotnet test dotnet/SkylinePrism.sln -c Debug` — 469 core + 192 Windows tests currently pass.
- Add tests for whatever logic you put in `PlotRenderer` (the existing density renderer tests are the
  model). The two Core methods are already covered.
- Manual: the Spectrum density tab against any output directory containing `merged_data.parquet`. A
  real one is at `V:\Radiation-Oncology\2026-03-Omar-FLASH-Colon\PRISM-Output\`, whose acquisition is
  167 windows of ~3.0014 Th over 400.4-901.7 m/z.
- Ship gate before release: `pwsh -File dotnet/build/package-and-verify.ps1 -Configuration Release`.

## Repository state at handoff

- Branch `feat/refanchored-combat-and-merge-fastpath`, PR **#34** open and green — the **26.10.1**
  patch release, which fixes a crash that makes 26.10.0 unusable with a connected Skyline. **Merge and
  tag that before starting this**, or the fix sits behind a feature.
- The two Core methods and their tests are already in that PR, unused. That is deliberate: the crash
  fix should not wait for a feature.
- After merging: `gh pr merge 34 --merge`, then
  `git tag dotnet-v26.10.1 origin/main && git push origin dotnet-v26.10.1`.
- Also outstanding, unrelated: the published 26.10.0 release body carries a superseded explanation of
  the isolation-import fix; `gh release edit dotnet-v26.10.0 --notes-file release-notes/RELEASE_NOTES_dotnet-v26.10.0.md`
  corrects it without re-tagging.

## Open question worth asking the user first

They said, of isolation schemes generally: *"the isolation lists don't really make sense with modern
DIA data collection"*. If their methods are not a fixed repeating cycle, then "rows = isolation
windows" may be the wrong axis for the heatmap — in which case the histogram and load-over-time views
matter more than the map, and the map may need rethinking rather than extending. Worth asking before
investing in heatmap-specific polish.
