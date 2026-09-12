using SkylinePrism.App;
using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// The precedence behind the Spectrum density picker's starting choice. The ORDER is the content.
/// </summary>
public class DensitySchemeDefaultTests
{
    /// <summary>
    /// THE case to protect: nothing outranks a scheme the user named. The map is re-binned when the
    /// picker moves, so a default that reasserted itself - when the data-file read landed, or when the
    /// run combo moved to another replicate of the same acquisition - would silently redraw a plot
    /// someone was in the middle of reading.
    /// </summary>
    [Fact]
    public void WhatTheUserPickedOutranksEveryGuess()
    {
        Assert.Equal(4, DensitySchemeDefault.Choose(
            documentIndex: -1, keptIndex: 4, measuredIndex: 1, builtInIndex: 2, fallbackIndex: 3));
    }

    /// <summary>
    /// A document that declares its own windows beats even that: the picker is locked on it, because
    /// the acquisition as Skyline recorded it is not something to have an opinion about.
    /// </summary>
    [Fact]
    public void ADocumentsOwnWindowsOutrankEverything()
    {
        Assert.Equal(0, DensitySchemeDefault.Choose(
            documentIndex: 0, keptIndex: 4, measuredIndex: 1, builtInIndex: 2, fallbackIndex: 3));
    }

    /// <summary>
    /// The change this rule exists for: with nothing chosen, windows read out of the instrument files
    /// win over a built-in layout. The built-in is a plausible modern DIA cycle and says nothing on the
    /// map about being a guess, which is what made it the wrong default.
    /// </summary>
    [Fact]
    public void MeasuredWindowsBeatABuiltInGuess()
    {
        Assert.Equal(1, DensitySchemeDefault.Choose(
            documentIndex: -1, keptIndex: -1, measuredIndex: 1, builtInIndex: 2, fallbackIndex: 3));
    }

    /// <summary>With no measurement, the built-in is still a better guess than uniform bins.</summary>
    [Fact]
    public void WithoutAMeasurementTheBuiltInIsStillPreferredToUniformBins()
    {
        Assert.Equal(2, DensitySchemeDefault.Choose(
            documentIndex: -1, keptIndex: -1, measuredIndex: -1, builtInIndex: 2, fallbackIndex: 3));
    }

    /// <summary>Uniform bins are the last resort, and the one entry that always exists.</summary>
    [Fact]
    public void UniformBinsAreTheLastResort()
    {
        Assert.Equal(3, DensitySchemeDefault.Choose(
            documentIndex: -1, keptIndex: -1, measuredIndex: -1, builtInIndex: -1, fallbackIndex: 3));
    }
}
