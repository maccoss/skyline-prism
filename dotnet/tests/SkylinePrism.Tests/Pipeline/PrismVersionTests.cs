using System;
using System.IO;
using System.Text.Json;
using SkylinePrism.Core.Config;
using SkylinePrism.Core.Pipeline;
using Xunit;

namespace SkylinePrism.Tests.Pipeline;

/// <summary>
/// PRISM versions as CalVer <c>YY.feature.patch</c>, and every surface that prints a version has to
/// say exactly that - no padded fourth component.
/// </summary>
public class PrismVersionTests
{
    /// <summary>
    /// The regression this file exists for: <c>PRISM v26.24.2.0</c> in the QC report footer, for a
    /// release tagged <c>dotnet-v26.24.2</c>. The fourth component was MSBuild padding
    /// <c>AssemblyVersion</c> out to four parts, surfaced by <c>Assembly.GetName().Version</c>.
    /// </summary>
    [Fact]
    public void TheVersionHasThreePartsAndNoPadding()
    {
        Assert.Matches(@"^\d+\.\d+\.\d+$", PrismVersion.Current);
        Assert.DoesNotContain("+", PrismVersion.Current);
    }

    /// <summary>
    /// It has to be the BUILD's version, not a literal that drifts. Compared against
    /// <c>AssemblyVersion</c> - which MSBuild also derives from <c>&lt;Version&gt;</c> - so a release
    /// bump moves both together, and this asserts only that the padding is what differs.
    /// </summary>
    [Fact]
    public void TheVersionIsTheAssemblyVersionWithoutItsPadding()
    {
        var assemblyVersion = typeof(PrismVersion).Assembly.GetName().Version;
        Assert.NotNull(assemblyVersion);
        Assert.Equal(
            $"{assemblyVersion!.Major}.{assemblyVersion.Minor}.{assemblyVersion.Build}",
            PrismVersion.Current);

        // The padded form is what used to be printed; keep the distinction visible here, because a
        // future "simplification" back to GetName().Version would still pass the test above.
        Assert.Equal(0, assemblyVersion.Revision);
        Assert.NotEqual(assemblyVersion.ToString(), PrismVersion.Current);
    }

    /// <summary>
    /// The version the user actually sees comes out of parameters.json, not out of the process that
    /// renders the report - <c>prism qc -d</c> re-renders from the file. So the stored form is the one
    /// that matters.
    /// </summary>
    [Fact]
    public void ProvenanceRecordsTheUnpaddedVersion()
    {
        var path = Path.Combine(
            Path.GetTempPath(), "prism_ver_" + Guid.NewGuid().ToString("N") + ".json");
        try
        {
            Provenance.Write(path, new PrismConfig(), new[] { "a.csv" },
                new Provenance.Stats(1, 10, 5, 5), "2026-01-01T00:00:00.0000000Z");

            using var doc = JsonDocument.Parse(File.ReadAllText(path));
            var stored = doc.RootElement.GetProperty("pipeline_version").GetString();
            Assert.Equal(PrismVersion.Current, stored);
            Assert.Matches(@"^\d+\.\d+\.\d+$", stored);

            var info = Provenance.ReadRunInfo(path);
            Assert.Equal(PrismVersion.Current, info!.PipelineVersion);
        }
        finally
        {
            if (File.Exists(path))
                File.Delete(path);
        }
    }
}
