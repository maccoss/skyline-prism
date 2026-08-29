using System;
using System.IO;
using System.Linq;
using SkylinePrism.Core.Config;
using SkylinePrism.Core.Pipeline;
using Xunit;

namespace SkylinePrism.Tests.Pipeline;

/// <summary>
/// Keeping the search database beside the results, so a run stays repeatable after the database moves.
///
/// <para>The behavior that matters is which path wins on the way back. The original, whenever it is
/// still there: the stage cache stamps path, size and write time, so preferring the archived copy would
/// invalidate every downstream stage on a re-run that changed nothing. The copy only when the original
/// has gone - and even then the caller is told, because substituting a database silently is the failure
/// this whole thing exists to prevent.</para>
/// </summary>
public class FastaArchiveTests
{
    private static string TempDir()
    {
        var dir = Path.Combine(Path.GetTempPath(), "prism-fasta-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        return dir;
    }

    private static string WriteFasta(string dir, string name, string body = ">sp|P1|A_HUMAN\nPEPTIDEK\n")
    {
        var path = Path.Combine(dir, name);
        Directory.CreateDirectory(dir);
        File.WriteAllText(path, body);
        return path;
    }

    [Fact]
    public void ItCopiesTheDatabaseBesideTheOutputs()
    {
        var root = TempDir();
        try
        {
            var source = WriteFasta(Path.Combine(root, "databases"), "search.fasta");
            var outDir = Path.Combine(root, "out");
            Directory.CreateDirectory(outDir);
            var config = new PrismConfig();
            config.Parsimony.FastaPath = source;

            var entries = FastaArchive.Archive(config, outDir);

            var entry = Assert.Single(entries);
            Assert.Equal("parsimony.fasta_path", entry.ConfigKey);
            Assert.Equal("fasta/search.fasta", entry.ArchivedPath);
            Assert.True(File.Exists(Path.Combine(outDir, "fasta", "search.fasta")));
            // The config still names the original - see the class summary.
            Assert.Equal(source, config.Parsimony.FastaPath);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void TheOriginalWinsWhileItStillExists()
    {
        var root = TempDir();
        try
        {
            var source = WriteFasta(Path.Combine(root, "databases"), "search.fasta");
            var outDir = Path.Combine(root, "out");
            Directory.CreateDirectory(outDir);
            var config = new PrismConfig();
            config.Parsimony.FastaPath = source;
            var entries = FastaArchive.Archive(config, outDir);

            var redirected = FastaArchive.Restore(config, outDir, entries);

            Assert.Empty(redirected);
            Assert.Equal(source, config.Parsimony.FastaPath);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void TheCopyIsUsedOnceTheOriginalHasGone()
    {
        var root = TempDir();
        try
        {
            var dbDir = Path.Combine(root, "databases");
            var source = WriteFasta(dbDir, "search.fasta");
            var outDir = Path.Combine(root, "out");
            Directory.CreateDirectory(outDir);
            var config = new PrismConfig();
            config.Parsimony.FastaPath = source;
            var entries = FastaArchive.Archive(config, outDir);

            Directory.Delete(dbDir, recursive: true); // the database is reorganized away

            var redirected = FastaArchive.Restore(config, outDir, entries);

            Assert.Equal(new[] { "parsimony.fasta_path" }, redirected);
            Assert.Equal(
                Path.GetFullPath(Path.Combine(outDir, "fasta", "search.fasta")),
                Path.GetFullPath(config.Parsimony.FastaPath!));
            Assert.True(File.Exists(config.Parsimony.FastaPath));
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    /// <summary>
    /// The two keys may name different databases - a curated digest set for iBAQ, the full search FASTA
    /// for parsimony - so each is archived and each is restored to its own copy.
    /// </summary>
    [Fact]
    public void TwoDifferentDatabasesAreArchivedSeparately()
    {
        var root = TempDir();
        try
        {
            var dbDir = Path.Combine(root, "databases");
            var full = WriteFasta(dbDir, "full.fasta", ">sp|P1|A_HUMAN\nPEPTIDEK\n");
            var curated = WriteFasta(dbDir, "curated.fasta", ">sp|P2|B_HUMAN\nOTHERPEPTIDEK\n");
            var outDir = Path.Combine(root, "out");
            Directory.CreateDirectory(outDir);
            var config = new PrismConfig();
            config.Parsimony.FastaPath = full;
            config.ProteinRollup.Ibaq.FastaPath = curated;

            var entries = FastaArchive.Archive(config, outDir);
            Directory.Delete(dbDir, recursive: true);
            var redirected = FastaArchive.Restore(config, outDir, entries);

            Assert.Equal(2, entries.Count);
            Assert.Equal(2, redirected.Count);
            Assert.EndsWith("full.fasta", config.Parsimony.FastaPath);
            Assert.EndsWith("curated.fasta", config.ProteinRollup.Ibaq.FastaPath);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    /// <summary>One file named by both keys is copied once and referenced twice, not duplicated.</summary>
    [Fact]
    public void OneDatabaseNamedTwiceIsArchivedOnce()
    {
        var root = TempDir();
        try
        {
            var source = WriteFasta(Path.Combine(root, "databases"), "search.fasta");
            var outDir = Path.Combine(root, "out");
            Directory.CreateDirectory(outDir);
            var config = new PrismConfig();
            config.Parsimony.FastaPath = source;
            config.ProteinRollup.Ibaq.FastaPath = source;

            var entries = FastaArchive.Archive(config, outDir);

            Assert.Equal(2, entries.Count);
            Assert.Single(entries.Select(e => e.ArchivedPath).Distinct());
            Assert.Single(Directory.GetFiles(Path.Combine(outDir, "fasta")));
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    /// <summary>
    /// A re-run into the same directory reads last run's archive. Copying it onto itself would truncate
    /// the file it is reading from.
    /// </summary>
    [Fact]
    public void AnAlreadyArchivedDatabaseIsNotRecopied()
    {
        var root = TempDir();
        try
        {
            var outDir = Path.Combine(root, "out");
            var inPlace = WriteFasta(Path.Combine(outDir, "fasta"), "search.fasta");
            var config = new PrismConfig();
            config.Parsimony.FastaPath = inPlace;

            var entries = FastaArchive.Archive(config, outDir);

            Assert.Equal("fasta/search.fasta", Assert.Single(entries).ArchivedPath);
            Assert.True(File.Exists(inPlace));
            Assert.NotEmpty(File.ReadAllText(inPlace));
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    /// <summary>A missing or unset database is not an error - most runs have neither.</summary>
    [Fact]
    public void NothingToArchiveIsNotAFailure()
    {
        var root = TempDir();
        try
        {
            Assert.Empty(FastaArchive.Archive(new PrismConfig(), root));

            var config = new PrismConfig();
            config.Parsimony.FastaPath = Path.Combine(root, "does-not-exist.fasta");
            Assert.Empty(FastaArchive.Archive(config, root));
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    /// <summary>End to end: the copy and the redirect survive a real parameters.json round trip.</summary>
    [Fact]
    public void ProvenanceRecordsTheArchiveAndResolvesItBack()
    {
        var root = TempDir();
        try
        {
            var dbDir = Path.Combine(root, "databases");
            var source = WriteFasta(dbDir, "search.fasta");
            var outDir = Path.Combine(root, "out");
            Directory.CreateDirectory(outDir);
            var config = new PrismConfig();
            config.Parsimony.FastaPath = source;

            var entries = FastaArchive.Archive(config, outDir);
            var json = Path.Combine(outDir, "parameters.json");
            Provenance.Write(
                json, config, new[] { "report.parquet" },
                new Provenance.Stats(1, 1, 1, 1), DateTime.UtcNow.ToString("o"), entries);

            Directory.Delete(dbDir, recursive: true);
            var reloaded = Provenance.LoadConfig(json, out var redirected);

            Assert.Equal(new[] { "parsimony.fasta_path" }, redirected);
            Assert.True(File.Exists(reloaded.Parsimony.FastaPath));
            Assert.EndsWith("search.fasta", reloaded.Parsimony.FastaPath);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }
}
