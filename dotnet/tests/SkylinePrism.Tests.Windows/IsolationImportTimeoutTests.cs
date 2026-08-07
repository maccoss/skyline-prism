using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading;
using SkylinePrism.Skyline;
using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// Reading isolation windows out of a data file is an ENRICHMENT: without it the density map falls
/// back to uniform bins and nothing else about the run changes. So it must never be able to stop a
/// run - which it could, because the runner blocked in ReadLine on Skyline's output pipe with no
/// deadline and no cancellation until a line arrived. A Skyline that printed nothing (a data file
/// behind a slow or half-mounted link) hung PRISM indefinitely, and Stop could not break it either.
/// </summary>
public class IsolationImportTimeoutTests
{
    /// <summary>A runner that never returns, standing in for a Skyline that has gone quiet.</summary>
    private sealed class HangingRunner : ISkylineCommandRunner
    {
        public string Description => "hanging runner";
        public bool SupportsParquet => false;
        public TimeSpan? SeenTimeout { get; private set; }

        public void Run(string[] args, Action<string> log, CancellationToken cancellationToken,
            TimeSpan? timeout = null)
        {
            SeenTimeout = timeout;
            // What the real runner does: wait for output that never comes, with cancellation and the
            // deadline racing - whichever lands first wins.
            if (cancellationToken.WaitHandle.WaitOne(timeout ?? Timeout.InfiniteTimeSpan))
                cancellationToken.ThrowIfCancellationRequested();
            throw new TimeoutException(
                $"Skyline-daily did not finish within {timeout?.TotalMinutes ?? 0:F0} min "
                + "and was stopped.");
        }
    }

    /// <summary>The import must be given a bound at all - not left to wait forever.</summary>
    [Fact]
    public void TheImportIsGivenATimeout()
    {
        var runner = new HangingRunner();
        var file = Path.Combine(Path.GetTempPath(), "prism_iso_" + Guid.NewGuid().ToString("N") + ".raw");
        File.WriteAllText(file, "not really a raw file");
        try
        {
            Environment.SetEnvironmentVariable(SkylineIsolationImporter.TimeoutEnvVar, "0.5");
            var log = new List<string>();

            var watch = Stopwatch.StartNew();
            var scheme = SkylineIsolationImporter.ImportFromDataFile(file, runner, log.Add);
            watch.Stop();

            Assert.NotNull(runner.SeenTimeout);
            Assert.Equal(TimeSpan.FromSeconds(0.5), runner.SeenTimeout);

            // Times out and returns nothing - it does NOT throw, because the run must continue.
            Assert.Null(scheme);
            Assert.True(watch.Elapsed < TimeSpan.FromSeconds(30), "the import did not give up");

            // And it has to say what was lost and how to wait longer, not just that time ran out.
            var message = string.Join("\n", log);
            Assert.Contains("uniform bins", message, StringComparison.OrdinalIgnoreCase);
            Assert.Contains(SkylineIsolationImporter.TimeoutEnvVar, message, StringComparison.Ordinal);
        }
        finally
        {
            Environment.SetEnvironmentVariable(SkylineIsolationImporter.TimeoutEnvVar, null);
            File.Delete(file);
        }
    }

    /// <summary>Cancellation still propagates - Stop must stop the run, not be swallowed as "no windows".</summary>
    [Fact]
    public void CancellationIsNotSwallowed()
    {
        var runner = new HangingRunner();
        var file = Path.Combine(Path.GetTempPath(), "prism_iso_" + Guid.NewGuid().ToString("N") + ".raw");
        File.WriteAllText(file, "not really a raw file");
        using var cts = new CancellationTokenSource();
        try
        {
            var thread = new Thread(() =>
            {
                Thread.Sleep(200);
                cts.Cancel();
            }) { IsBackground = true };
            thread.Start();

            Assert.ThrowsAny<OperationCanceledException>(() =>
                SkylineIsolationImporter.ImportFromDataFile(file, runner, _ => { }, cts.Token));
        }
        finally
        {
            File.Delete(file);
        }
    }

    /// <summary>The environment override has to actually be read, or it is not an escape hatch.</summary>
    [Theory]
    [InlineData("120", 120)]
    [InlineData("900", 900)]
    public void TimeoutCanBeRaisedByEnvironmentVariable(string value, int expectedSeconds)
    {
        try
        {
            Environment.SetEnvironmentVariable(SkylineIsolationImporter.TimeoutEnvVar, value);
            Assert.Equal(TimeSpan.FromSeconds(expectedSeconds), SkylineIsolationImporter.Timeout);
        }
        finally
        {
            Environment.SetEnvironmentVariable(SkylineIsolationImporter.TimeoutEnvVar, null);
        }
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("not-a-number")]
    [InlineData("-5")]
    [InlineData("0")]
    public void NonsenseInTheEnvironmentVariableFallsBackToTheDefault(string? value)
    {
        try
        {
            Environment.SetEnvironmentVariable(SkylineIsolationImporter.TimeoutEnvVar, value);
            Assert.Equal(TimeSpan.FromMinutes(5), SkylineIsolationImporter.Timeout);
        }
        finally
        {
            Environment.SetEnvironmentVariable(SkylineIsolationImporter.TimeoutEnvVar, null);
        }
    }
}
