using System;
using System.IO;
using System.Reflection;

namespace SkylinePrism.Core.Pipeline;

/// <summary>
/// PRISM's own version as a string, in the form the project actually versions in: CalVer
/// <c>YY.feature.patch</c>, so <c>26.24.2</c> - never <c>26.24.2.0</c>.
/// </summary>
/// <remarks>
/// <para>The obvious spelling, <c>Assembly.GetName().Version</c>, is wrong for anything a human
/// reads. It returns a <see cref="Version"/>, which always normalizes to four components, and
/// MSBuild pads <c>&lt;Version&gt;26.24.2&lt;/Version&gt;</c> out to an <c>AssemblyVersion</c> of
/// <c>26.24.2.0</c>. That trailing <c>.0</c> is the padding, not a release component: PRISM's scheme
/// has no fourth number, so nothing can ever make it anything but zero, and it read as a version
/// nobody had tagged.</para>
///
/// <para><c>AssemblyInformationalVersion</c> is MSBuild's verbatim copy of <c>&lt;Version&gt;</c>, so
/// it is the one assembly version attribute that can carry <c>26.24.2</c> - and the only one that
/// could carry a prerelease suffix such as <c>26.25.0-rc1</c>, which truncating
/// <c>AssemblyVersion</c> to three parts would silently drop. PRISM has never tagged one, and
/// <c>PrismVersionTests</c> pins the three-part form, so introducing a suffix is a deliberate change
/// to both that test and the release workflow's tag comparison.</para>
///
/// <para>Read from THIS assembly rather than the caller's, so every surface answers with one string:
/// <c>prism --version</c>, the QC report footer, provenance <c>pipeline_version</c>, the stage-cache
/// fingerprint, and the export and extraction sidecars. They are all built from the same
/// <c>&lt;Version&gt;</c> today; reading a single assembly means they cannot disagree if that ever
/// stops being true.</para>
/// </remarks>
public static class PrismVersion
{
    /// <summary>The running PRISM version, e.g. <c>26.24.2</c>. Never null, never empty.</summary>
    public static string Current { get; } = Resolve();

    /// <summary>
    /// The running build, in the terms someone checking a fix needs: the version, when the assembly
    /// was built, and the folder it was loaded from.
    /// </summary>
    /// <remarks>
    /// <para><b>Deliberately not part of <see cref="Current"/>.</b> That string feeds the
    /// stage-cache fingerprint, provenance and the export sidecar - a build time in it would
    /// invalidate all three on every rebuild, which is the same reason the source-revision suffix is
    /// stripped above. Nothing computes anything from this one; it is for the log and the window.
    /// </para>
    ///
    /// <para><b>Why it exists.</b> The version alone cannot answer "am I running the fix?". PRISM
    /// versions at RELEASE time, so every build between two releases reports the same number, and a
    /// user testing a fix has nothing to check. Two 48-file measurements - several hours each - were
    /// run against a build that predated the fix they were testing, and neither the window nor the
    /// log could have said so. The folder matters as much as the time: the Skyline tool runs the
    /// copy installed under Skyline's Tools directory, which a newly built zip sitting on disk does
    /// not touch.</para>
    ///
    /// <para>The timestamp is the assembly file's own, which survives being zipped and extracted, so
    /// an installed copy still reports when it was built rather than when it was installed.</para>
    /// </remarks>
    public static string BuildStamp { get; } = ResolveStamp();

    /// <summary>The folder the running PRISM assemblies were loaded from, or empty.</summary>
    /// <inheritdoc cref="BuildStamp"/>
    public static string BuildLocation { get; } = ResolveLocation();

    /// <inheritdoc cref="Resolve"/>
    private static string ResolveStamp()
    {
        try
        {
            var path = typeof(PrismVersion).Assembly.Location;
            if (string.IsNullOrEmpty(path) || !File.Exists(path))
                return Current;
            return $"{Current}, built {File.GetLastWriteTime(path):yyyy-MM-dd HH:mm}";
        }
        catch
        {
            return Current;
        }
    }

    /// <inheritdoc cref="Resolve"/>
    private static string ResolveLocation()
    {
        try
        {
            var path = typeof(PrismVersion).Assembly.Location;
            return string.IsNullOrEmpty(path) ? "" : Path.GetDirectoryName(path) ?? "";
        }
        catch
        {
            return "";
        }
    }

    /// <summary>
    /// Never throws, deliberately. This runs inside a static initializer, so an exception here
    /// becomes a <c>TypeInitializationException</c> on EVERY later read of <see cref="Current"/> -
    /// including the ones inside <c>Provenance.Write</c> and <c>StageCache.Fingerprint</c>, which
    /// would abort a run over a version string. The two properties this replaced could not throw
    /// (both ended in a <c>?? "0"</c> fallback) and that contract is worth keeping: reflection over
    /// assembly attributes is not quite total - <c>GetCustomAttribute</c> can raise
    /// <c>CustomAttributeFormatException</c> or <c>TypeLoadException</c>, and <c>GetName</c> throws
    /// for some dynamically generated assemblies. An unknown version is worth degrading over; a
    /// failed run is not.
    /// </summary>
    private static string Resolve()
    {
        try
        {
            return ResolveCore();
        }
        catch
        {
            return "0.0.0";
        }
    }

    private static string ResolveCore()
    {
        var assembly = typeof(PrismVersion).Assembly;

        var informational = assembly
            .GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion;
        if (!string.IsNullOrWhiteSpace(informational))
        {
            // Strip SemVer build metadata. Directory.Build.props turns the source-revision suffix off,
            // so nothing appends one today - but this string now feeds the stage-cache fingerprint and
            // the export sidecar, and a "+<commit>" in it would invalidate both on every rebuild.
            var plus = informational.IndexOf('+');
            var trimmed = (plus >= 0 ? informational[..plus] : informational).Trim();
            if (trimmed.Length > 0)
                return trimmed;
        }

        // No informational attribute at all (a hand-built assembly): fall back to the padded
        // AssemblyVersion with the padding dropped.
        var version = assembly.GetName().Version;
        return version is null
            ? "0.0.0"
            : $"{version.Major}.{version.Minor}.{Math.Max(version.Build, 0)}";
    }
}
