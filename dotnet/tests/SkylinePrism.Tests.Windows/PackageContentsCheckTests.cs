using System;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Text;
using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// The ship gate's package check, exercised against zips built to fail it.
/// </summary>
/// <remarks>
/// <para>It guards things whose absence is SILENT - the Skyline Tool Store logo, the manifest, the
/// instrument-file reader. A zip missing any of them still builds, extracts, launches and installs,
/// and CI stays green; the loss shows up later in the Tool Store or on someone's machine.</para>
///
/// <para>Only the missing-logo path had ever been exercised, once, by hand. A check nobody has seen
/// fail is a check nobody knows works - and this one is the last thing standing between a broken
/// package and a published release, since the release workflow runs it before the upload.</para>
/// </remarks>
public class PackageContentsCheckTests : IDisposable
{
    private readonly string _dir = Path.Combine(
        Path.GetTempPath(), "prism-pkgcheck-" + Guid.NewGuid().ToString("N"));

    private static string Script => Path.GetFullPath(Path.Combine(
        AppContext.BaseDirectory, "..", "..", "..", "..", "..", "build", "verify-zip-contents.ps1"));

    public PackageContentsCheckTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try
        {
            Directory.Delete(_dir, recursive: true);
        }
        catch (IOException)
        {
        }
    }

    [Fact]
    public void ACompletePackagePasses()
    {
        var (code, output) = Check(Zip("complete", Png(4096)));
        Assert.True(code == 0, output);
        Assert.Contains("package contents OK", output, StringComparison.Ordinal);
    }

    [Fact]
    public void AMissingLogoFails()
    {
        var (code, output) = Check(Zip("no-logo", logo: null));
        Assert.Equal(1, code);
        Assert.Contains("no image in tool-inf/", output, StringComparison.Ordinal);
        // The message has to name the line that puts it there, or the reader has to go looking.
        Assert.Contains("SkylinePrism.App.csproj", output, StringComparison.Ordinal);
    }

    /// <summary>
    /// Two images is not belt and braces - the Store picks one and it is not defined which.
    /// </summary>
    [Fact]
    public void TwoImagesFail()
    {
        var (code, output) = Check(Zip("two-logos", Png(4096), second: Png(4096)));
        Assert.Equal(1, code);
        Assert.Contains("2 images", output, StringComparison.Ordinal);
    }

    /// <summary>A placeholder passes an existence check and still shows as nothing.</summary>
    [Fact]
    public void ATinyLogoFails()
    {
        var (code, output) = Check(Zip("tiny-logo", Png(64)));
        Assert.Equal(1, code);
        Assert.Contains("not an image", output, StringComparison.Ordinal);
    }

    /// <summary>Right name, right size, not a PNG - which is why the magic bytes are checked.</summary>
    [Fact]
    public void ALogoThatIsNotAPngFails()
    {
        var bytes = new byte[4096];
        Array.Fill(bytes, (byte)0x41);
        var (code, output) = Check(Zip("not-a-png", bytes));
        Assert.Equal(1, code);
        Assert.Contains("not a PNG", output, StringComparison.Ordinal);
    }

    [Fact]
    public void AMissingManifestFails()
    {
        var (code, output) = Check(Zip("no-manifest", Png(4096), manifest: false));
        Assert.Equal(1, code);
        Assert.Contains("tool-inf/info.properties", output, StringComparison.Ordinal);
    }

    /// <summary>
    /// The reader is demanded only when asked for, because the CI pack job packages without it on
    /// purpose and must still pass.
    /// </summary>
    [Fact]
    public void TheReaderIsOnlyRequiredWhenAskedFor()
    {
        var zip = Zip("no-reader", Png(4096));

        Assert.Equal(0, Check(zip).Code);
        var (code, output) = Check(zip, requireReader: true);
        Assert.Equal(1, code);
        Assert.Contains("SkylinePrism.Pwiz.dll", output, StringComparison.Ordinal);
    }

    private static byte[] Png(int length)
    {
        var bytes = new byte[length];
        // The eight-byte PNG signature; the rest is padding, since only the magic is checked.
        new byte[] { 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A }.CopyTo(bytes, 0);
        return bytes;
    }

    private string Zip(string name, byte[]? logo, byte[]? second = null, bool manifest = true)
    {
        var path = Path.Combine(_dir, name + ".zip");
        using var archive = ZipFile.Open(path, ZipArchiveMode.Create);

        void Add(string entry, byte[] content)
        {
            using var stream = archive.CreateEntry(entry).Open();
            stream.Write(content, 0, content.Length);
        }

        Add("SkylinePrism.exe", Encoding.UTF8.GetBytes("MZ"));
        if (manifest)
        {
            Add("tool-inf/info.properties", Encoding.UTF8.GetBytes("Version = 26.24.3"));
            Add("tool-inf/SkylinePrism.properties", Encoding.UTF8.GetBytes("Name = PRISM"));
        }
        if (logo is not null)
            Add("tool-inf/skyline-prism-logo.png", logo);
        if (second is not null)
            Add("tool-inf/extra-logo.png", second);
        return path;
    }

    private static (int Code, string Output) Check(string zip, bool requireReader = false)
    {
        var args = $"-NoProfile -File \"{Script}\" -Zip \"{zip}\""
            + (requireReader ? " -RequireReader" : "");
        using var process = Process.Start(new ProcessStartInfo("pwsh", args)
        {
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        })!;
        var output = process.StandardOutput.ReadToEnd() + process.StandardError.ReadToEnd();
        process.WaitForExit();
        return (process.ExitCode, output);
    }
}
