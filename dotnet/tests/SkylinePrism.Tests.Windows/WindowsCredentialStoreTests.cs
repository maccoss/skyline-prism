using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using SkylinePrism.Core.Panorama;
using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// <see cref="WindowsCredentialStore"/> against the real Windows Credential Manager, since what it
/// must get right - the UTF-16 secret, the 512-byte limit, "not found" as null rather than an error -
/// is the native API's behavior, not PRISM's.
/// </summary>
/// <remarks>
/// These write real entries, so every one is named for a host that cannot exist
/// (<c>https://prism-test-&lt;guid&gt;.invalid</c>), deleted by the test that wrote it, and swept again
/// before and after the class runs in case a crashed run left one behind. That is how PanoramaBridge's
/// tests once left <c>pb-test</c> entries in a real Credential Manager; it must not happen here, and
/// nothing here touches an entry for a real server.
/// </remarks>
public sealed class WindowsCredentialStoreTests : IDisposable
{
    private static readonly string[] Prefixes = { "Skyline-PRISM", "PanoramaBridge", "LabOps" };

    private readonly WindowsCredentialStore _store = new();
    private readonly Uri _server = new($"https://prism-test-{Guid.NewGuid():N}.invalid");

    public WindowsCredentialStoreTests() => Sweep();

    public void Dispose() => Sweep();

    private string Target(string prefix = "Skyline-PRISM") => $"{prefix}:{_server.Scheme}://{_server.Host}";

    private static string TestSecret() => "test-key-" + Guid.NewGuid().ToString("N");

    [Fact]
    public void AWrittenEntry_ReadsBack_AsWritten()
    {
        var secret = TestSecret() + "-é漢字"; // stored as UTF-16, so anything a password can hold
        _store.Write(Target(), new StoredCredential("apikey", secret), "test");

        Assert.Equal(new StoredCredential("apikey", secret), _store.Read(Target()));
    }

    [Fact]
    public void AMissingEntry_ReadsAsNull_AndDeletingOneIsNotAnError()
    {
        Assert.Null(_store.Read(Target()));
        _store.Delete(Target());
    }

    [Fact]
    public void WritingAgain_ReplacesTheEntry_AndDeleteRemovesIt()
    {
        _store.Write(Target(), new StoredCredential("someone", TestSecret()), "test");
        var second = TestSecret();
        _store.Write(Target(), new StoredCredential("someone else", second), "test");
        Assert.Equal(new StoredCredential("someone else", second), _store.Read(Target()));

        _store.Delete(Target());
        Assert.Null(_store.Read(Target()));
    }

    [Fact]
    public void ASecretAtTheLimit_IsKept_AndOneOverItIsRefused_WithNothingWritten()
    {
        // 512 bytes of UTF-16 is 256 characters.
        var atLimit = new string('k', 256);
        _store.Write(Target(), new StoredCredential("apikey", atLimit), "test");
        Assert.Equal(atLimit, _store.Read(Target())!.Value.Secret);
        _store.Delete(Target());

        var ex = Assert.Throws<ArgumentException>(() =>
            _store.Write(Target(), new StoredCredential("apikey", new string('k', 257)), "test"));
        Assert.Contains("the limit is 512", ex.Message, StringComparison.Ordinal);
        Assert.Null(_store.Read(Target()));
    }

    [Fact]
    public void AnEmptySecret_ReadsBackEmpty_AndANameIsRequired()
    {
        _store.Write(Target(), new StoredCredential("someone", ""), "test");
        Assert.Equal(new StoredCredential("someone", ""), _store.Read(Target()));

        Assert.Throws<ArgumentException>(() => _store.Write(Target(), new StoredCredential(" ", TestSecret()), "test"));
    }

    [Fact]
    public void SignIn_SavesAndForgetsOnlyItsOwnEntry_AndPrefersPanoramaBridges()
    {
        var signIn = new PanoramaSignIn(_store, _server, _ => null);
        var bridgeKey = TestSecret();
        _store.Write(Target("PanoramaBridge"), new StoredCredential("apikey", bridgeKey), "test");

        signIn.Save(PanoramaCredential.ApiKey(TestSecret()));
        Assert.Equal(new[] { "PanoramaBridge", "Skyline-PRISM" }, signIn.Candidates().Select(c => c.Source));
        Assert.Equal(bridgeKey, signIn.Candidates()[0].Secret);

        signIn.Forget();
        Assert.Null(_store.Read(Target()));
        // Never PanoramaBridge's, which belongs to another program.
        Assert.NotNull(_store.Read(Target("PanoramaBridge")));
    }

    /// <summary>Deletes every entry a test here could have written, this run's or an earlier one's.</summary>
    private void Sweep()
    {
        foreach (var target in Prefixes.SelectMany(p => Enumerate($"{p}:https://prism-test-*")))
            _store.Delete(target);
    }

    private static IReadOnlyList<string> Enumerate(string filter)
    {
        if (!CredEnumerateW(filter, 0, out var count, out var list))
            return Array.Empty<string>(); // ERROR_NOT_FOUND: nothing matches
        try
        {
            var targets = new List<string>(count);
            for (var i = 0; i < count; i++)
            {
                var entry = Marshal.PtrToStructure<CredentialHead>(Marshal.ReadIntPtr(list, i * IntPtr.Size));
                if (Marshal.PtrToStringUni(entry.TargetName) is { } name)
                    targets.Add(name);
            }

            return targets;
        }
        finally
        {
            CredFree(list);
        }
    }

    // The start of CREDENTIALW, as far as its target name.
    [StructLayout(LayoutKind.Sequential)]
    private struct CredentialHead
    {
        public uint Flags;
        public uint Type;
        public IntPtr TargetName;
    }

    [DllImport("advapi32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool CredEnumerateW(string filter, uint flags, out int count, out IntPtr credentials);

    [DllImport("advapi32.dll")]
    private static extern void CredFree(IntPtr buffer);
}
