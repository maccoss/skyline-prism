using System;
using System.Collections.Generic;
using System.Linq;

namespace SkylinePrism.Core.Panorama;

/// <summary>
/// The Panorama sign-ins to try, in order: an API key in <see cref="ApiKeyVariable"/>, then the
/// sign-in PanoramaBridge saved on this computer, then LabOps's, then PRISM's own. Saving and
/// forgetting touch only PRISM's entry; the others belong to their apps.
/// </summary>
/// <remarks>
/// <para><b>PanoramaBridge first</b> because it is the app that keeps a key renewed, and it is the
/// one a lab machine that acquires raw files already has; LabOps does the same for the same reason.
/// PRISM saves its own only when none of the others worked.</para>
/// <para><b>The environment variable before all of them</b> because it is the only thing a headless
/// run can set, and the only source at all on Linux and macOS, which have no Credential Manager. It
/// is read, never written: putting a key into a variable is the caller's decision.</para>
/// </remarks>
public sealed class PanoramaSignIn
{
    /// <summary>An API key here is tried first: the headless path, and the only one off Windows.</summary>
    public const string ApiKeyVariable = "PRISM_PANORAMA_API_KEY";

    private readonly ICredentialStore? _store;
    private readonly Func<string, string?> _environment;

    public PanoramaSignIn(ICredentialStore? store, Uri server, Func<string, string?>? environment = null)
    {
        _store = store;
        Server = server;
        _environment = environment ?? Environment.GetEnvironmentVariable;
    }

    /// <summary>The credential store this computer has: Credential Manager on Windows, none elsewhere.</summary>
    public static PanoramaSignIn ForThisComputer(Uri server) =>
        new(OperatingSystem.IsWindows() ? new WindowsCredentialStore() : null, server);

    public Uri Server { get; }

    /// <summary>"https://panoramaweb.org": scheme and host, which is how all three apps name entries.</summary>
    public string Host => $"{Server.Scheme}://{Server.Host}";

    /// <summary>PanoramaBridge's entry (its WindowsCredentialStore.TargetFor with no account).</summary>
    public string PanoramaBridgeTarget => $"PanoramaBridge:{Host}";

    /// <summary>LabOps's entry, read and never written.</summary>
    public string LabOpsTarget => $"LabOps:{Host}";

    /// <summary>PRISM's own entry: the only one this class writes or deletes.</summary>
    public string OwnTarget => $"Skyline-PRISM:{Host}";

    /// <summary>Whether a sign-in can be saved here (false off Windows).</summary>
    public bool CanSave => _store is not null;

    /// <summary>The saved sign-ins, in the order to try them.</summary>
    public IReadOnlyList<PanoramaCredential> Candidates()
    {
        var found = new List<PanoramaCredential?>();
        if (_environment(ApiKeyVariable) is { } key && !string.IsNullOrWhiteSpace(key))
            found.Add(PanoramaCredential.ApiKey(key, ApiKeyVariable));
        found.Add(PanoramaCredential.FromStored(Read(PanoramaBridgeTarget), "PanoramaBridge"));
        found.Add(PanoramaCredential.FromStored(Read(LabOpsTarget), "LabOps"));
        found.Add(PanoramaCredential.FromStored(Read(OwnTarget), "Skyline-PRISM"));
        return found.OfType<PanoramaCredential>().ToList();
    }

    /// <summary>Saves a sign-in typed into PRISM, for next time.</summary>
    public PanoramaCredential Save(PanoramaCredential credential)
    {
        if (_store is null)
            throw new InvalidOperationException(
                $"This computer has no credential store; set {ApiKeyVariable} to an API key instead.");
        _store.Write(OwnTarget, credential.ToStored(), "Panorama sign-in saved by Skyline-PRISM");
        return PanoramaCredential.FromStored(credential.ToStored(), "Skyline-PRISM")!;
    }

    /// <summary>Forgets PRISM's own sign-in (never PanoramaBridge's or LabOps's).</summary>
    public void Forget() => _store?.Delete(OwnTarget);

    // An entry that cannot be read is treated as missing: the person is asked to sign in instead.
    private StoredCredential? Read(string target)
    {
        if (_store is null)
            return null;
        try
        {
            return _store.Read(target);
        }
        catch (InvalidOperationException)
        {
            return null;
        }
    }
}
