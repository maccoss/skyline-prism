using System;
using System.Collections.Generic;
using System.Linq;
using SkylinePrism.Core.Panorama;
using Xunit;

namespace SkylinePrism.Tests.Panorama;

/// <summary>
/// Which saved sign-in PRISM tries, in what order, and which one it may change. PanoramaBridge's and
/// LabOps's entries belong to those apps: read, never written or deleted.
/// </summary>
public class PanoramaSignInTests
{
    private sealed class MemoryStore : ICredentialStore
    {
        public Dictionary<string, StoredCredential> Entries { get; } = new(StringComparer.Ordinal);
        public HashSet<string> Unreadable { get; } = new(StringComparer.Ordinal);

        public StoredCredential? Read(string target) => Unreadable.Contains(target)
            ? throw new InvalidOperationException("damaged")
            : Entries.TryGetValue(target, out var c) ? c : null;

        public void Write(string target, StoredCredential credential, string comment) => Entries[target] = credential;

        public void Delete(string target) => Entries.Remove(target);
    }

    private static readonly Uri Server = new("https://panoramaweb.org/MacCoss/x");

    [Fact]
    public void Order_IsTheEnvironmentThenPanoramaBridgeThenLabOpsThenPrism()
    {
        var store = new MemoryStore();
        store.Entries["Skyline-PRISM:https://panoramaweb.org"] = new("apikey", "k-prism");
        store.Entries["LabOps:https://panoramaweb.org"] = new("someone@uw.edu", "pw");
        store.Entries["PanoramaBridge:https://panoramaweb.org"] = new("apikey", "k-bridge");
        var signIn = new PanoramaSignIn(store, Server, name => name == PanoramaSignIn.ApiKeyVariable ? "k-env" : null);

        var sources = signIn.Candidates().Select(c => c.Source).ToList();

        Assert.Equal(new[] { PanoramaSignIn.ApiKeyVariable, "PanoramaBridge", "LabOps", "Skyline-PRISM" }, sources);
        Assert.True(signIn.Candidates()[0].IsApiKey);
        Assert.False(signIn.Candidates()[2].IsApiKey); // LabOps's was a user name and password
    }

    [Fact]
    public void Entries_AreKeyedOnSchemeAndHost_LikeTheOtherApps()
    {
        var signIn = new PanoramaSignIn(new MemoryStore(), Server, _ => null);
        Assert.Equal("PanoramaBridge:https://panoramaweb.org", signIn.PanoramaBridgeTarget);
        Assert.Equal("Skyline-PRISM:https://panoramaweb.org", signIn.OwnTarget);
    }

    [Fact]
    public void SavingAndForgetting_TouchOnlyPrismsOwnEntry()
    {
        var store = new MemoryStore();
        store.Entries["PanoramaBridge:https://panoramaweb.org"] = new("apikey", "k-bridge");
        var signIn = new PanoramaSignIn(store, Server, _ => null);

        var saved = signIn.Save(PanoramaCredential.ApiKey("apikey|k-typed"));
        Assert.Equal("Skyline-PRISM", saved.Source);
        Assert.Equal("k-typed", store.Entries["Skyline-PRISM:https://panoramaweb.org"].Secret); // legacy prefix stripped

        signIn.Forget();
        Assert.False(store.Entries.ContainsKey("Skyline-PRISM:https://panoramaweb.org"));
        Assert.True(store.Entries.ContainsKey("PanoramaBridge:https://panoramaweb.org"));
    }

    [Fact]
    public void AnUnreadableEntry_IsSkipped_NotFatal()
    {
        var store = new MemoryStore();
        store.Unreadable.Add("PanoramaBridge:https://panoramaweb.org");
        store.Entries["LabOps:https://panoramaweb.org"] = new("apikey", "k");
        var signIn = new PanoramaSignIn(store, Server, _ => null);

        Assert.Equal("LabOps", Assert.Single(signIn.Candidates()).Source);
    }

    [Fact]
    public void WithoutACredentialStore_OnlyTheEnvironmentIsTried_AndNothingCanBeSaved()
    {
        var signIn = new PanoramaSignIn(null, Server, name => name == PanoramaSignIn.ApiKeyVariable ? "k" : null);

        Assert.Equal(PanoramaSignIn.ApiKeyVariable, Assert.Single(signIn.Candidates()).Source);
        Assert.False(signIn.CanSave);
        Assert.Throws<InvalidOperationException>(() => signIn.Save(PanoramaCredential.ApiKey("k")));
    }

    [Fact]
    public void ACredential_NeverRendersItsSecret()
    {
        Assert.Equal("an API key", PanoramaCredential.ApiKey("secret-key").ToString());
        Assert.Equal("user a@b.org", PanoramaCredential.Login("a@b.org", "secret-pw").ToString());
        Assert.DoesNotContain("secret", new StoredCredential("u", "secret-pw").ToString(), StringComparison.Ordinal);
    }
}
