using System;
using System.IO;
using System.Net;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using SkylinePrism.Core.Panorama;
using Xunit;

namespace SkylinePrism.Tests.Panorama;

/// <summary>The client's own failure handling: a dropped upload retried, a refusal said plainly.</summary>
public sealed class PanoramaClientTests : IDisposable
{
    private readonly string _file = Path.Combine(Path.GetTempPath(), "prism_client_" + Guid.NewGuid().ToString("N") + ".bin");
    private readonly TimeSpan _retryDelayUnit = PanoramaClient.RetryDelayUnit;

    public PanoramaClientTests()
    {
        File.WriteAllBytes(_file, new byte[3000]);
        // The real pause is seconds. It is a static, so it is put back afterwards rather than left
        // shortened for every test that runs later.
        PanoramaClient.RetryDelayUnit = TimeSpan.FromMilliseconds(10);
    }

    public void Dispose()
    {
        PanoramaClient.RetryDelayUnit = _retryDelayUnit;
        try { File.Delete(_file); }
        catch (IOException) { /* best-effort */ }
    }

    /// <summary>Answers by method, and fails the first <see cref="FailPuts"/> PUTs at the transport.</summary>
    private sealed class ScriptedServer : HttpMessageHandler
    {
        public int FailPuts { get; set; }
        public int GatewayErrors { get; set; }
        public int Puts { get; private set; }
        public HttpStatusCode Mkcol { get; set; } = HttpStatusCode.Created;

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            if (request.Method.Method == "MKCOL")
                return Task.FromResult(new HttpResponseMessage(Mkcol));
            if (request.Method == HttpMethod.Put)
            {
                Puts++;
                if (Puts <= FailPuts)
                    throw new HttpRequestException("connection reset");
                if (Puts <= FailPuts + GatewayErrors)
                    return Task.FromResult(new HttpResponseMessage(HttpStatusCode.ServiceUnavailable));
                return Task.FromResult(new HttpResponseMessage(HttpStatusCode.Created));
            }

            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.NotFound));
        }
    }

    private static PanoramaClient Client(HttpMessageHandler server) =>
        new(new Uri("https://panorama.test"), PanoramaCredential.ApiKey("k", "PanoramaBridge"), server);

    [Fact]
    public async Task ADroppedUpload_IsRetriedFromTheStart()
    {
        var server = new ScriptedServer { FailPuts = 1 };
        using var client = Client(server);
        await client.UploadFileAsync(_file, "/P/@files/out", "a.bin");
        Assert.Equal(2, server.Puts);
    }

    [Fact]
    public async Task AGatewayErrorDuringAnUpload_IsRetried_LikeADroppedConnection()
    {
        var server = new ScriptedServer { GatewayErrors = 2 };
        using var client = Client(server);
        await client.UploadFileAsync(_file, "/P/@files/out", "a.bin");
        Assert.Equal(3, server.Puts);
    }

    [Fact]
    public async Task AnUploadThatKeepsFailing_SaysHowManyTimesItWasTried()
    {
        var server = new ScriptedServer { FailPuts = 99 };
        using var client = Client(server);
        var ex = await Assert.ThrowsAsync<PanoramaException>(() => client.UploadFileAsync(_file, "/P/@files/out", "a.bin"));
        Assert.Contains("after 3 attempts", ex.Message, StringComparison.Ordinal);
        Assert.Equal(3, server.Puts);
    }

    [Fact]
    public async Task AFolderThisAccountMayNotCreate_IsSaidPlainly()
    {
        using var client = Client(new ScriptedServer { Mkcol = HttpStatusCode.Forbidden });
        var ex = await Assert.ThrowsAsync<PanoramaException>(() => client.EnsureFolderAsync("/P/@files/out"));
        Assert.Contains("may not create /P/@files/out", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ARejectedSignIn_NamesTheAppThatSavedIt()
    {
        using var client = Client(new FakePanorama { RejectSignIn = true });
        var problem = await client.CheckSignInAsync();
        Assert.NotNull(problem);
        Assert.True(problem!.IsSignInProblem);
        Assert.Contains("the sign-in from PanoramaBridge (an API key)", problem.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void PanoramaJson_RoundTrips_AndADamagedOneIsAForgottenPreference()
    {
        var dir = Path.Combine(Path.GetTempPath(), "prism_targets_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var targets = new PanoramaTargets("https://panoramaweb.org", new WikiTarget("/P", "PRISM-QC-x"), null, "/P/@files/RawFiles", "/P/@files/x");
            targets.Save(dir);
            Assert.Equal(targets, PanoramaTargets.Load(dir));
            Assert.DoesNotContain("server_uri", File.ReadAllText(Path.Combine(dir, PanoramaTargets.FileName)), StringComparison.Ordinal);

            File.WriteAllText(Path.Combine(dir, PanoramaTargets.FileName), "{ not json");
            Assert.Equal(PanoramaTargets.Empty, PanoramaTargets.Load(dir));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }
}
