using System.IO;
using SkylinePrism.Core.Config;
using SkylinePrism.Core.Pipeline;
using Xunit;

namespace SkylinePrism.Tests.Pipeline;

/// <summary>
/// Which batch estimation a run falls back to when no batch label distinguishes the samples.
/// </summary>
public class BatchEstimationMethodTests
{
    /// <summary>
    /// An unset method means the default, none - never gap detection. Inventing batches is the
    /// failure the default exists to prevent: ComBat then "corrects" between batches that do not
    /// exist and every abundance moves. A blank YAML key no longer reaches here as null (Parse gives
    /// it the default), but a parameters.json or a config built in code still can.
    /// </summary>
    [Fact]
    public void AnUnsetMethod_EstimatesNoBatches()
    {
        var config = new PrismConfig();
        config.BatchEstimation.Method = null!;

        Assert.Equal("none", PrismPipeline.BatchEstimationMethod(config));
    }

    [Fact]
    public void ABlankMethodInAConfig_EstimatesNoBatches()
    {
        var config = PrismConfig.Parse("batch_estimation:\n  method:\n");

        Assert.Equal("none", PrismPipeline.BatchEstimationMethod(config));
    }

    [Fact]
    public void AMethodIsReadCaseInsensitively()
    {
        var config = new PrismConfig();
        config.BatchEstimation.Method = "Gap";

        Assert.Equal("gap", PrismPipeline.BatchEstimationMethod(config));
    }
}
