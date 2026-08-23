using System.Runtime.InteropServices;
using Xunit;

namespace SkylinePrism.Tests.TestSupport;

/// <summary>
/// A <see cref="TheoryAttribute"/> that runs only on the bit-parity reference platform (Windows).
/// <para>
/// Bit-exact comparison of the pipeline's quantities is only meaningful against a fixed platform.
/// IEEE-754 pins the basic operations, so those agree everywhere - but <c>Math.Log</c>,
/// <c>Math.Exp</c> and <c>Math.Pow</c> delegate to the platform's libm and are NOT required to
/// return identical bits. Measured on this repo's CI: the same commit that is bit-exact on Windows
/// differs on ubuntu and macOS in 5-6 columns for the ComBat-disabled fixtures, and in 389 columns
/// for <c>e2e-sum</c>, the one fixture with ComBat enabled - whose empirical-Bayes estimation is
/// dense in exp/log. That is a property of the C runtime, not a defect in the pipeline.
/// </para>
/// <para>
/// Windows is the reference because the Skyline external tool is Windows-only, so it is the
/// platform PRISM's numbers are most often produced on. The other platforms are not left
/// unguarded: the cross-language parity tests compare every stage against the Python goldens at
/// 1e-9 everywhere, which catches any change large enough to alter a reported result. What this
/// gate adds on top - catching drift BELOW that tolerance - is inherently platform-specific, and a
/// C# change that moves a quantity will move it on Windows too.
/// </para>
/// </summary>
public sealed class ReferencePlatformTheoryAttribute : TheoryAttribute
{
    public ReferencePlatformTheoryAttribute()
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            Skip = "Bit-exact quantity digests are pinned to the Windows reference platform "
                 + "(Math.Log/Exp/Pow differ by libm; see ReferencePlatformTheoryAttribute). "
                 + "Cross-language parity at 1e-9 still runs here.";
    }
}
