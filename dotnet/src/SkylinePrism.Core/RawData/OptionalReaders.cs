using System;
using System.Linq;
using System.Reflection;

namespace SkylinePrism.Core.RawData;

/// <summary>
/// Loads the optional instrument-file reader, if this build has one.
///
/// <para><b>Reflection rather than a reference.</b> <c>SkylinePrism.Pwiz</c> is referenced only when
/// a pwiz-sharp checkout was present at build time, so naming its types would stop a pwiz-less build
/// compiling - and that build is what a developer without the checkout, and the published
/// cross-platform CLI, both use. Going through the assembly name instead means the same code path
/// serves both, and the reader appears the moment the assembly is beside the executable.</para>
///
/// <para>Shared by the WPF tool and the CLI deliberately. The CLI used to have no bootstrap at all,
/// so every reader-dependent command failed with "this build has no instrument-file reader" even in
/// a build that had one sitting next to it - the reader was present and simply never registered.</para>
/// </summary>
public static class OptionalReaders
{
    private const string RegistrationType =
        "SkylinePrism.Pwiz.PwizReaderRegistration, SkylinePrism.Pwiz";

    private static bool _attempted;

    /// <summary>
    /// Register whatever readers this build carries. Idempotent and safe to call from any entry
    /// point; never throws - a reader that cannot load is a missing denominator, not a reason to
    /// fail startup.
    /// </summary>
    /// <param name="log">Told what happened, in one line, whichever way it goes.</param>
    public static void Register(Action<string>? log = null)
    {
        if (_attempted)
            return;
        _attempted = true;

        try
        {
            var type = Type.GetType(RegistrationType, throwOnError: false);
            if (type is null)
            {
                log?.Invoke(
                    "Instrument-file reader: not in this build; acquired ion counts will read as "
                    + "unknown.");
                return;
            }

            type.GetMethod("Register", BindingFlags.Public | BindingFlags.Static)
                ?.Invoke(null, null);
            log?.Invoke(
                "Instrument-file reader: registered ("
                + string.Join(", ", Ms2SignalReaders.All.Select(r => r.Describe())) + ").");
        }
        catch (Exception ex)
        {
            log?.Invoke("Instrument-file reader: could not be registered - " + ex.Message);
        }
    }

    /// <summary>Forget that registration was attempted. For tests only.</summary>
    internal static void Reset() => _attempted = false;
}
