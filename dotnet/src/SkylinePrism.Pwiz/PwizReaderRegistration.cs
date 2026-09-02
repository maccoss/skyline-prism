using SkylinePrism.Core.RawData;

namespace SkylinePrism.Pwiz;

/// <summary>
/// Registers this build's instrument-file reader with <see cref="Ms2SignalReaders"/>.
///
/// <para>Always present, whether or not the build has pwiz-sharp: a host calls
/// <see cref="Register"/> once at startup and does not have to know which build it is. Without
/// pwiz this is a no-op, and the accounting reports the acquired MS2 total as unknown - which is the
/// correct answer, not a failure.</para>
/// </summary>
public static class PwizReaderRegistration
{
    /// <summary>True when this build can actually read instrument files.</summary>
    public static bool IsAvailable =>
#if PRISM_NO_PWIZ
        false;
#else
        true;
#endif

    /// <summary>
    /// Register the reader. Idempotent - <see cref="Ms2SignalReaders.Register"/> ignores an instance
    /// it already holds, and this keeps one instance so repeated calls cannot stack readers.
    /// </summary>
    public static void Register()
    {
#if !PRISM_NO_PWIZ
        Ms2SignalReaders.Register(Instance);
#endif
    }

#if !PRISM_NO_PWIZ
    private static readonly PwizMs2SignalReader Instance = new();
#endif
}
