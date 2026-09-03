using Pwiz.Data.MsData.Readers;
using Pwiz.Vendor.Thermo;

namespace SkylinePrism.Pwiz;

/// <summary>
/// Plugs the vendor readers this build carries into pwiz's format dispatcher.
/// </summary>
/// <remarks>
/// <para><c>Pwiz.Data.MsData</c> deliberately does not reference the vendor projects - that would
/// drag every encrypted vendor SDK into everything touching the core data model - so
/// <c>ReaderList.Default</c> knows only the open formats until a host adds the rest. Without this,
/// opening a <c>.raw</c> fails with "no registered reader recognized the file" even though the reader
/// is sitting in the same output directory.</para>
/// <para>Registered from a static constructor rather than from an entry point, because a registration
/// that depends on someone remembering to call it is one that eventually gets missed. A module
/// initializer would be tidier but CA2255 objects to one in a library, and is right to: it would run
/// on assembly load, which a caller has no way to anticipate.</para>
/// </remarks>
internal static class VendorReaders
{
    static VendorReaders() => Register();

    /// <summary>
    /// Ensure the vendor readers are registered. Cheap and idempotent: the work is in the static
    /// constructor, which the runtime runs once on first touch of this type.
    /// </summary>
    internal static void EnsureRegistered()
    {
        // Referencing the type triggers the static constructor; there is nothing to do here.
    }

    private static void Register()
    {
        // Appended once. ReaderList.Default rebuilds a list on every access and copies
        // AdditionalReaders into it, so registering twice would double every vendor reader.
        if (ReaderList.AdditionalReaders.Count > 0)
            return;

        // Thermo only - see the ProjectReference comment in SkylinePrism.Pwiz.csproj for why, and
        // what it costs. The open formats (mzML, mzXML, mz5) come from ReaderList.Default already.
        ReaderList.AdditionalReaders.Add(new Reader_Thermo());
    }
}
