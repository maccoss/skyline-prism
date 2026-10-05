using System;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using System.Text;

namespace SkylinePrism.Core.Panorama;

/// <summary>A user name and secret as a credential store holds them.</summary>
public readonly record struct StoredCredential(string UserName, string Secret)
{
    /// <summary>Never the secret.</summary>
    public override string ToString() => $"StoredCredential {{ UserName = {UserName} }}";
}

/// <summary>Where sign-ins are kept, by target name.</summary>
public interface ICredentialStore
{
    StoredCredential? Read(string target);

    void Write(string target, StoredCredential credential, string comment);

    void Delete(string target);
}

/// <summary>
/// Windows Credential Manager, as PanoramaBridge and LabOps use it: generic credentials, the secret
/// as UTF-16, kept on this computer only. Any program running as the same Windows user can read an
/// entry (which is how PRISM finds PanoramaBridge's sign-in); other accounts cannot.
/// </summary>
/// <remarks>
/// Ported from LabOps, which ported it from PanoramaBridge's WindowsCredentialStore (Apache-2.0,
/// MacCoss Lab). The secret is never written to the settings file or to <c>panorama.json</c>: a
/// plain-text key in a results folder would be uploaded to Panorama with the rest of it.
/// </remarks>
[SupportedOSPlatform("windows")]
public sealed class WindowsCredentialStore : ICredentialStore
{
    private const int ErrorNotFound = 1168;
    private const int MaximumBlobBytes = 512;

    public StoredCredential? Read(string target)
    {
        if (!CredReadW(target, CredentialType.Generic, 0, out var handle))
        {
            var error = Marshal.GetLastWin32Error();
            return error == ErrorNotFound
                ? null
                : throw new InvalidOperationException($"Reading {target} failed (Win32 error {error}).");
        }

        try
        {
            var native = Marshal.PtrToStructure<NativeCredential>(handle);
            var secret = native.CredentialBlobSize > 0 && native.CredentialBlob != IntPtr.Zero
                ? Marshal.PtrToStringUni(native.CredentialBlob, (int)(native.CredentialBlobSize / 2))
                : "";
            return new StoredCredential(native.UserName ?? "", secret ?? "");
        }
        finally
        {
            CredFree(handle);
        }
    }

    public void Write(string target, StoredCredential credential, string comment)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(credential.UserName);
        var blob = Encoding.Unicode.GetBytes(credential.Secret);
        if (blob.Length > MaximumBlobBytes)
            throw new ArgumentException(
                $"The secret is too long to store ({blob.Length} bytes; the limit is {MaximumBlobBytes}).");

        var blobHandle = Marshal.AllocHGlobal(blob.Length);
        try
        {
            Marshal.Copy(blob, 0, blobHandle, blob.Length);
            var native = new NativeCredential
            {
                Type = CredentialType.Generic,
                TargetName = target,
                CredentialBlob = blobHandle,
                CredentialBlobSize = (uint)blob.Length,
                Persist = CredentialPersistence.LocalMachine, // never roams to another computer
                UserName = credential.UserName,
                Comment = comment,
            };
            if (!CredWriteW(ref native, 0))
                throw new InvalidOperationException(
                    $"Could not save the sign-in (Win32 error {Marshal.GetLastWin32Error()}).");
        }
        finally
        {
            for (var i = 0; i < blob.Length; i++)
                Marshal.WriteByte(blobHandle, i, 0);
            Marshal.FreeHGlobal(blobHandle);
            Array.Clear(blob);
        }
    }

    public void Delete(string target)
    {
        if (!CredDeleteW(target, CredentialType.Generic, 0)
            && Marshal.GetLastWin32Error() is var error and not ErrorNotFound)
            throw new InvalidOperationException($"Deleting {target} failed (Win32 error {error}).");
    }

    private enum CredentialType : uint
    {
        Generic = 1,
    }

    private enum CredentialPersistence : uint
    {
        LocalMachine = 2,
    }

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    private struct NativeCredential
    {
        public uint Flags;
        public CredentialType Type;

        [MarshalAs(UnmanagedType.LPWStr)]
        public string TargetName;

        [MarshalAs(UnmanagedType.LPWStr)]
        public string Comment;

        public long LastWritten;
        public uint CredentialBlobSize;
        public IntPtr CredentialBlob;
        public CredentialPersistence Persist;
        public uint AttributeCount;
        public IntPtr Attributes;

        [MarshalAs(UnmanagedType.LPWStr)]
        public string TargetAlias;

        [MarshalAs(UnmanagedType.LPWStr)]
        public string UserName;
    }

    [DllImport("advapi32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool CredReadW(string target, CredentialType type, uint flags, out IntPtr credential);

    [DllImport("advapi32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool CredWriteW(ref NativeCredential credential, uint flags);

    [DllImport("advapi32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool CredDeleteW(string target, CredentialType type, uint flags);

    [DllImport("advapi32.dll")]
    private static extern void CredFree(IntPtr buffer);
}
