using System;

namespace SkylinePrism.Cli;

/// <summary>
/// Entry point for the cross-platform `prism` CLI. Subcommands (run / merge / qc /
/// config-template / compare) are wired up in Layer 9; this scaffold establishes the
/// dispatch surface and version banner.
/// </summary>
public static class Program
{
    public static int Main(string[] args)
    {
        if (args.Length == 0)
        {
            PrintUsage();
            return 1;
        }

        var command = args[0];
        return command switch
        {
            "--version" or "-v" or "version" => PrintVersion(),
            "--help" or "-h" or "help" => PrintUsage(),
            _ => Unimplemented(command),
        };
    }

    private static int PrintVersion()
    {
        var version = typeof(Program).Assembly.GetName().Version?.ToString() ?? "unknown";
        Console.WriteLine($"prism (Skyline-PRISM C#) {version}");
        return 0;
    }

    private static int PrintUsage()
    {
        Console.WriteLine("Skyline-PRISM CLI");
        Console.WriteLine();
        Console.WriteLine("Usage: prism <command> [options]");
        Console.WriteLine();
        Console.WriteLine("Commands:");
        Console.WriteLine("  run              Run the full PRISM pipeline");
        Console.WriteLine("  merge            Merge Skyline reports into a unified parquet");
        Console.WriteLine("  qc               Regenerate the QC report from an output directory");
        Console.WriteLine("  config-template  Emit an annotated configuration template");
        Console.WriteLine("  compare          Compare rollup methods");
        Console.WriteLine("  version          Print the version");
        return 0;
    }

    private static int Unimplemented(string command)
    {
        Console.Error.WriteLine($"Command '{command}' is not implemented yet.");
        return 2;
    }
}
