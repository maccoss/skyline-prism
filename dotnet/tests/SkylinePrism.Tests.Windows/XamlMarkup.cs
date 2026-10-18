using System;
using System.IO;
using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>Reading MainWindow.xaml as text, for the tests that check its markup directly.</summary>
internal static class XamlMarkup
{
    /// <summary>The whole of MainWindow.xaml.</summary>
    public static string MainWindow => File.ReadAllText(Path.GetFullPath(Path.Combine(
        AppContext.BaseDirectory, "..", "..", "..", "..", "..",
        "src", "SkylinePrism.App", "MainWindow.xaml")));

    /// <summary>The markup of one named element's opening tag, from its '&lt;' to the tag's close.</summary>
    public static string Element(string name)
    {
        var xaml = MainWindow;
        var start = xaml.IndexOf($"x:Name=\"{name}\"", StringComparison.Ordinal);
        Assert.True(start >= 0, $"{name} was not found in MainWindow.xaml - was it renamed?");
        var open = xaml.LastIndexOf('<', start);
        var end = xaml.IndexOf('>', start);
        Assert.True(open >= 0 && end > open, $"could not delimit the {name} element");
        return xaml[open..end];
    }
}
