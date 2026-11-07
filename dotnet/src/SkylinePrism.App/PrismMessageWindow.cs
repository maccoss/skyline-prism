using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;

namespace SkylinePrism.App;

/// <summary>
/// A message the user can act on and COPY: a short statement up top, the full details (a stack
/// trace, a list of files) in a selectable box underneath, a Copy button, and the caller's own
/// buttons - OK, or Retry and Cancel.
/// </summary>
/// <remarks>
/// <para>Replaces <see cref="MessageBox"/> for errors. A message box does copy its text on Ctrl+C,
/// but nothing on it says so, none of it can be selected, and a stack trace overflows it - so what
/// reached us was a screenshot of a stack trace. It also cannot offer Retry, which is the one useful
/// answer to "the file is open in Excel".</para>
/// <para>Built in code rather than XAML: it is a handful of controls, and it has to be safe to raise
/// from the unhandled-exception path, where the less there is to go wrong the better.</para>
/// </remarks>
internal sealed class PrismMessageWindow : Window
{
    private readonly string _copyText;
    private readonly TextBlock _copiedNote;

    internal PrismMessageWindow(string title, string message, string? details, bool isError,
        IReadOnlyList<string> buttons, string? logPath)
    {
        Title = title;
        Width = 680;
        SizeToContent = SizeToContent.Height;
        MaxHeight = SystemParameters.WorkArea.Height * 0.9;
        WindowStartupLocation = WindowStartupLocation.CenterOwner;
        ResizeMode = ResizeMode.CanResizeWithGrip;
        ShowInTaskbar = false;
        Result = buttons.Count > 0 ? buttons[^1] : null; // closing with the X is the last button: Cancel, or OK

        _copyText = title + Environment.NewLine + Environment.NewLine + message
                    + (string.IsNullOrEmpty(details) ? "" : Environment.NewLine + Environment.NewLine + details)
                    + (logPath is null ? "" : Environment.NewLine + Environment.NewLine + "Log: " + logPath);

        var root = new DockPanel { Margin = new Thickness(16), LastChildFill = true };

        // Buttons: Copy (and Open log) on the left, the caller's choices on the right.
        var bar = new DockPanel { Margin = new Thickness(0, 12, 0, 0), LastChildFill = false };
        DockPanel.SetDock(bar, Dock.Bottom);
        var copy = new Button
        {
            Content = "Copy to clipboard", Padding = new Thickness(10, 3, 10, 3),
            ToolTip = "Copy the whole message - including the details and the log location - to paste into an email or an issue.",
        };
        copy.Click += (_, _) => Copy();
        DockPanel.SetDock(copy, Dock.Left);
        bar.Children.Add(copy);
        if (logPath is not null)
        {
            var openLog = new Button
            {
                Content = "Open log", Padding = new Thickness(10, 3, 10, 3), Margin = new Thickness(8, 0, 0, 0),
                ToolTip = logPath,
            };
            openLog.Click += (_, _) => OpenLog(logPath);
            DockPanel.SetDock(openLog, Dock.Left);
            bar.Children.Add(openLog);
        }

        _copiedNote = new TextBlock
        {
            Text = "Copied.", Foreground = Brushes.Gray, VerticalAlignment = VerticalAlignment.Center,
            Margin = new Thickness(8, 0, 0, 0), Visibility = Visibility.Collapsed,
        };
        DockPanel.SetDock(_copiedNote, Dock.Left);
        bar.Children.Add(_copiedNote);

        // Right-docked children stack from the right edge inward, so add them in reverse.
        for (var i = buttons.Count - 1; i >= 0; i--)
        {
            var label = buttons[i];
            var b = new Button
            {
                Content = label, MinWidth = 80, Padding = new Thickness(10, 3, 10, 3),
                Margin = new Thickness(8, 0, 0, 0),
                IsDefault = i == 0,
                IsCancel = i == buttons.Count - 1,
            };
            b.Click += (_, _) =>
            {
                Result = label;
                DialogResult = true;
            };
            DockPanel.SetDock(b, Dock.Right);
            bar.Children.Add(b);
        }

        root.Children.Add(bar);

        var head = new DockPanel { Margin = new Thickness(0, 0, 0, 10) };
        DockPanel.SetDock(head, Dock.Top);
        var icon = new TextBlock
        {
            Text = isError ? "Error" : "Action needed", FontWeight = FontWeights.SemiBold,
            Foreground = isError ? Brushes.Firebrick : Brushes.DarkGoldenrod,
            Margin = new Thickness(0, 0, 12, 0),
        };
        DockPanel.SetDock(icon, Dock.Left);
        head.Children.Add(icon);
        head.Children.Add(new TextBox
        {
            // A read-only TextBox rather than a TextBlock, so the message itself can be selected.
            Text = message, IsReadOnly = true, TextWrapping = TextWrapping.Wrap,
            BorderThickness = new Thickness(0), Background = Brushes.Transparent,
        });
        root.Children.Add(head);

        if (!string.IsNullOrEmpty(details))
        {
            root.Children.Add(new TextBox
            {
                Text = details, IsReadOnly = true, FontFamily = new FontFamily("Consolas"), FontSize = 11,
                TextWrapping = TextWrapping.NoWrap, MinHeight = 120, MaxHeight = 420,
                VerticalScrollBarVisibility = ScrollBarVisibility.Auto,
                HorizontalScrollBarVisibility = ScrollBarVisibility.Auto,
            });
        }

        Content = root;
    }

    /// <summary>Everything the Copy button puts on the clipboard.</summary>
    internal string CopyText => _copyText;

    /// <summary>The button the user chose, or the last one (Cancel / OK) if the window was closed.</summary>
    public string? Result { get; private set; }

    /// <summary>
    /// An error: what went wrong, the full exception to select or copy, and where the log is.
    /// </summary>
    public static void ShowError(Window? owner, string title, string message, Exception? ex, string? logPath) =>
        Show(owner, title, message, ex?.ToString(), isError: true, new[] { "OK" }, logPath);

    /// <summary>Show the window modally and return the label of the button chosen.</summary>
    public static string? Show(Window? owner, string title, string message, string? details, bool isError,
        IReadOnlyList<string> buttons, string? logPath = null)
    {
        var window = new PrismMessageWindow(title, message, details, isError, buttons, logPath);
        if (owner is { IsVisible: true })
            window.Owner = owner;
        else
            window.WindowStartupLocation = WindowStartupLocation.CenterScreen;
        window.ShowDialog();
        return window.Result;
    }

    private void Copy()
    {
        // The clipboard is briefly held by whatever else touched it last (CLIPBRD_E_CANT_OPEN); a
        // couple of quick retries clear nearly every case, and a copy button that silently did
        // nothing would be worse than one that says it failed.
        for (var attempt = 0; attempt < 5; attempt++)
        {
            try
            {
                Clipboard.SetText(_copyText);
                _copiedNote.Text = "Copied.";
                _copiedNote.Visibility = Visibility.Visible;
                return;
            }
            catch (System.Runtime.InteropServices.COMException)
            {
                System.Threading.Thread.Sleep(50);
            }
        }

        _copiedNote.Text = "The clipboard is busy - try again.";
        _copiedNote.Visibility = Visibility.Visible;
    }

    private static void OpenLog(string path)
    {
        try
        {
            if (File.Exists(path))
                Process.Start(new ProcessStartInfo(path) { UseShellExecute = true });
        }
        catch
        {
            // No associated viewer: the path is in the copied text and the tooltip.
        }
    }
}
