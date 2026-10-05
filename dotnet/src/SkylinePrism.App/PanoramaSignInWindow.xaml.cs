using System;
using System.Windows;
using SkylinePrism.Core.Panorama;

namespace SkylinePrism.App;

/// <summary>
/// Asks for a Panorama sign-in when none saved on this computer works, checks it against the server
/// before accepting it, and saves it as PRISM's own Credential Manager entry when asked to.
/// </summary>
/// <remarks>
/// Only PRISM's entry is ever written. PanoramaBridge's and LabOps's belong to those apps, and PRISM
/// reads them first (<see cref="PanoramaSignIn"/>), so a lab machine that already runs PanoramaBridge
/// never sees this window.
/// </remarks>
public partial class PanoramaSignInWindow : Window
{
    private readonly PanoramaSignIn _signIn;

    private PanoramaSignInWindow(PanoramaSignIn signIn, string why)
    {
        InitializeComponent();
        _signIn = signIn;
        IntroText.Text = why;
        RememberCheck.IsEnabled = signIn.CanSave;
    }

    /// <summary>A client signed in with what the person typed, or null when they canceled.</summary>
    public PanoramaClient? Client { get; private set; }

    public static PanoramaClient? Ask(Window owner, PanoramaSignIn signIn, string why)
    {
        var window = new PanoramaSignInWindow(signIn, why) { Owner = owner };
        return window.ShowDialog() == true ? window.Client : null;
    }

    private void OnHowChanged(object sender, RoutedEventArgs e)
    {
        // ApiKeyChoice is checked in the XAML, so this fires inside InitializeComponent, before the
        // boxes declared after it exist (see XamlInitializationOrderTests).
        if (!IsInitialized)
            return;
        var key = ApiKeyChoice.IsChecked == true;
        ApiKeyBox.IsEnabled = key;
        UserBox.IsEnabled = PasswordBox.IsEnabled = !key;
    }

    private async void OnSignIn(object sender, RoutedEventArgs e)
    {
        PanoramaCredential credential;
        try
        {
            credential = ApiKeyChoice.IsChecked == true
                ? PanoramaCredential.ApiKey(ApiKeyBox.Password)
                : PanoramaCredential.Login(UserBox.Text, PasswordBox.Password);
        }
        catch (ArgumentException)
        {
            StatusText.Text = ApiKeyChoice.IsChecked == true ? "Paste the API key." : "Enter the email and password.";
            return;
        }

        SignInButton.IsEnabled = false;
        StatusText.Text = "Checking with Panorama...";
        var client = new PanoramaClient(_signIn.Server, credential);
        PanoramaException? problem;
        try
        {
            problem = await client.CheckSignInAsync();
        }
        catch (Exception ex)
        {
            // async void: an escaping exception would end the tool, not just this window.
            client.Dispose();
            SignInButton.IsEnabled = true;
            StatusText.Text = ex.Message;
            return;
        }

        SignInButton.IsEnabled = true;
        if (problem is not null)
        {
            client.Dispose();
            StatusText.Text = problem.Message;
            return;
        }

        if (RememberCheck.IsChecked == true && _signIn.CanSave)
        {
            try
            {
                var saved = _signIn.Save(credential);
                client.Dispose();
                client = new PanoramaClient(_signIn.Server, saved);
            }
            catch (Exception ex) when (ex is InvalidOperationException or ArgumentException)
            {
                // Signed in either way; it just will not be remembered. ArgumentException is a secret too
                // long for Credential Manager (over 256 characters), which must not escape this async void.
                MessageBox.Show(this, $"Signed in, but the sign-in could not be saved: {ex.Message}", Title,
                    MessageBoxButton.OK, MessageBoxImage.Warning);
            }
        }

        Client = client;
        DialogResult = true;
    }
}
