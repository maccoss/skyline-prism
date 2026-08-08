# Skyline-PRISM (C#) dotnet-v26.10.1 Release Notes

A patch for one bug in 26.10.0 that made the Skyline tool unusable. **Anyone on 26.10.0 should update.**

## Bug Fixes

- **The Dynamic Range tab produced endless error dialogs.** Opening it against a connected Skyline
  raised `The calling thread cannot access this object because a different thread owns it` - and then
  again every 0.75 seconds, one dialog per tick, until the tool was killed.

  Following Skyline's selection (new in 26.10.0) builds its lookup index on a worker thread so a
  150,000-entry index cannot freeze the window. One method on that path still read the level from the
  combo box, and WPF controls belong to the UI thread. It now takes the level as a parameter and
  touches no control at all.

  The dialog *storm* was a second, separate fault: the poll runs on a timer from an `async void`
  method, where an escaping exception becomes an unhandled application exception - so one failure
  repeated forever. Every such handler in the tool now contains its own failures, and the selection
  poll gives up after three consecutive ones rather than retrying a broken state. Following the
  selection is a convenience; it should never have been able to interrupt you, let alone repeatedly.

  Both rules are now enforced by tests that fail on the exact code that shipped: nothing reachable
  from that worker thread may touch a control, and no `async void` handler may be able to throw.

## New Features

- **The Spectrum density tab no longer offers Skyline's saved isolation schemes.** They are generic
  templates - SWATH (15 m/z), SWATH (25 m/z), SWATH (VW 64) - unrelated to how any given data was
  acquired, and binning a 3 Th acquisition on a 25 Th grid gives a map that looks plausible and is
  wrong. The fallback is now a built-in **Astral 3 Th, 400-900 m/z**, used when an acquisition's own
  windows cannot be read. Its edges are deliberately not round (400.4319, stepping by ~3.0014 Th),
  because that is where a real forbidden-zone scheme places them. A document's own windows still win
  wherever they can be read.
