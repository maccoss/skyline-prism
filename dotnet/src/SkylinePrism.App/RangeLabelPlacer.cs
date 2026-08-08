using System;
using System.Collections.Generic;
using ScottPlot;
using SkylinePrism.Core.Qc;

namespace SkylinePrism.App;

/// <summary>
/// Chooses where each label sits so they collide as little as possible.
/// <para>
/// Every label used to go up-and-right at a fixed offset, which is fine for one and unreadable
/// for twenty on a dense curve - they landed on top of each other. Each label now tries a ring
/// of candidate positions around its point (above and BELOW, left and right, then further out)
/// and takes the first that clears the labels already placed; if none is clear it takes the
/// least-overlapping one, because dropping a label the user asked for is worse than crowding.
/// </para>
/// <para>
/// Boxes are estimated rather than measured: ScottPlot does not lay text out until it renders,
/// and the estimate only has to be good enough to keep labels apart. Sizes are computed in
/// pixels and converted to data units through the plot's own extents, so the spacing looks the
/// same whatever the abundance range or window size.
/// </para>
/// </summary>
internal sealed class RangeLabelPlacer
{
    private const double FontSize = 16;
    private const double CharWidthFactor = 0.58;   // mean glyph width / point size, bold sans
    private const double LineHeightFactor = 1.35;

    private readonly List<Box> _placed = new();
    private readonly double _xPerPx;
    private readonly double _yPerPx;
    private readonly double _xSpan;
    private readonly double _yMin;
    private readonly double _yMax;

    public RangeLabelPlacer(
        int nPoints, double yTop, double yBottom, double widthDip, double heightDip)
    {
        _xSpan = Math.Max(1, nPoints);
        _yMax = yTop;
        _yMin = yBottom;
        var ySpan = Math.Max(0.5, yTop - yBottom);

        // Fall back to a typical size when the control has not been measured yet (first render).
        var wPx = widthDip > 50 ? widthDip : 900;
        var hPx = heightDip > 50 ? heightDip : 600;
        _xPerPx = _xSpan / wPx;
        _yPerPx = ySpan / hPx;
    }

    public Placement Place(AbundanceEntry entry)
    {
        var w = entry.Label.Length * FontSize * CharWidthFactor * _xPerPx;
        var h = FontSize * LineHeightFactor * _yPerPx;

        Placement? best = null;
        var bestCost = double.MaxValue;

        foreach (var candidate in Candidates(entry, w, h))
        {
            var cost = Cost(candidate.Box);
            if (cost <= 0)
            {
                _placed.Add(candidate.Box);
                return candidate;
            }
            if (cost < bestCost)
            {
                bestCost = cost;
                best = candidate;
            }
        }

        var chosen = best!.Value;
        _placed.Add(chosen.Box);
        return chosen;
    }

    /// <summary>
    /// Positions to try, nearest first: the four diagonals at one leader length, then the same
    /// four further out. Down is as good as up - there is as much empty space under the curve as
    /// over it, and insisting on "above" is what made them stack.
    /// </summary>
    private IEnumerable<Placement> Candidates(AbundanceEntry entry, double w, double h)
    {
        var leadX = _xSpan * 0.030;
        var leadY = (_yMax - _yMin) * 0.040;

        foreach (var scale in new[] { 1.0, 1.9, 3.0 })
        foreach (var (sx, sy) in new[] { (1, 1), (-1, 1), (1, -1), (-1, -1) })
        {
            var x = entry.Rank + sx * leadX * scale;
            var y = entry.Log10Abundance + sy * leadY * scale;

            // Keep the label inside the plotted region; a label off-canvas is not a placement.
            if (x < 0 || x > _xSpan)
                continue;
            var box = sx > 0
                ? new Box(x, x + w, sy > 0 ? y : y - h, sy > 0 ? y + h : y)
                : new Box(x - w, x, sy > 0 ? y : y - h, sy > 0 ? y + h : y);

            yield return new Placement(
                x, y, box,
                (sx > 0, sy > 0) switch
                {
                    (true, true) => Alignment.LowerLeft,
                    (false, true) => Alignment.LowerRight,
                    (true, false) => Alignment.UpperLeft,
                    _ => Alignment.UpperRight,
                });
        }
    }

    /// <summary>Total overlap area with the labels already placed; 0 means clear.</summary>
    private double Cost(Box box)
    {
        var cost = 0.0;
        foreach (var other in _placed)
        {
            var overlapX = Math.Min(box.X2, other.X2) - Math.Max(box.X1, other.X1);
            var overlapY = Math.Min(box.Y2, other.Y2) - Math.Max(box.Y1, other.Y1);
            if (overlapX > 0 && overlapY > 0)
                cost += (overlapX / _xPerPx) * (overlapY / _yPerPx); // in px^2, so x and y compare
        }
        return cost;
    }

    internal readonly record struct Box(double X1, double X2, double Y1, double Y2);

    internal readonly record struct Placement(double X, double Y, Box Box, Alignment Alignment);
}
