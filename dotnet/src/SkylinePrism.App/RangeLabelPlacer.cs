using System;
using System.Collections.Generic;
using System.Linq;
using ScottPlot;
using SkylinePrism.Core.Qc;

namespace SkylinePrism.App;

/// <summary>
/// Chooses where the Dynamic Range plot's labels sit, all of them together, so that leader lines
/// do not cross, labels do not overlap, and neither labels nor leaders sit on the curve.
/// <para>
/// The first version placed labels one at a time, each at the first spot around its point that
/// cleared the labels already placed. That kept labels off each other, but with twenty-odd labels
/// on the steep top of the curve their leaders crossed one another and ran through other labels.
/// </para>
/// <para>
/// Each label now has a few hundred candidate spots on every side of its point - a ring of
/// directions and distances, and a grid of positions beside it, which is what lets a crowd stack
/// into a column - and a layout is scored as a whole. In order of how bad they are: two leaders
/// crossing or a leader through a label; two labels overlapping; a label covering the curve's
/// points; a leader running across them; and finally leader length, superlinear so that two medium
/// leaders beat one short and one very long. The ratios matter more than the values: priced alike,
/// an overlap and a leader across the curve were traded for each other, and the "best" layout was a
/// pile of overlapping labels.
/// </para>
/// <para>
/// Greedy placement settles in a local minimum that depends on who chooses first - labels on the
/// steep top of the curve took the open space to the right and left the crowd at its bend nothing
/// but spots across the curve. So the greedy layout is improved by simulated annealing, the method
/// Christensen, Marks and Shieber (1995) found best for map labelling: random single-label moves,
/// accepting a worse layout with a probability that falls as the run cools, which lets it climb out
/// of those minima. Seeded, so a plot always comes out the same. A label is never dropped: with no
/// clear spot it takes the least bad one, because losing a label the user asked for is worse.
/// </para>
/// <para>
/// Work is done in device-independent pixels of the data area, so spacing looks the same whatever
/// the abundance range or window size. Text boxes are estimated rather than measured - ScottPlot
/// does not lay text out until it renders - which only has to be good enough to keep them apart.
/// </para>
/// </summary>
internal sealed class RangeLabelPlacer
{
    private const double FontSize = 16;
    private const double CharWidthFactor = 0.62;   // mean glyph width / point size, bold sans, generous
    private const double LineHeightFactor = 1.35;
    private const double EstimatePaddingPx = 8;    // the label's background and border, estimated
    private const double MeasuredPaddingPx = 4;    // the same, around text the caller measured

    // The ring: directions in degrees counter-clockwise from +x (nearest the up-right diagonal
    // first, so a tie goes to the conventional spot) at a range of distances.
    private static readonly double[] AnglesDeg = { 45, 30, 60, 15, 75, 0, 90, 225, 210, 240, 195, 255, 180, 270, 120, 150, 300, 330 };
    private static readonly double[] RadiiPx = { 28, 45, 70, 100, 140, 190, 250, 320 };

    // The grid: left-aligned to the right of the point and right-aligned to its left, every half
    // line height up and down.
    private static readonly double[] GridDxPx = { 30, 60, 95, 135, 180, 230, 290, 360 };
    private const int GridHalfSteps = 20;

    // Costs, worst first (see the remarks on why the ratios matter).
    private const double CrossingCost = 1e7;
    private const double OverlapCost = 1e7;
    private const double OnCurveCost = 1e5;
    private const double LeaderOverCurveCost = 1e5;

    // Measured on the 26-label plasma list: 4,000 moves a label settles to no crossings, overlaps
    // or labels on the curve in about half a second; 1,500 left one overlap and one label on the
    // curve. The total is capped, and beyond the label limit only greedy and local moves run,
    // because each move costs time in proportion to the label count and this runs on the UI thread.
    private const int AnnealMovesPerLabel = 4000;
    private const int AnnealMaxMoves = 150_000;
    private const int AnnealLabelLimit = 100;
    private const int PairShortlist = 60;          // spots per label tried in a joint move
    private const double SwapShare = 0.2;          // of annealing moves: swaps between neighbours
    private const double LocalShare = 0.4;         // ... and moves to a spot near the current one
    private const double LocalMovePx = 60;

    private readonly double[] _curveX;   // px, non-decreasing
    private readonly double[] _curveY;   // px, non-increasing
    private readonly double _xMin, _yMin, _xPerPx, _yPerPx, _widthPx, _heightPx, _markerRadiusPx;
    private readonly IReadOnlyList<Box> _obstacles;
    private readonly Func<string, (double Width, double Height)>? _measureText;

    /// <summary>
    /// A label's box: its text as the caller measured it, plus the background's padding - or, with
    /// no measurer, an estimate from the character count. The estimate is generous (an "ALB" label
    /// is 30 x 21 px of text, estimated at 46 x 30 with padding), and on a crowded pane those extra
    /// pixels are the difference between room and none, so the app measures.
    /// </summary>
    private (double W, double H) LabelSize(string text)
    {
        if (_measureText is null)
            return (text.Length * FontSize * CharWidthFactor + 2 * EstimatePaddingPx, FontSize * LineHeightFactor + EstimatePaddingPx);
        var (w, h) = _measureText(text);
        return (w + 2 * MeasuredPaddingPx, h + 2 * MeasuredPaddingPx);
    }

    /// <param name="curve">Every plotted point, in any order: what labels and leaders must not cover.</param>
    /// <param name="xMin">Left axis limit, in data units (rank).</param>
    /// <param name="xMax">Right axis limit.</param>
    /// <param name="yMin">Bottom axis limit, in data units (log10 abundance).</param>
    /// <param name="yMax">Top axis limit.</param>
    /// <param name="widthPx">The data area's width, in device-independent pixels.</param>
    /// <param name="heightPx">The data area's height.</param>
    /// <param name="markerRadiusPx">Half a point's drawn size: how close a label may come to a point.</param>
    /// <param name="obstacles">Further boxes to keep clear, in data-area pixels (the legend).</param>
    public RangeLabelPlacer(
        IReadOnlyList<AbundanceEntry> curve, double xMin, double xMax, double yMin, double yMax,
        double widthPx, double heightPx, double markerRadiusPx, IReadOnlyList<Box>? obstacles = null,
        Func<string, (double Width, double Height)>? measureText = null)
    {
        _widthPx = widthPx > 50 ? widthPx : 900;
        _heightPx = heightPx > 50 ? heightPx : 600;
        _xMin = xMin;
        _yMin = yMin;
        _xPerPx = Math.Max(xMax - xMin, 1e-9) / _widthPx;
        _yPerPx = Math.Max(yMax - yMin, 1e-9) / _heightPx;
        _markerRadiusPx = Math.Max(0, markerRadiusPx);
        _obstacles = obstacles ?? Array.Empty<Box>();
        _measureText = measureText;

        var sorted = curve.OrderBy(e => e.Rank).ToArray();
        _curveX = sorted.Select(e => ToPxX(e.Rank)).ToArray();
        _curveY = sorted.Select(e => ToPxY(e.Log10Abundance)).ToArray();
    }

    /// <summary>
    /// Place every label in <paramref name="labels"/>, which come in order of priority (the first
    /// chooses first in the greedy start). Returns one placement per label, in the same order.
    /// </summary>
    public IReadOnlyList<Placement> PlaceAll(IReadOnlyList<AbundanceEntry> labels)
    {
        var n = labels.Count;
        if (n == 0)
            return Array.Empty<Placement>();
        var points = labels.Select(e => (X: ToPxX(e.Rank), Y: ToPxY(e.Log10Abundance))).ToArray();
        var candidates = labels
            .Select((e, i) => Candidates(points[i], LabelSize(e.Label), withGrid: n <= AnnealLabelLimit).ToArray())
            .ToArray();

        var chosen = new int[n];
        var placed = new bool[n];
        for (var i = 0; i < n; i++)
        {
            chosen[i] = BestIndex(i, candidates, chosen, placed, points);
            placed[i] = true;
        }

        // Two schedules, the cheaper layout kept: starting hot finds the layouts that take several
        // worsening moves to reach (it did better on a small, crowded pane), starting cool keeps
        // more of a good greedy start (it did better on a large crowd). Neither won everywhere.
        if (n <= AnnealLabelLimit)
        {
            var cool = (int[])chosen.Clone();
            var hotTotal = Anneal(candidates, chosen, points, tStart: OverlapCost * 3);
            var coolTotal = Anneal(candidates, cool, points, tStart: OnCurveCost / 3);
            if (coolTotal < hotTotal)
                Array.Copy(cool, chosen, n);
        }
        Polish(candidates, chosen, points);
        if (n <= AnnealLabelLimit)
        {
            RepairPairs(candidates, chosen, points);
            Polish(candidates, chosen, points);
        }

        return chosen.Select((k, i) => ToPlacement(candidates[i][k])).ToArray();
    }

    /// <summary>
    /// Simulated annealing over the candidates, returning the best total cost it reached: move one
    /// random label to one random candidate,
    /// keep the move if it lowers the total cost, or with probability exp(-increase / T) if not.
    /// A move changes only the moved label's own terms, so its effect on the total is the change in
    /// that label's cost. T falls geometrically from <paramref name="tStart"/> to below one pixel of
    /// leader, so late in the run only improvements survive. The best layout seen is kept.
    /// Moves are drawn only from spots whose label is clear of the curve (all of them, if none is):
    /// most spots put the label on it, and a run spent proposing those did not settle. A spot whose
    /// leader merely crosses the curve stays in: on a small pane those are often all there is.
    /// </summary>
    private double Anneal(Candidate[][] candidates, int[] chosen, (double X, double Y)[] points, double tStart)
    {
        var n = chosen.Length;
        var all = Enumerable.Repeat(true, n).ToArray();
        var rng = new Random(20261112);
        var moves = Math.Min(AnnealMovesPerLabel * n, AnnealMaxMoves);
        const double tEnd = 0.5;
        var viable = candidates.Select(cs =>
        {
            var clear = Enumerable.Range(0, cs.Length).Where(k => cs[k].FixedCost < OnCurveCost).ToArray();
            return clear.Length > 0 ? clear : Enumerable.Range(0, cs.Length).ToArray();
        }).ToArray();
        var cooling = Math.Pow(tEnd / tStart, 1.0 / moves);

        // A label's cost is its own terms plus every pair it is in, so summing label costs counts
        // each pair twice: the total is the own terms plus half the rest.
        var total = Enumerable.Range(0, n).Sum(i =>
        {
            var own = candidates[i][chosen[i]].FixedCost;
            return own + (Cost(i, candidates[i][chosen[i]], candidates, chosen, all, points) - own) / 2;
        });
        // Each label's nearest few by point: the labels a swap is tried with.
        var neighbours = Enumerable.Range(0, n)
            .Select(i => Enumerable.Range(0, n).Where(j => j != i)
                .OrderBy(j => Sq(points[i].X - points[j].X) + Sq(points[i].Y - points[j].Y))
                .Take(6).ToArray())
            .ToArray();

        var best = (int[])chosen.Clone();
        var bestTotal = total;
        var t = tStart;
        for (var m = 0; m < moves; m++, t *= cooling)
        {
            var kind = rng.NextDouble();
            double delta;
            int i, j = -1, ki, kj = -1, oldI, oldJ = -1;
            if (kind < SwapShare && n > 1)
            {
                // Swap: each of two neighbouring labels to its spot nearest the other's anchor. A
                // crossing between two neighbours' leaders is always undone by exchanging their
                // ends, and a crowd on nearly one point can only fan out in the right order this way:
                // by single moves, every step on the way crosses a leader.
                i = rng.Next(n);
                j = neighbours[i][rng.Next(neighbours[i].Length)];
                (oldI, oldJ) = (chosen[i], chosen[j]);
                var (ci, cj) = (candidates[i][oldI], candidates[j][oldJ]);
                ki = NearestAnchor(candidates[i], viable[i], cj.AnchorX, cj.AnchorY);
                kj = NearestAnchor(candidates[j], viable[j], ci.AnchorX, ci.AnchorY);
                if (ki == oldI && kj == oldJ)
                    continue;
                var same = points[i] == points[j];
                var before = Cost(i, ci, candidates, chosen, all, points) + Cost(j, cj, candidates, chosen, all, points)
                             - PairCost(ci, cj, same);
                (chosen[i], chosen[j]) = (ki, kj);
                var (ni, nj) = (candidates[i][ki], candidates[j][kj]);
                var after = Cost(i, ni, candidates, chosen, all, points) + Cost(j, nj, candidates, chosen, all, points)
                            - PairCost(ni, nj, same);
                delta = after - before;
            }
            else
            {
                // One label to another spot: half the time one near where it is now, which is how
                // a nearly right layout gets finished.
                i = rng.Next(n);
                oldI = chosen[i];
                ki = kind < SwapShare + LocalShare ? NearbySpot(candidates[i], viable[i], oldI, rng) : viable[i][rng.Next(viable[i].Length)];
                if (ki == oldI)
                    continue;
                // Only label i's own terms and its pairs change, so this is the change in the total.
                delta = Cost(i, candidates[i][ki], candidates, chosen, all, points)
                        - Cost(i, candidates[i][oldI], candidates, chosen, all, points);
                chosen[i] = ki;
            }

            if (delta <= 0 || rng.NextDouble() < Math.Exp(-delta / t))
            {
                total += delta;
                if (total < bestTotal - 1e-9)
                {
                    bestTotal = total;
                    Array.Copy(chosen, best, n);
                }
            }
            else
            {
                chosen[i] = oldI;
                if (j >= 0)
                    chosen[j] = oldJ;
            }
        }
        Array.Copy(best, chosen, n);
        return bestTotal;
    }

    /// <summary>Of the <paramref name="allowed"/> spots, the one whose anchor is nearest (x, y).</summary>
    private static int NearestAnchor(Candidate[] spots, int[] allowed, double x, double y)
    {
        var best = allowed[0];
        var bestD = double.MaxValue;
        foreach (var k in allowed)
        {
            var d = Sq(spots[k].AnchorX - x) + Sq(spots[k].AnchorY - y);
            if (d < bestD)
            {
                bestD = d;
                best = k;
            }
        }
        return best;
    }

    /// <summary>A random allowed spot within <see cref="LocalMovePx"/> of the current anchor, if one turns up in a few tries.</summary>
    private static int NearbySpot(Candidate[] spots, int[] allowed, int current, Random rng)
    {
        var (x, y) = (spots[current].AnchorX, spots[current].AnchorY);
        var k = current;
        for (var attempt = 0; attempt < 12; attempt++)
        {
            k = allowed[rng.Next(allowed.Length)];
            if (Sq(spots[k].AnchorX - x) + Sq(spots[k].AnchorY - y) < LocalMovePx * LocalMovePx)
                return k;
        }
        return k;
    }

    /// <summary>Move each label to its best spot given all the others, until no move helps.</summary>
    private void Polish(Candidate[][] candidates, int[] chosen, (double X, double Y)[] points)
    {
        var all = Enumerable.Repeat(true, chosen.Length).ToArray();
        for (var pass = 0; pass < 25; pass++)
        {
            var moved = false;
            for (var i = 0; i < chosen.Length; i++)
            {
                var current = Cost(i, candidates[i][chosen[i]], candidates, chosen, all, points);
                var best = BestIndex(i, candidates, chosen, all, points);
                if (Cost(i, candidates[i][best], candidates, chosen, all, points) < current - 1e-6)
                {
                    chosen[i] = best;
                    moved = true;
                }
            }
            if (!moved)
                break;
        }
    }

    private int BestIndex(int i, Candidate[][] candidates, int[] chosen, bool[] placed, (double X, double Y)[] points)
    {
        var best = 0;
        var bestCost = double.MaxValue;
        for (var k = 0; k < candidates[i].Length; k++)
        {
            var cost = Cost(i, candidates[i][k], candidates, chosen, placed, points);
            // Spots at the same distance tie; without a tolerance, rounding in the leader length
            // decided which, and a lone label went down-left instead of the conventional up-right.
            if (cost < bestCost - 1e-9)
            {
                bestCost = cost;
                best = k;
            }
        }
        return best;
    }

    /// <summary>What label <paramref name="i"/> at <paramref name="c"/> costs, given the others placed.</summary>
    private static double Cost(
        int i, Candidate c, Candidate[][] candidates, int[] chosen, bool[] placed, (double X, double Y)[] points)
    {
        var cost = c.FixedCost;
        for (var j = 0; j < chosen.Length; j++)
            if (j != i && placed[j])
                cost += PairCost(c, candidates[j][chosen[j]], points[i] == points[j]);
        return cost;
    }

    /// <summary>What two labels cost each other: overlapping, crossing leaders, or a leader through the other.</summary>
    private static double PairCost(Candidate a, Candidate b, bool samePoint)
    {
        var cost = 0.0;
        if (Overlaps(a.Box, b.Box))
            cost += OverlapCost + OverlapArea(a.Box, b.Box);
        // Leaders from the same point (the selection and a list member) cannot cross each other.
        if (!samePoint && SegmentsCross(a.PointX, a.PointY, a.AnchorX, a.AnchorY, b.PointX, b.PointY, b.AnchorX, b.AnchorY))
            cost += CrossingCost;
        if (SegmentHitsBox(a.PointX, a.PointY, a.AnchorX, a.AnchorY, b.Box))
            cost += CrossingCost;
        if (SegmentHitsBox(b.PointX, b.PointY, b.AnchorX, b.AnchorY, a.Box))
            cost += CrossingCost;
        return cost;
    }

    /// <summary>
    /// For every two labels still in conflict - overlapping, or a leader crossing or running through
    /// the other - try moving both at once, over each one's cheapest spots, and keep the best pair
    /// if it lowers the total. Single-label moves cannot undo a conflict that needs both labels to
    /// move, which is the usual leftover after annealing: two neighbours whose leaders must swap
    /// sides of each other.
    /// </summary>
    private void RepairPairs(Candidate[][] candidates, int[] chosen, (double X, double Y)[] points)
    {
        var n = chosen.Length;
        var all = Enumerable.Repeat(true, n).ToArray();
        var shortlist = candidates
            .Select(cs => Enumerable.Range(0, cs.Length).OrderBy(k => cs[k].FixedCost).Take(PairShortlist).ToArray())
            .ToArray();
        for (var round = 0; round < 5; round++)
        {
            var improved = false;
            for (var i = 0; i < n; i++)
                for (var j = i + 1; j < n; j++)
                {
                    var same = points[i] == points[j];
                    if (PairCost(candidates[i][chosen[i]], candidates[j][chosen[j]], same) < OverlapCost)
                        continue;
                    var (oldI, oldJ) = (chosen[i], chosen[j]);
                    double Total(int ki, int kj)
                    {
                        chosen[i] = ki;
                        chosen[j] = kj;
                        return Cost(i, candidates[i][ki], candidates, chosen, all, points)
                               + Cost(j, candidates[j][kj], candidates, chosen, all, points)
                               - PairCost(candidates[i][ki], candidates[j][kj], same);
                    }
                    var bestTotal = Total(oldI, oldJ);
                    var (bestI, bestJ) = (oldI, oldJ);
                    foreach (var ki in shortlist[i])
                        foreach (var kj in shortlist[j])
                        {
                            var total = Total(ki, kj);
                            if (total < bestTotal - 1e-6)
                                (bestTotal, bestI, bestJ) = (total, ki, kj);
                        }
                    (chosen[i], chosen[j]) = (bestI, bestJ);
                    if (bestI != oldI || bestJ != oldJ)
                        improved = true;
                }
            if (!improved)
                break;
        }
    }

    /// <summary>
    /// The spots to try around <paramref name="point"/>, each with the box it implies; the box
    /// extends AWAY from the point, so the leader meets its near edge or corner. Spots whose box
    /// leaves the data area are dropped; if that leaves nothing (a tiny plot), all are kept.
    /// </summary>
    private IEnumerable<Candidate> Candidates((double X, double Y) point, (double W, double H) size, bool withGrid)
    {
        var all = new List<Candidate>();
        foreach (var r in RadiiPx)
            foreach (var deg in AnglesDeg)
            {
                var rad = deg * Math.PI / 180;
                var (cos, sin) = (Math.Cos(rad), Math.Sin(rad));
                var horizontal = cos > 0.4 ? HorizontalAlignment.Left : cos < -0.4 ? HorizontalAlignment.Right : HorizontalAlignment.Center;
                var vertical = sin > 0.4 ? VerticalAlignment.Bottom : sin < -0.4 ? VerticalAlignment.Top : VerticalAlignment.Middle;
                all.Add(Make(point, size, point.X + r * cos, point.Y + r * sin, horizontal, vertical));
            }
        if (withGrid)
            foreach (var dx in GridDxPx)
                for (var step = -GridHalfSteps; step <= GridHalfSteps; step++)
                {
                    var dy = step * size.H / 2;
                    all.Add(Make(point, size, point.X + dx, point.Y + dy, HorizontalAlignment.Left, VerticalAlignment.Middle));
                    all.Add(Make(point, size, point.X - dx, point.Y + dy, HorizontalAlignment.Right, VerticalAlignment.Middle));
                }

        var inside = all.Where(c => c.Box.X1 >= 0 && c.Box.X2 <= _widthPx && c.Box.Y1 >= 0 && c.Box.Y2 <= _heightPx).ToList();
        return inside.Count > 0 ? inside : all;
    }

    private Candidate Make(
        (double X, double Y) point, (double W, double H) size, double ax, double ay,
        HorizontalAlignment horizontal, VerticalAlignment vertical)
    {
        var x1 = horizontal switch
        {
            HorizontalAlignment.Left => ax,
            HorizontalAlignment.Right => ax - size.W,
            _ => ax - size.W / 2,
        };
        var y1 = vertical switch
        {
            VerticalAlignment.Bottom => ay,
            VerticalAlignment.Top => ay - size.H,
            _ => ay - size.H / 2,
        };
        var box = new Box(x1, x1 + size.W, y1, y1 + size.H);

        // Everything that does not depend on the other labels, worked out once per candidate.
        var length = Math.Sqrt(Sq(ax - point.X) + Sq(ay - point.Y));
        var fixedCost = length + length * length / 200;
        if (CoversCurve(box))
            fixedCost += OnCurveCost;
        if (LeaderRunsOverCurve(point.X, point.Y, ax, ay))
            fixedCost += LeaderOverCurveCost;
        foreach (var o in _obstacles)
            if (Overlaps(box, o))
                fixedCost += OverlapCost;
        return new Candidate(point.X, point.Y, ax, ay, box, AlignmentOf(horizontal, vertical), fixedCost);
    }

    internal static bool Overlaps(Box a, Box b) =>
        Math.Min(a.X2, b.X2) > Math.Max(a.X1, b.X1) && Math.Min(a.Y2, b.Y2) > Math.Max(a.Y1, b.Y1);

    private static double OverlapArea(Box a, Box b) =>
        (Math.Min(a.X2, b.X2) - Math.Max(a.X1, b.X1)) * (Math.Min(a.Y2, b.Y2) - Math.Max(a.Y1, b.Y1));

    /// <summary>
    /// True when <paramref name="box"/> (in data-area pixels) comes within a marker's radius of any
    /// plotted point. The curve only falls, so over the box's x range (widened by the radius) its
    /// points span from the height at the right end to the height at the left end; two binary
    /// searches find both.
    /// </summary>
    internal bool CoversCurve(Box box)
    {
        if (_curveX.Length == 0)
            return false;
        var lo = LowerBound(_curveX, box.X1 - _markerRadiusPx);
        var hi = LowerBound(_curveX, box.X2 + _markerRadiusPx + 1e-9) - 1;
        if (lo > hi)
            return false;
        var top = _curveY[lo];      // leftmost in range: highest
        var bottom = _curveY[hi];   // rightmost in range: lowest
        return box.Y1 - _markerRadiusPx < top && box.Y2 + _markerRadiusPx > bottom;
    }

    /// <summary>
    /// True when a leader from a point to its label (in data-area pixels) passes over the curve's
    /// other points - which hides them and reads as pointing at them. Sampled every few pixels from
    /// beyond the point's own marker, each sample tested as a small box against the curve.
    /// </summary>
    internal bool LeaderRunsOverCurve(double x0, double y0, double x1, double y1)
    {
        var length = Math.Sqrt(Sq(x1 - x0) + Sq(y1 - y0));
        const double margin = 2;
        for (var d = _markerRadiusPx * 2 + margin; d <= length; d += 3)
        {
            var x = x0 + d / length * (x1 - x0);
            var y = y0 + d / length * (y1 - y0);
            if (CoversCurve(new Box(x - margin, x + margin, y - margin, y + margin)))
                return true;
        }
        return false;
    }

    private static int LowerBound(double[] xs, double x)
    {
        int lo = 0, hi = xs.Length;
        while (lo < hi)
        {
            var mid = (lo + hi) / 2;
            if (xs[mid] < x)
                lo = mid + 1;
            else
                hi = mid;
        }
        return lo;
    }

    /// <summary>True when the two segments properly cross (touching at an end does not count).</summary>
    internal static bool SegmentsCross(
        double ax, double ay, double bx, double by, double cx, double cy, double dx, double dy)
    {
        var d1 = Orientation(cx, cy, dx, dy, ax, ay);
        var d2 = Orientation(cx, cy, dx, dy, bx, by);
        var d3 = Orientation(ax, ay, bx, by, cx, cy);
        var d4 = Orientation(ax, ay, bx, by, dx, dy);
        return d1 * d2 < 0 && d3 * d4 < 0;
    }

    private static double Orientation(double ax, double ay, double bx, double by, double cx, double cy) =>
        (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);

    /// <summary>True when the segment passes through the inside of <paramref name="box"/> (Liang-Barsky).</summary>
    internal static bool SegmentHitsBox(double x0, double y0, double x1, double y1, Box box)
    {
        double t0 = 0, t1 = 1;
        var dx = x1 - x0;
        var dy = y1 - y0;
        foreach (var (p, q) in new[] { (-dx, x0 - box.X1), (dx, box.X2 - x0), (-dy, y0 - box.Y1), (dy, box.Y2 - y0) })
        {
            if (Math.Abs(p) < 1e-12)
            {
                if (q <= 0)
                    return false;
                continue;
            }
            var t = q / p;
            if (p < 0)
                t0 = Math.Max(t0, t);
            else
                t1 = Math.Min(t1, t);
            if (t0 >= t1)
                return false;
        }
        return t1 - t0 > 1e-9;
    }

    private static double Sq(double v) => v * v;

    private double ToPxX(double x) => (x - _xMin) / _xPerPx;
    private double ToPxY(double y) => (y - _yMin) / _yPerPx;

    private Placement ToPlacement(Candidate c) => new(
        _xMin + c.AnchorX * _xPerPx, _yMin + c.AnchorY * _yPerPx,
        new Box(_xMin + c.Box.X1 * _xPerPx, _xMin + c.Box.X2 * _xPerPx,
            _yMin + c.Box.Y1 * _yPerPx, _yMin + c.Box.Y2 * _yPerPx),
        c.Alignment, c.Box, c.PointX, c.PointY, c.AnchorX, c.AnchorY);

    private static Alignment AlignmentOf(HorizontalAlignment h, VerticalAlignment v) => (v, h) switch
    {
        (VerticalAlignment.Bottom, HorizontalAlignment.Left) => Alignment.LowerLeft,
        (VerticalAlignment.Bottom, HorizontalAlignment.Center) => Alignment.LowerCenter,
        (VerticalAlignment.Bottom, HorizontalAlignment.Right) => Alignment.LowerRight,
        (VerticalAlignment.Middle, HorizontalAlignment.Left) => Alignment.MiddleLeft,
        (VerticalAlignment.Middle, HorizontalAlignment.Center) => Alignment.MiddleCenter,
        (VerticalAlignment.Middle, HorizontalAlignment.Right) => Alignment.MiddleRight,
        (VerticalAlignment.Top, HorizontalAlignment.Left) => Alignment.UpperLeft,
        (VerticalAlignment.Top, HorizontalAlignment.Center) => Alignment.UpperCenter,
        _ => Alignment.UpperRight,
    };

    private enum HorizontalAlignment { Left, Center, Right }

    private enum VerticalAlignment { Bottom, Middle, Top }

    private readonly record struct Candidate(
        double PointX, double PointY, double AnchorX, double AnchorY, Box Box, Alignment Alignment,
        double FixedCost);

    /// <summary>A rectangle: in data units on a <see cref="Placement"/>, in data-area pixels for an obstacle.</summary>
    internal readonly record struct Box(double X1, double X2, double Y1, double Y2);

    /// <summary>
    /// Where a label's anchor goes, in data units, with its box and the text alignment at the
    /// anchor - and, for checking a layout, the same box and the leader's ends in data-area pixels.
    /// </summary>
    internal readonly record struct Placement(
        double X, double Y, Box Box, Alignment Alignment,
        Box BoxPx, double PointXPx, double PointYPx, double AnchorXPx, double AnchorYPx);
}
